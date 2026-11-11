"""End-to-end CMAPSS FD001 two-stage probabilistic RUL training and evaluation.

Loads the FD001 training and test splits, builds standardized sliding
windows and standardized RUL targets, and trains a two-stage probabilistic pipeline through `stagereg`'s
explicit supervised stages:

* Stage 1 trains a deterministic 1D-CNN mean model via `MeanStage`.
* Stage 2 trains an independently-initialized 1D-CNN Gamma model on the
  frozen Stage-1 model's squared residuals via `GammaVarianceStage`.

The frozen Stage-1 point prediction and the Stage-2 Gamma mean are then
combined into a composite `Gaussian` predictive distribution, mapped back
to RUL cycles with `CyclesPredictive`, and evaluated end to end against the official FD001 test split with `stagereg`'s
`MetricSuite`/`GaussianPredictor`/`evaluate_loader` machinery, reporting
RMSE, 95% interval coverage, and the closed-form CRPS. The test split follows a
single-window-per-trajectory protocol: one window per test unit, ending at
its last observed (possibly truncated) cycle, scored against that unit's
ground-truth RUL from `RUL_FD001.txt`.

Run it with:

    uv run --group example-cmapss python examples/jax/cmapss/run.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx

# Allow sibling-module imports (`data`, `model`, `preprocessing`) both when
# run as a script and when loaded from an arbitrary working directory, e.g.
# via `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from data import (
    load_fd001_data,
    load_fd001_test_data,
    load_fd001_test_rul,
)
from model import (
    Cnn1DGammaModel,
    Cnn1DGaussianModel,
    Cnn1DMeanModel,
    CompositeGaussianModel,
)
from plots import LabelledModel, plot_validation_rul_curves
from preprocessing import (
    SensorStandardization,
    TargetStandardization,
    apply_standardization,
    build_last_windows,
    build_windows,
    fit_standardization,
    fit_target_standardization,
    select_lifetime_spanning_units,
    split_by_unit,
)

from stagereg.core.losses import SquaredErrorLoss
from stagereg.core.metric_registry import (
    ClosedFormContinuousRankedProbabilityScore,
    EpochMetric,
    IntervalCoverage,
    RootMeanSquaredError,
)
from stagereg.core.protocols import LoaderFactory
from stagereg.core.tracking import EventSink, TrainingEvent
from stagereg.core.types import Batch, ParameterRole, StageState, TrainingState
from stagereg.jax import (
    GammaVarianceStage,
    Gaussian,
    GaussianPredictor,
    HeldOutValidation,
    MCDropout,
    MeanStage,
    MetricSuite,
    PosteriorPredictive,
    PosteriorStage,
    PosteriorStageOptions,
    Predictor,
    SupervisedStageOptions,
    create_optimizer,
    evaluate_loader,
    make_supervised_loss,
)

_SENSOR_NAMES = [f"sensor_{i}" for i in (11, 12, 4, 7, 15, 20, 21, 2, 17)]

TEST_METRICS: tuple[EpochMetric, ...] = (
    RootMeanSquaredError(),
    IntervalCoverage(level=0.95),
    ClosedFormContinuousRankedProbabilityScore(),
)
"""The staged-only run's test metrics: RMSE, 95% coverage, closed-form CRPS."""


@dataclass(frozen=True)
class CmapssConfig:
    """Configuration for the CMAPSS FD001 point-RUL example.

    Attributes:
        window_length: Number of cycles per sliding window.
        validation_fraction: Fraction of training units held out for
            validation, grouped by whole unit ID.
        batch_size: Number of windows per mini-batch.
        hidden_channels: Convolutional channel width of the mean, joint and
            MC-Dropout models.
        kernel_size: Convolution kernel width along the time axis.
        learning_rate: Adam learning rate for both stages.
        mean_epochs: Number of Stage-1 mean-model training epochs.
        variance_epochs: Number of Stage-2 Gamma-model training epochs.
        gamma_hidden_channels: Convolutional channel width of the Gamma
            model, kept below `hidden_channels` so the variance does not
            overfit the in-sample squared residuals.
        gamma_weight_decay: AdamW decoupled weight decay of the Gamma
            model's training, which shrinks it toward a constant variance.
        joint_epochs: Number of training epochs for the joint model of the
            staged-vs-joint comparison, with or without MC-Dropout.
        dropout_rate: Dropout rate, in `[0, 1)`, of the MC-Dropout models'
            backbones: the joint MC-Dropout network and the staged
            MC-Dropout model's mean network, both trained and predicted with
            dropout active. Dropout is their only regularizer.
        num_draws: Number of dropout-mask sets an MC-Dropout model's posterior
            predictive mixes.
        seed: Base JAX random seed for splitting, model init, and training.
    """

    window_length: int = 30
    validation_fraction: float = 0.2
    batch_size: int = 32
    hidden_channels: int = 16
    kernel_size: int = 5
    learning_rate: float = 1e-3
    mean_epochs: int = 100
    variance_epochs: int = 100
    gamma_hidden_channels: int = 4
    gamma_weight_decay: float = 0.1
    joint_epochs: int = 100
    dropout_rate: float = 0.1
    num_draws: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate size, fraction, epoch, and MC-Dropout configuration.

        Raises:
            ValueError: If sizes, epochs or draws are not positive, the
                Gamma weight decay is negative, the validation fraction is
                not in (0, 1), or the dropout rate is not in [0, 1).
        """
        if self.window_length <= 0 or self.batch_size <= 0:
            raise ValueError("window_length and batch_size must be positive.")
        if self.hidden_channels <= 0 or self.kernel_size <= 0:
            raise ValueError("hidden_channels and kernel_size must be positive.")
        if self.gamma_hidden_channels <= 0:
            raise ValueError("gamma_hidden_channels must be positive.")
        if not self.gamma_weight_decay >= 0.0:
            raise ValueError("gamma_weight_decay must be non-negative.")
        if self.mean_epochs <= 0 or self.variance_epochs <= 0:
            raise ValueError("mean_epochs and variance_epochs must be positive.")
        if self.joint_epochs <= 0:
            raise ValueError("joint_epochs must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within (0, 1).")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be within [0, 1).")
        if self.num_draws <= 0:
            raise ValueError("num_draws must be positive.")


def make_window_loader(
    windows: np.ndarray, targets: np.ndarray, *, batch_size: int
) -> LoaderFactory:
    """Build a `LoaderFactory` yielding shuffled window/target mini-batches.

    Args:
        windows: Sliding-window inputs, shape `(n_windows, window_length,
            n_sensors)`.
        targets: Aligned RUL targets, shape `(n_windows,)`.
        batch_size: Number of windows per mini-batch.

    Returns:
        A callable `loader(split, epoch)` producing an iterable of
        `Batch` objects for that epoch, freshly shuffled from `epoch`.
    """
    inputs = jnp.asarray(windows, dtype=jnp.float32)
    targets_array = jnp.asarray(targets, dtype=jnp.float32).reshape(-1, 1)

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        permutation = jax.random.permutation(jax.random.key(epoch), inputs.shape[0])
        shuffled_inputs = inputs[permutation]
        shuffled_targets = targets_array[permutation]
        return [
            Batch(
                inputs=shuffled_inputs[start : start + batch_size],
                targets=shuffled_targets[start : start + batch_size],
            )
            for start in range(0, inputs.shape[0], batch_size)
        ]

    return loader


class PrintingEventSink:
    """An event sink that prints a one-line summary per epoch."""

    def on_event(self, event: TrainingEvent) -> None:
        """Print the stage-qualified epoch metrics for one training event.

        Args:
            event: The training event to summarize.
        """
        values = dict(event.metrics)
        if event.decision is not None:
            values[event.decision.metric] = event.decision.value
        metrics = ", ".join(f"{name}={value:.4f}" for name, value in values.items())
        print(f"[{event.name}] epoch={event.step} split={event.split} {metrics}")


@dataclass(frozen=True)
class SupervisedTrainingSetup[ModelT: nnx.Module]:
    """A freshly initialized CNN model and what supervised training needs for it.

    Attributes:
        model: The freshly initialized model.
        optimizer: An Adam optimizer over all of `model`'s parameters.
        rng_key: The training key split off alongside the model's
            initialization key.
        train_loader: Shuffled mini-batches of the training windows.
        validation_loader: Shuffled mini-batches of the validation windows.
    """

    model: ModelT
    optimizer: nnx.Optimizer
    rng_key: jax.Array
    train_loader: LoaderFactory
    validation_loader: LoaderFactory


def build_supervised_training_setup[ModelT: nnx.Module](
    model_type: Callable[..., ModelT],
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    validation_windows: np.ndarray,
    validation_targets: np.ndarray,
    config: CmapssConfig,
    *,
    dropout_rate: float = 0.0,
) -> SupervisedTrainingSetup[ModelT]:
    """Initialize a 1D-CNN model and its optimizer and loaders from `config`.

    The key `config.seed` is split into the model's initialization key and
    the training key. The model is built with the configured channel width
    and kernel size, and optimized with Adam at the configured learning rate.
    Dropout adds no parameters, so the initial parameters do not depend on
    `dropout_rate`.

    Args:
        model_type: The model class, constructed as `model_type(n_sensors, *,
            hidden_channels, kernel_size, dropout_rate, rngs)`.
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation RUL targets.
        config: Example configuration controlling model size and training.
        dropout_rate: The model's dropout rate; 0, the default, disables
            dropout.

    Returns:
        The model, its optimizer, the training key, and both loaders.
    """
    model_key, rng_key = jax.random.split(jax.random.key(config.seed))
    model = model_type(
        train_windows.shape[-1],
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        dropout_rate=dropout_rate,
        rngs=nnx.Rngs(model_key),
    )
    return SupervisedTrainingSetup(
        model=model,
        optimizer=create_optimizer(model, optax.adam(config.learning_rate)),
        rng_key=rng_key,
        train_loader=make_window_loader(
            train_windows, train_targets, batch_size=config.batch_size
        ),
        validation_loader=make_window_loader(
            validation_windows, validation_targets, batch_size=config.batch_size
        ),
    )


def train_mean_model(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    validation_windows: np.ndarray,
    validation_targets: np.ndarray,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
    dropout_rate: float = 0.0,
) -> tuple[Cnn1DMeanModel, TrainingState]:
    """Train a 1D-CNN mean model through `MeanStage` on windowed RUL data.

    Args:
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation RUL targets.
        config: Example configuration controlling model size and training.
        event_sinks: Sinks notified of the stage's training events; pass
            none to train silently.
        dropout_rate: The mean network's dropout rate, active during
            training; 0, the default, trains it without dropout.

    Returns:
        A tuple of the trained `Cnn1DMeanModel` and the shared staged
        `TrainingState`, ready for Stage 2 to continue on.
    """
    setup = build_supervised_training_setup(
        Cnn1DMeanModel,
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        dropout_rate=dropout_rate,
    )
    loss = make_supervised_loss(SquaredErrorLoss())
    stage = MeanStage(
        model=setup.model,
        optimizer=setup.optimizer,
        train_loader=setup.train_loader,
        loss=loss,
        options=SupervisedStageOptions(
            epochs=config.mean_epochs,
            validation=HeldOutValidation(
                model=setup.model,
                loader=setup.validation_loader,
                loss=loss,
            ),
            event_sinks=event_sinks,
        ),
    )
    state = TrainingState(rng_state=setup.rng_key)
    stage.prepare(state)
    stage.train(state)
    return setup.model, state


def train_gamma_model(
    state: TrainingState,
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> Cnn1DGammaModel:
    """Train a 1D-CNN Gamma model through `GammaVarianceStage`.

    Consumes the frozen, already-trained Stage-1 mean model's squared
    residuals on the training split, per `GammaVarianceStage`'s contract.
    The network is `config.gamma_hidden_channels` wide and trained with
    AdamW at `config.gamma_weight_decay`: in-sample residuals understate the
    held-out error, and a full-width, unregularized Gamma network overfits
    them into a spiky, overconfident variance.

    Args:
        state: The `MEAN_READY` staged training state produced by
            `train_mean_model`.
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        config: Example configuration controlling model size and training.
        event_sinks: Sinks notified of the stage's training events; pass
            none to train silently.

    Returns:
        The trained `Cnn1DGammaModel`.
    """
    n_sensors = train_windows.shape[-1]
    model_key = seed_stream_key(config, GAMMA_NETWORK_STREAM)
    model = Cnn1DGammaModel(
        n_sensors,
        hidden_channels=config.gamma_hidden_channels,
        kernel_size=config.kernel_size,
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(
        model,
        optax.adamw(config.learning_rate, weight_decay=config.gamma_weight_decay),
    )
    train_loader = make_window_loader(
        train_windows, train_targets, batch_size=config.batch_size
    )

    stage = GammaVarianceStage(
        model=model,
        optimizer=optimizer,
        source_loader=train_loader,
        options=SupervisedStageOptions(
            epochs=config.variance_epochs,
            event_sinks=event_sinks,
        ),
        splits=("train",),
    )
    stage.prepare(state)
    stage.train(state)
    return model


GAMMA_NETWORK_STREAM = 1
"""The stream folded into `config.seed`'s key for the Gamma network's init."""

MC_DROPOUT_DRAW_STREAM = 2
"""The stream folded into `config.seed`'s key for the MC-Dropout masks."""

POSTERIOR_NETWORK_STREAM = 3
"""The stream folded into `config.seed`'s key for the posterior network's init.

The posterior stage overwrites the network's parameters with the trained mean network's,
so only its dropout streams' initial keys come from here.
"""


def seed_stream_key(config: CmapssConfig, stream: int) -> jax.Array:
    """Return the key of one named random stream under `config.seed`.

    Args:
        config: Example configuration; its `seed` is read.
        stream: The stream folded into the seed's key, e.g.
            `MC_DROPOUT_DRAW_STREAM`.

    Returns:
        A JAX PRNG key, independent of every other stream's.
    """
    return jax.random.fold_in(jax.random.key(config.seed), stream)


def mc_dropout_draw_key(config: CmapssConfig) -> jax.Array:
    """Return the fixed key an MC-Dropout model's mask sets are drawn from.

    Derived from `config.seed` alone, so test scoring and the figure score
    the same draws (ADR 0014), and a seed's draws are reproducible run to
    run.

    Args:
        config: Example configuration; its `seed` is read.

    Returns:
        A JAX PRNG key.
    """
    return seed_stream_key(config, MC_DROPOUT_DRAW_STREAM)


class McDropoutPredictive(nnx.Module):
    """A Gaussian-head network with active dropout, as a posterior predictive.

    Evaluates the network under `num_draws` dropout-mask sets drawn from a
    fixed key and stacks each draw's mean and its own aleatoric variance
    into a [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive]. Each
    draw is one mask set applied to the whole batch, so a draw is the same
    function at every window, and the same key gives the same draws on any
    batch. Dropout stays active whatever mode the wrapper is put in.
    """

    def __init__(
        self, network: Cnn1DGaussianModel, *, num_draws: int, key: jax.Array
    ) -> None:
        """Wrap a network, without copying it.

        Args:
            network: The Gaussian-head network; later changes to its
                parameters are seen.
            num_draws: Number of dropout-mask sets to draw.
            key: The key the mask sets are drawn from.
        """
        self.network: Cnn1DGaussianModel = network
        self.num_draws: int = num_draws
        self.key: jax.Array = nnx.data(key)

    def __call__(self, inputs: jax.Array) -> PosteriorPredictive:
        """Predict the posterior predictive for each windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            A [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive] whose
            draws and per-draw aleatoric variances are shaped
            ``(num_draws, batch, 1)``.
        """
        graph, state = nnx.split(self.network)

        def draw(key: jax.Array) -> jax.Array:
            network = nnx.merge(graph, state)
            network.set_attributes(
                nnx.Dropout, deterministic=False, raise_if_not_found=False
            )
            gaussian = network(inputs, dropout_key=key)
            return jnp.stack([gaussian.loc, gaussian.scale**2])

        outputs = jax.vmap(draw)(jax.random.split(self.key, self.num_draws))
        return PosteriorPredictive(
            draws=outputs[:, 0], aleatoric_variance=outputs[:, 1]
        )


def build_mc_dropout_predictive(
    network: Cnn1DGaussianModel, config: CmapssConfig
) -> McDropoutPredictive:
    """Wrap a clone of a trained network as its MC-Dropout posterior predictive.

    Args:
        network: The trained Gaussian-head network with dropout.
        config: Example configuration; `num_draws` and `seed` are read.

    Returns:
        An eval-mode `McDropoutPredictive` over a clone of `network`,
        independent of later training, drawing its masks from
        `mc_dropout_draw_key`.
    """
    predictive = McDropoutPredictive(
        nnx.clone(network), num_draws=config.num_draws, key=mc_dropout_draw_key(config)
    )
    predictive.eval()
    return predictive


class PosteriorStagePredictive(nnx.Module, pytree=False):
    """A trained posterior stage's posterior predictive under one fixed key.

    Every call takes the same draws, so test scoring and the figure see the same mask
    sets (ADR 0014).
    """

    def __init__(self, stage: PosteriorStage, *, key: jax.Array) -> None:
        """Wrap a trained posterior stage.

        Args:
            stage: A trained posterior stage.
            key: The key selecting the posterior's draws.
        """
        self.stage: PosteriorStage = stage
        self.key: jax.Array = key

    def __call__(self, inputs: jax.Array) -> PosteriorPredictive:
        """Predict the posterior predictive for each windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            The stage's [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive]:
            the mean network's draws under dropout masks, sharing the Gamma
            stage's aleatoric variance.
        """
        return self.stage.predict(inputs, self.key)


def build_staged_mcd_posterior(
    state: TrainingState,
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> PosteriorStagePredictive:
    """Run an inference-only MC-Dropout posterior stage after the Gamma stage.

    The mean network was already trained with dropout active, so the
    posterior stage does not train it again: its `MCDropout` inference method
    steps with `optax.set_to_zero()`, which leaves every parameter at the
    trained mean network's value. `PosteriorStage` needs at least one epoch,
    so it runs one, which only evaluates the training loss; with no updates
    neither its prior nor validation nor early stopping has any effect, and
    none is configured. The posterior is the trained mean network, rebuilt
    at `config.dropout_rate`, under random dropout masks, sharing the Gamma
    stage's aleatoric variance.

    Args:
        state: The `VARIANCE_READY` staged training state produced by
            `train_gamma_model`, after a mean stage trained with dropout at
            `config.dropout_rate`. It moves on to `POSTERIOR_READY`.
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        config: Example configuration; `config.seed` sets the prediction
            draws.
        event_sinks: Sinks notified of the stage's training events; pass
            none to run silently.

    Returns:
        The stage's posterior predictive under `mc_dropout_draw_key`.
    """
    network = Cnn1DMeanModel(
        train_windows.shape[-1],
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout_rate,
        rngs=nnx.Rngs(seed_stream_key(config, POSTERIOR_NETWORK_STREAM)),
    )
    stage = PosteriorStage(
        inference_method=MCDropout(optimizer=optax.set_to_zero()),
        train_loader=make_window_loader(
            train_windows, train_targets, batch_size=config.batch_size
        ),
        dataset_size=train_windows.shape[0],
        model=network,
        options=PosteriorStageOptions(
            epochs=1, num_draws=config.num_draws, event_sinks=event_sinks
        ),
    )
    stage.prepare(state)
    stage.train(state)
    return PosteriorStagePredictive(stage, key=mc_dropout_draw_key(config))


class PredictiveMean(nnx.Module, pytree=False):
    """The posterior predictive's mean, as a deterministic mean model."""

    def __init__(self, predictive: PosteriorStagePredictive) -> None:
        """Wrap a trained posterior stage's predictive.

        Args:
            predictive: The posterior predictive under its fixed key.
        """
        self.predictive: PosteriorStagePredictive = predictive

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Return the mixture mean, shape ``(batch, 1)``."""
        return self.predictive(inputs).mean()


class RefitVariancePredictive(nnx.Module, pytree=False):
    """A posterior predictive's draws under a re-fitted Gamma variance."""

    def __init__(
        self, predictive: PosteriorStagePredictive, variance_model: Cnn1DGammaModel
    ) -> None:
        """Pair the posterior's draws with the re-fitted variance network.

        Args:
            predictive: The posterior predictive under its fixed key.
            variance_model: The Gamma network re-fitted to the predictive
                mean's squared residuals; cloned and put in eval mode.
        """
        self.predictive: PosteriorStagePredictive = predictive
        variance_model = nnx.clone(variance_model)
        variance_model.eval()
        self.variance_model: Cnn1DGammaModel = variance_model

    def __call__(self, inputs: jax.Array) -> PosteriorPredictive:
        """Return the draws, all sharing the re-fitted aleatoric variance."""
        return PosteriorPredictive(
            draws=self.predictive(inputs).draws,
            aleatoric_variance=self.variance_model(inputs).mean(),
        )


def refit_gamma_to_posterior_mean(
    predictive: PosteriorStagePredictive,
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> RefitVariancePredictive:
    """Re-fit the Gamma network to the posterior predictive's mean.

    The Gamma variance the posterior stage holds fixed was fit to the
    residuals of the mean network with dropout off, while the posterior
    predicts with the mean of its draws under dropout masks. This trains a
    fresh Gamma network, from the same initialization, on the squared
    residuals of the posterior predictive's mean, through a new `MEAN_READY`
    state, and pairs it with the posterior's draws.

    Args:
        predictive: The posterior stage's predictive.
        train_windows: Training sliding windows.
        train_targets: Training RUL targets.
        config: Example configuration.
        event_sinks: Sinks notified of the Gamma stage's training events.

    Returns:
        The posterior's draws under the re-fitted aleatoric variance.
    """
    state = TrainingState(rng_state=seed_stream_key(config, GAMMA_NETWORK_STREAM))
    state.register_component("mean_model", PredictiveMean(predictive))
    state.parameter_roles["mean_model"] = ParameterRole.MEAN
    state.lifecycle_state = StageState.MEAN_READY
    variance_model = train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=event_sinks
    )
    return RefitVariancePredictive(predictive, variance_model)


def build_composite_model(
    mean_model: Cnn1DMeanModel, variance_model: Cnn1DGammaModel
) -> CompositeGaussianModel:
    """Assemble an eval-mode composite Gaussian model from two trained stages.

    Both stage models are cloned, so the returned model is independent of
    the ones the training stages keep mutating, and the composite is put in
    eval mode. Every consumer of the composite predictive distribution —
    metrics and plots alike — should obtain it here, so that the
    clone-and-eval detail cannot drift between call sites.

    Args:
        mean_model: Trained Stage-1 point-RUL regressor.
        variance_model: Trained Stage-2 Gamma residual regressor.

    Returns:
        A `CompositeGaussianModel` over clones of both stage models, in
        eval mode.
    """
    composite = CompositeGaussianModel(nnx.clone(mean_model), nnx.clone(variance_model))
    composite.eval()
    return composite


class CyclesPredictive(nnx.Module, pytree=False):
    """A model trained on standardized RUL targets, predicting in cycles.

    Maps the wrapped model's standardized predictive back to RUL cycles: a
    `Gaussian`'s location and scale, or a `PosteriorPredictive`'s draws and
    aleatoric variance, through the affine inverse of the targets'
    standardization. Test scoring and the figure compare against RUL in
    cycles, so every model they see is wrapped.
    """

    def __init__(
        self, model: nnx.Module, standardization: TargetStandardization
    ) -> None:
        """Wrap a model, without copying it.

        Args:
            model: A model mapping windows to a standardized `Gaussian` or
                `PosteriorPredictive`.
            standardization: The statistics its training targets were
                standardized with.
        """
        self.model: nnx.Module = model
        self.standardization: TargetStandardization = standardization

    def __call__(self, inputs: jax.Array) -> Gaussian | PosteriorPredictive:
        """Predict the wrapped model's distribution, in RUL cycles.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            The wrapped model's output, of the same type, in cycles.
        """
        mean = self.standardization.mean
        scale = self.standardization.scale
        output = self.model(inputs)
        if isinstance(output, PosteriorPredictive):
            return PosteriorPredictive(
                draws=output.draws * scale + mean,
                aleatoric_variance=output.aleatoric_variance * scale**2,
                draw_mask=output.draw_mask,
            )
        return Gaussian(loc=output.loc * scale + mean, scale=output.scale * scale)


def evaluate_test_metrics(
    model: nnx.Module,
    test_windows: np.ndarray,
    test_rul: np.ndarray,
    *,
    epoch_metrics: tuple[EpochMetric, ...] = TEST_METRICS,
    predictor: Predictor | None = None,
) -> dict[str, float]:
    """Score a model against the FD001 test split.

    By default works for any model whose output is a `Gaussian`, such as the
    staged model from `build_composite_model`; with a
    `PosteriorPredictivePredictor()` it scores a model returning a
    `PosteriorPredictive` on the exact mixture. Scoring goes through
    `stagereg`'s `MetricSuite`, the predictor, and `evaluate_loader`; the
    closed-form CRPS needs neither predictive samples nor an evaluation grid,
    so the scores carry no Monte Carlo noise. No loss is computed.

    Args:
        model: An eval-mode model mapping windows to a `Gaussian` (the
            default predictor) or to a `PosteriorPredictive`, such as the
            MC-Dropout models' predictives (with a
            `PosteriorPredictivePredictor()` as `predictor`).
        test_windows: One trailing window per test unit, shape `(n_units,
            window_length, n_sensors)`.
        test_rul: Ground-truth RUL per test unit, shape `(n_units,)`.
        epoch_metrics: The metrics to report. Defaults to `TEST_METRICS`:
            RMSE (`"rmse"`), 95% interval coverage (`"coverage"`) and the
            closed-form CRPS (`"closed_form_crps"`).
        predictor: Materializes the metrics' data from the model's output.
            Defaults to `GaussianPredictor()`; pass a
            `PosteriorPredictivePredictor()` for a model returning a
            `PosteriorPredictive`.

    Returns:
        A mapping from each metric's name to its value on the test split.
    """
    inputs = jnp.asarray(test_windows, dtype=jnp.float32)
    targets = jnp.asarray(test_rul, dtype=jnp.float32).reshape(-1, 1)
    batch = Batch(inputs=inputs, targets=targets)

    metric_suite = MetricSuite(
        epoch=epoch_metrics,
        predictor=GaussianPredictor() if predictor is None else predictor,
    )
    # No metric draws predictive samples, so the key never affects a score.
    metrics, _ = evaluate_loader(
        model, [batch], key=jax.random.key(0), metrics=metric_suite
    )
    return metrics


@dataclass(frozen=True)
class PreparedCmapssData:
    """The standardized FD001 arrays and trajectories one training run needs.

    The sensor features and the training and validation RUL targets are
    standardized with statistics fitted on the training subset only; the
    test RUL stays in cycles, so a model is scored through `CyclesPredictive`.

    Carries both the windowed arrays the stages train and score on and the
    standardized trajectories they were windowed from, so that a downstream
    consumer — a per-cycle RUL curve, say — can re-window a unit under
    exactly the feature scaling the models were trained with, without
    re-loading the archive or re-fitting the statistics.

    Attributes:
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Standardized training linear-RUL targets, shape
            `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation linear-RUL targets,
            standardized with the training statistics.
        test_windows: One trailing window per test unit, shape `(n_units,
            window_length, n_sensors)`.
        test_rul: Ground-truth RUL per test unit, in cycles, shape
            `(n_units,)`.
        train_trajectories: The standardized training-subset trajectories
            `train_windows` was built from.
        validation_trajectories: The standardized validation-subset
            trajectories `validation_windows` was built from.
        standardization: The sensor statistics fitted on the training
            subset only and applied to every subset.
        target_standardization: The RUL-target statistics fitted on the
            training targets only and applied to the validation targets.
    """

    train_windows: np.ndarray
    train_targets: np.ndarray
    validation_windows: np.ndarray
    validation_targets: np.ndarray
    test_windows: np.ndarray
    test_rul: np.ndarray
    train_trajectories: pd.DataFrame
    validation_trajectories: pd.DataFrame
    standardization: SensorStandardization
    target_standardization: TargetStandardization


def prepare_cmapss_windows(config: CmapssConfig) -> PreparedCmapssData:
    """Load, split, standardize, and window the FD001 train/test splits.

    Args:
        config: Example configuration controlling the window length and
            the train/validation split.

    Returns:
        A `PreparedCmapssData` holding the windowed arrays, the
        standardized trajectories they came from, and the sensor and target
        standardization statistics. Those statistics are fit on the
        training subset only and reused for the other subsets; the test RUL
        stays in cycles.
    """
    train_data = load_fd001_data()
    train_df, validation_df = split_by_unit(
        train_data, test_size=config.validation_fraction, random_state=config.seed
    )
    stats = fit_standardization(train_df, _SENSOR_NAMES)
    train_df = apply_standardization(train_df, stats)
    validation_df = apply_standardization(validation_df, stats)

    train_windows, train_targets = build_windows(
        train_df, _SENSOR_NAMES, window_length=config.window_length
    )
    validation_windows, validation_targets = build_windows(
        validation_df, _SENSOR_NAMES, window_length=config.window_length
    )

    test_data = apply_standardization(load_fd001_test_data(), stats)
    test_windows = build_last_windows(
        test_data, _SENSOR_NAMES, window_length=config.window_length
    )
    test_rul = load_fd001_test_rul()
    target_stats = fit_target_standardization(train_targets)

    return PreparedCmapssData(
        train_windows=train_windows,
        train_targets=target_stats.standardize(train_targets),
        validation_windows=validation_windows,
        validation_targets=target_stats.standardize(validation_targets),
        test_windows=test_windows,
        test_rul=test_rul,
        train_trajectories=train_df,
        validation_trajectories=validation_df,
        standardization=stats,
        target_standardization=target_stats,
    )


def main() -> None:
    """Train the two-stage CMAPSS pipeline, print test metrics, and plot RUL curves.

    Pass `--plot-path` to save the RUL-curves figure to that path instead of displaying
    it interactively.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Save the validation RUL-curves figure to this path instead of "
            "displaying it."
        ),
    )
    args = parser.parse_args()

    config = CmapssConfig()
    prepared = prepare_cmapss_windows(config)

    mean_model, state = train_mean_model(
        prepared.train_windows,
        prepared.train_targets,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
        event_sinks=[PrintingEventSink()],
    )
    variance_model = train_gamma_model(
        state,
        prepared.train_windows,
        prepared.train_targets,
        config,
        event_sinks=[PrintingEventSink()],
    )

    staged_model = CyclesPredictive(
        build_composite_model(mean_model, variance_model),
        prepared.target_standardization,
    )
    metrics = evaluate_test_metrics(
        staged_model,
        prepared.test_windows,
        prepared.test_rul,
    )
    print(f"FD001 test RMSE: {metrics['rmse']:.4f}")
    print(f"FD001 test 95% interval coverage: {metrics['coverage']:.4f}")
    print(f"FD001 test closed-form CRPS: {metrics['closed_form_crps']:.4f}")

    validation_trajectories = prepared.validation_trajectories
    plot_validation_rul_curves(
        validation_trajectories,
        _SENSOR_NAMES,
        [LabelledModel("Staged model", staged_model)],
        units=select_lifetime_spanning_units(validation_trajectories),
        window_length=config.window_length,
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
