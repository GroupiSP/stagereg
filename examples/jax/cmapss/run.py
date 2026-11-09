"""End-to-end CMAPSS FD001 two-stage probabilistic RUL training and evaluation.

Loads the FD001 training and test splits, builds standardized sliding
windows, and trains a two-stage probabilistic pipeline through `stagereg`'s
explicit supervised stages:

* Stage 1 trains a deterministic 1D-CNN mean model via `MeanStage`.
* Stage 2 trains an independently-initialized 1D-CNN Gamma model on the
  frozen Stage-1 model's squared residuals via `GammaVarianceStage`.

The frozen Stage-1 point prediction and the Stage-2 Gamma mean are then
combined into a composite `Gaussian` predictive distribution, evaluated
end to end against the official FD001 test split with `stagereg`'s
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
    apply_standardization,
    build_last_windows,
    build_windows,
    fit_standardization,
    select_lifetime_spanning_units,
    split_by_unit,
)

from stagereg.core import EarlyStopper, OptimizationMode, Split
from stagereg.core.checkpoints import InMemoryCheckpointStore
from stagereg.core.losses import NegativeLogLikelihoodLoss, SquaredErrorLoss
from stagereg.core.metric_registry import (
    ClosedFormContinuousRankedProbabilityScore,
    EpochMetric,
    IntervalCoverage,
    RootMeanSquaredError,
)
from stagereg.core.protocols import LoaderFactory
from stagereg.core.tracking import EventSink, TrainingEvent
from stagereg.core.types import Batch, TrainingState
from stagereg.jax import (
    GammaVarianceStage,
    GaussianPredictor,
    HeldOutValidation,
    IsotropicGaussianPrior,
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
    initialize_training_state,
    make_supervised_loss,
    restore_checkpoint,
    run_supervised,
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
        hidden_channels: Convolutional channel width of the mean and Gamma
            models.
        kernel_size: Convolution kernel width along the time axis.
        learning_rate: Adam learning rate for both stages.
        mean_epochs: Number of Stage-1 mean-model training epochs.
        variance_epochs: Number of Stage-2 Gamma-model training epochs.
        joint_epochs: Number of training epochs for the joint model of the
            staged-vs-joint comparison.
        dropout_rate: Dropout rate of the MC-Dropout models' backbones, in
            `[0, 1)`.
        joint_mcd_epochs: Maximum number of training epochs of the joint
            MC-Dropout model, which stops early on its validation NLL.
        mcd_learning_rate: Adam learning rate of the MC-Dropout training,
            shared by both MC-Dropout models; tuned on FD001 (see the
            example's README).
        posterior_epochs: Maximum number of epochs of the staged MC-Dropout
            model's posterior stage, which stops early on its validation
            mixture NLL; tuned on FD001.
        prior_precision: Precision of the isotropic Gaussian prior over the
            mean network's parameters in the staged MC-Dropout model's
            posterior stage; tuned on FD001.
        num_draws: Number of dropout-mask sets an MC-Dropout model's posterior
            predictive mixes.
        early_stopping_patience: Number of epochs without a better validation
            NLL an MC-Dropout model trains on before it stops.
        seed: Base JAX random seed for splitting, model init, and training.
    """

    window_length: int = 30
    validation_fraction: float = 0.2
    batch_size: int = 32
    hidden_channels: int = 16
    kernel_size: int = 5
    learning_rate: float = 1e-3
    mean_epochs: int = 20
    variance_epochs: int = 20
    joint_epochs: int = 20
    dropout_rate: float = 0.1
    joint_mcd_epochs: int = 60
    mcd_learning_rate: float = 1e-3
    posterior_epochs: int = 20
    prior_precision: float = 1.0
    num_draws: int = 100
    early_stopping_patience: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate size, fraction, epoch, and MC-Dropout configuration.

        Raises:
            ValueError: If sizes, epochs, draws, patience, the MC-Dropout
                learning rate or the prior precision are not positive, the validation fraction is
                not in (0, 1), or the dropout rate is not in [0, 1).
        """
        if self.window_length <= 0 or self.batch_size <= 0:
            raise ValueError("window_length and batch_size must be positive.")
        if self.hidden_channels <= 0 or self.kernel_size <= 0:
            raise ValueError("hidden_channels and kernel_size must be positive.")
        if self.mean_epochs <= 0 or self.variance_epochs <= 0:
            raise ValueError("mean_epochs and variance_epochs must be positive.")
        if self.joint_epochs <= 0:
            raise ValueError("joint_epochs must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within (0, 1).")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be within [0, 1).")
        for name in (
            "joint_mcd_epochs",
            "posterior_epochs",
            "num_draws",
            "early_stopping_patience",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("mcd_learning_rate", "prior_precision"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive.")


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
    learning_rate: float | None = None,
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
        learning_rate: Adam learning rate; `None`, the default, uses
            `config.learning_rate`.

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
    if learning_rate is None:
        learning_rate = config.learning_rate
    return SupervisedTrainingSetup(
        model=model,
        optimizer=create_optimizer(model, optax.adam(learning_rate)),
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
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(model, optax.adam(config.learning_rate))
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

JOINT_MCD_STAGE = "joint_mcd"
"""The stage segment the joint MC-Dropout model's training is reported under."""


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

    Derived from `config.seed` alone, so validation, test scoring and the
    figure all score the same draws (ADR 0014), and a seed's draws are
    reproducible run to run.

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


def _mc_dropout_predictive_over(
    network: Cnn1DGaussianModel, config: CmapssConfig
) -> McDropoutPredictive:
    """Wrap `network` itself, not a clone, as its MC-Dropout posterior predictive.

    Args:
        network: The Gaussian-head network with dropout; later changes to
            its parameters are seen.
        config: Example configuration; `num_draws` and `seed` are read.

    Returns:
        An `McDropoutPredictive` over `network` drawing its masks from
        `mc_dropout_draw_key`.
    """
    return McDropoutPredictive(
        network, num_draws=config.num_draws, key=mc_dropout_draw_key(config)
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
    predictive = _mc_dropout_predictive_over(nnx.clone(network), config)
    predictive.eval()
    return predictive


def _validation_nll_early_stopper(config: CmapssConfig, *, metric: str) -> EarlyStopper:
    """Build the early stopper an MC-Dropout model trains under.

    Args:
        config: Example configuration; `early_stopping_patience` is read.
        metric: The name the validation NLL is reported under, `"loss"`
            for `run_supervised` or `"nll"` for a posterior stage.

    Returns:
        An `EarlyStopper` minimizing `metric` on the validation split.
    """
    return EarlyStopper(
        metric=metric,
        mode=OptimizationMode.MIN,
        patience=config.early_stopping_patience,
        source=Split.VALIDATION,
    )


def _fixed_order_loader(
    windows: np.ndarray, targets: np.ndarray, *, batch_size: int
) -> LoaderFactory:
    """Build a `LoaderFactory` yielding the same unshuffled batches every epoch.

    The batches are built once, so every epoch sees the same windows in the
    same batches; an MC-Dropout model's validation NLL is then comparable
    epoch to epoch.

    Args:
        windows: Sliding-window inputs, shape `(n_windows, window_length,
            n_sensors)`.
        targets: Aligned RUL targets, shape `(n_windows,)`.
        batch_size: Number of windows per mini-batch.

    Returns:
        A callable `loader(split, epoch)` returning the fixed batches.
    """
    inputs = jnp.asarray(windows, dtype=jnp.float32)
    targets_array = jnp.asarray(targets, dtype=jnp.float32).reshape(-1, 1)
    batches = [
        Batch(
            inputs=inputs[start : start + batch_size],
            targets=targets_array[start : start + batch_size],
        )
        for start in range(0, inputs.shape[0], batch_size)
    ]

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return batches

    return loader


def train_joint_mcd_model(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    validation_windows: np.ndarray,
    validation_targets: np.ndarray,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> Cnn1DGaussianModel:
    """Train a joint MC-Dropout network on the Gaussian NLL with dropout active.

    A Gaussian-head CNN at `config.dropout_rate`, from a random
    initialization, trained with `run_supervised` at
    `config.mcd_learning_rate` and no prior term. After every epoch it is
    validated on the NLL of its MC mixture (`McDropoutPredictive` under
    `mc_dropout_draw_key`), and it stops once that NLL has not improved for
    `config.early_stopping_patience` epochs, after at most
    `config.joint_mcd_epochs`.

    Args:
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation RUL targets.
        config: Example configuration; `config.seed` sets the
            initialization, the training masks and the validation draws.
        event_sinks: Sinks notified of the training events; pass none to
            train silently.

    Returns:
        The network with its best validation epoch's parameters; wrap it
        with `build_mc_dropout_predictive` to predict.
    """
    setup = build_supervised_training_setup(
        Cnn1DGaussianModel,
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        dropout_rate=config.dropout_rate,
        learning_rate=config.mcd_learning_rate,
    )
    loss = make_supervised_loss(NegativeLogLikelihoodLoss())
    state = initialize_training_state(
        setup.model, setup.optimizer, rng_key=setup.rng_key
    )
    validation_loader = _fixed_order_loader(
        validation_windows, validation_targets, batch_size=config.batch_size
    )
    store = InMemoryCheckpointStore()
    run_supervised(
        model=setup.model,
        optimizer=setup.optimizer,
        train_loader=setup.train_loader,
        loss=loss,
        state=state,
        epochs=config.joint_mcd_epochs,
        validation=HeldOutValidation(
            model=_mc_dropout_predictive_over(setup.model, config),
            loader=validation_loader,
            loss=loss,
        ),
        early_stopper=_validation_nll_early_stopper(config, metric="loss"),
        event_sinks=event_sinks,
        checkpoint_store=store,
        stage=JOINT_MCD_STAGE,
    )
    restore_checkpoint(
        store.load(f"{JOINT_MCD_STAGE}/best"),
        state=state,
        model=setup.model,
        optimizer=setup.optimizer,
    )
    return setup.model


class PosteriorStagePredictive(nnx.Module, pytree=False):
    """A trained posterior stage's posterior predictive under one fixed key.

    Every call takes the same draws, so validation, test scoring and the figure all see
    the same mask sets (ADR 0014).
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
            the draws of the fine-tuned mean network, sharing the Gamma
            stage's aleatoric variance.
        """
        return self.stage.predict(inputs, self.key)


def train_staged_mcd_posterior(
    state: TrainingState,
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    validation_windows: np.ndarray,
    validation_targets: np.ndarray,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> PosteriorStagePredictive:
    """Run an MC-Dropout posterior stage after the mean and Gamma stages.

    The posterior network is the mean network rebuilt at
    `config.dropout_rate`; the stage warm-starts it from the trained mean
    network and fine-tunes all of it with `MCDropout` at
    `config.mcd_learning_rate`, under an isotropic Gaussian prior at
    `config.prior_precision`, holding the Gamma stage's aleatoric variance
    fixed. After every epoch the posterior predictive is scored on the
    validation windows under `mc_dropout_draw_key`, and the stage stops once
    its mixture NLL has not improved for `config.early_stopping_patience`
    epochs, after at most `config.posterior_epochs`, then finalizes the best
    epoch.

    Args:
        state: The `VARIANCE_READY` staged training state produced by
            `train_gamma_model`. It moves on to `POSTERIOR_READY`.
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training RUL targets, shape `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation RUL targets.
        config: Example configuration; `config.seed` sets the validation and
            prediction draws.
        event_sinks: Sinks notified of the stage's training events; pass
            none to train silently.

    Returns:
        The trained stage's posterior predictive under
        `mc_dropout_draw_key`.
    """
    network = Cnn1DMeanModel(
        train_windows.shape[-1],
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout_rate,
        rngs=nnx.Rngs(seed_stream_key(config, POSTERIOR_NETWORK_STREAM)),
    )
    key = mc_dropout_draw_key(config)
    validation_loader = _fixed_order_loader(
        validation_windows, validation_targets, batch_size=config.batch_size
    )
    stage = PosteriorStage(
        inference_method=MCDropout(optimizer=optax.adam(config.mcd_learning_rate)),
        train_loader=make_window_loader(
            train_windows, train_targets, batch_size=config.batch_size
        ),
        dataset_size=train_windows.shape[0],
        prior=IsotropicGaussianPrior(precision=config.prior_precision),
        model=network,
        options=PosteriorStageOptions(
            epochs=config.posterior_epochs,
            num_draws=config.num_draws,
            validation_loader=validation_loader,
            early_stopper=_validation_nll_early_stopper(config, metric="nll"),
            event_sinks=event_sinks,
            checkpoint_store=InMemoryCheckpointStore(),
            validation_key=key,
        ),
    )
    stage.prepare(state)
    stage.train(state)
    return PosteriorStagePredictive(stage, key=key)


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

    Carries both the windowed arrays the stages train and score on and the
    standardized trajectories they were windowed from, so that a downstream
    consumer — a per-cycle RUL curve, say — can re-window a unit under
    exactly the feature scaling the models were trained with, without
    re-loading the archive or re-fitting the statistics.

    Attributes:
        train_windows: Training sliding windows, shape `(n_train,
            window_length, n_sensors)`.
        train_targets: Training linear-RUL targets, shape `(n_train,)`.
        validation_windows: Held-out validation windows, same layout as
            `train_windows`.
        validation_targets: Held-out validation linear-RUL targets.
        test_windows: One trailing window per test unit, shape `(n_units,
            window_length, n_sensors)`.
        test_rul: Ground-truth RUL per test unit, shape `(n_units,)`.
        train_trajectories: The standardized training-subset trajectories
            `train_windows` was built from.
        validation_trajectories: The standardized validation-subset
            trajectories `validation_windows` was built from.
        standardization: The statistics fitted on the training subset only
            and applied to every subset.
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


def prepare_cmapss_windows(config: CmapssConfig) -> PreparedCmapssData:
    """Load, split, standardize, and window the FD001 train/test splits.

    Args:
        config: Example configuration controlling the window length and
            the train/validation split.

    Returns:
        A `PreparedCmapssData` holding the windowed arrays, the
        standardized trajectories they came from, and the standardization
        statistics. Those statistics are fit on the training subset only
        and reused for the validation and test subsets.
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

    return PreparedCmapssData(
        train_windows=train_windows,
        train_targets=train_targets,
        validation_windows=validation_windows,
        validation_targets=validation_targets,
        test_windows=test_windows,
        test_rul=test_rul,
        train_trajectories=train_df,
        validation_trajectories=validation_df,
        standardization=stats,
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

    metrics = evaluate_test_metrics(
        build_composite_model(mean_model, variance_model),
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
        [
            LabelledModel(
                "Staged model", build_composite_model(mean_model, variance_model)
            )
        ],
        units=select_lifetime_spanning_units(validation_trajectories),
        window_length=config.window_length,
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
