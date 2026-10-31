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
from collections.abc import Sequence
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
from model import Cnn1DGammaModel, Cnn1DMeanModel, CompositeGaussianModel
from plots import plot_validation_rul_curves
from preprocessing import (
    SensorStandardization,
    apply_standardization,
    build_last_windows,
    build_windows,
    fit_standardization,
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
from stagereg.core.types import Batch, TrainingState
from stagereg.jax import (
    GammaVarianceStage,
    GaussianPredictor,
    HeldOutValidation,
    MeanStage,
    MetricSuite,
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
        hidden_channels: Convolutional channel width of the mean and Gamma
            models.
        kernel_size: Convolution kernel width along the time axis.
        learning_rate: Adam learning rate for both stages.
        mean_epochs: Number of Stage-1 mean-model training epochs.
        variance_epochs: Number of Stage-2 Gamma-model training epochs.
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
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate size, fraction, and epoch configuration.

        Raises:
            ValueError: If sizes/epochs are not positive or the validation
                fraction is not in (0, 1).
        """
        if self.window_length <= 0 or self.batch_size <= 0:
            raise ValueError("window_length and batch_size must be positive.")
        if self.hidden_channels <= 0 or self.kernel_size <= 0:
            raise ValueError("hidden_channels and kernel_size must be positive.")
        if self.mean_epochs <= 0 or self.variance_epochs <= 0:
            raise ValueError("mean_epochs and variance_epochs must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be within (0, 1).")


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
    n_sensors = train_windows.shape[-1]
    model_key, rng_key = jax.random.split(jax.random.key(config.seed))
    model = Cnn1DMeanModel(
        n_sensors,
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(model, optax.adam(config.learning_rate))
    loss = make_supervised_loss(SquaredErrorLoss())
    train_loader = make_window_loader(
        train_windows, train_targets, batch_size=config.batch_size
    )
    validation_loader = make_window_loader(
        validation_windows, validation_targets, batch_size=config.batch_size
    )

    stage = MeanStage(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss=loss,
        options=SupervisedStageOptions(
            epochs=config.mean_epochs,
            validation=HeldOutValidation(
                model=model,
                loader=validation_loader,
                loss=loss,
            ),
            event_sinks=event_sinks,
        ),
    )
    state = TrainingState(rng_state=rng_key)
    stage.prepare(state)
    stage.train(state)
    return model, state


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
    model_key = jax.random.fold_in(jax.random.key(config.seed), 1)
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
) -> dict[str, float]:
    """Score a Gaussian-output model against the FD001 test split.

    Works for any model whose output is a `Gaussian`, such as the staged
    model from `build_composite_model`. Scoring goes through `stagereg`'s
    `MetricSuite`, `GaussianPredictor`, and `evaluate_loader`; the closed-form
    CRPS needs neither predictive samples nor an evaluation grid, so the
    scores carry no Monte Carlo noise. No loss is computed.

    Args:
        model: An eval-mode model mapping windows to a `Gaussian`.
        test_windows: One trailing window per test unit, shape `(n_units,
            window_length, n_sensors)`.
        test_rul: Ground-truth RUL per test unit, shape `(n_units,)`.
        epoch_metrics: The metrics to report. Defaults to `TEST_METRICS`:
            RMSE (`"rmse"`), 95% interval coverage (`"coverage"`) and the
            closed-form CRPS (`"closed_form_crps"`).

    Returns:
        A mapping from each metric's name to its value on the test split.
    """
    inputs = jnp.asarray(test_windows, dtype=jnp.float32)
    targets = jnp.asarray(test_rul, dtype=jnp.float32).reshape(-1, 1)
    batch = Batch(inputs=inputs, targets=targets)

    metric_suite = MetricSuite(epoch=epoch_metrics, predictor=GaussianPredictor())
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
        build_composite_model(mean_model, variance_model),
        units=select_lifetime_spanning_units(validation_trajectories),
        window_length=config.window_length,
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
