"""Compare the staged model with a joint model on CMAPSS FD001 over several seeds.

Prepares the FD001 data once, exactly as `run.py` does, then for each seed
trains two models on it:

* the **staged model**: a 1D-CNN mean stage followed by a 1D-CNN Gamma
  variance stage on its squared residuals, assembled into one Gaussian;
* a **joint model**: one 1D-CNN backbone with the same architecture and
  hyperparameters, followed by a Gaussian head, trained with `run_supervised`
  on the Gaussian negative log-likelihood.

Both train on the raw linear-RUL targets and are scored against the
official FD001 test split on RMSE, 95% interval coverage, closed-form CRPS
and NLL. The train/validation split stays fixed across seeds; only model
initialization changes.

It then draws the first seed's validation RUL curves: the staged model's row
above the joint model's, for the same lifetime-spanning validation units.

Run it with:

    uv run --group example-cmapss python examples/jax/cmapss/compare.py --seeds 5

Pass `--plot-path` to save the figure instead of displaying it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx

# Allow sibling-module imports both when run as a script and when loaded
# from an arbitrary working directory, e.g. via `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from model import Cnn1DGaussianModel, CompositeGaussianModel
from plots import LabelledModel, plot_validation_rul_curves
from preprocessing import select_lifetime_spanning_units
from run import (
    TEST_METRICS,
    CmapssConfig,
    PreparedCmapssData,
    build_composite_model,
    evaluate_test_metrics,
    make_window_loader,
    prepare_cmapss_windows,
    train_gamma_model,
    train_mean_model,
)

from stagereg.core import NegativeLogLikelihood, NegativeLogLikelihoodLoss
from stagereg.core.metric_registry import EpochMetric
from stagereg.core.tracking import EventSink
from stagereg.jax import (
    HeldOutValidation,
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

COMPARISON_METRICS: tuple[EpochMetric, ...] = (*TEST_METRICS, NegativeLogLikelihood())
"""The comparison's test metrics: RMSE, 95% coverage, closed-form CRPS and NLL."""

JOINT_STAGE = "joint"
"""The stage segment the joint model's training is reported under."""

STAGED = "staged"
"""The staged model's name in the per-run lines and the table."""

JOINT = "joint"
"""The joint model's name in the per-run lines and the table."""


def train_joint_model(
    prepared: PreparedCmapssData,
    config: CmapssConfig,
    *,
    event_sinks: Sequence[EventSink],
) -> Cnn1DGaussianModel:
    """Train a joint model with `run_supervised` on the Gaussian NLL.

    The model is validated on the held-out validation windows after every
    epoch and trained for `config.joint_epochs` epochs on the raw RUL
    targets.

    Args:
        prepared: The prepared FD001 windows to train and validate on.
        config: Example configuration controlling model size and training;
            `config.seed` sets the model's initialization.
        event_sinks: Sinks notified of the training events; pass none to
            train silently.

    Returns:
        The trained `Cnn1DGaussianModel`.
    """
    n_sensors = prepared.train_windows.shape[-1]
    model_key, rng_key = jax.random.split(jax.random.key(config.seed))
    model = Cnn1DGaussianModel(
        n_sensors,
        hidden_channels=config.hidden_channels,
        kernel_size=config.kernel_size,
        rngs=nnx.Rngs(model_key),
    )
    optimizer = create_optimizer(model, optax.adam(config.learning_rate))
    loss = make_supervised_loss(NegativeLogLikelihoodLoss())
    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=make_window_loader(
            prepared.train_windows,
            prepared.train_targets,
            batch_size=config.batch_size,
        ),
        loss=loss,
        state=initialize_training_state(model, optimizer, rng_key=rng_key),
        epochs=config.joint_epochs,
        validation=HeldOutValidation(
            model=model,
            loader=make_window_loader(
                prepared.validation_windows,
                prepared.validation_targets,
                batch_size=config.batch_size,
            ),
            loss=loss,
        ),
        event_sinks=event_sinks,
        stage=JOINT_STAGE,
    )
    return model


@dataclass(frozen=True)
class SeedComparison:
    """Both models trained from one seed, and their test-split scores.

    Attributes:
        seed: The seed both models were initialized from.
        staged_model: The eval-mode staged model.
        joint_model: An eval-mode clone of the trained joint model.
        staged_metrics: The staged model's `COMPARISON_METRICS` on the test
            split, by metric name.
        joint_metrics: The joint model's `COMPARISON_METRICS` on the test
            split, by metric name.
    """

    seed: int
    staged_model: CompositeGaussianModel
    joint_model: Cnn1DGaussianModel
    staged_metrics: dict[str, float]
    joint_metrics: dict[str, float]


RunReporter = Callable[[str, int, dict[str, float]], None]
"""Called as `report(model, seed, metrics)` once each (seed, model) run is scored."""


def _ignore_run(model: str, seed: int, metrics: dict[str, float]) -> None:
    """Report nothing about a finished run.

    Args:
        model: The model's name, ignored.
        seed: The seed the model was initialized from, ignored.
        metrics: The model's test metrics by name, ignored.
    """
    del model, seed, metrics


def _train_and_score_seed(
    prepared: PreparedCmapssData, config: CmapssConfig, *, on_run: RunReporter
) -> SeedComparison:
    """Train and score the staged and then the joint model from `config.seed`.

    Args:
        prepared: The prepared FD001 windows to train, validate and score on.
        config: Example configuration; its `seed` initializes both models.
        on_run: Called with each model's name, the seed and its test metrics
            as soon as that model is scored.

    Returns:
        The seed's `SeedComparison`.
    """

    def score(model: nnx.Module) -> dict[str, float]:
        return evaluate_test_metrics(
            model,
            prepared.test_windows,
            prepared.test_rul,
            epoch_metrics=COMPARISON_METRICS,
        )

    mean_model, state = train_mean_model(
        prepared.train_windows,
        prepared.train_targets,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
        event_sinks=(),
    )
    variance_model = train_gamma_model(
        state,
        prepared.train_windows,
        prepared.train_targets,
        config,
        event_sinks=(),
    )
    staged_model = build_composite_model(mean_model, variance_model)
    staged_metrics = score(staged_model)
    on_run(STAGED, config.seed, staged_metrics)

    joint_model = nnx.clone(train_joint_model(prepared, config, event_sinks=()))
    joint_model.eval()
    joint_metrics = score(joint_model)
    on_run(JOINT, config.seed, joint_metrics)

    return SeedComparison(
        seed=config.seed,
        staged_model=staged_model,
        joint_model=joint_model,
        staged_metrics=staged_metrics,
        joint_metrics=joint_metrics,
    )


def compare_models(
    prepared: PreparedCmapssData,
    config: CmapssConfig,
    *,
    seeds: int,
    on_run: RunReporter = _ignore_run,
) -> list[SeedComparison]:
    """Train and score the staged and the joint model once per seed.

    Every seed trains on the same `prepared` data, so only initialization
    varies: seed index `i` initializes both models from `config.seed + i`.
    Training is silent. Each `SeedComparison` keeps both trained models, so
    a caller can plot, say, the first seed's models without retraining.

    Args:
        prepared: The prepared FD001 windows, shared by every seed.
        config: Example configuration; its `seed` is the first seed.
        seeds: Number of seeds to train both models from.
        on_run: Called with the model's name (`"staged"` or `"joint"`), the
            seed and the test metrics as soon as each run is scored. Does
            nothing by default.

    Returns:
        One `SeedComparison` per seed, in seed order.
    """
    return [
        _train_and_score_seed(
            prepared,
            dataclasses.replace(config, seed=config.seed + index),
            on_run=on_run,
        )
        for index in range(seeds)
    ]


_TABLE_ROWS = (
    ("RMSE", "rmse"),
    ("95% coverage", "coverage"),
    ("closed-form CRPS", "closed_form_crps"),
    ("NLL", "nll"),
)


def _mean_std(values: Sequence[float]) -> str:
    """Format values as `mean ± std`: the sample std, or 0 for a single value.

    Args:
        values: One metric's values, one per seed; at least one.

    Returns:
        The mean and the sample standard deviation (`ddof=1`), both to four
        decimals, joined by `±`.
    """
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return f"{float(array.mean()):.4f} ± {std:.4f}"


def format_comparison_table(results: Sequence[SeedComparison]) -> str:
    """Tabulate both models' test metrics as `mean ± std` over seeds.

    Args:
        results: The per-seed comparisons from `compare_models`.

    Returns:
        A table with one row per metric (RMSE, 95% coverage, closed-form
        CRPS, NLL) and a `staged` and a `joint` column.
    """
    cells = [
        (
            label,
            _mean_std([result.staged_metrics[name] for result in results]),
            _mean_std([result.joint_metrics[name] for result in results]),
        )
        for label, name in _TABLE_ROWS
    ]
    label_width = max(len(label) for label, _, _ in cells)
    cell_width = max(len(cell) for row in cells for cell in row[1:])
    lines = [
        f"{'':<{label_width}}  {STAGED:>{cell_width}}  {JOINT:>{cell_width}}",
        *(
            f"{label:<{label_width}}  {staged:>{cell_width}}  {joint:>{cell_width}}"
            for label, staged, joint in cells
        ),
    ]
    return "\n".join(lines)


def print_run(model: str, seed: int, metrics: dict[str, float]) -> None:
    """Print one finished (seed, model) run's test metrics on one line.

    Args:
        model: The model's name, `"staged"` or `"joint"`.
        seed: The seed the model was initialized from.
        metrics: The model's test metrics by name.
    """
    values = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"[{model} seed={seed}] {values}")


def main() -> None:
    """Compare the staged and the joint model over seeds, print and plot the results.

    Pass `--seeds` to set the number of seeds (default 5), and `--plot-path` to save the
    first seed's RUL-curves figure to that path instead of displaying it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to train both models from (default: 5).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Save the staged-vs-joint validation RUL-curves figure to this path "
            "instead of displaying it."
        ),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive.")

    config = CmapssConfig()
    prepared = prepare_cmapss_windows(config)
    results = compare_models(prepared, config, seeds=args.seeds, on_run=print_run)
    print(f"\nFD001 test metrics over {args.seeds} seeds (mean ± std):")
    print(format_comparison_table(results))

    first = results[0]
    validation_trajectories = prepared.validation_trajectories
    plot_validation_rul_curves(
        validation_trajectories,
        prepared.standardization.feature_columns,
        [
            LabelledModel("Staged model", first.staged_model),
            LabelledModel("Joint model", first.joint_model),
        ],
        units=select_lifetime_spanning_units(validation_trajectories),
        window_length=config.window_length,
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
