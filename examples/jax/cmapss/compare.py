"""Compare the staged model with a joint model on CMAPSS FD001 over several seeds.

Prepares the FD001 data once, exactly as `run.py` does, then for each seed
trains two models on it:

* the **staged model**: a 1D-CNN mean stage followed by a narrower 1D-CNN
  Gamma variance stage, trained with weight decay, on its squared residuals,
  assembled into one Gaussian;
* a **joint model**: one 1D-CNN backbone with the same architecture and
  hyperparameters, followed by a Gaussian head, trained with `run_supervised`
  on the Gaussian negative log-likelihood.

Both train on the raw linear-RUL targets and are scored against the
official FD001 test split on RMSE, 95% interval coverage, closed-form CRPS
and NLL. The train/validation split stays fixed across seeds; only model
initialization changes.

It then draws the first seed's validation RUL curves: the staged model's row
above the joint model's, for the same lifetime-spanning validation units.

With `--posterior-stage` it adds two MC-Dropout models, each fine-tuning a
network first trained with dropout off. The **joint MC-Dropout model**
(`joint+mcd`) warm-starts a Gaussian-head CNN with dropout active from the
`joint` model, fine-tunes it on the Gaussian NLL under the posterior stage's
prior and early-stops on its MC mixture's validation NLL; its prediction is a
posterior predictive over fixed dropout-mask sets with each draw's own
aleatoric variance. The **staged MC-Dropout model** (`staged+mcd`) is a
posterior stage with the `MCDropout` inference method that fine-tunes the
`staged` model's trained mean network with dropout active, holding its Gamma
stage's aleatoric variance fixed, and early-stops on the validation mixture
NLL; a fresh Gamma network is then re-fitted to the squared residuals of the
posterior predictive's mean, and the draws share that variance. The models are
then `joint`, `joint+mcd`, `staged`
and `staged+mcd`, in that order, in the per-run lines, the table and the
figure.

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

import jax.numpy as jnp
import numpy as np
from flax import nnx

# Allow sibling-module imports both when run as a script and when loaded
# from an arbitrary working directory, e.g. via `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from model import Cnn1DGammaModel, Cnn1DGaussianModel, Cnn1DMeanModel
from plots import LabelledModel, plot_validation_rul_curves
from preprocessing import select_lifetime_spanning_units
from run import (
    TEST_METRICS,
    CmapssConfig,
    PreparedCmapssData,
    build_composite_model,
    build_mc_dropout_predictive,
    build_supervised_training_setup,
    evaluate_test_metrics,
    prepare_cmapss_windows,
    refit_gamma_to_posterior_mean,
    train_gamma_model,
    train_joint_mcd_model,
    train_mean_model,
    train_staged_mcd_posterior,
)

from stagereg.core import NegativeLogLikelihood, NegativeLogLikelihoodLoss
from stagereg.core.metric_registry import EpochMetric
from stagereg.core.tracking import EventSink
from stagereg.core.types import TrainingState
from stagereg.jax import (
    GaussianPredictor,
    HeldOutValidation,
    PosteriorPredictive,
    PosteriorPredictivePredictor,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

COMPARISON_METRICS: tuple[EpochMetric, ...] = (*TEST_METRICS, NegativeLogLikelihood())
"""The comparison's test metrics: RMSE, 95% coverage, closed-form CRPS and NLL."""

JOINT_STAGE = "joint"
"""The stage segment the joint model's training is reported under."""


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
    setup = build_supervised_training_setup(
        Cnn1DGaussianModel,
        prepared.train_windows,
        prepared.train_targets,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
    )
    loss = make_supervised_loss(NegativeLogLikelihoodLoss())
    run_supervised(
        model=setup.model,
        optimizer=setup.optimizer,
        train_loader=setup.train_loader,
        loss=loss,
        state=initialize_training_state(
            setup.model, setup.optimizer, rng_key=setup.rng_key
        ),
        epochs=config.joint_epochs,
        validation=HeldOutValidation(
            model=setup.model, loader=setup.validation_loader, loss=loss
        ),
        event_sinks=event_sinks,
        stage=JOINT_STAGE,
    )
    return setup.model


def score_on_test_split(
    model: nnx.Module, prepared: PreparedCmapssData
) -> dict[str, float]:
    """Score a model on the FD001 test split with `COMPARISON_METRICS`.

    The predictor is chosen by the model's output: `GaussianPredictor` for a
    `Gaussian`, `PosteriorPredictivePredictor` for a `PosteriorPredictive`,
    which scores the exact mixture (mixture log-density, mixture-quantile
    coverage and the mixture's closed-form CRPS).

    Args:
        model: An eval-mode model mapping windows to a `Gaussian` or a
            `PosteriorPredictive`.
        prepared: The prepared FD001 windows whose test split is scored.

    Returns:
        The model's `COMPARISON_METRICS` on the test split, by metric name.
    """
    output = model(jnp.asarray(prepared.test_windows[:1], dtype=jnp.float32))
    predictor = (
        PosteriorPredictivePredictor()
        if isinstance(output, PosteriorPredictive)
        else GaussianPredictor()
    )
    return evaluate_test_metrics(
        model,
        prepared.test_windows,
        prepared.test_rul,
        epoch_metrics=COMPARISON_METRICS,
        predictor=predictor,
    )


@dataclass(frozen=True)
class ModelResult:
    """One trained model of a comparison and its test-split scores.

    Attributes:
        name: The model's name in the per-run lines and the table's column
            header, e.g. `"staged"`.
        label: The model's row label in the figure, e.g. `"Staged model"`.
        model: The trained, eval-mode model, returning a `Gaussian` or a
            `PosteriorPredictive`.
        metrics: The model's `COMPARISON_METRICS` on the test split, by
            metric name.
    """

    name: str
    label: str
    model: nnx.Module
    metrics: dict[str, float]


@dataclass(frozen=True)
class SeedComparison:
    """The models trained from one seed, and their test-split scores.

    Attributes:
        seed: The seed every model was initialized from.
        models: One result per model, in the table's column order and the
            figure's row order. Without `--posterior-stage`: `staged`, then
            `joint`. With it: `joint`, `joint+mcd`, `staged`, then
            `staged+mcd`.
    """

    seed: int
    models: tuple[ModelResult, ...]


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


def _train_staged_networks(
    prepared: PreparedCmapssData, config: CmapssConfig
) -> tuple[Cnn1DMeanModel, Cnn1DGammaModel, TrainingState]:
    """Silently train the staged model's mean and Gamma variance networks.

    Args:
        prepared: The prepared FD001 windows to train and validate on.
        config: Example configuration; its `seed` initializes both networks.

    Returns:
        The trained mean network, Gamma variance network, and the
        `VARIANCE_READY` training state a posterior stage continues on.
    """
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
    return mean_model, variance_model, state


def _train_joint(prepared: PreparedCmapssData, config: CmapssConfig) -> nnx.Module:
    """Silently train the joint model and return an eval-mode clone of it.

    Args:
        prepared: The prepared FD001 windows to train and validate on.
        config: Example configuration; its `seed` initializes the model.

    Returns:
        An eval-mode clone of the trained `Cnn1DGaussianModel`.
    """
    joint_model = nnx.clone(train_joint_model(prepared, config, event_sinks=()))
    joint_model.eval()
    return joint_model


def _train_joint_mcd(
    prepared: PreparedCmapssData, config: CmapssConfig, joint_model: nnx.Module
) -> nnx.Module:
    """Silently fine-tune the joint MC-Dropout model and return its predictive.

    Args:
        prepared: The prepared FD001 windows to train and validate on.
        config: Example configuration; its `seed` sets the network's
            training masks and the predictive's draws.
        joint_model: The trained joint model the fine-tune starts from.

    Returns:
        The trained network's eval-mode `McDropoutPredictive`, from
        `build_mc_dropout_predictive`.
    """
    network = train_joint_mcd_model(
        joint_model,
        prepared.train_windows,
        prepared.train_targets,
        prepared.validation_windows,
        prepared.validation_targets,
        config,
        event_sinks=(),
    )
    return build_mc_dropout_predictive(network, config)


def _train_and_score_seed(
    prepared: PreparedCmapssData,
    config: CmapssConfig,
    *,
    posterior_stage: bool,
    on_run: RunReporter,
) -> SeedComparison:
    """Train and score every model of the comparison from `config.seed`.

    Without a posterior stage the models are the staged and then the joint
    model; with one, the joint, the joint MC-Dropout, the staged and the
    staged MC-Dropout model. Each model is trained and scored in that order;
    the joint MC-Dropout model is fine-tuned from the trained joint model,
    and the staged MC-Dropout model's posterior stage continues on the
    staged model's trained networks.

    Args:
        prepared: The prepared FD001 windows to train, validate and score on.
        config: Example configuration; its `seed` initializes every model.
        posterior_stage: Whether to add the MC-Dropout models.
        on_run: Called with each model's name, the seed and its test metrics
            as soon as that model is scored.

    Returns:
        The seed's `SeedComparison`.
    """

    def score(name: str, label: str, model: nnx.Module) -> ModelResult:
        metrics = score_on_test_split(model, prepared)
        on_run(name, config.seed, metrics)
        return ModelResult(name=name, label=label, model=model, metrics=metrics)

    def train_and_score_staged() -> tuple[ModelResult, TrainingState]:
        mean_model, variance_model, state = _train_staged_networks(prepared, config)
        composite = build_composite_model(mean_model, variance_model)
        return score("staged", "Staged model", composite), state

    def train_and_score_joint() -> ModelResult:
        return score("joint", "Joint model", _train_joint(prepared, config))

    if not posterior_stage:
        staged_result, _ = train_and_score_staged()
        return SeedComparison(
            seed=config.seed, models=(staged_result, train_and_score_joint())
        )

    def train_and_score_joint_mcd(joint_model: nnx.Module) -> ModelResult:
        return score(
            "joint+mcd",
            "Joint MC-Dropout model",
            _train_joint_mcd(prepared, config, joint_model),
        )

    def train_and_score_staged_mcd(state: TrainingState) -> ModelResult:
        predictive = train_staged_mcd_posterior(
            state,
            prepared.train_windows,
            prepared.train_targets,
            prepared.validation_windows,
            prepared.validation_targets,
            config,
            event_sinks=(),
        )
        refit = refit_gamma_to_posterior_mean(
            predictive,
            prepared.train_windows,
            prepared.train_targets,
            config,
            event_sinks=(),
        )
        return score("staged+mcd", "Staged MC-Dropout model", refit)

    joint_result = train_and_score_joint()
    joint_mcd_result = train_and_score_joint_mcd(joint_result.model)
    staged_result, state = train_and_score_staged()
    models = (
        joint_result,
        joint_mcd_result,
        staged_result,
        train_and_score_staged_mcd(state),
    )
    return SeedComparison(seed=config.seed, models=models)


def compare_models(
    prepared: PreparedCmapssData,
    config: CmapssConfig,
    *,
    seeds: int,
    posterior_stage: bool = False,
    on_run: RunReporter = _ignore_run,
) -> list[SeedComparison]:
    """Train and score the comparison's models once per seed.

    Every seed trains on the same `prepared` data, so only initialization
    varies: seed index `i` initializes every model, and the MC-Dropout
    models' masks, from `config.seed + i`. Training is silent. Each
    `SeedComparison` keeps its trained models, so a caller can plot, say,
    the first seed's models without retraining.

    Args:
        prepared: The prepared FD001 windows, shared by every seed.
        config: Example configuration; its `seed` is the first seed.
        seeds: Number of seeds to train the models from.
        posterior_stage: Off, the default, compares the staged and the joint
            model, in that order. On, it compares the joint, the joint
            MC-Dropout (`"joint+mcd"`), the staged and the staged MC-Dropout
            (`"staged+mcd"`) model, in that order; the joint and staged
            models are the same as with it off, and the staged MC-Dropout
            model starts from exactly the staged model's networks.
        on_run: Called with the model's name, the seed and the test metrics
            as soon as each run is scored. Does nothing by default.

    Returns:
        One `SeedComparison` per seed, in seed order.
    """
    return [
        _train_and_score_seed(
            prepared,
            dataclasses.replace(config, seed=config.seed + index),
            posterior_stage=posterior_stage,
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
    """Tabulate every model's test metrics as `mean ± std` over seeds.

    Args:
        results: The per-seed comparisons from `compare_models`; at least
            one.

    Returns:
        A table with one row per metric (RMSE, 95% coverage, closed-form
        CRPS, NLL) and one column per model result, headed by its name, in
        the comparisons' model order.
    """
    names = [model.name for model in results[0].models]
    rows = [
        (
            label,
            [
                _mean_std([result.models[index].metrics[metric] for result in results])
                for index in range(len(names))
            ],
        )
        for label, metric in _TABLE_ROWS
    ]
    label_width = max(len(label) for label, _ in rows)
    cell_width = max(len(cell) for _, cells in rows for cell in cells)

    def line(label: str, cells: Sequence[str]) -> str:
        return "  ".join(
            [f"{label:<{label_width}}", *(f"{cell:>{cell_width}}" for cell in cells)]
        )

    return "\n".join([line("", names), *(line(label, cells) for label, cells in rows)])


def print_run(model: str, seed: int, metrics: dict[str, float]) -> None:
    """Print one finished (seed, model) run's test metrics on one line.

    Args:
        model: The model's name, e.g. `"staged"` or `"joint+mcd"`.
        seed: The seed the model was initialized from.
        metrics: The model's test metrics by name.
    """
    values = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"[{model} seed={seed}] {values}")


def main() -> None:
    """Compare the CMAPSS models over seeds, print and plot the results.

    Compares `staged` and `joint`; with `--posterior-stage`, `joint`,
    `joint+mcd`, `staged` and `staged+mcd`, in that order.

    Pass `--seeds` to set the number of seeds (default 5), `--plot-path` to save the
    first seed's RUL-curves figure to that path instead of displaying it, and
    `--posterior-stage` to add the joint and the staged MC-Dropout models.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to train the models from (default: 5).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Save the first seed's validation RUL-curves figure, one row per "
            "model, to this path instead of displaying it."
        ),
    )
    parser.add_argument(
        "--posterior-stage",
        action="store_true",
        help=(
            "Add the joint and staged MC-Dropout models (joint+mcd, "
            "staged+mcd); the models are then joint, joint+mcd, staged and "
            "staged+mcd."
        ),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive.")

    config = CmapssConfig()
    prepared = prepare_cmapss_windows(config)
    results = compare_models(
        prepared,
        config,
        seeds=args.seeds,
        posterior_stage=args.posterior_stage,
        on_run=print_run,
    )
    print(f"\nFD001 test metrics over {args.seeds} seeds (mean ± std):")
    print(format_comparison_table(results))

    first = results[0]
    validation_trajectories = prepared.validation_trajectories
    plot_validation_rul_curves(
        validation_trajectories,
        prepared.standardization.feature_columns,
        [LabelledModel(result.label, result.model) for result in first.models],
        units=select_lifetime_spanning_units(validation_trajectories),
        window_length=config.window_length,
        save_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
