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

Run it with:

    uv run --group example-cmapss python examples/jax/cmapss/compare.py --seeds 5
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import optax
from flax import nnx

# Allow sibling-module imports both when run as a script and when loaded
# from an arbitrary working directory, e.g. via `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from model import Cnn1DGaussianModel, CompositeGaussianModel
from run import (
    TEST_METRICS,
    CmapssConfig,
    PreparedCmapssData,
    build_composite_model,
    evaluate_test_metrics,
    make_window_loader,
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


def _train_and_score_seed(
    prepared: PreparedCmapssData, config: CmapssConfig
) -> SeedComparison:
    """Train and score the staged and the joint model from `config.seed`."""
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
    joint_model = nnx.clone(train_joint_model(prepared, config, event_sinks=()))
    joint_model.eval()

    def score(model: nnx.Module) -> dict[str, float]:
        return evaluate_test_metrics(
            model,
            prepared.test_windows,
            prepared.test_rul,
            epoch_metrics=COMPARISON_METRICS,
        )

    return SeedComparison(
        seed=config.seed,
        staged_model=staged_model,
        joint_model=joint_model,
        staged_metrics=score(staged_model),
        joint_metrics=score(joint_model),
    )


def compare_models(
    prepared: PreparedCmapssData, config: CmapssConfig, *, seeds: int
) -> list[SeedComparison]:
    """Train and score the staged and the joint model once per seed.

    Every seed trains on the same `prepared` data, so only initialization
    varies: seed index `i` initializes both models from `config.seed + i`.
    Training is silent.

    Args:
        prepared: The prepared FD001 windows, shared by every seed.
        config: Example configuration; its `seed` is the first seed.
        seeds: Number of seeds to train both models from.

    Returns:
        One `SeedComparison` per seed, in seed order.
    """
    return [
        _train_and_score_seed(
            prepared, dataclasses.replace(config, seed=config.seed + index)
        )
        for index in range(seeds)
    ]
