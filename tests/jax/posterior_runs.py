"""Shared problems, networks and runs of the posterior tests.

The posterior-stage tests of every inference method start from the same
variance-ready state, so its data, loader and stage builders live here once.
The `make_stages` and `variance_ready_run` fixtures in `conftest.py` wrap the
builders; the plain names below are imported directly because module-level
helpers and Hypothesis `@given` bodies cannot take fixtures.

The inference methods' own tests share a hand-built Bayesian linear
regression problem with a known posterior, and the dropout tests share the
dropout networks below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core.checkpoints import (
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
)
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.protocols import LoaderFactory
from stagereg.core.tracking import EventSink
from stagereg.core.types import Batch, PyTree, TrainingState
from stagereg.jax import (
    GammaHead,
    GammaVarianceStage,
    HeldOutValidation,
    InferenceMethod,
    MeanStage,
    PosteriorProblem,
    Prior,
    SupervisedLoss,
    SupervisedStageOptions,
    create_optimizer,
)

#: The number of training examples ``N`` of `regression_loader`.
DATASET_SIZE = 16
_BATCH_SIZE = 8


def regression_data() -> tuple[jax.Array, jax.Array]:
    """Sixteen noisy points on the line ``y = 2x``."""
    data_key = jax.random.key(3)
    inputs = jnp.linspace(-1.0, 1.0, DATASET_SIZE)[:, None]
    targets = 2.0 * inputs + 0.3 * jax.random.normal(data_key, inputs.shape)
    return inputs, targets


def regression_loader(*, split: str, epoch: int) -> list[Batch]:
    """Two batches of eight; the validation split is shifted by 0.25."""
    del epoch
    inputs, targets = regression_data()
    offset = 0.25 if split == "validation" else 0.0
    return [
        Batch(
            inputs=inputs[start : start + _BATCH_SIZE],
            targets=targets[start : start + _BATCH_SIZE] + offset,
        )
        for start in range(0, DATASET_SIZE, _BATCH_SIZE)
    ]


class RecordingStore(InMemoryCheckpointStore):
    """A store that also records every checkpoint saved, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.saves: list[tuple[str, Checkpoint]] = []

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.saves.append((key, checkpoint))
        super().save(key, checkpoint)


@dataclass
class VarianceReadyRun:
    """A training state after real mean and variance stages."""

    state: TrainingState
    mean_model: nnx.Module
    variance_model: nnx.Module


MakeVarianceReadyRun = Callable[..., VarianceReadyRun]


@dataclass
class MeanAndVarianceStages:
    """A mean and a variance stage with fresh models, ready to train or restore."""

    mean: MeanStage
    variance: GammaVarianceStage


MakeStages = Callable[..., MeanAndVarianceStages]


def build_stages(
    linear_model: type[Any],
    squared_error: SupervisedLoss,
    *sinks: EventSink,
    seed: int = 0,
    checkpoint_store: CheckpointStore | None = None,
    loader: LoaderFactory = regression_loader,
) -> MeanAndVarianceStages:
    """Build both stages; a checkpoint store also adds patient early stoppers.

    Args:
        linear_model: The mean model's class.
        squared_error: The mean stage's validation loss.
        *sinks: The event sinks attached to both stages.
        seed: Seeds the fresh models' initial weights.
        checkpoint_store: Where both stages save their checkpoints.
        loader: The mean stage's training and validation loader, and the
            variance stage's source loader.

    Returns:
        The two stages.
    """
    stopper = (
        None
        if checkpoint_store is None
        else EarlyStopper(metric="loss", mode="min", patience=100)
    )
    mean_model = linear_model(rngs=nnx.Rngs(seed))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.adam(0.1)),
        train_loader=loader,
        options=SupervisedStageOptions(
            epochs=20,
            validation=HeldOutValidation(
                model=mean_model, loader=loader, loss=squared_error
            ),
            event_sinks=sinks,
            early_stopper=stopper,
            checkpoint_store=checkpoint_store,
        ),
    )
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(seed + 2))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.adam(0.05)),
        source_loader=loader,
        options=SupervisedStageOptions(
            epochs=5,
            event_sinks=sinks,
            early_stopper=stopper,
            checkpoint_store=checkpoint_store,
        ),
        validation_factory=lambda residuals: HeldOutValidation(
            model=variance_model, loader=residuals, loss=variance_stage.loss
        ),
    )
    return MeanAndVarianceStages(mean_stage, variance_stage)


def run_to_variance_ready(
    stages: MeanAndVarianceStages, *, mean_only: bool = False
) -> VarianceReadyRun:
    """Train a validated mean stage and, unless ``mean_only``, a variance stage.

    Args:
        stages: Fresh stages from `build_stages`.
        mean_only: Stop after the mean stage.

    Returns:
        The shared state and the two trained models.
    """
    state = TrainingState(rng_state=jax.random.key(1))
    stages.mean.prepare(state)
    stages.mean.train(state)
    if not mean_only:
        stages.variance.prepare(state)
        stages.variance.train(state)
    return VarianceReadyRun(state, stages.mean.model, stages.variance.model)


def restore_mean_and_variance(
    stages: MeanAndVarianceStages, store: InMemoryCheckpointStore
) -> TrainingState:
    """Restore the finalized mean and variance checkpoints into a fresh state."""
    state = TrainingState()
    stages.mean.restore(state, store.load("mean/best"))
    stages.variance.restore(state, store.load("variance/best"))
    return state


def leaves_equal(left: PyTree, right: PyTree) -> bool:
    """Whether two trees hold exactly equal leaves."""
    return all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


# Bayesian linear regression y = w x + b + noise with known noise. The inputs
# are symmetric about zero, so the design's Gram matrix is diagonal and the
# exact posterior over (w, b) factorizes.
LINE_INPUTS = jnp.linspace(-1.0, 1.0, 20)[:, None]
LINE_NOISE_STD = 0.5
LINE_PRIOR_PRECISION = 2.0
_LINE_BATCH_SIZE = 10


def line_targets() -> jax.Array:
    """Noisy targets of the line ``y = 1.5 x - 0.5`` at `LINE_INPUTS`."""
    noise = LINE_NOISE_STD * jax.random.normal(jax.random.key(0), LINE_INPUTS.shape)
    return 1.5 * LINE_INPUTS - 0.5 + noise


def line(parameters: PyTree, inputs: PyTree, key: jax.Array) -> jax.Array:
    """The line ``w x + b``; the key is ignored."""
    del key
    return inputs * parameters["w"] + parameters["b"]


def line_log_likelihood(parameters: PyTree, batch: Batch, key: jax.Array) -> jax.Array:
    """The batch's Gaussian log-likelihood of `line`, scaled by ``N / B``."""
    targets = jnp.asarray(batch.targets)
    log_density = jax.scipy.stats.norm.logpdf(
        targets, line(parameters, batch.inputs, key), LINE_NOISE_STD
    )
    return LINE_INPUTS.shape[0] / targets.shape[0] * jnp.sum(log_density)


def line_loader(*, split: str, epoch: int) -> list[Batch]:
    """Two batches of ten of the line's data, whatever the split and epoch."""
    del split, epoch
    targets = line_targets()
    return [
        Batch(
            inputs=LINE_INPUTS[start : start + _LINE_BATCH_SIZE],
            targets=targets[start : start + _LINE_BATCH_SIZE],
        )
        for start in range(0, LINE_INPUTS.shape[0], _LINE_BATCH_SIZE)
    ]


def line_problem(prior: Prior) -> PosteriorProblem:
    """The line's posterior problem under ``prior``, warm-started at zero."""
    return PosteriorProblem(
        initial_parameters={"w": jnp.zeros(()), "b": jnp.zeros(())},
        mean_function=line,
        log_likelihood=line_log_likelihood,
        prior=prior,
        train_loader=line_loader,
        dataset_size=LINE_INPUTS.shape[0],
    )


def line_posterior() -> tuple[jax.Array, jax.Array]:
    """The exact posterior mean and covariance of ``(w, b)``."""
    design = jnp.concatenate([LINE_INPUTS, jnp.ones_like(LINE_INPUTS)], axis=1)
    precision = (
        LINE_PRIOR_PRECISION * jnp.eye(2) + design.T @ design / LINE_NOISE_STD**2
    )
    covariance = jnp.linalg.inv(precision)
    mean = covariance @ design.T @ line_targets()[:, 0] / LINE_NOISE_STD**2
    return mean, covariance


def fit(method: InferenceMethod, problem: PosteriorProblem, epochs: int) -> None:
    """Initialize ``method`` on ``problem`` and update it for ``epochs`` epochs."""
    method.init(problem)
    key = jax.random.key(1)
    for epoch in range(epochs):
        for batch in problem.train_loader(split="train", epoch=epoch):
            key, step_key = jax.random.split(key)
            method.update(batch, step_key)


class DropoutLinearModel(nnx.Module):
    """A linear layer behind an always-active input dropout.

    It has the parameter tree of the ``linear_model`` fixture's class.
    """

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(self.dropout(inputs))


MLP_HIDDEN = 8


class MLP(nnx.Module):
    """A mean network with one hidden layer of `MLP_HIDDEN` units."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(1, MLP_HIDDEN, rngs=rngs)
        self.output = nnx.Linear(MLP_HIDDEN, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(nnx.relu(self.hidden(inputs)))


class DropoutMLP(MLP):
    """The `MLP` with one hidden dropout mask shared by the whole batch."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(rngs=rngs)
        self.dropout = nnx.Dropout(0.5, broadcast_dims=(0,), rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(self.dropout(nnx.relu(self.hidden(inputs))))
