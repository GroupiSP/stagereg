from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from posterior_runs import DropoutLinearModel

from stagereg.core.checkpoints import (
    Checkpoint,
    CheckpointStore,
    InMemoryCheckpointStore,
)
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.losses import (
    NegativeLogLikelihoodLoss,
    SquaredErrorLoss,
    add_epsilon,
)
from stagereg.core.naming import Split
from stagereg.core.protocols import LoaderFactory, ValidationStrategy
from stagereg.core.tracking import EventSink, TrackerEventSink
from stagereg.core.types import (
    Batch,
    ParameterRole,
    StageResult,
    StageState,
    TrainingState,
    ValidationResult,
)
from stagereg.jax import HeldOutValidation, SupervisedLoss
from stagereg.jax.distributions import GammaHead
from stagereg.jax.losses import make_supervised_loss
from stagereg.jax.state import create_optimizer, restore_checkpoint
from stagereg.jax.supervised_staged import (
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
    materialize_residual_loader,
)

Tracker = Callable[[], Any]
MakeMeanStage = Callable[..., tuple[MeanStage, TrainingState]]
StagedRun = Callable[..., tuple[TrainingState, StageResult, StageResult]]


class SqueezedLinearModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(1, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.linear(inputs).squeeze(-1)


class LogPrediction:
    batch_shape = (2, 1)
    event_shape = ()

    def __init__(self, values: jax.Array) -> None:
        self.values = values

    def log_prob(self, targets: jax.Array) -> jax.Array:
        return jnp.log(targets) + self.values

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        del key
        return jnp.zeros(sample_shape + self.batch_shape)

    def mean(self) -> jax.Array:
        return self.values

    def variance(self) -> jax.Array:
        return jnp.ones_like(self.values)


class LogPredictionModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> LogPrediction:
        return LogPrediction(jnp.zeros_like(inputs))


def test_mean_squared_error_loss_supports_weights_and_reduction(
    linear_model: type[Any],
) -> None:
    model = linear_model(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    loss = make_supervised_loss(SquaredErrorLoss(), reduction=jnp.sum)

    value = loss(
        model,
        jnp.array([[1.0], [2.0]]),
        jnp.array([[1.0], [5.0]]),
        jnp.array([2.0, 3.0]),
        jax.random.key(0),
        True,
    )

    assert value == pytest.approx(5.0)


def test_gamma_residual_loss_supports_weights_and_reduction() -> None:
    loss = make_supervised_loss(
        NegativeLogLikelihoodLoss(target_transform=add_epsilon(1e-6)),
        reduction=jnp.sum,
    )

    value = loss(
        LogPredictionModel(),
        jnp.ones((2, 1)),
        jnp.array([[0.0], [1.0]]),
        jnp.array([2.0, 3.0]),
        jax.random.key(0),
        True,
    )

    expected = -2.0 * jnp.log(1e-6) - 3.0 * jnp.log(1.000001)
    assert value == pytest.approx(float(expected))


def test_supervised_loss_rejects_mismatched_sample_weight_batch(
    linear_model: type[Any],
) -> None:
    loss = make_supervised_loss(SquaredErrorLoss())

    with pytest.raises(ValueError, match="matching batch sizes"):
        loss(
            linear_model(rngs=nnx.Rngs(0)),
            jnp.ones((2, 1)),
            jnp.ones((2, 1)),
            jnp.ones((3,)),
            jax.random.key(0),
            True,
        )


def test_supervised_loss_squeezes_column_weights_for_vector_losses() -> None:
    model = SqueezedLinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    loss = make_supervised_loss(SquaredErrorLoss(), reduction=jnp.sum)

    value = loss(
        model,
        jnp.array([[1.0], [2.0]]),
        jnp.array([1.0, 5.0]),
        jnp.array([[2.0], [3.0]]),
        jax.random.key(0),
        True,
    )

    assert value == pytest.approx(5.0)


def test_gamma_residual_loss_has_finite_gradient_at_zero() -> None:
    loss = make_supervised_loss(
        NegativeLogLikelihoodLoss(target_transform=add_epsilon(1e-6))
    )

    def evaluate(target: jax.Array) -> jax.Array:
        return loss(
            LogPredictionModel(),
            jnp.ones((1, 1)),
            target,
            None,
            jax.random.key(0),
            True,
        )

    gradient = jax.grad(evaluate)(jnp.zeros((1, 1)))

    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_materialize_residual_loader_caches_exact_detached_residuals(
    linear_model: type[Any],
) -> None:
    model = linear_model(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 2.0
    model.linear.bias[...] = 0.0
    calls: list[tuple[str, int]] = []
    sample_weight = jnp.array([[0.5], [1.0]])
    metadata = {"source": "xsin"}

    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        calls.append((split, epoch))
        return [
            Batch(
                inputs=jnp.array([[1.0], [2.0]]),
                targets=jnp.array([[3.0], [2.0]]),
                sample_weight=sample_weight,
                metadata=metadata,
            )
        ]

    residual_loader = materialize_residual_loader(
        model,
        source_loader,
        splits=("train",),
        source_epoch=7,
    )
    first = residual_loader(split="train", epoch=0)
    second = residual_loader(split="train", epoch=99)

    assert calls == [("train", 7)]
    assert first is second
    assert jnp.array_equal(first[0].targets, jnp.array([[1.0], [4.0]]))
    assert first[0].sample_weight is sample_weight
    assert first[0].metadata is metadata


def test_materialized_residual_targets_stop_source_target_gradients(
    linear_model: type[Any],
) -> None:
    model = linear_model(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 0.0
    model.linear.bias[...] = 0.0

    def residual_sum(targets: jax.Array) -> jax.Array:
        def source_loader(*, split: str, epoch: int) -> list[Batch]:
            del split, epoch
            return [Batch(inputs=jnp.ones_like(targets), targets=targets)]

        loader = materialize_residual_loader(
            model,
            source_loader,
            splits=("train",),
        )
        return jnp.sum(loader(split="train", epoch=0)[0].targets)

    gradient = jax.grad(residual_sum)(jnp.array([[2.0]]))

    assert jnp.array_equal(gradient, jnp.zeros((1, 1)))


def test_materialize_residual_loader_uses_inference_clone() -> None:
    model = DropoutLinearModel(rngs=nnx.Rngs(0))
    model.linear.kernel[...] = 1.0
    model.linear.bias[...] = 0.0
    model.train()

    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        values = jnp.ones((2, 1))
        return [Batch(inputs=values, targets=values)]

    loader = materialize_residual_loader(
        model,
        source_loader,
        splits=("train",),
    )

    assert model.dropout.deterministic is False
    assert jnp.array_equal(
        loader(split="train", epoch=0)[0].targets,
        jnp.zeros((2, 1)),
    )


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ((), "at least one"),
        (("",), "must not be empty"),
        (("train", "train"), "unique"),
    ],
)
def test_materialize_residual_loader_rejects_invalid_splits(
    linear_model: type[Any], splits: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_residual_loader(
            linear_model(rngs=nnx.Rngs(0)),
            lambda **kwargs: [],
            splits=splits,
        )


def test_materialize_residual_loader_rejects_empty_source_split(
    linear_model: type[Any],
) -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        materialize_residual_loader(
            linear_model(rngs=nnx.Rngs(0)),
            lambda **kwargs: [],
            splits=("train",),
        )


def test_materialize_residual_loader_requires_targets(linear_model: type[Any]) -> None:
    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [Batch(inputs=jnp.ones((1, 1)))]

    with pytest.raises(ValueError, match="provide targets"):
        materialize_residual_loader(
            linear_model(rngs=nnx.Rngs(0)),
            source_loader,
            splits=("train",),
        )


def test_materialize_residual_loader_requires_matching_shapes(
    linear_model: type[Any],
) -> None:
    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [
            Batch(
                inputs=jnp.ones((2, 1)),
                targets=jnp.ones((2,)),
            )
        ]

    with pytest.raises(ValueError, match="matching shapes"):
        materialize_residual_loader(
            linear_model(rngs=nnx.Rngs(0)),
            source_loader,
            splits=("train",),
        )


def test_materialized_residual_loader_rejects_unknown_split(
    linear_model: type[Any],
) -> None:
    def source_loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [Batch(inputs=jnp.ones((1, 1)), targets=jnp.ones((1, 1)))]

    loader = materialize_residual_loader(
        linear_model(rngs=nnx.Rngs(0)),
        source_loader,
        splits=("train",),
    )

    with pytest.raises(ValueError, match="was not materialized"):
        loader(split="validation", epoch=0)


def mean_loader(*, split: str, epoch: int) -> list[Batch]:
    del split, epoch
    return [
        Batch(
            inputs=jnp.array([[-1.0], [0.0], [1.0]]),
            targets=jnp.array([[-2.0], [0.0], [2.0]]),
        )
    ]


@pytest.fixture(scope="session")
def make_mean_stage(
    linear_model: type[Any],
) -> MakeMeanStage:
    """Return a factory of a mean stage on a linear model, and fresh state."""

    def make(
        *,
        epochs: int = 10,
        learning_rate: float = 0.1,
        checkpoint_store: CheckpointStore | None = None,
        early_stopper: EarlyStopper | None = None,
        validation: ValidationStrategy | None = None,
    ) -> tuple[MeanStage, TrainingState]:
        """Build a mean stage on `mean_loader`, and fresh state.

        Args:
            epochs: The stage's maximum number of epochs.
            learning_rate: The SGD learning rate.
            checkpoint_store: Where the stage saves its best checkpoint, if anywhere.
            early_stopper: The stage's early stopper, if any.
            validation: The stage's validation strategy, if any.

        Returns:
            The unprepared stage and a training state seeded with key 1.
        """
        model = linear_model(rngs=nnx.Rngs(0))
        optimizer = create_optimizer(model, optax.sgd(learning_rate))
        stage = MeanStage(
            model=model,
            optimizer=optimizer,
            train_loader=mean_loader,
            options=SupervisedStageOptions(
                epochs=epochs,
                checkpoint_store=checkpoint_store,
                checkpoint_key="mean-best",
                early_stopper=early_stopper,
                validation=validation,
            ),
        )
        return stage, TrainingState(rng_state=jax.random.key(1))

    return make


def validation_loss_is_epoch(
    current_state: TrainingState,
    *,
    epoch: int,
) -> ValidationResult:
    """Report a validation loss equal to the epoch, so only epoch 0 improves."""
    del current_state
    return ValidationResult(passed=True, metrics={"loss": float(epoch)})


StopAtFirstEpoch = Callable[[], dict[str, Any]]


@pytest.fixture(scope="session")
def stop_at_first_epoch() -> StopAtFirstEpoch:
    """Return a factory of stage options whose best epoch is epoch 0."""

    def make() -> dict[str, Any]:
        """Return a fresh patience-0 loss stopper and `validation_loss_is_epoch`.

        Returns:
            The ``early_stopper`` and ``validation`` keyword arguments of a
            stage factory.
        """
        return {
            "early_stopper": EarlyStopper(metric="loss", mode="min", patience=0),
            "validation": validation_loss_is_epoch,
        }

    return make


def test_mean_stage_prepares_trains_and_validates_lifecycle(
    make_mean_stage: MakeMeanStage,
) -> None:
    stage, state = make_mean_stage()

    stage.prepare(state)
    initial_loss = float(
        stage.loss(
            stage.model,
            mean_loader(split="train", epoch=0)[0].inputs,
            mean_loader(split="train", epoch=0)[0].targets,
            None,
            jax.random.key(2),
            False,
        )
    )
    result = stage.train(state)

    assert state.lifecycle_state is StageState.MEAN_READY
    assert state.active_stage == "mean"
    assert state.model_components["mean_model"] is stage.model
    assert state.optimizer_states["mean_optimizer"] is stage.optimizer
    assert state.parameter_roles["mean_model"] is ParameterRole.MEAN
    assert result.loss is not None and result.loss < initial_loss
    assert len(state.metric_history["mean/train/loss"]) == 10
    assert "supervised/train/loss" not in state.metric_history
    assert stage.validate(state).passed


def test_mean_stage_rejects_invalid_lifecycle_without_advancing_state(
    make_mean_stage: MakeMeanStage,
) -> None:
    stage, state = make_mean_stage()
    state.lifecycle_state = StageState.VARIANCE_READY

    with pytest.raises(ValueError, match="requires NEW or INITIALIZED"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.VARIANCE_READY


def test_mean_stage_rejects_conflicting_parameter_role(
    make_mean_stage: MakeMeanStage,
) -> None:
    stage, state = make_mean_stage()
    state.parameter_roles["mean_model"] = ParameterRole.VARIANCE

    with pytest.raises(ValueError, match="already has role"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.NEW


def test_mean_stage_selects_existing_checkpoint(
    make_mean_stage: MakeMeanStage,
) -> None:
    store = InMemoryCheckpointStore()
    stopper = EarlyStopper(
        metric="loss",
        mode="min",
        patience=0,
        source=Split.TRAIN,
    )
    stage, state = make_mean_stage(
        learning_rate=0.0,
        checkpoint_store=store,
        early_stopper=stopper,
    )

    stage.prepare(state)
    stage.train(state)
    reference = stage.select_checkpoint(state)

    assert reference.key == "mean-best"
    assert reference.metadata == {"stage": "mean"}
    assert "mean/train/loss" in store.load("mean-best").state.metric_history


def test_mean_stage_restores_and_finalizes_best_checkpoint(
    make_mean_stage: MakeMeanStage,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> None:
    store = InMemoryCheckpointStore()
    stage, state = make_mean_stage(checkpoint_store=store, **stop_at_first_epoch())
    expected_stage, expected_state = make_mean_stage()
    expected_stage.options = SupervisedStageOptions(epochs=1)

    stage.prepare(state)
    result = stage.train(state)
    expected_stage.prepare(expected_state)
    expected_stage.train(expected_state)

    checkpoint = store.load("mean-best")
    assert checkpoint.epoch == 0
    assert checkpoint.state.lifecycle_state is StageState.MEAN_READY
    assert checkpoint.metadata == {"stage": "mean", "stage_complete": True}
    assert int(stage.optimizer.step.get_value()) == 1
    assert len(state.metric_history["mean/train/loss"]) == 1
    assert result.loss == state.metric_history["mean/train/loss"][-1]
    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(
            jax.tree.leaves(nnx.state(stage.model)),
            jax.tree.leaves(nnx.state(expected_stage.model)),
            strict=True,
        )
    )


def test_mean_stage_restore_skips_history_keys_that_are_not_tags(
    make_mean_stage: MakeMeanStage,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> None:
    stage, state = make_mean_stage(
        checkpoint_store=InMemoryCheckpointStore(),
        **stop_at_first_epoch(),
    )
    stage.prepare(state)
    state.metric_history["lr"] = [0.1]

    result = stage.train(state)

    assert set(result.metrics) == {"loss"}
    assert result.loss == state.metric_history["mean/train/loss"][-1]


def test_finalized_mean_checkpoint_can_resume_variance_preparation(
    make_mean_stage: MakeMeanStage,
    linear_model: type[Any],
) -> None:
    store = InMemoryCheckpointStore()
    stopper = EarlyStopper(
        metric="loss",
        mode="min",
        patience=0,
        source=Split.TRAIN,
    )
    trained_stage, trained_state = make_mean_stage(
        learning_rate=0.0,
        checkpoint_store=store,
        early_stopper=stopper,
    )
    trained_stage.prepare(trained_state)
    trained_stage.train(trained_state)

    resumed_model = linear_model(rngs=nnx.Rngs(8))
    resumed_optimizer = create_optimizer(resumed_model, optax.sgd(0.0))
    resumed_state = TrainingState(rng_state=jax.random.key(9))
    restore_checkpoint(
        store.load("mean-best"),
        state=resumed_state,
        model=resumed_model,
        optimizer=resumed_optimizer,
        model_name="mean_model",
        optimizer_name="mean_optimizer",
    )
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(10))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.01)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        splits=("train",),
    )

    variance_stage.prepare(resumed_state)

    assert resumed_state.lifecycle_state is StageState.MEAN_READY
    assert resumed_state.model_components["mean_model"] is resumed_model
    assert "mean_model" in resumed_state.frozen_components


def test_stages_sharing_a_store_save_under_their_own_default_keys(
    linear_model: type[Any],
) -> None:
    store = InMemoryCheckpointStore()

    def stopper() -> EarlyStopper:
        return EarlyStopper(metric="loss", mode="min", patience=0, source=Split.TRAIN)

    mean_model = linear_model(rngs=nnx.Rngs(0))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.sgd(0.0)),
        train_loader=mean_loader,
        options=SupervisedStageOptions(
            epochs=2,
            early_stopper=stopper(),
            checkpoint_store=store,
        ),
    )
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(1))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.0)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(
            epochs=2,
            early_stopper=stopper(),
            checkpoint_store=store,
        ),
        splits=("train",),
    )
    state = TrainingState(rng_state=jax.random.key(2))

    mean_stage.prepare(state)
    mean_stage.train(state)
    variance_stage.prepare(state)
    variance_stage.train(state)

    assert not store.exists("best")
    mean_checkpoint = store.load("mean/best")
    assert mean_checkpoint.state.lifecycle_state is StageState.MEAN_READY
    assert mean_checkpoint.metadata == {"stage": "mean", "stage_complete": True}
    assert "variance/train/loss" in store.load("variance/best").state.metric_history
    assert mean_stage.select_checkpoint(state).key == "mean/best"
    assert variance_stage.select_checkpoint(state).key == "variance/best"


def test_mean_stage_rejects_missing_checkpoint(
    make_mean_stage: MakeMeanStage,
) -> None:
    stage, state = make_mean_stage()

    with pytest.raises(ValueError, match="not available"):
        stage.select_checkpoint(state)


def make_two_step_loader(
    inputs: jax.Array,
    targets: jax.Array,
) -> LoaderFactory:
    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split, epoch
        return [Batch(inputs=inputs, targets=targets)]

    return loader


def test_gamma_variance_stage_updates_variance_and_preserves_mean(
    in_memory_tracker: Tracker, linear_model: type[Any]
) -> None:
    data_key, mean_key, variance_key, rng_key = jax.random.split(
        jax.random.key(20),
        4,
    )
    inputs = jax.random.uniform(data_key, (256, 1), minval=-2.0, maxval=2.0)
    noise_scale = 0.1 + 0.3 * (inputs + 2.0)
    targets = 2.0 * inputs + noise_scale * jax.random.normal(
        jax.random.fold_in(data_key, 1),
        inputs.shape,
    )
    source_loader = make_two_step_loader(inputs, targets)
    events = in_memory_tracker()

    mean_model = linear_model(rngs=nnx.Rngs(mean_key))
    mean_optimizer = create_optimizer(mean_model, optax.adam(0.05))
    state = TrainingState(rng_state=rng_key)
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=mean_optimizer,
        train_loader=source_loader,
        options=SupervisedStageOptions(epochs=100, event_sinks=(events,)),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)
    mean_state_before = jax.tree.map(lambda value: value.copy(), nnx.state(mean_model))

    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(variance_key))
    variance_optimizer = create_optimizer(variance_model, optax.adam(0.03))
    variance_state_before = jax.tree.map(
        lambda value: value.copy(),
        nnx.state(variance_model),
    )
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=variance_optimizer,
        source_loader=source_loader,
        options=SupervisedStageOptions(epochs=150, event_sinks=(events,)),
        splits=("train",),
    )

    variance_stage.prepare(state)
    result = variance_stage.train(state)

    assert result.loss is not None and math.isfinite(result.loss)
    assert state.lifecycle_state is StageState.VARIANCE_READY
    assert state.parameter_roles["variance_model"] is ParameterRole.VARIANCE
    assert "mean_model" in state.frozen_components
    assert len(state.metric_history["mean/train/loss"]) == 100
    assert len(state.metric_history["variance/train/loss"]) == 150
    assert "supervised/train/loss" not in state.metric_history
    assert {event.stage for event in events.events} == {"mean", "variance"}
    assert variance_stage.validate(state).passed
    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(mean_state_before),
            jax.tree.leaves(nnx.state(mean_model)),
            strict=True,
        )
    )
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(variance_state_before),
            jax.tree.leaves(nnx.state(variance_model)),
            strict=True,
        )
    )
    low_variance = variance_model(jnp.array([[-2.0]])).mean()
    high_variance = variance_model(jnp.array([[2.0]])).mean()
    assert bool(jnp.all(high_variance > low_variance))


class ZeroMeanModel(nnx.Module):
    """A mean model predicting zero, whose squared residuals are the targets squared."""

    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.zeros((1,)))

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return inputs * self.weight


def _variance_trained_on_targets_scaled_by(
    factor: float, query_inputs: jax.Array
) -> jax.Array:
    """Train a Gamma head on squared residuals scaled by ``factor**2``.

    The mean model predicts zero and never moves, so the residual targets are
    exactly the source targets squared. Data, initialization and optimizer are
    the same for every ``factor``.

    Returns:
        The trained head's variance at ``query_inputs``.
    """
    data_key, noise_key = jax.random.split(jax.random.key(7))
    inputs = jax.random.uniform(data_key, (128, 1), minval=-1.0, maxval=1.0)
    targets = factor * (0.5 + inputs) * jax.random.normal(noise_key, inputs.shape)
    source_loader = make_two_step_loader(inputs, targets)

    mean_model = ZeroMeanModel()
    state = TrainingState(rng_state=jax.random.key(0))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.sgd(0.0)),
        train_loader=source_loader,
        options=SupervisedStageOptions(epochs=1),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)

    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(3))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.adam(0.05)),
        source_loader=source_loader,
        options=SupervisedStageOptions(epochs=50),
        splits=("train",),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)
    return variance_model(query_inputs).mean()


@pytest.mark.parametrize("factor", [0.03, 30.0])
def test_gamma_variance_scales_with_the_squared_residuals(factor: float) -> None:
    # Scaling the targets by `factor` scales the squared residuals by
    # `factor**2`; the trained variance must follow by the same factor, inside
    # and outside the data. Without it, small residuals force large rate
    # logits that blow the variance up beyond the data.
    query_inputs = jnp.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])

    unit = _variance_trained_on_targets_scaled_by(1.0, query_inputs)
    scaled = _variance_trained_on_targets_scaled_by(factor, query_inputs)

    assert jnp.allclose(scaled, factor**2 * unit, rtol=1e-3)


def test_gamma_variance_stage_requires_ready_mean() -> None:
    model = GammaHead(1, 1, rngs=nnx.Rngs(0))
    stage = GammaVarianceStage(
        model=model,
        optimizer=create_optimizer(model, optax.sgd(0.1)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        splits=("train",),
    )
    state = TrainingState(rng_state=jax.random.key(0))

    with pytest.raises(ValueError, match="MEAN_READY"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.NEW


def test_gamma_variance_stage_requires_mean_role(linear_model: type[Any]) -> None:
    mean_model = linear_model(rngs=nnx.Rngs(0))
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(1))
    state = TrainingState(
        model_components={"mean_model": mean_model},
        parameter_roles={"mean_model": ParameterRole.AUXILIARY},
        lifecycle_state=StageState.MEAN_READY,
        rng_state=jax.random.key(0),
    )
    stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.1)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        splits=("train",),
    )

    with pytest.raises(ValueError, match="MEAN parameter role"):
        stage.prepare(state)

    assert "variance_model" not in state.model_components


def test_gamma_variance_stage_builds_validation_from_residual_loader(
    make_mean_stage: MakeMeanStage,
) -> None:
    mean_stage, state = make_mean_stage()
    mean_stage.prepare(state)
    mean_stage.train(state)
    variance_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
    observed_targets: list[jax.Array] = []

    def validation_factory(loader: LoaderFactory):
        def validation(
            current_state: TrainingState,
            *,
            epoch: int,
        ) -> ValidationResult:
            del current_state
            observed_targets.append(loader(split="validation", epoch=epoch)[0].targets)
            return ValidationResult(passed=True, metrics={"loss": 0.0})

        return validation

    stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.sgd(0.01)),
        source_loader=mean_loader,
        options=SupervisedStageOptions(epochs=1),
        validation_factory=validation_factory,
    )

    stage.prepare(state)
    stage.train(state)

    assert len(observed_targets) == 1
    assert bool(jnp.all(observed_targets[0] >= 0.0))


def offset_validation_loader(*, split: str, epoch: int) -> list[Batch]:
    """`mean_loader`, with validation targets shifted by 0.5.

    The shift keeps validation loss distinct from train loss, so a test reading either
    split cannot pass by reading the other.
    """
    offset = 0.5 if split == Split.VALIDATION else 0.0
    (batch,) = mean_loader(split=split, epoch=epoch)
    assert batch.targets is not None
    return [Batch(inputs=batch.inputs, targets=batch.targets + offset)]


@pytest.fixture(scope="session")
def staged_run(linear_model: type[Any], squared_error: SupervisedLoss) -> StagedRun:
    """Return a driver of a real staged mean-then-variance run."""

    def run(
        *sinks: EventSink,
        early_stopper: EarlyStopper | None = None,
        validate: bool = True,
        learning_rate: float = 0.1,
    ) -> tuple[TrainingState, StageResult, StageResult]:
        """Train a mean stage, then a Gamma variance stage on its residuals.

        Args:
            *sinks: The event sinks to attach to both stages.
            early_stopper: The mean stage's early stopper, if any.
            validate: Whether both stages validate every epoch.
            learning_rate: The mean stage's SGD learning rate.

        Returns:
            The shared training state and the two stages' results.
        """
        mean_model = linear_model(rngs=nnx.Rngs(0))
        state = TrainingState(rng_state=jax.random.key(1))
        mean_stage = MeanStage(
            model=mean_model,
            optimizer=create_optimizer(mean_model, optax.sgd(learning_rate)),
            train_loader=offset_validation_loader,
            options=SupervisedStageOptions(
                epochs=4,
                validation=(
                    HeldOutValidation(
                        model=mean_model,
                        loader=offset_validation_loader,
                        loss=squared_error,
                    )
                    if validate
                    else None
                ),
                early_stopper=early_stopper,
                event_sinks=sinks,
                checkpoint_store=InMemoryCheckpointStore(),
                checkpoint_key="mean-best",
            ),
        )
        mean_stage.prepare(state)
        mean_result = mean_stage.train(state)

        variance_model = GammaHead(1, 1, rngs=nnx.Rngs(2))
        variance_stage = GammaVarianceStage(
            model=variance_model,
            optimizer=create_optimizer(variance_model, optax.sgd(0.01)),
            source_loader=offset_validation_loader,
            options=SupervisedStageOptions(epochs=3, event_sinks=sinks),
            validation_factory=(
                (
                    lambda residuals: HeldOutValidation(
                        model=variance_model,
                        loader=residuals,
                        loss=variance_stage.loss,
                    )
                )
                if validate
                else None
            ),
        )
        variance_stage.prepare(state)
        variance_result = variance_stage.train(state)
        return state, mean_result, variance_result

    return run


def test_the_two_stages_of_a_staged_run_never_share_a_tag(
    in_memory_tracker: Tracker,
    staged_run: StagedRun,
) -> None:
    recorder = in_memory_tracker()
    tracker = in_memory_tracker()

    state, _, _ = staged_run(recorder, TrackerEventSink(tracker))

    expected = {
        "mean/train/loss",
        "mean/validation/loss",
        "variance/train/loss",
        "variance/validation/loss",
    }
    assert {
        f"{event.stage}/{event.split}/{metric}"
        for event in recorder.events
        for metric in event.metrics
    } == expected
    assert tracker.tags == expected
    assert set(state.metric_history) == expected


@pytest.mark.parametrize("validate", [False, True])
def test_staged_run_without_a_stopper_returns_bare_train_only_metrics(
    staged_run: StagedRun,
    validate: bool,
) -> None:
    state, mean_result, _ = staged_run(validate=validate)

    assert set(mean_result.metrics) == {"loss"}
    assert mean_result.loss == state.metric_history["mean/train/loss"][-1]


@given(source=st.sampled_from(Split), validate=st.booleans())
@settings(deadline=None, max_examples=4)
def test_staged_restore_returns_bare_train_only_metrics(
    staged_run: StagedRun,
    source: Split,
    validate: bool,
) -> None:
    # A zero learning rate makes epoch 0 the best, so the stopper halts at
    # epoch 1 and the stage restores the epoch-0 checkpoint.
    validate = validate or source is Split.VALIDATION
    stopper = EarlyStopper(metric="loss", mode="min", patience=0, source=source)

    state, mean_result, _ = staged_run(
        early_stopper=stopper, validate=validate, learning_rate=0.0
    )

    restored_history = state.metric_history["mean/train/loss"]
    assert len(restored_history) == 1
    assert set(mean_result.metrics) == {"loss"}
    assert mean_result.loss == restored_history[-1]


MakeVarianceStage = Callable[..., tuple[GammaVarianceStage, TrainingState]]


@pytest.fixture(scope="session")
def make_variance_stage(make_mean_stage: MakeMeanStage) -> MakeVarianceStage:
    """Return a factory of a variance stage on a trained mean stage's state."""

    def make(
        *,
        epochs: int = 5,
        checkpoint_store: CheckpointStore | None = None,
        early_stopper: EarlyStopper | None = None,
        validation: ValidationStrategy | None = None,
        mean: tuple[MeanStage, TrainingState] | None = None,
    ) -> tuple[GammaVarianceStage, TrainingState]:
        """Train a mean stage, then build an unprepared variance stage after it.

        Args:
            epochs: The variance stage's maximum number of epochs.
            checkpoint_store: Where the variance stage saves its best checkpoint.
            early_stopper: The variance stage's early stopper, if any.
            validation: The variance stage's validation strategy, if any.
            mean: The unprepared mean stage and fresh state to train first.
                Defaults to a plain `make_mean_stage()`.

        Returns:
            The unprepared variance stage and the `MEAN_READY` training state.
        """
        mean_stage, state = mean if mean is not None else make_mean_stage()
        mean_stage.prepare(state)
        mean_stage.train(state)
        model = GammaHead(1, 1, rngs=nnx.Rngs(2))
        stage = GammaVarianceStage(
            model=model,
            optimizer=create_optimizer(model, optax.sgd(0.1)),
            source_loader=mean_loader,
            options=SupervisedStageOptions(
                epochs=epochs,
                checkpoint_store=checkpoint_store,
                checkpoint_key="variance-best",
                early_stopper=early_stopper,
                validation=validation,
            ),
            splits=("train",),
        )
        return stage, state

    return make


def test_variance_stage_restores_and_finalizes_best_checkpoint(
    make_variance_stage: MakeVarianceStage,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> None:
    store = InMemoryCheckpointStore()
    stage, state = make_variance_stage(
        checkpoint_store=store,
        **stop_at_first_epoch(),
    )
    expected_stage, expected_state = make_variance_stage(epochs=1)
    mean_model = state.model_components["mean_model"]
    mean_optimizer = state.optimizer_states["mean_optimizer"]

    stage.prepare(state)
    result = stage.train(state)
    expected_stage.prepare(expected_state)
    expected_stage.train(expected_state)

    checkpoint = store.load("variance-best")
    assert checkpoint.epoch == 0
    assert checkpoint.state.lifecycle_state is StageState.VARIANCE_READY
    assert checkpoint.metadata == {"stage": "variance", "stage_complete": True}
    assert int(stage.optimizer.step.get_value()) == 1
    assert len(state.metric_history["variance/train/loss"]) == 1
    assert result.loss == state.metric_history["variance/train/loss"][-1]
    assert stage.validate(state).passed
    assert state.model_components["mean_model"] is mean_model
    assert state.optimizer_states["mean_optimizer"] is mean_optimizer
    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(
            jax.tree.leaves(nnx.state(stage.model)),
            jax.tree.leaves(nnx.state(expected_stage.model)),
            strict=True,
        )
    )


def test_variance_stage_without_a_store_keeps_its_last_epoch(
    make_variance_stage: MakeVarianceStage,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> None:
    stage, state = make_variance_stage(
        **stop_at_first_epoch(),
    )

    stage.prepare(state)
    result = stage.train(state)

    assert int(stage.optimizer.step.get_value()) == 2
    assert len(state.metric_history["variance/train/loss"]) == 2
    assert result.loss == state.metric_history["variance/train/loss"][-1]
    assert stage.validate(state).passed


def test_variance_stage_restore_leaves_an_unregistered_mean_optimizer_out(
    make_variance_stage: MakeVarianceStage,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> None:
    stage, state = make_variance_stage(
        checkpoint_store=InMemoryCheckpointStore(),
        **stop_at_first_epoch(),
    )
    del state.optimizer_states["mean_optimizer"]

    stage.prepare(state)
    stage.train(state)

    assert "mean_optimizer" not in state.optimizer_states
    assert stage.validate(state).passed


FinalizedRun = tuple[MeanStage, GammaVarianceStage, InMemoryCheckpointStore]
MakeFreshStages = Callable[[], tuple[MeanStage, GammaVarianceStage]]


@pytest.fixture(scope="session")
def finalized_run(
    make_mean_stage: MakeMeanStage,
    make_variance_stage: MakeVarianceStage,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> FinalizedRun:
    """Train both stages into one store, leaving a finalized checkpoint for each."""
    store = InMemoryCheckpointStore()
    mean = make_mean_stage(checkpoint_store=store, **stop_at_first_epoch())
    variance_stage, state = make_variance_stage(
        epochs=3,
        checkpoint_store=store,
        mean=mean,
        **stop_at_first_epoch(),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)
    return mean[0], variance_stage, store


@pytest.fixture(scope="session")
def make_fresh_stages(linear_model: type[Any]) -> MakeFreshStages:
    """Return a factory of untrained stages with freshly initialized objects."""

    def make() -> tuple[MeanStage, GammaVarianceStage]:
        """Build both stages as a later process would, from new objects only."""
        mean_model = linear_model(rngs=nnx.Rngs(7))
        variance_model = GammaHead(1, 1, rngs=nnx.Rngs(8))
        mean_stage = MeanStage(
            model=mean_model,
            optimizer=create_optimizer(mean_model, optax.sgd(0.1)),
            train_loader=mean_loader,
            options=SupervisedStageOptions(epochs=1),
        )
        variance_stage = GammaVarianceStage(
            model=variance_model,
            optimizer=create_optimizer(variance_model, optax.sgd(0.1)),
            source_loader=mean_loader,
            options=SupervisedStageOptions(epochs=1),
            splits=("train",),
        )
        return mean_stage, variance_stage

    return make


def test_stages_resume_a_fresh_state_from_their_finalized_checkpoints(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
) -> None:
    trained_mean, trained_variance, store = finalized_run
    mean_stage, variance_stage = make_fresh_stages()
    state = TrainingState()
    inputs = mean_loader(split="train", epoch=0)[0].inputs

    mean_stage.restore(state, store.load("mean-best"))
    variance_stage.restore(state, store.load("variance-best"))

    assert variance_stage.validate(state).passed
    assert state.model_components["mean_model"] is mean_stage.model
    assert state.optimizer_states["mean_optimizer"] is mean_stage.optimizer
    assert jnp.array_equal(mean_stage.model(inputs), trained_mean.model(inputs))
    restored, trained = variance_stage.model(inputs), trained_variance.model(inputs)
    assert jnp.array_equal(restored.concentration, trained.concentration)
    assert jnp.array_equal(restored.rate, trained.rate)


@settings(deadline=None, max_examples=5)
@given(epochs=st.integers(min_value=1, max_value=4))
def test_a_mean_stage_without_a_stopper_finalizes_its_last_epoch(
    make_mean_stage: MakeMeanStage,
    make_fresh_stages: MakeFreshStages,
    epochs: int,
) -> None:
    store = InMemoryCheckpointStore()
    trained, state = make_mean_stage(epochs=epochs, checkpoint_store=store)
    trained.prepare(state)
    result = trained.train(state)
    fresh, _ = make_fresh_stages()
    fresh_state = TrainingState()
    inputs = mean_loader(split="train", epoch=0)[0].inputs

    checkpoint = store.load("mean-best")
    fresh.restore(fresh_state, checkpoint)

    assert checkpoint.epoch == epochs - 1
    assert len(state.metric_history["mean/train/loss"]) == epochs
    assert result.loss == state.metric_history["mean/train/loss"][-1]
    assert fresh.validate(fresh_state).passed
    assert jnp.array_equal(fresh.model(inputs), trained.model(inputs))


@settings(deadline=None, max_examples=5)
@given(epochs=st.integers(min_value=1, max_value=4))
def test_a_variance_stage_without_a_stopper_finalizes_its_last_epoch(
    make_mean_stage: MakeMeanStage,
    make_variance_stage: MakeVarianceStage,
    make_fresh_stages: MakeFreshStages,
    epochs: int,
) -> None:
    store = InMemoryCheckpointStore()
    trained, state = make_variance_stage(
        epochs=epochs,
        checkpoint_store=store,
        mean=make_mean_stage(epochs=epochs, checkpoint_store=store),
    )
    trained.prepare(state)
    result = trained.train(state)
    fresh_mean, fresh = make_fresh_stages()
    fresh_state = TrainingState()
    fresh_mean.restore(fresh_state, store.load("mean-best"))
    inputs = mean_loader(split="train", epoch=0)[0].inputs

    checkpoint = store.load("variance-best")
    fresh.restore(fresh_state, checkpoint)

    assert checkpoint.epoch == epochs - 1
    assert len(state.metric_history["variance/train/loss"]) == epochs
    assert result.loss == state.metric_history["variance/train/loss"][-1]
    assert fresh.validate(fresh_state).passed
    restored, live = fresh.model(inputs), trained.model(inputs)
    assert jnp.array_equal(restored.concentration, live.concentration)
    assert jnp.array_equal(restored.rate, live.rate)


def test_a_stage_restore_round_trips_the_active_stage_and_the_random_key(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
) -> None:
    _, _, store = finalized_run
    mean_stage, variance_stage = make_fresh_stages()
    fresh_key = jax.random.key(123)
    state = TrainingState(rng_state=fresh_key)

    for stage, key in ((mean_stage, "mean-best"), (variance_stage, "variance-best")):
        saved = store.load(key)
        stage.restore(state, saved)

        assert state.active_stage == stage.name
        assert jnp.array_equal(
            jax.random.key_data(state.rng_state),
            jax.random.key_data(saved.state.rng_state),
        )
        assert not jnp.array_equal(
            jax.random.key_data(state.rng_state), jax.random.key_data(fresh_key)
        )


def test_variance_restore_before_mean_restore_is_refused(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
) -> None:
    _, _, store = finalized_run
    _, variance_stage = make_fresh_stages()
    state = TrainingState()

    with pytest.raises(ValueError, match="mean model component 'mean_model'"):
        variance_stage.restore(state, store.load("variance-best"))

    assert state.lifecycle_state is StageState.NEW
    assert state.model_components == {}


class FirstSaveStore(InMemoryCheckpointStore):
    """A store that also keeps the first checkpoint saved under each key."""

    def __init__(self) -> None:
        super().__init__()
        self.first: dict[str, Checkpoint] = {}

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.first.setdefault(key, checkpoint)
        super().save(key, checkpoint)


def test_stage_restore_refuses_a_best_checkpoint_that_was_not_finalized(
    make_mean_stage: MakeMeanStage,
    make_fresh_stages: MakeFreshStages,
    stop_at_first_epoch: StopAtFirstEpoch,
) -> None:
    store = FirstSaveStore()
    trained, trained_state = make_mean_stage(
        checkpoint_store=store,
        **stop_at_first_epoch(),
    )
    trained.prepare(trained_state)
    trained.train(trained_state)
    mean_stage, _ = make_fresh_stages()
    state = TrainingState()

    with pytest.raises(ValueError, match="finalized"):
        mean_stage.restore(state, store.first["mean-best"])

    assert state.lifecycle_state is StageState.NEW
    assert state.model_components == {}


def test_mean_stage_refuses_the_variance_checkpoint(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
) -> None:
    _, _, store = finalized_run
    mean_stage, _ = make_fresh_stages()
    state = TrainingState()

    with pytest.raises(ValueError, match="finalized"):
        mean_stage.restore(state, store.load("variance-best"))

    assert state.lifecycle_state is StageState.NEW


def test_variance_stage_refuses_the_mean_checkpoint(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
) -> None:
    _, _, store = finalized_run
    mean_stage, variance_stage = make_fresh_stages()
    state = TrainingState()
    mean_stage.restore(state, store.load("mean-best"))

    with pytest.raises(ValueError, match="finalized"):
        variance_stage.restore(state, store.load("mean-best"))

    assert mean_stage.validate(state).passed


@pytest.mark.parametrize(
    ("stage_name", "renamed"),
    [
        ("mean", "model_name"),
        ("variance", "model_name"),
        ("variance", "mean_model_name"),
    ],
)
def test_stage_restore_refuses_a_checkpoint_saved_under_other_component_names(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
    stage_name: str,
    renamed: str,
) -> None:
    _, _, store = finalized_run
    stages = make_fresh_stages()
    mean_stage, variance_stage = stages
    state = TrainingState()
    if stage_name == "variance":
        mean_stage.restore(state, store.load("mean-best"))
    stage = mean_stage if stage_name == "mean" else variance_stage
    setattr(stage, renamed, "renamed")
    before = _observable_state(state, stages)

    with pytest.raises(ValueError, match="other component names"):
        stage.restore(state, store.load(f"{stage_name}-best"))

    assert _observable_state(state, stages) == before


def _observable_state(
    state: TrainingState,
    stages: tuple[MeanStage, GammaVarianceStage],
) -> tuple[Any, ...]:
    """Return what a refused restore must leave unchanged, as comparable values."""
    weights = tuple(
        tuple(
            jnp.asarray(leaf).tolist() for leaf in jax.tree.leaves(nnx.state(s.model))
        )
        for s in stages
    )
    return (
        dict(state.model_components),
        dict(state.optimizer_states),
        dict(state.parameter_roles),
        frozenset(state.frozen_components),
        state.lifecycle_state,
        weights,
    )


@given(
    stage_name=st.sampled_from(["mean", "variance"]),
    lifecycle=st.sampled_from(StageState),
    metadata_stage=st.sampled_from(["mean", "variance", "other", None]),
    stage_complete=st.sampled_from([True, False, None]),
    mean_restored=st.booleans(),
)
@settings(deadline=None, max_examples=30)
def test_a_refused_restore_leaves_the_state_and_live_models_unchanged(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
    stage_name: str,
    lifecycle: StageState,
    metadata_stage: str | None,
    stage_complete: bool | None,
    mean_restored: bool,
) -> None:
    _, _, store = finalized_run
    stages = make_fresh_stages()
    mean_stage, variance_stage = stages
    stage = mean_stage if stage_name == "mean" else variance_stage
    ready = StageState.MEAN_READY if stage_name == "mean" else StageState.VARIANCE_READY
    finalized = lifecycle is ready and metadata_stage == stage_name and stage_complete
    mean_missing = stage_name == "variance" and not mean_restored
    assume(not finalized or mean_missing)
    finalized_checkpoint = store.load(f"{stage_name}-best")
    metadata = {
        key: value
        for key, value in {
            "stage": metadata_stage,
            "stage_complete": stage_complete,
        }.items()
        if value is not None
    }
    checkpoint = dataclasses.replace(
        finalized_checkpoint,
        state=dataclasses.replace(
            finalized_checkpoint.state, lifecycle_state=lifecycle
        ),
        metadata=metadata,
    )
    state = TrainingState()
    if mean_restored:
        mean_stage.restore(state, store.load("mean-best"))
    before = _observable_state(state, stages)

    with pytest.raises(ValueError):
        stage.restore(state, checkpoint)

    assert _observable_state(state, stages) == before


@given(
    stage_name=st.sampled_from(["mean", "variance"]),
    prior_lifecycle=st.sampled_from(StageState),
    keep_mean_optimizer=st.booleans(),
    stray_component=st.booleans(),
)
@settings(deadline=None, max_examples=20)
def test_a_restored_finalized_checkpoint_passes_the_stage_validation(
    finalized_run: FinalizedRun,
    make_fresh_stages: MakeFreshStages,
    stage_name: str,
    prior_lifecycle: StageState,
    keep_mean_optimizer: bool,
    stray_component: bool,
) -> None:
    _, _, store = finalized_run
    mean_stage, variance_stage = make_fresh_stages()
    stage = mean_stage if stage_name == "mean" else variance_stage
    state = TrainingState()
    mean_stage.restore(state, store.load("mean-best"))
    if not keep_mean_optimizer:
        del state.optimizer_states["mean_optimizer"]
    if stray_component:
        state.register_component("stray", nnx.Linear(1, 1, rngs=nnx.Rngs(3)))
    state.lifecycle_state = prior_lifecycle

    stage.restore(state, store.load(f"{stage_name}-best"))

    assert stage.validate(state).passed
    assert state.model_components["mean_model"] is mean_stage.model
    if stage_name == "variance":
        assert ("mean_optimizer" in state.optimizer_states) is keep_mean_optimizer
