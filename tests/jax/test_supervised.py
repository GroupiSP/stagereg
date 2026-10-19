from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from stagereg.core.checkpoints import InMemoryCheckpointStore
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.losses import NegativeLogLikelihoodLoss
from stagereg.core.metric_registry import (
    EpochPredictionData,
    EvaluationGrid,
    PointContinuousRankedProbabilityScore,
    RootMeanSquaredError,
)
from stagereg.core.naming import Split
from stagereg.core.protocols import LoaderFactory
from stagereg.core.tracking import Decision, TrackerEventSink
from stagereg.core.types import Batch, StageResult, TrainingState, ValidationResult
from stagereg.jax import (
    BatchMetricSpec,
    GaussianHead,
    GaussianPredictor,
    HeldOutValidation,
    MetricSuite,
    PredictionRequirements,
    SupervisedLoss,
    create_optimizer,
    evaluate_loader,
    initialize_training_state,
    make_supervised_loss,
    make_train_step,
    run_supervised,
    split_key,
)

DECISION_EVENTS = {"best_model", "early_stop"}

Tracker = Callable[[], Any]
Run = Callable[..., StageResult]
MakeComponents = Callable[..., tuple[Any, nnx.Optimizer, TrainingState]]


def linear_predictor(
    model: nnx.Module,
    batch: Batch,
    requirements: PredictionRequirements,
    key: jax.Array,
    /,
) -> EpochPredictionData:
    del requirements, key
    if batch.targets is None:
        raise ValueError("batch.targets must be provided for epoch metrics.")
    return EpochPredictionData(
        targets=np.asarray(jax.device_get(batch.targets), dtype=float).reshape(-1),
        mean=np.asarray(jax.device_get(model(batch.inputs)), dtype=float).reshape(-1),
    )


def mean_absolute_error(
    model: nnx.Module,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    del key, training
    errors = jnp.abs(model(inputs) - targets)
    if sample_weight is not None:
        errors = errors * sample_weight
    return jnp.mean(errors)


def training_flag_metric(
    model: nnx.Module,
    inputs: jax.Array,
    targets: jax.Array,
    sample_weight: jax.Array | None,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    del model, inputs, targets, sample_weight, key
    return jnp.asarray(1.0 if training else 0.0)


@pytest.fixture(scope="session")
def make_components(
    linear_model: type[Any],
) -> MakeComponents:
    """Return a factory of a linear model, its SGD optimizer and fresh state."""

    def make(
        learning_rate: float = 0.1,
    ) -> tuple[Any, nnx.Optimizer, TrainingState]:
        """Build a fresh linear model, its SGD optimizer and training state.

        Args:
            learning_rate: The SGD learning rate.

        Returns:
            The model, its optimizer and a training state seeded with key 1.
        """
        model = linear_model(rngs=nnx.Rngs(0))
        optimizer = create_optimizer(model, optax.sgd(learning_rate))
        state = initialize_training_state(model, optimizer, rng_key=jax.random.key(1))
        return model, optimizer, state

    return make


def test_fixed_epoch_training_without_validation_updates_parameters_and_rng(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components()
    initial_key = state.rng_state

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=3,
    )

    assert len(result.state.metric_history["supervised/train/loss"]) == 3
    assert result.loss is not None
    assert not bool(jnp.array_equal(initial_key, result.state.rng_state))
    assert int(optimizer.step.get_value()) == 3


def test_run_supervised_registers_named_model_and_optimizer(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components()
    state.model_components.clear()
    state.optimizer_states.clear()

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=1,
        stage="mean",
        model_name="mean_model",
        optimizer_name="mean_optimizer",
    )

    assert result.state.model_components == {"mean_model": model}
    assert result.state.optimizer_states == {"mean_optimizer": optimizer}
    assert result.state.active_stage == "mean"


def test_run_supervised_default_registration_names_remain_compatible(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components()

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=1,
    )

    assert result.state.model_components == {"model": model}
    assert result.state.optimizer_states == {"optimizer": optimizer}


def test_run_supervised_tags_history_with_its_stage_and_keeps_events_bare(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
    in_memory_tracker: Tracker,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    events = in_memory_tracker()
    validation = HeldOutValidation(
        model=model, loader=constant_loader, loss=squared_error
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=1,
        validation=validation,
        event_sinks=[events],
        stage="mean",
    )

    assert set(result.state.metric_history) == {
        "mean/train/loss",
        "mean/validation/loss",
    }
    assert result.metrics == {"loss": result.loss}
    assert [event.metrics.keys() for event in events.events] == [{"loss"}, {"loss"}]


def test_make_train_step_without_registered_metrics_returns_loss_mapping(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
) -> None:
    model, optimizer, _ = make_components()
    train_step = make_train_step(squared_error)

    output = train_step(
        model,
        optimizer,
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
        None,
        jax.random.key(0),
    )

    assert set(output) == {"loss"}


def test_make_train_step_evaluates_batch_metrics_in_inference_mode(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
) -> None:
    model, optimizer, _ = make_components()
    train_step = make_train_step(
        squared_error,
        metrics=(BatchMetricSpec(name="training_flag", metric=training_flag_metric),),
    )

    output = train_step(
        model,
        optimizer,
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
        None,
        jax.random.key(0),
    )

    assert output["training_flag"] == pytest.approx(0.0)


class TiedLinearModel(nnx.Module):
    """Two attributes sharing one linear layer, so the module graph aliases."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.first = nnx.Linear(1, 1, rngs=rngs)
        self.second = self.first

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.second(self.first(inputs))


@pytest.mark.parametrize("tied", [False, True], ids=["plain", "shared-submodule"])
def test_run_supervised_trains_like_calling_make_train_step_per_batch(
    linear_model: type[Any],
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
    tied: bool,
) -> None:
    model_class = TiedLinearModel if tied else linear_model

    def components() -> tuple[Any, nnx.Optimizer]:
        model = model_class(rngs=nnx.Rngs(0))
        return model, create_optimizer(model, optax.adam(0.1))

    model, optimizer = components()
    state = initialize_training_state(model, optimizer, rng_key=jax.random.key(1))
    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=3,
    )

    expected_model, expected_optimizer = components()
    train_step = make_train_step(squared_error)
    key = jax.random.key(1)
    for epoch in range(3):
        for batch in constant_loader(split="train", epoch=epoch):
            key, batch_key = split_key(key)
            train_step(
                expected_model,
                expected_optimizer,
                batch.inputs,
                batch.targets,
                batch.sample_weight,
                batch_key,
            )

    assert jax.tree.all(
        jax.tree.map(
            np.array_equal,
            nnx.state((model, optimizer)),
            nnx.state((expected_model, expected_optimizer)),
        )
    )


def test_evaluate_loader_without_registered_metrics_returns_loss_mapping(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, _, state = make_components()

    metrics, next_key = evaluate_loader(
        model,
        constant_loader(split="train", epoch=0),
        key=state.rng_state,
        loss=squared_error,
    )

    assert set(metrics) == {"loss"}
    assert not bool(jnp.array_equal(next_key, state.rng_state))


def test_evaluate_loader_without_loss_omits_loss_and_scores_metrics(
    make_components: MakeComponents,
    constant_loader: LoaderFactory,
) -> None:
    model, _, state = make_components()
    suite = MetricSuite(
        batch=(BatchMetricSpec(name="training_flag", metric=training_flag_metric),)
    )

    metrics, next_key = evaluate_loader(
        model,
        constant_loader(split="train", epoch=0),
        key=state.rng_state,
        metrics=suite,
    )

    assert set(metrics) == {"training_flag"}
    assert metrics["training_flag"] == pytest.approx(0.0)
    assert not bool(jnp.array_equal(next_key, state.rng_state))


def test_evaluate_loader_with_loss_and_metrics_includes_both(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, _, state = make_components()
    suite = MetricSuite(
        batch=(BatchMetricSpec(name="training_flag", metric=training_flag_metric),)
    )

    metrics, _ = evaluate_loader(
        model,
        constant_loader(split="train", epoch=0),
        key=state.rng_state,
        metrics=suite,
        loss=squared_error,
    )

    assert set(metrics) == {"loss", "training_flag"}


def test_training_metric_stopping_saves_best_checkpoint_and_events(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
    in_memory_tracker: Tracker,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    store = InMemoryCheckpointStore()
    events = in_memory_tracker()
    stopper = EarlyStopper(
        metric="loss",
        mode="min",
        patience=0,
        source=Split.TRAIN,
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=4,
        early_stopper=stopper,
        event_sinks=[events],
        checkpoint_store=store,
        checkpoint_key="mean-best",
    )

    assert len(result.state.metric_history["supervised/train/loss"]) == 2
    assert store.exists("mean-best")
    checkpoint = store.load("mean-best")
    assert checkpoint.epoch == 0
    assert checkpoint.early_stopping_state.best_epoch == 0
    assert checkpoint.early_stopping_state.stopped is False
    assert [event.name for event in events.events] == [
        "epoch_end",
        "best_model",
        "epoch_end",
        "early_stop",
    ]


@pytest.mark.parametrize(
    ("checkpoint_key", "expected_key"),
    [(None, "mean/best"), ("best", "best")],
)
def test_best_checkpoint_key_defaults_to_the_stage_scoped_key(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
    checkpoint_key: str | None,
    expected_key: str,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    store = InMemoryCheckpointStore()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0, source=Split.TRAIN)

    run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=2,
        early_stopper=stopper,
        checkpoint_store=store,
        checkpoint_key=checkpoint_key,
        stage="mean",
    )

    assert store.exists(expected_key)
    assert not store.exists("best" if checkpoint_key is None else "mean/best")


def test_best_checkpoint_state_is_frozen_and_unaffected_by_later_epochs(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    # Learning rate equal to zero is ensured that the checkpoint is saved after
    # the first epoch, which allows to test if the checkpoint remains unaffected by later epochs.
    model, optimizer, state = make_components(learning_rate=0.0)
    store = InMemoryCheckpointStore()
    # High patience so training keeps running (and keeps mutating ``state``)
    # for several epochs after the one-and-only improvement is checkpointed.
    stopper = EarlyStopper(metric="loss", mode="min", patience=5, source=Split.TRAIN)

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=3,
        early_stopper=stopper,
        checkpoint_store=store,
    )

    checkpoint = store.load("supervised/best")

    # The checkpoint was saved after epoch 0; its frozen state must only
    # contain that single observation, even though the live state keeps
    # accumulating history for all 3 epochs.
    assert checkpoint.state.metric_history["supervised/train/loss"] == [
        result.state.metric_history["supervised/train/loss"][0]
    ]
    assert len(result.state.metric_history["supervised/train/loss"]) == 3

    # Mutating the live training state after the fact (as further training,
    # or a resumed run, would) must not leak into the already-saved
    # checkpoint's state.
    result.state.metric_history["supervised/train/loss"].append(999.0)
    result.state.checkpoint_registry["late"] = object()

    assert checkpoint.state.metric_history["supervised/train/loss"] == [
        result.state.metric_history["supervised/train/loss"][0]
    ]
    assert "late" not in checkpoint.state.checkpoint_registry

    # The checkpoint's snapshot must also be independent of the live model
    # and optimizer, which the training loop keeps mutating in place.
    assert checkpoint.state.model_components == {}
    assert checkpoint.state.optimizer_states == {}


def test_held_out_validation_drives_validation_metric_stopping(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    validation = HeldOutValidation(
        model=model, loader=constant_loader, loss=squared_error
    )
    stopper = EarlyStopper(
        metric="loss", mode="min", patience=0, source=Split.VALIDATION
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=4,
        validation=validation,
        early_stopper=stopper,
    )

    assert len(result.state.metric_history["supervised/validation/loss"]) == 2


def test_custom_fold_validation_strategy_is_accepted(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)

    def fold_validation(
        current_state: TrainingState, *, epoch: int
    ) -> ValidationResult:
        del current_state
        return ValidationResult(
            passed=True,
            metrics={"fold_1_loss": float(epoch)},
            message="fold-1",
        )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=2,
        validation=fold_validation,
    )

    assert result.state.metric_history["supervised/validation/fold_1_loss"] == [
        0.0,
        1.0,
    ]


def test_validation_stopping_requires_a_validation_strategy(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components()
    stopper = EarlyStopper(
        metric="loss", mode="min", patience=1, source=Split.VALIDATION
    )

    with pytest.raises(ValueError, match="requires a validation strategy"):
        run_supervised(
            model=model,
            optimizer=optimizer,
            train_loader=constant_loader,
            loss=squared_error,
            state=state,
            epochs=1,
            early_stopper=stopper,
        )


@pytest.mark.parametrize("stage", ["", "mean/train"])
def test_an_invalid_stage_is_rejected_before_any_state_changes(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
    stage: str,
) -> None:
    model, optimizer, state = make_components()
    state.model_components.clear()
    state.optimizer_states.clear()
    initial_kernel = np.asarray(model.linear.kernel.get_value()).copy()
    initial_key = state.rng_state
    initial_stage = state.active_stage

    with pytest.raises(ValueError, match="stage"):
        run_supervised(
            model=model,
            optimizer=optimizer,
            train_loader=constant_loader,
            loss=squared_error,
            state=state,
            epochs=1,
            stage=stage,
        )

    np.testing.assert_array_equal(model.linear.kernel.get_value(), initial_kernel)
    assert int(optimizer.step.get_value()) == 0
    assert state.rng_state is initial_key
    assert state.active_stage == initial_stage
    assert state.model_components == {}
    assert state.optimizer_states == {}
    assert state.metric_history == {}


def test_split_key_is_reproducible() -> None:
    key = jax.random.key(42)

    next_key, operation_key = split_key(key)
    duplicate_next_key, duplicate_operation_key = split_key(key)

    assert bool(jnp.array_equal(next_key, duplicate_next_key))
    assert bool(jnp.array_equal(operation_key, duplicate_operation_key))


def test_run_supervised_records_registered_batch_and_epoch_metrics(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
    in_memory_tracker: Tracker,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    events = in_memory_tracker()
    metric_suite = MetricSuite(
        batch=(BatchMetricSpec(name="mae", metric=mean_absolute_error),),
        epoch=(RootMeanSquaredError(),),
        predictor=linear_predictor,
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=2,
        metrics=metric_suite,
        event_sinks=[events],
    )

    assert set(result.metrics) == {"loss", "mae", "rmse"}
    assert len(result.state.metric_history["supervised/train/loss"]) == 2
    assert len(result.state.metric_history["supervised/train/mae"]) == 2
    assert len(result.state.metric_history["supervised/train/rmse"]) == 2
    assert all(
        {"loss", "mae", "rmse"} <= set(event.metrics)
        for event in events.events
        if event.name == "epoch_end"
    )


def test_run_supervised_reports_epoch_metric_under_its_declared_name(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    metric_suite = MetricSuite(
        epoch=(RootMeanSquaredError(name="root_mse"),),
        predictor=linear_predictor,
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=1,
        metrics=metric_suite,
    )

    assert set(result.metrics) == {"loss", "root_mse"}


def test_held_out_validation_returns_registered_metrics_under_bare_names(
    make_components: MakeComponents,
    squared_error: SupervisedLoss,
    constant_loader: LoaderFactory,
) -> None:
    model, optimizer, state = make_components(learning_rate=0.0)
    validation = HeldOutValidation(
        model=model,
        loader=constant_loader,
        loss=squared_error,
        metrics=MetricSuite(
            batch=(BatchMetricSpec(name="mae", metric=mean_absolute_error),),
            epoch=(RootMeanSquaredError(),),
            predictor=linear_predictor,
        ),
    )

    result = run_supervised(
        model=model,
        optimizer=optimizer,
        train_loader=constant_loader,
        loss=squared_error,
        state=state,
        epochs=2,
        validation=validation,
    )

    assert len(result.state.metric_history["supervised/validation/loss"]) == 2
    assert len(result.state.metric_history["supervised/validation/mae"]) == 2
    assert len(result.state.metric_history["supervised/validation/rmse"]) == 2


def test_metric_suite_rejects_reserved_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="reserved"):
        MetricSuite(batch=(BatchMetricSpec(name="loss", metric=mean_absolute_error),))

    with pytest.raises(ValueError, match="duplicate"):
        MetricSuite(
            batch=(BatchMetricSpec(name="mae", metric=mean_absolute_error),),
            epoch=(RootMeanSquaredError(name="mae"),),
        )


def test_epoch_metrics_require_explicit_predictor() -> None:
    with pytest.raises(ValueError, match="explicit suite predictor"):
        MetricSuite(epoch=(RootMeanSquaredError(),))


def test_sampled_epoch_metrics_preserve_loss_batch_metric_and_rng_trajectory(
    constant_loader: LoaderFactory,
) -> None:
    def random_key_metric(
        model: nnx.Module,
        inputs: jax.Array,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
        training: bool,
    ) -> jax.Array:
        del model, inputs, targets, sample_weight, training
        return jax.random.uniform(key)

    batch_metric = BatchMetricSpec(name="key_probe", metric=random_key_metric)
    grid = EvaluationGrid(np.linspace(-5.0, 5.0, 51))
    base_suite = MetricSuite(batch=(batch_metric,))
    sampled_suite = MetricSuite(
        batch=(batch_metric,),
        epoch=(PointContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
        predictive_sample_count=8,
        evaluation_grid=grid,
    )

    def run(metrics: MetricSuite) -> tuple[StageResult, GaussianHead]:
        model = GaussianHead(1, 1, rngs=nnx.Rngs(7))
        optimizer = create_optimizer(model, optax.sgd(0.01))
        state = initialize_training_state(model, optimizer, rng_key=jax.random.key(11))
        result = run_supervised(
            model=model,
            optimizer=optimizer,
            train_loader=constant_loader,
            loss=make_supervised_loss(NegativeLogLikelihoodLoss()),
            state=state,
            epochs=2,
            metrics=metrics,
        )
        return result, model

    base_result, base_model = run(base_suite)
    sampled_result, sampled_model = run(sampled_suite)

    assert base_result.metrics["loss"] == pytest.approx(sampled_result.metrics["loss"])
    assert base_result.metrics["key_probe"] == pytest.approx(
        sampled_result.metrics["key_probe"]
    )
    assert bool(
        jnp.array_equal(base_result.state.rng_state, sampled_result.state.rng_state)
    )
    for base_leaf, sampled_leaf in zip(
        jax.tree.leaves(nnx.state(base_model)),
        jax.tree.leaves(nnx.state(sampled_model)),
        strict=True,
    ):
        assert jnp.array_equal(base_leaf, sampled_leaf)


def test_run_with_the_default_stage_records_supervised_tags(
    in_memory_tracker: Tracker, supervised_run: Run
) -> None:
    tracker = in_memory_tracker()

    result = supervised_run(TrackerEventSink(tracker), validate=False)

    assert tracker.tags == {"supervised/train/loss"}
    assert set(result.state.metric_history) == {"supervised/train/loss"}
    assert len(result.state.metric_history["supervised/train/loss"]) == 3


@given(epochs=st.integers(min_value=1, max_value=4), validate=st.booleans())
@settings(deadline=None, max_examples=6)
def test_a_default_stage_run_keeps_event_and_result_metrics_bare(
    in_memory_tracker: Tracker, supervised_run: Run, epochs: int, validate: bool
) -> None:
    recorder = in_memory_tracker()

    result = supervised_run(recorder, epochs=epochs, validate=validate)

    assert len(recorder.events) == epochs * (2 if validate else 1)
    assert all(set(event.metrics) == {"loss"} for event in recorder.events)
    assert result.metrics == {"loss": result.loss}


def test_events_carry_the_split_they_concern(
    in_memory_tracker: Tracker, supervised_run: Run
) -> None:
    recorder = in_memory_tracker()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0)

    supervised_run(recorder, learning_rate=0.0, epochs=4, early_stopper=stopper)

    splits = {event.name: event.split for event in recorder.events}
    assert splits == {
        "epoch_end": Split.TRAIN,
        "validation_end": Split.VALIDATION,
        "best_model": Split.VALIDATION,
        "early_stop": Split.VALIDATION,
    }


@pytest.mark.parametrize("validate", [False, True])
def test_decision_events_name_the_train_split_when_the_stopper_reads_it(
    in_memory_tracker: Tracker, supervised_run: Run, validate: bool
) -> None:
    recorder = in_memory_tracker()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0, source=Split.TRAIN)

    result = supervised_run(
        recorder,
        learning_rate=0.0,
        epochs=4,
        validate=validate,
        early_stopper=stopper,
    )

    decisions = [event for event in recorder.events if event.name in DECISION_EVENTS]
    assert [event.name for event in decisions] == ["best_model", "early_stop"]
    assert all(event.split is Split.TRAIN for event in decisions)
    history = result.state.metric_history["supervised/train/loss"]
    assert [event.decision for event in decisions] == [
        Decision(metric="loss", value=history[0]),
        Decision(metric="loss", value=history[1]),
    ]


def test_only_decision_events_carry_a_decision(
    in_memory_tracker: Tracker, supervised_run: Run
) -> None:
    recorder = in_memory_tracker()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0)

    supervised_run(recorder, learning_rate=0.0, epochs=4, early_stopper=stopper)

    assert recorder.events
    for event in recorder.events:
        assert (event.decision is not None) == (event.name in DECISION_EVENTS)


def test_decision_events_carry_no_metrics_and_add_no_tracker_series(
    in_memory_tracker: Tracker, supervised_run: Run
) -> None:
    recorder = in_memory_tracker()
    tracker = in_memory_tracker()
    stopper = EarlyStopper(metric="loss", mode="min", patience=0)

    result = supervised_run(
        recorder,
        TrackerEventSink(tracker),
        learning_rate=0.0,
        epochs=4,
        early_stopper=stopper,
    )

    decisions = [event for event in recorder.events if event.name in DECISION_EVENTS]
    assert {event.name for event in decisions} == DECISION_EVENTS
    assert all(event.metrics == {} for event in decisions)
    history = result.state.metric_history["supervised/validation/loss"]
    assert [event.decision for event in decisions] == [
        Decision(metric="loss", value=history[0]),
        Decision(metric="loss", value=history[1]),
    ]
    measured = sum(len(event.metrics) for event in recorder.events)
    assert sum(len(values) for values, _ in tracker.metrics) == measured
    assert tracker.tags == {
        "supervised/train/loss",
        "supervised/validation/loss",
    }
