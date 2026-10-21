from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from posterior_runs import (
    DATASET_SIZE,
    DropoutLinearModel,
    MakeStages,
    MakeVarianceReadyRun,
    RecordingStore,
    VarianceReadyRun,
    leaves_equal,
    regression_data,
    regression_loader,
    restore_mean_and_variance,
)

from stagereg.core.checkpoints import (
    CachedCheckpointStore,
    Checkpoint,
    finalized_only,
)
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.naming import Split, metric_tag
from stagereg.core.metric_registry import (
    EvaluationGrid,
    NegativeLogLikelihood,
    PointContinuousRankedProbabilityScore,
)
from stagereg.core.tracking import TrackerEventSink
from stagereg.core.types import (
    Batch,
    ParameterRole,
    PyTree,
    StageState,
    TrainingState,
)
from stagereg.jax import (
    BayesByBackprop,
    MCDropout,
    MetricSuite,
    PosteriorPredictive,
    PosteriorPredictivePredictor,
    PreconditionedSGLD,
)
from stagereg.jax.posterior import (
    InferenceMethod,
    PosteriorProblem,
    TraceablePosterior,
)
from stagereg.jax.posterior_stage import PosteriorStage, PosteriorStageOptions

Tracker = Callable[[], Any]


@pytest.fixture(scope="session")
def shared_variance_ready_run(
    variance_ready_run: MakeVarianceReadyRun,
) -> VarianceReadyRun:
    """One variance-ready run; take an ``_isolated`` copy before preparing it."""
    return variance_ready_run()


def _isolated(run: VarianceReadyRun) -> VarianceReadyRun:
    """A deep copy of ``run``, so preparing it cannot leak into other tests."""
    return copy.deepcopy(run)


@dataclass(frozen=True)
class OffsetDrawsPosterior:
    """Draws that shift the posterior network's mean by fixed offsets.

    Finite when ``offsets`` is given: then every call returns one draw per
    offset and refuses an explicit draw count. Unlimited otherwise: each draw
    shifts the mean by a standard normal offset drawn from the key.
    """

    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    parameters: PyTree
    offsets: tuple[float, ...] | None = None
    calls: list[tuple[jax.Array, int | None]] = field(default_factory=list)

    @property
    def num_draws(self) -> int | None:
        return None if self.offsets is None else len(self.offsets)

    def sample_means(
        self, inputs: PyTree, key: jax.Array, num_samples: int | None = None
    ) -> jax.Array:
        self.calls.append((key, num_samples))
        if self.offsets is not None:
            if num_samples is not None:
                raise ValueError("a finite posterior uses all its draws.")
            offsets = jnp.asarray(self.offsets)
        else:
            if num_samples is None:
                raise ValueError("an unlimited posterior needs num_samples.")
            offsets = jax.random.normal(key, (num_samples,))
        means = self.mean_function(self.parameters, inputs, key)
        return means[None] + offsets.reshape((-1,) + (1,) * means.ndim)


@dataclass
class GradientAscentMethod:
    """A deterministic inference method: gradient ascent on the log posterior."""

    learning_rate: float = 0.01
    offsets: tuple[float, ...] | None = (-0.5, 0.5)
    early_stopping: bool = True
    diagnostics: tuple[str, ...] = ()
    problem: PosteriorProblem | None = None
    parameters: PyTree = None
    steps: int = 0
    posteriors: list[OffsetDrawsPosterior] = field(default_factory=list)

    @property
    def supports_early_stopping(self) -> bool:
        return self.early_stopping

    @property
    def has_posterior(self) -> bool:
        return self.problem is not None

    def init(self, problem: PosteriorProblem) -> None:
        self.problem = problem
        self.parameters = problem.initial_parameters
        self.steps = 0

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        assert self.problem is not None
        problem = self.problem

        def negative_log_posterior(parameters: PyTree) -> jax.Array:
            return -(
                problem.log_likelihood(parameters, batch, key)
                + problem.prior.log_prob(parameters)
            )

        loss, grads = jax.value_and_grad(negative_log_posterior)(self.parameters)
        self.parameters = jax.tree.map(
            lambda value, grad: value - self.learning_rate * grad,
            self.parameters,
            grads,
        )
        self.steps += 1
        # Each step also reports its own count, under a declared and an undeclared key.
        return {"loss": loss, "step_count": self.steps, "internal": self.steps}

    def posterior(self) -> OffsetDrawsPosterior:
        assert self.problem is not None
        posterior = OffsetDrawsPosterior(
            self.problem.mean_function, self.parameters, self.offsets
        )
        self.posteriors.append(posterior)
        return posterior

    def state(self) -> PyTree:
        return {"parameters": self.parameters, "steps": jnp.asarray(self.steps)}

    def load_state(self, state: PyTree) -> None:
        self.parameters = state["parameters"]
        self.steps = int(state["steps"])

    def posterior_state(self) -> PyTree:
        return self.parameters

    def load_posterior(self, state: PyTree) -> None:
        self.parameters = state


def _posterior_stage(method: GradientAscentMethod, **options: Any) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(**{"epochs": 3, **options}),
    )


def _observable(state: TrainingState) -> tuple[Any, ...]:
    """Everything a refused prepare must leave unchanged, copied."""
    return (
        dict(state.model_components),
        dict(state.parameter_roles),
        state.frozen_components,
        dict(state.optimizer_states),
        state.posterior_state,
        state.rng_state,
        state.lifecycle_state,
        state.active_stage,
        {tag: list(values) for tag, values in state.metric_history.items()},
    )


def _copy(module: nnx.Module) -> PyTree:
    return jax.tree.map(jnp.copy, nnx.state(module))


@pytest.mark.parametrize("mean_only", [False, True])
def test_posterior_stage_refuses_to_run_without_ready_mean_and_variance(
    variance_ready_run: MakeVarianceReadyRun, mean_only: bool
) -> None:
    run = variance_ready_run(mean_only=mean_only)
    if not mean_only:
        run.state.lifecycle_state = StageState.MEAN_READY
    stage = _posterior_stage(GradientAscentMethod())
    before = _observable(run.state)

    with pytest.raises(ValueError, match="VARIANCE_READY"):
        stage.prepare(run.state)

    assert _observable(run.state) == before


def test_posterior_stage_refuses_a_fresh_state() -> None:
    state = TrainingState(rng_state=jax.random.key(0))

    with pytest.raises(ValueError, match="VARIANCE_READY"):
        _posterior_stage(GradientAscentMethod()).prepare(state)

    assert _observable(state) == _observable(TrainingState(rng_state=state.rng_state))


@pytest.mark.parametrize("network_seed", [None, 42])
def test_the_warm_start_equals_the_mean_weights(
    shared_variance_ready_run: VarianceReadyRun,
    linear_model: type[Any],
    network_seed: int | None,
) -> None:
    run = _isolated(shared_variance_ready_run)
    method = GradientAscentMethod()
    network = (
        None if network_seed is None else linear_model(rngs=nnx.Rngs(network_seed))
    )
    stage = dataclasses.replace(_posterior_stage(method), model=network)

    stage.prepare(run.state)

    assert method.problem is not None
    assert leaves_equal(
        method.problem.initial_parameters, nnx.state(run.mean_model, nnx.Param)
    )
    inputs, _ = regression_data()
    assert jnp.allclose(
        method.problem.mean_function(
            method.problem.initial_parameters, inputs, jax.random.key(0)
        ),
        run.mean_model(inputs),
    )


def test_training_leaves_the_mean_and_variance_weights_unchanged(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    mean_before = _copy(run.mean_model)
    variance_before = _copy(run.variance_model)
    method = GradientAscentMethod()
    stage = _posterior_stage(method)

    stage.prepare(run.state)
    initial = method.parameters
    stage.train(run.state)

    assert not leaves_equal(method.parameters, initial)
    assert leaves_equal(nnx.state(run.mean_model), mean_before)
    assert leaves_equal(nnx.state(run.variance_model), variance_before)


def _independent_log_likelihood(
    run: VarianceReadyRun, batch: Batch, dataset_size: int
) -> float:
    """``N`` times the mean per-row Gaussian log-density under the trained models."""
    assert batch.targets is not None
    variance = run.variance_model(batch.inputs).mean()
    rows = jax.scipy.stats.norm.logpdf(
        batch.targets, run.mean_model(batch.inputs), jnp.sqrt(variance)
    )
    return dataset_size * float(jnp.mean(rows))


@settings(deadline=None, max_examples=10)
@given(
    repeats=st.integers(1, 4),
    dataset_size=st.integers(1, 1000),
    rows=st.integers(1, DATASET_SIZE),
)
def test_the_likelihood_is_scaled_by_dataset_size_over_batch_size(
    shared_variance_ready_run: VarianceReadyRun,
    repeats: int,
    dataset_size: int,
    rows: int,
) -> None:
    run = _isolated(shared_variance_ready_run)
    method = GradientAscentMethod()
    stage = dataclasses.replace(_posterior_stage(method), dataset_size=dataset_size)
    stage.prepare(run.state)
    assert method.problem is not None
    inputs, targets = regression_data()
    batch = Batch(inputs=inputs[:rows], targets=targets[:rows])
    repeated = Batch(
        inputs=jnp.tile(batch.inputs, (repeats, 1)),
        targets=jnp.tile(batch.targets, (repeats, 1)),
    )

    log_likelihood = method.problem.log_likelihood(
        method.problem.initial_parameters, repeated, jax.random.key(0)
    )

    assert float(log_likelihood) == pytest.approx(
        _independent_log_likelihood(run, batch, dataset_size), rel=1e-4
    )


def test_the_likelihood_weights_each_row_by_its_sample_weight(
    shared_variance_ready_run: VarianceReadyRun,
) -> None:
    run = _isolated(shared_variance_ready_run)
    method = GradientAscentMethod()
    _posterior_stage(method).prepare(run.state)
    assert method.problem is not None
    inputs, targets = regression_data()
    weights = jnp.arange(inputs.shape[0]) % 3 / 2.0
    weighted = Batch(inputs=inputs, targets=targets, sample_weight=weights)

    log_likelihood = method.problem.log_likelihood(
        method.problem.initial_parameters, weighted, jax.random.key(0)
    )

    variance = run.variance_model(inputs).mean()
    rows = jax.scipy.stats.norm.logpdf(
        targets, run.mean_model(inputs), jnp.sqrt(variance)
    )
    expected = DATASET_SIZE * float(jnp.mean(weights[:, None] * rows))
    assert float(log_likelihood) == pytest.approx(expected, rel=1e-4)


@settings(deadline=None, max_examples=10)
@given(
    inputs=st.lists(
        st.floats(-3.0, 3.0, allow_nan=False, width=32), min_size=1, max_size=8
    )
)
def test_the_problem_carries_the_stages_aleatoric_variance(
    shared_variance_ready_run: VarianceReadyRun, inputs: list[float]
) -> None:
    run = _isolated(shared_variance_ready_run)
    method = GradientAscentMethod()
    _posterior_stage(method).prepare(run.state)
    assert method.problem is not None
    aleatoric_variance = method.problem.aleatoric_variance
    assert aleatoric_variance is not None
    points = jnp.asarray(inputs)[:, None]

    variance = aleatoric_variance(points)

    assert jnp.allclose(variance, run.variance_model(points).mean(), rtol=1e-6)
    assert jnp.array_equal(
        jax.grad(lambda x: aleatoric_variance(x).sum())(points),
        jnp.zeros_like(points),
    )


def _validation_suite() -> MetricSuite:
    return MetricSuite(
        epoch=(
            NegativeLogLikelihood(),
            PointContinuousRankedProbabilityScore(name="crps"),
        ),
        predictor=PosteriorPredictivePredictor(),
        predictive_sample_count=64,
        evaluation_grid=EvaluationGrid(np.linspace(-6.0, 6.0, 121)),
    )


def test_the_three_stages_of_a_posterior_run_never_share_a_tag(
    in_memory_tracker: Tracker,
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    recorder = in_memory_tracker()
    tracker = in_memory_tracker()
    sinks = (recorder, TrackerEventSink(tracker))
    run = variance_ready_run(*sinks)
    stage = _posterior_stage(
        GradientAscentMethod(),
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        event_sinks=sinks,
    )

    stage.prepare(run.state)
    stage.train(run.state)

    expected = {
        "mean/train/loss",
        "mean/validation/loss",
        "variance/train/loss",
        "variance/validation/loss",
        "posterior/train/loss",
        "posterior/validation/nll",
        "posterior/validation/crps",
    }
    assert {
        f"{event.stage}/{event.split}/{metric}"
        for event in recorder.events
        for metric in event.metrics
    } == expected
    assert tracker.tags == expected
    assert set(run.state.metric_history) == expected


def test_declared_diagnostics_are_recorded_as_epoch_means_and_reach_trackers(
    in_memory_tracker: Tracker,
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    tracker = in_memory_tracker()
    run = variance_ready_run()
    stage = _posterior_stage(
        GradientAscentMethod(diagnostics=("step_count",)),
        event_sinks=(TrackerEventSink(tracker),),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    # Two batches per epoch: steps 1 and 2, then 3 and 4, then 5 and 6.
    assert run.state.metric_history["posterior/train/step_count"] == [1.5, 3.5, 5.5]
    assert "posterior/train/step_count" in tracker.tags
    assert not any("internal" in tag for tag in run.state.metric_history)
    assert not any("internal" in tag for tag in tracker.tags)


@pytest.mark.parametrize(
    "method",
    [BayesByBackprop(), PreconditionedSGLD(step_size=1e-3)],
    ids=["bbb", "psgld"],
)
def test_the_shipped_methods_record_only_the_training_loss(
    variance_ready_run: MakeVarianceReadyRun, method: InferenceMethod
) -> None:
    run = variance_ready_run()
    stage = dataclasses.replace(
        _posterior_stage(GradientAscentMethod()), inference_method=method
    )

    stage.prepare(run.state)
    stage.train(run.state)

    assert {
        tag for tag in run.state.metric_history if tag.startswith("posterior/")
    } == {"posterior/train/loss"}


@pytest.mark.parametrize("diagnostics", [("loss",), ("step_count", "step_count")])
def test_a_diagnostic_colliding_with_a_recorded_tag_is_refused(
    variance_ready_run: MakeVarianceReadyRun, diagnostics: tuple[str, ...]
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(GradientAscentMethod(diagnostics=diagnostics))
    before = _observable(run.state)

    with pytest.raises(ValueError, match="diagnostic"):
        stage.prepare(run.state)

    assert _observable(run.state) == before


def test_a_declared_diagnostic_that_update_omits_fails_training(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(GradientAscentMethod(diagnostics=("absent",)))
    stage.prepare(run.state)

    with pytest.raises(ValueError, match="diagnostic 'absent'"):
        stage.train(run.state)


def test_validation_scores_the_current_posterior_predictive(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    method = GradientAscentMethod()
    stage = _posterior_stage(
        method,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    history = run.state.metric_history
    assert len(history["posterior/validation/nll"]) == 3
    assert all(math.isfinite(value) for value in history["posterior/validation/crps"])
    final = method.posteriors[-1]
    targets = jnp.concatenate(
        [b.targets for b in regression_loader(split="validation", epoch=2)]
    )
    inputs = jnp.concatenate(
        [b.inputs for b in regression_loader(split="validation", epoch=2)]
    )
    draws = (
        final.mean_function(final.parameters, inputs, jax.random.key(0))[None]
        + jnp.array([-0.5, 0.5])[:, None, None]
    )
    scale = jnp.sqrt(run.variance_model(inputs).mean())
    mixture = jsp.special.logsumexp(
        jax.scipy.stats.norm.logpdf(targets, draws, scale), axis=0
    ) - math.log(2.0)
    assert history["posterior/validation/nll"][-1] == pytest.approx(
        -float(jnp.mean(mixture)), rel=1e-4
    )


def test_the_default_validation_reports_the_nll_and_the_crps_without_a_grid(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(
        GradientAscentMethod(), validation_loader=regression_loader
    )

    stage.prepare(run.state)
    stage.train(run.state)

    history = run.state.metric_history
    for name in ("nll", "crps"):
        values = history[f"posterior/validation/{name}"]
        assert len(values) == 3
        assert all(math.isfinite(value) and value > 0.0 for value in values)


def test_an_unlimited_posterior_draws_the_configured_count_with_the_same_key_every_epoch(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    method = GradientAscentMethod(offsets=None)
    stage = _posterior_stage(
        method,
        num_draws=7,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    validated = [posterior for posterior in method.posteriors if posterior.calls]
    assert len(validated) == 3
    keys = [
        jax.random.key_data(key)
        for posterior in validated
        for key, _ in posterior.calls
    ]
    assert len(keys) == 3 * 2
    assert all(jnp.array_equal(key, keys[0]) for key in keys)
    assert {count for p in validated for _, count in p.calls} == {7}


def _validation_history(
    state: TrainingState, stage: PosteriorStage
) -> dict[str, list[float]]:
    """Prepare and train ``stage`` on ``state``; return its default validation history.

    Args:
        state: The variance-ready state to train on.
        stage: A posterior stage with a validation loader and default metrics.

    Returns:
        The per-epoch validation ``nll`` and ``crps``, keyed by metric name.
    """
    stage.prepare(state)
    stage.train(state)
    return {
        name: list(state.metric_history[metric_tag(stage.name, Split.VALIDATION, name)])
        for name in ("nll", "crps")
    }


def test_an_unchanged_posterior_scores_identical_metrics_every_epoch(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    history = _validation_history(
        variance_ready_run().state,
        _posterior_stage(
            GradientAscentMethod(learning_rate=0.0, offsets=None),
            validation_loader=regression_loader,
        ),
    )

    for values in history.values():
        assert len(values) == 3
        assert values == [values[0]] * 3


def test_the_validation_key_selects_the_crps(
    shared_variance_ready_run: VarianceReadyRun,
) -> None:
    def crps(key: jax.Array) -> list[float]:
        return _validation_history(
            _isolated(shared_variance_ready_run).state,
            _posterior_stage(
                GradientAscentMethod(offsets=None),
                validation_loader=regression_loader,
                validation_key=key,
            ),
        )["crps"]

    first = crps(jax.random.key(1))

    assert crps(jax.random.key(1)) == first
    assert crps(jax.random.key(2)) != first


def test_validation_leaves_the_training_randomness_unchanged(
    shared_variance_ready_run: VarianceReadyRun,
) -> None:
    def train(**options: Any) -> tuple[TrainingState, GradientAscentMethod]:
        run = _isolated(shared_variance_ready_run)
        method = GradientAscentMethod(offsets=None)
        stage = _posterior_stage(method, **options)
        stage.prepare(run.state)
        stage.train(run.state)
        return run.state, method

    validated, validated_method = train(validation_loader=regression_loader)
    unvalidated, unvalidated_method = train()

    assert "posterior/validation/crps" in validated.metric_history
    assert jnp.array_equal(
        jax.random.key_data(validated.rng_state),
        jax.random.key_data(unvalidated.rng_state),
    )
    assert leaves_equal(validated_method.parameters, unvalidated_method.parameters)


@pytest.mark.parametrize(
    "key",
    [
        jnp.zeros(2),
        jax.random.key_data(jax.random.key(0)),
        jax.random.split(jax.random.key(0), 2),
    ],
    ids=["float array", "raw uint32 key", "batched typed keys"],
)
def test_a_validation_key_that_is_not_a_single_typed_prng_key_is_refused(
    key: jax.Array,
) -> None:
    with pytest.raises(TypeError, match="validation_key"):
        PosteriorStageOptions(epochs=1, validation_key=key)


def test_an_early_stopper_is_refused_for_a_method_without_early_stopping(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(
        GradientAscentMethod(early_stopping=False),
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        early_stopper=EarlyStopper(metric="nll", mode="min", patience=0),
    )
    before = _observable(run.state)

    with pytest.raises(ValueError, match="early stopping"):
        stage.prepare(run.state)

    assert _observable(run.state) == before


def test_an_early_stopper_stops_a_method_that_supports_it(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = _posterior_stage(
        GradientAscentMethod(learning_rate=0.0),
        epochs=10,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        early_stopper=EarlyStopper(metric="nll", mode="min", patience=0),
    )

    stage.prepare(run.state)
    stage.train(run.state)

    assert len(run.state.metric_history["posterior/train/loss"]) < 10


def test_a_trained_posterior_stage_passes_validate(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    method = GradientAscentMethod()
    stage = _posterior_stage(method)

    stage.prepare(run.state)
    assert not stage.validate(run.state).passed
    result = stage.train(run.state)

    assert result.loss is not None and math.isfinite(result.loss)
    assert stage.validate(run.state).passed
    assert run.state.lifecycle_state is StageState.POSTERIOR_READY
    assert run.state.model_components["posterior"] is method.posteriors[-1]
    assert run.state.parameter_roles["posterior"] is ParameterRole.POSTERIOR
    assert {"mean_model", "variance_model"} <= run.state.frozen_components


def test_a_posterior_stage_must_be_prepared_before_training(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()

    with pytest.raises(ValueError, match="prepared"):
        _posterior_stage(GradientAscentMethod()).train(run.state)


def test_a_posterior_network_with_another_parameter_tree_is_refused(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    stage = dataclasses.replace(
        _posterior_stage(GradientAscentMethod()),
        model=nnx.Linear(1, 2, rngs=nnx.Rngs(0)),
    )

    with pytest.raises(ValueError, match="parameter tree"):
        stage.prepare(run.state)


def test_posterior_stage_declares_its_place_in_the_lifecycle() -> None:
    stage = _posterior_stage(GradientAscentMethod())

    assert stage.name == "posterior"
    assert stage.requires == frozenset({"mean", "variance"})
    assert stage.produces == frozenset({"posterior"})


def test_a_method_without_early_stopping_writes_only_a_finalized_checkpoint(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    store = RecordingStore()
    method = GradientAscentMethod(early_stopping=False)
    stage = _posterior_stage(
        method,
        validation_loader=regression_loader,
        validation_metrics=_validation_suite(),
        checkpoint_store=store,
    )

    stage.prepare(run.state)
    stage.train(run.state)

    assert [key for key, _ in store.saves] == ["posterior/best"]
    finalized = store.load("posterior/best")
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.state.lifecycle_state is StageState.POSTERIOR_READY
    assert finalized.epoch == 2
    assert leaves_equal(finalized.state.posterior_state, method.parameters)
    assert finalized.parameters is None
    assert finalized.optimizer_state is None
    assert finalized.state.model_components == {}


def test_the_best_checkpoint_holds_the_method_state_and_is_finalized_as_its_posterior(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    run = variance_ready_run()
    store = RecordingStore()
    method = GradientAscentMethod()
    stage = _posterior_stage(
        method,
        early_stopper=EarlyStopper(
            metric="loss", mode="max", patience=10, source="train"
        ),
        checkpoint_store=store,
    )

    stage.prepare(run.state)
    result = stage.train(run.state)

    *bests, (final_key, finalized) = store.saves
    assert final_key == "posterior/best"
    assert bests and {key for key, _ in bests} == {"posterior/best"}
    _, best = bests[-1]
    assert best.epoch < 2
    assert best.metadata == {}
    assert best.state.lifecycle_state is StageState.VARIANCE_READY
    assert int(best.parameters["steps"]) == 2 * (best.epoch + 1)
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.state.lifecycle_state is StageState.POSTERIOR_READY
    assert finalized.epoch == best.epoch
    assert finalized.early_stopping_state == best.early_stopping_state
    assert finalized.parameters is None
    assert leaves_equal(finalized.state.posterior_state, best.parameters["parameters"])
    assert leaves_equal(method.posteriors[-1].parameters, best.parameters["parameters"])
    assert result.loss == best.state.metric_history["posterior/train/loss"][-1]
    assert stage.validate(run.state).passed


@dataclass
class FinishedRun:
    """A three-stage run whose stages saved into one store."""

    run: VarianceReadyRun
    method: GradientAscentMethod
    store: RecordingStore


@pytest.fixture(scope="session")
def finished_run(variance_ready_run: MakeVarianceReadyRun) -> FinishedRun:
    """One mean, variance and posterior run that shares a store; never mutate it."""
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    method = GradientAscentMethod()
    stage = _posterior_stage(method, checkpoint_store=store)
    stage.prepare(run.state)
    stage.train(run.state)
    return FinishedRun(run, method, store)


def test_three_stages_sharing_a_store_never_overwrite_each_other(
    finished_run: FinishedRun,
) -> None:
    finals = {
        key: checkpoint.metadata.get("stage")
        for key, checkpoint in finished_run.store.saves
        if checkpoint.metadata.get("stage_complete")
    }

    assert finals == {
        "mean/best": "mean",
        "variance/best": "variance",
        "posterior/best": "posterior",
    }
    for key, stage in finals.items():
        assert finished_run.store.load(key).metadata["stage"] == stage


def test_a_cached_store_persists_only_the_finalized_checkpoints_of_a_full_run(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    persistent = RecordingStore()
    store = CachedCheckpointStore(persistent, persist_when=finalized_only)
    run = variance_ready_run(checkpoint_store=store)
    stage = _posterior_stage(
        GradientAscentMethod(),
        early_stopper=EarlyStopper(
            metric="loss", mode="max", patience=10, source="train"
        ),
        checkpoint_store=store,
    )

    stage.prepare(run.state)
    stage.train(run.state)

    assert [(key, checkpoint.metadata) for key, checkpoint in persistent.saves] == [
        ("mean/best", {"stage": "mean", "stage_complete": True}),
        ("variance/best", {"stage": "variance", "stage_complete": True}),
        ("posterior/best", {"stage": "posterior", "stage_complete": True}),
    ]


def test_a_fresh_state_resumes_through_the_three_stage_restore(
    finished_run: FinishedRun, make_stages: MakeStages
) -> None:
    stages = make_stages(seed=7)
    state = restore_mean_and_variance(stages, finished_run.store)
    method = GradientAscentMethod()
    stage = _posterior_stage(method)

    stage.restore(state, finished_run.store.load("posterior/best"))

    assert stage.validate(state).passed
    assert state.lifecycle_state is StageState.POSTERIOR_READY
    assert state.active_stage == "posterior"
    assert jnp.array_equal(
        jax.random.key_data(state.rng_state),
        jax.random.key_data(finished_run.run.state.rng_state),
    )
    assert state.model_components["mean_model"] is stages.mean.model
    assert state.model_components["variance_model"] is stages.variance.model
    assert state.optimizer_states["mean_optimizer"] is stages.mean.optimizer
    assert state.optimizer_states["variance_optimizer"] is stages.variance.optimizer
    restored = state.model_components["posterior"]
    trained = finished_run.run.state.model_components["posterior"]
    inputs, _ = regression_data()
    key = jax.random.key(0)
    assert jnp.allclose(
        restored.sample_means(inputs, key), trained.sample_means(inputs, key)
    )
    assert state.metric_history == finished_run.run.state.metric_history


def _unfinalized(checkpoint: Checkpoint) -> Checkpoint:
    return dataclasses.replace(checkpoint, metadata={})


def _variance_ready_lifecycle(checkpoint: Checkpoint) -> Checkpoint:
    saved = dataclasses.replace(
        checkpoint.state, lifecycle_state=StageState.VARIANCE_READY
    )
    return dataclasses.replace(checkpoint, state=saved)


def _without_posterior_state(checkpoint: Checkpoint) -> Checkpoint:
    saved = dataclasses.replace(checkpoint.state, posterior_state=None)
    return dataclasses.replace(checkpoint, state=saved)


_REFUSALS: dict[str, tuple[str, Callable[[FinishedRun], Checkpoint], str]] = {
    "mean checkpoint": ("variance", lambda r: r.store.load("mean/best"), "finalized"),
    "variance checkpoint": (
        "variance",
        lambda r: r.store.load("variance/best"),
        "finalized",
    ),
    "unfinalized metadata": (
        "variance",
        lambda r: _unfinalized(r.store.load("posterior/best")),
        "finalized",
    ),
    "unfinalized lifecycle": (
        "variance",
        lambda r: _variance_ready_lifecycle(r.store.load("posterior/best")),
        "finalized",
    ),
    "no posterior state": (
        "variance",
        lambda r: _without_posterior_state(r.store.load("posterior/best")),
        "posterior state",
    ),
    "variance not restored": (
        "mean",
        lambda r: r.store.load("posterior/best"),
        "variance model",
    ),
    "nothing restored": (
        "none",
        lambda r: r.store.load("posterior/best"),
        "mean model",
    ),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_a_refused_restore_leaves_the_state_and_live_models_unchanged(
    finished_run: FinishedRun, make_stages: MakeStages, case: str
) -> None:
    restored_through, checkpoint_of, message = _REFUSALS[case]
    stages = make_stages(seed=7)
    state = TrainingState()
    if restored_through != "none":
        stages.mean.restore(state, finished_run.store.load("mean/best"))
    if restored_through == "variance":
        stages.variance.restore(state, finished_run.store.load("variance/best"))
    before = _observable(state)
    mean_before = _copy(stages.mean.model)
    variance_before = _copy(stages.variance.model)
    method = GradientAscentMethod()

    with pytest.raises(ValueError, match=message):
        _posterior_stage(method).restore(state, checkpoint_of(finished_run))

    assert _observable(state) == before
    assert leaves_equal(nnx.state(stages.mean.model), mean_before)
    assert leaves_equal(nnx.state(stages.variance.model), variance_before)
    assert method.problem is None


@dataclass
class RefusingMethod(GradientAscentMethod):
    """A method that refuses every saved posterior state."""

    def load_posterior(self, state: PyTree) -> None:
        raise ValueError("incompatible posterior state")


def test_a_refused_posterior_state_leaves_the_method_and_model_unchanged(
    finished_run: FinishedRun, make_stages: MakeStages, linear_model: type[Any]
) -> None:
    state = restore_mean_and_variance(make_stages(seed=7), finished_run.store)
    before = _observable(state)
    network = linear_model(rngs=nnx.Rngs(42))
    network_before = _copy(network)
    method = RefusingMethod()
    stage = dataclasses.replace(_posterior_stage(method), model=network)

    with pytest.raises(ValueError, match="incompatible"):
        stage.restore(state, finished_run.store.load("posterior/best"))

    assert _observable(state) == before
    assert method.problem is None
    assert leaves_equal(nnx.state(network), network_before)


@pytest.mark.parametrize(
    "names",
    [
        {"model_name": "other_posterior"},
        {"mean_model_name": "other_mean"},
        {"variance_model_name": "other_variance"},
    ],
)
def test_restore_refuses_a_checkpoint_saved_under_other_component_names(
    finished_run: FinishedRun, make_stages: MakeStages, names: dict[str, str]
) -> None:
    state = restore_mean_and_variance(make_stages(seed=7), finished_run.store)
    before = _observable(state)
    stage = dataclasses.replace(_posterior_stage(GradientAscentMethod()), **names)

    with pytest.raises(ValueError, match="component names"):
        stage.restore(state, finished_run.store.load("posterior/best"))

    assert _observable(state) == before


# Inferred subnetwork. The mean network of `posterior_runs` is one linear
# layer. Inferring its kernel alone leaves the bias as the frozen remainder,
# which every draw then shares: at input zero each draw is exactly the bias.

_KERNEL = nnx.PathContains("kernel")

_SUBNETWORK_METHODS: dict[str, Callable[[], InferenceMethod]] = {
    "bayes by backprop": lambda: BayesByBackprop(
        optimizer=optax.adam(0.01), initial_std=0.05
    ),
    "psgld": lambda: PreconditionedSGLD(step_size=1e-3, burn_in=1, thinning=2),
}


def _subnetwork_stage(
    method: InferenceMethod, inferred_subnetwork: Any = nnx.Param, **options: Any
) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(**{"epochs": 3, "num_draws": 4, **options}),
        inferred_subnetwork=inferred_subnetwork,
    )


@dataclass
class SubnetworkRun:
    """A posterior stage trained on a variance-ready run."""

    run: VarianceReadyRun
    method: InferenceMethod


def _train_subnetwork(
    variance_ready_run: MakeVarianceReadyRun,
    method_name: str,
    inferred_subnetwork: Any = nnx.Param,
) -> SubnetworkRun:
    run = variance_ready_run()
    method = _SUBNETWORK_METHODS[method_name]()
    stage = _subnetwork_stage(method, inferred_subnetwork)
    stage.prepare(run.state)
    stage.train(run.state)
    return SubnetworkRun(run, method)


def _parameter_names(tree: PyTree) -> set[str]:
    """The names of the linear layer's parameters in a parameter tree."""
    return set(tree["linear"].keys())


@pytest.fixture(scope="module")
def kernel_runs(variance_ready_run: MakeVarianceReadyRun) -> dict[str, SubnetworkRun]:
    """Each method trained on the kernel alone; never mutate them."""
    return {
        name: _train_subnetwork(variance_ready_run, name, _KERNEL)
        for name in _SUBNETWORK_METHODS
    }


@pytest.mark.parametrize("method_name", sorted(_SUBNETWORK_METHODS))
def test_a_method_infers_only_the_inferred_subnetwork(
    kernel_runs: dict[str, SubnetworkRun], method_name: str
) -> None:
    posterior_state = kernel_runs[method_name].method.posterior_state()
    trees = (
        [posterior_state["means"], posterior_state["stds"]]
        if method_name == "bayes by backprop"
        else [posterior_state]
    )

    assert all(_parameter_names(tree) == {"kernel"} for tree in trees)


@settings(deadline=None, max_examples=10)
@given(
    method_name=st.sampled_from(sorted(_SUBNETWORK_METHODS)),
    seed=st.integers(0, 2**31 - 1),
    rows=st.integers(1, 5),
)
def test_the_frozen_remainder_keeps_its_trained_mean_values_bit_for_bit(
    kernel_runs: dict[str, SubnetworkRun], method_name: str, seed: int, rows: int
) -> None:
    trained = kernel_runs[method_name]
    posterior = trained.run.state.model_components["posterior"]
    num_samples = None if posterior.num_draws is not None else 4

    draws = posterior.sample_means(
        jnp.zeros((rows, 1)), jax.random.key(seed), num_samples
    )

    bias = trained.run.mean_model.linear.bias.get_value()
    assert draws.shape[1:] == (rows, 1)
    np.testing.assert_array_equal(draws, jnp.broadcast_to(bias, draws.shape))


# The training losses of the default filter under stagereg 0.1.0, which
# inferred the whole network.
_LOSSES_OF_0_1_0 = {
    "bayes by backprop": [17.569884300231934, 17.512011528015137, 17.620736122131348],
    "psgld": [14.574500560760498, 14.39689016342163, 14.398993968963623],
}


@pytest.mark.parametrize("method_name", sorted(_SUBNETWORK_METHODS))
def test_the_default_filter_reproduces_stagereg_0_1_0(
    variance_ready_run: MakeVarianceReadyRun, method_name: str
) -> None:
    trained = _train_subnetwork(variance_ready_run, method_name)

    np.testing.assert_allclose(
        trained.run.state.metric_history["posterior/train/loss"],
        _LOSSES_OF_0_1_0[method_name],
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "inferred_subnetwork",
    [nnx.PathContains("conv"), nnx.RngState, nnx.Not(nnx.Param)],
    ids=["unknown path", "rng state", "no param"],
)
def test_a_filter_selecting_no_param_leaf_is_refused_at_prepare(
    variance_ready_run: MakeVarianceReadyRun, inferred_subnetwork: Any
) -> None:
    state = variance_ready_run().state
    lifecycle, frozen, stage_name = (
        state.lifecycle_state,
        state.frozen_components,
        state.active_stage,
    )
    stage = _subnetwork_stage(
        _SUBNETWORK_METHODS["bayes by backprop"](), inferred_subnetwork
    )

    with pytest.raises(ValueError, match="selects no Param leaf"):
        stage.prepare(state)

    assert (state.lifecycle_state, state.frozen_components, state.active_stage) == (
        lifecycle,
        frozen,
        stage_name,
    )


@pytest.mark.parametrize("method_name", sorted(_SUBNETWORK_METHODS))
@pytest.mark.parametrize(
    ("trained_with", "restored_with"),
    [(_KERNEL, nnx.Param), (nnx.Param, _KERNEL), (_KERNEL, nnx.PathContains("bias"))],
    ids=["kernel then all", "all then kernel", "kernel then bias"],
)
def test_restoring_under_another_filter_is_refused(
    variance_ready_run: MakeVarianceReadyRun,
    make_stages: MakeStages,
    method_name: str,
    trained_with: Any,
    restored_with: Any,
) -> None:
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    trained = _subnetwork_stage(
        _SUBNETWORK_METHODS[method_name](), trained_with, checkpoint_store=store
    )
    trained.prepare(run.state)
    trained.train(run.state)
    state = restore_mean_and_variance(make_stages(seed=7), store)
    stage = _subnetwork_stage(_SUBNETWORK_METHODS[method_name](), restored_with)

    with pytest.raises(ValueError, match="inferred_subnetwork"):
        stage.restore(state, store.load("posterior/best"))

    assert stage.inference_method.has_posterior is False
    assert "posterior" not in state.model_components


@pytest.mark.parametrize("method_name", sorted(_SUBNETWORK_METHODS))
def test_restoring_under_the_same_filter_rebuilds_the_posterior(
    variance_ready_run: MakeVarianceReadyRun,
    make_stages: MakeStages,
    method_name: str,
) -> None:
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    trained = _subnetwork_stage(
        _SUBNETWORK_METHODS[method_name](), _KERNEL, checkpoint_store=store
    )
    trained.prepare(run.state)
    trained.train(run.state)
    state = restore_mean_and_variance(make_stages(seed=7), store)
    stage = _subnetwork_stage(_SUBNETWORK_METHODS[method_name](), _KERNEL)

    stage.restore(state, store.load("posterior/best"))

    inputs = jnp.linspace(-1.0, 1.0, 5)[:, None]
    key = jax.random.key(0)
    restored = state.model_components["posterior"]
    num_samples = None if restored.num_draws is not None else 4
    np.testing.assert_array_equal(
        restored.sample_means(inputs, key, num_samples),
        run.state.model_components["posterior"].sample_means(inputs, key, num_samples),
    )


# The keyed seam: the stage hands its inference method a mean function and a
# log-likelihood that take a PRNG key. A deterministic network ignores it; a
# network with ``nnx.Dropout`` draws its masks from it.


@dataclass
class KeyedProbe:
    """Delegates to an inference method, and records each batch's mean function.

    In every ``update`` it evaluates the problem's mean function at the warm
    start on ``probe_inputs`` under the batch's key, so a test can see which
    masks the stage's keys select.
    """

    inner: InferenceMethod
    probe_inputs: jax.Array
    problem: PosteriorProblem | None = None
    probes: list[jax.Array] = field(default_factory=list)

    @property
    def supports_early_stopping(self) -> bool:
        return self.inner.supports_early_stopping

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return self.inner.diagnostics

    @property
    def has_posterior(self) -> bool:
        return self.inner.has_posterior

    def init(self, problem: PosteriorProblem) -> None:
        self.problem = problem
        self.inner.init(problem)

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        assert self.problem is not None
        self.probes.append(
            self.problem.mean_function(
                self.problem.initial_parameters, self.probe_inputs, key
            )
        )
        return self.inner.update(batch, key)

    def posterior(self) -> Any:
        return self.inner.posterior()

    def state(self) -> PyTree:
        return self.inner.state()

    def load_state(self, state: PyTree) -> None:
        self.inner.load_state(state)

    def posterior_state(self) -> PyTree:
        return self.inner.posterior_state()

    def load_posterior(self, state: PyTree) -> None:
        self.inner.load_posterior(state)


def _probe() -> KeyedProbe:
    return KeyedProbe(
        inner=BayesByBackprop(optimizer=optax.adam(0.01), initial_std=0.05),
        probe_inputs=jnp.ones((64, 1)),
    )


def _keyed_stage(
    method: InferenceMethod, model: nnx.Module | None = None
) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(epochs=3, num_draws=4),
        model=model,
    )


@pytest.fixture(scope="module")
def deterministic_problem(
    variance_ready_run: MakeVarianceReadyRun,
) -> PosteriorProblem:
    """The posterior problem a stage builds for the deterministic mean network."""
    run = variance_ready_run()
    probe = _probe()
    _keyed_stage(probe).prepare(run.state)
    assert probe.problem is not None
    return probe.problem


@pytest.fixture(scope="module")
def dropout_run(variance_ready_run: MakeVarianceReadyRun) -> tuple[Any, KeyedProbe]:
    """A dropout posterior network trained by Bayes by Backprop; never mutate it."""
    run: VarianceReadyRun = variance_ready_run()
    probe = _probe()
    stage = _keyed_stage(probe, model=DropoutLinearModel(rngs=nnx.Rngs(0)))
    stage.prepare(run.state)
    result = stage.train(run.state)
    return result, probe


@settings(deadline=None, max_examples=20)
@given(
    first_seed=st.integers(0, 2**31 - 1),
    second_seed=st.integers(0, 2**31 - 1),
    rows=st.integers(1, DATASET_SIZE),
)
def test_a_deterministic_network_gives_identical_results_for_any_key(
    deterministic_problem: PosteriorProblem,
    first_seed: int,
    second_seed: int,
    rows: int,
) -> None:
    problem = deterministic_problem
    inputs, targets = regression_data()
    batch = Batch(inputs=inputs[:rows], targets=targets[:rows])
    first, second = jax.random.key(first_seed), jax.random.key(second_seed)
    parameters = problem.initial_parameters

    assert jnp.array_equal(
        problem.mean_function(parameters, batch.inputs, first),
        problem.mean_function(parameters, batch.inputs, second),
    )
    assert jnp.array_equal(
        problem.log_likelihood(parameters, batch, first),
        problem.log_likelihood(parameters, batch, second),
    )


def test_a_dropout_network_trains_under_the_posterior_stage(
    dropout_run: tuple[Any, KeyedProbe],
) -> None:
    result, _ = dropout_run

    assert result.loss is not None
    assert jnp.isfinite(result.loss)


def test_a_dropout_networks_masks_vary_across_batches(
    dropout_run: tuple[Any, KeyedProbe],
) -> None:
    _, probe = dropout_run

    assert len(probe.probes) == 6
    assert all(
        not jnp.array_equal(earlier, later)
        for index, earlier in enumerate(probe.probes)
        for later in probe.probes[index + 1 :]
    )


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(0, 2**31 - 1))
def test_a_dropout_networks_masks_come_from_the_key(
    dropout_run: tuple[Any, KeyedProbe], seed: int
) -> None:
    _, probe = dropout_run
    problem = probe.problem
    assert problem is not None
    key = jax.random.key(seed)
    parameters = problem.initial_parameters

    first = problem.mean_function(parameters, probe.probe_inputs, key)
    again = jax.jit(problem.mean_function)(parameters, probe.probe_inputs, key)

    assert jnp.allclose(first, again)


_COMPILED_METHODS: dict[str, tuple[Callable[[], InferenceMethod], Any]] = {
    "bayes by backprop": (
        lambda: BayesByBackprop(optimizer=optax.adam(0.01), initial_std=0.05),
        None,
    ),
    "psgld": (lambda: PreconditionedSGLD(step_size=1e-3), None),
    "mc dropout": (
        lambda: MCDropout(optimizer=optax.adam(0.01)),
        lambda: DropoutLinearModel(rngs=nnx.Rngs(5)),
    ),
}


def _compiled_stage(method_name: str, **options: Any) -> PosteriorStage:
    make_method, make_model = _COMPILED_METHODS[method_name]
    return PosteriorStage(
        inference_method=make_method(),
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(**{"epochs": 4, "num_draws": 4, **options}),
        model=None if make_model is None else make_model(),
    )


@pytest.mark.parametrize("method_name", sorted(_COMPILED_METHODS))
def test_posterior_predictive_validation_compiles_nothing_after_warmup(
    shared_variance_ready_run: VarianceReadyRun,
    compile_counter: Callable[[], Any],
    method_name: str,
) -> None:
    state = _isolated(shared_variance_ready_run).state
    with compile_counter() as counter:
        stage = _compiled_stage(
            method_name, validation_loader=regression_loader, event_sinks=(counter,)
        )
        stage.prepare(state)
        stage.train(state)

    assert counter.during_warmup("posterior") > 0
    assert counter.after_warmup("posterior") == 0
    assert len(state.metric_history["posterior/validation/nll"]) == 4


@dataclass(frozen=True)
class OpaquePosterior:
    """A built-in posterior behind a wrapper that is not a TraceablePosterior.

    Attributes:
        inner: The built-in posterior every call is forwarded to.
    """

    inner: Any

    @property
    def num_draws(self) -> int | None:
        return self.inner.num_draws

    def sample_means(
        self, inputs: PyTree, key: jax.Array, num_samples: int | None = None
    ) -> jax.Array:
        return self.inner.sample_means(inputs, key, num_samples)


@dataclass
class OpaquePosteriorMethod:
    """A built-in inference method whose posteriors come wrapped as opaque.

    Attributes:
        inner: The built-in inference method every attribute is forwarded to.
    """

    inner: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def posterior(self) -> OpaquePosterior:
        return OpaquePosterior(self.inner.posterior())


@pytest.mark.parametrize("method_name", sorted(_COMPILED_METHODS))
def test_compiled_and_uncompiled_validation_agree_under_the_same_key(
    shared_variance_ready_run: VarianceReadyRun, method_name: str
) -> None:
    compiled_stage = _compiled_stage(method_name, validation_loader=regression_loader)
    wrapped_stage = _compiled_stage(method_name, validation_loader=regression_loader)
    opaque = OpaquePosteriorMethod(wrapped_stage.inference_method)
    wrapped_stage.inference_method = opaque

    compiled = _validation_history(
        _isolated(shared_variance_ready_run).state, compiled_stage
    )
    uncompiled = _validation_history(
        _isolated(shared_variance_ready_run).state, wrapped_stage
    )

    assert isinstance(opaque.inner.posterior(), TraceablePosterior)
    assert not isinstance(opaque.posterior(), TraceablePosterior)
    for name, values in compiled.items():
        assert len(values) == 4
        assert uncompiled[name] == pytest.approx(values, rel=1e-5)


@dataclass
class TrainedStage:
    """A posterior stage trained, without validation, on a variance-ready run."""

    stage: PosteriorStage
    run: VarianceReadyRun


@pytest.fixture(scope="module")
def trained_stages(variance_ready_run: MakeVarianceReadyRun) -> dict[str, TrainedStage]:
    """Each method's trained posterior stage; never mutate them."""
    trained = {}
    for name in _COMPILED_METHODS:
        run = variance_ready_run()
        stage = _compiled_stage(name)
        stage.prepare(run.state)
        stage.train(run.state)
        trained[name] = TrainedStage(stage, run)
    return trained


@pytest.mark.parametrize("method_name", sorted(_COMPILED_METHODS))
def test_a_trained_posterior_predicts_through_one_compiled_call(
    trained_stages: dict[str, TrainedStage],
    compile_counter: Callable[[], Any],
    method_name: str,
) -> None:
    stage = trained_stages[method_name].stage
    # An input shape no other test predicts at, so the first call compiles.
    inputs = jnp.linspace(-1.0, 1.0, 13)[:, None]
    shifted = inputs + 0.5
    key, other_key = jax.random.key(3), jax.random.key(4)

    with compile_counter() as first:
        predictive = stage.predict(inputs, key)
    with compile_counter() as again:
        repeated = stage.predict(shifted, other_key)

    assert isinstance(predictive, PosteriorPredictive)
    assert isinstance(repeated, PosteriorPredictive)
    assert first.total == 1
    assert again.total == 0


@pytest.mark.parametrize("method_name", sorted(_COMPILED_METHODS))
def test_a_trained_posteriors_prediction_is_its_posterior_predictive(
    trained_stages: dict[str, TrainedStage], method_name: str
) -> None:
    trained = trained_stages[method_name]
    posterior = trained.run.state.model_components["posterior"]
    inputs = jnp.linspace(-1.5, 1.5, 9)[:, None]
    targets = 2.0 * inputs
    key = jax.random.key(11)

    predictive = trained.stage.predict(inputs, key)
    expected = PosteriorPredictive(
        draws=posterior.sample_means(
            inputs, key, None if posterior.num_draws is not None else 4
        ),
        aleatoric_variance=trained.run.variance_model(inputs).mean(),
    )

    np.testing.assert_allclose(predictive.mean(), expected.mean(), rtol=1e-5)
    np.testing.assert_allclose(predictive.variance(), expected.variance(), rtol=1e-5)
    np.testing.assert_allclose(
        predictive.log_prob(targets), expected.log_prob(targets), rtol=1e-5
    )


@pytest.mark.parametrize("method_name", sorted(_COMPILED_METHODS))
def test_a_restored_posterior_stage_predicts_as_the_trained_one(
    variance_ready_run: MakeVarianceReadyRun,
    make_stages: MakeStages,
    method_name: str,
) -> None:
    store = RecordingStore()
    run = variance_ready_run(checkpoint_store=store)
    trained = _compiled_stage(method_name, checkpoint_store=store)
    trained.prepare(run.state)
    trained.train(run.state)
    restored_state = restore_mean_and_variance(make_stages(seed=7), store)
    restored = _compiled_stage(method_name)
    restored.restore(restored_state, store.load("posterior/best"))
    inputs = jnp.linspace(-1.0, 1.0, 6)[:, None]
    key = jax.random.key(2)

    expected = trained.predict(inputs, key)
    actual = restored.predict(inputs, key)

    np.testing.assert_allclose(actual.mean(), expected.mean(), rtol=1e-6)
    np.testing.assert_allclose(actual.variance(), expected.variance(), rtol=1e-6)


def test_an_untrained_posterior_stage_refuses_to_predict() -> None:
    stage = _compiled_stage("psgld")

    with pytest.raises(ValueError, match="no posterior"):
        stage.predict(jnp.zeros((3, 1)), jax.random.key(0))


class _StaticPosteriorHolder(nnx.Module):
    """Holds a posterior as static data, as NNX modules held them before #125."""

    def __init__(self, posterior: Any) -> None:
        self.posterior = nnx.static(posterior)

    def __call__(self, inputs: jax.Array, key: jax.Array) -> jax.Array:
        num_samples = None if self.posterior.num_draws is not None else 4
        return self.posterior.sample_means(inputs, key, num_samples)


@pytest.mark.parametrize("method_name", sorted(_COMPILED_METHODS))
def test_a_built_in_posterior_can_be_held_as_static_data_in_an_nnx_module(
    trained_stages: dict[str, TrainedStage], method_name: str
) -> None:
    posterior = trained_stages[method_name].run.state.model_components["posterior"]
    inputs = jnp.linspace(-1.0, 1.0, 5)[:, None]
    key = jax.random.key(1)

    holder = _StaticPosteriorHolder(posterior)

    num_samples = None if posterior.num_draws is not None else 4
    np.testing.assert_array_equal(
        holder(inputs, key), posterior.sample_means(inputs, key, num_samples)
    )
