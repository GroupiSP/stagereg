from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from posterior_runs import (
    DATASET_SIZE,
    MakeStages,
    MakeVarianceReadyRun,
    RecordingStore,
    regression_loader,
    restore_mean_and_variance,
)

from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.types import Batch, PyTree, StageState
from stagereg.jax import (
    IsotropicGaussianPrior,
    PosteriorProblem,
    PosteriorStage,
    PosteriorStageOptions,
    PreconditionedSGLD,
    RetainedSamplesPosterior,
)

_NOISE_SCALE = 0.5
_PRIOR_PRECISION = 1.0


def _linear_mean(parameters: PyTree, inputs: jax.Array, key: jax.Array) -> jax.Array:
    """A line through the inputs: ``weight * x + bias``; it ignores the key."""
    del key
    return parameters["weight"] * inputs[:, 0] + parameters["bias"]


def _linear_regression_batch() -> Batch:
    """Thirty-two noisy points on ``y = 1.5x - 0.5``."""
    inputs = jnp.linspace(-1.0, 1.0, 32)[:, None]
    noise = _NOISE_SCALE * jax.random.normal(jax.random.key(11), (32,))
    return Batch(inputs=inputs, targets=1.5 * inputs[:, 0] - 0.5 + noise)


def _linear_regression_problem(batch: Batch) -> PosteriorProblem:
    """Bayesian linear regression with known noise, one full batch per step."""

    def log_likelihood(parameters: PyTree, batch: Batch, key: jax.Array) -> jax.Array:
        means = _linear_mean(parameters, batch.inputs, key)
        return jnp.sum(jstats.norm.logpdf(batch.targets, means, _NOISE_SCALE))

    return PosteriorProblem(
        initial_parameters={"weight": jnp.asarray(0.0), "bias": jnp.asarray(0.0)},
        mean_function=_linear_mean,
        log_likelihood=log_likelihood,
        prior=IsotropicGaussianPrior(precision=_PRIOR_PRECISION),
        train_loader=lambda *, split, epoch: [batch],
        dataset_size=int(batch.inputs.shape[0]),
    )


def _analytic_predictive(
    batch: Batch, inputs: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """The exact posterior mean and spread of the line at ``inputs``.

    Returns:
        The mean and the standard deviation of ``weight * x + bias`` under the
        Gaussian posterior.
    """
    features = np.column_stack([np.asarray(batch.inputs[:, 0]), np.ones(32)])
    targets = np.asarray(batch.targets)
    precision = _PRIOR_PRECISION * np.eye(2) + features.T @ features / _NOISE_SCALE**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ features.T @ targets / _NOISE_SCALE**2
    query = np.column_stack([np.asarray(inputs[:, 0]), np.ones(inputs.shape[0])])
    spread = np.sqrt(np.einsum("ij,jk,ik->i", query, covariance, query))
    return query @ mean, spread


def _run_chain(
    method: PreconditionedSGLD, problem: PosteriorProblem, steps: int, seed: int = 0
) -> None:
    method.init(problem)
    batch = next(iter(problem.train_loader(split="train", epoch=0)))
    for key in jax.random.split(jax.random.key(seed), steps):
        method.update(batch, key)


def test_retained_samples_match_the_bayesian_linear_regression_posterior() -> None:
    batch = _linear_regression_batch()
    method = PreconditionedSGLD(step_size=0.03, burn_in=1_000, thinning=20)
    inputs = jnp.linspace(-2.0, 2.0, 9)[:, None]

    _run_chain(method, _linear_regression_problem(batch), steps=41_000)
    draws = np.asarray(method.posterior().sample_means(inputs, jax.random.key(0)))

    expected_mean, expected_spread = _analytic_predictive(batch, inputs)
    assert draws.shape == (2_000, 9)
    np.testing.assert_allclose(
        draws.mean(axis=0), expected_mean, atol=0.25 * expected_spread.min()
    )
    np.testing.assert_allclose(draws.std(axis=0), expected_spread, rtol=0.15)


@pytest.mark.parametrize(
    ("hyperparameters", "message"),
    [
        ({"step_size": 0.0}, "step_size"),
        ({"step_size": float("nan")}, "step_size"),
        ({"stability": 0.0}, "stability"),
        ({"burn_in": -1}, "burn_in"),
        ({"thinning": 0}, "thinning"),
        ({"decay": 1.0}, "decay"),
        ({"decay": -0.1}, "decay"),
    ],
)
def test_invalid_hyperparameters_are_refused(
    hyperparameters: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PreconditionedSGLD(**{"step_size": 0.01, **hyperparameters})


_MAX_STEPS = 16


@pytest.fixture(scope="session")
def linear_regression_problem() -> PosteriorProblem:
    """One Bayesian linear regression problem shared by the retention tests."""
    return _linear_regression_problem(_linear_regression_batch())


@pytest.fixture(scope="session")
def every_position(linear_regression_problem: PosteriorProblem) -> PyTree:
    """The first ``_MAX_STEPS`` positions of the chain, stacked."""
    method = PreconditionedSGLD(step_size=0.01)
    _run_chain(method, linear_regression_problem, _MAX_STEPS)
    return method.posterior_state()


@settings(deadline=None, max_examples=12)
@given(
    burn_in=st.integers(min_value=0, max_value=6),
    thinning=st.integers(min_value=1, max_value=4),
    steps=st.integers(min_value=0, max_value=_MAX_STEPS),
)
def test_burn_in_and_thinning_retain_exactly_the_due_positions(
    linear_regression_problem: PosteriorProblem,
    every_position: PyTree,
    burn_in: int,
    thinning: int,
    steps: int,
) -> None:
    method = PreconditionedSGLD(step_size=0.01, burn_in=burn_in, thinning=thinning)

    _run_chain(method, linear_regression_problem, steps)

    due = list(range(burn_in + thinning, steps + 1, thinning))
    if not due:
        with pytest.raises(ValueError, match="no sample has been retained"):
            method.posterior()
        return
    expected = jax.tree.map(lambda leaf: leaf[np.asarray(due) - 1], every_position)
    assert method.posterior().num_draws == len(due)
    jax.tree.map(np.testing.assert_array_equal, method.posterior().samples, expected)
    jax.tree.map(np.testing.assert_array_equal, method.posterior_state(), expected)


def test_samples_load_before_init_without_a_tree_check(every_position: PyTree) -> None:
    method = PreconditionedSGLD(step_size=0.01)

    method.load_posterior(every_position)

    jax.tree.map(
        np.testing.assert_array_equal, method.posterior_state(), every_position
    )


def test_a_chain_resumed_from_its_state_continues_as_if_uninterrupted(
    linear_regression_problem: PosteriorProblem,
) -> None:
    batch = next(iter(linear_regression_problem.train_loader(split="train", epoch=0)))
    keys = jax.random.split(jax.random.key(4), 10)
    uninterrupted = PreconditionedSGLD(step_size=0.01, burn_in=3, thinning=2)
    interrupted = PreconditionedSGLD(step_size=0.01, burn_in=3, thinning=2)
    resumed = PreconditionedSGLD(step_size=0.01, burn_in=3, thinning=2)
    uninterrupted.init(linear_regression_problem)
    interrupted.init(linear_regression_problem)
    resumed.init(linear_regression_problem)

    for key in keys:
        uninterrupted.update(batch, key)
    for key in keys[:6]:
        interrupted.update(batch, key)
    resumed.load_state(interrupted.state())
    for key in keys[6:]:
        resumed.update(batch, key)

    jax.tree.map(
        np.testing.assert_array_equal,
        resumed.state(),
        uninterrupted.state(),
    )


def _five_lines() -> RetainedSamplesPosterior:
    """A posterior of five retained lines."""
    weights, biases = jax.random.normal(jax.random.key(5), (2, 5))
    return RetainedSamplesPosterior(
        mean_function=_linear_mean, samples={"weight": weights, "bias": biases}
    )


def test_a_retained_samples_posterior_refuses_an_explicit_draw_count() -> None:
    posterior = _five_lines()

    with pytest.raises(ValueError, match="num_samples must be None"):
        posterior.sample_means(jnp.zeros((3, 1)), jax.random.key(0), num_samples=5)


@settings(deadline=None, max_examples=20)
@given(
    split=st.integers(min_value=0, max_value=7),
    seeds=st.tuples(st.integers(0, 2**31 - 1), st.integers(0, 2**31 - 1)),
)
def test_each_draw_is_the_same_function_across_inputs_and_keys(
    split: int, seeds: tuple[int, int]
) -> None:
    posterior = _five_lines()
    inputs = jnp.linspace(-1.0, 1.0, 7)[:, None]
    first, second = (jax.random.key(seed) for seed in seeds)

    whole = posterior.sample_means(inputs, first)
    parts = jnp.concatenate(
        [
            posterior.sample_means(inputs[:split], first),
            posterior.sample_means(inputs[split:], second),
        ],
        axis=1,
    )

    assert posterior.num_draws == 5
    assert whole.shape == (5, 7)
    np.testing.assert_allclose(parts, whole, rtol=1e-6)


def _psgld_stage(method: PreconditionedSGLD, **options: Any) -> PosteriorStage:
    return PosteriorStage(
        inference_method=method,
        train_loader=regression_loader,
        dataset_size=DATASET_SIZE,
        options=PosteriorStageOptions(**{"epochs": 3, **options}),
    )


def test_the_posterior_stage_refuses_an_early_stopper_for_psgld(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    state = variance_ready_run().state
    stage = _psgld_stage(
        PreconditionedSGLD(step_size=1e-3),
        early_stopper=EarlyStopper(metric="loss", mode="min", patience=1),
    )

    with pytest.raises(ValueError, match="does not support early stopping"):
        stage.prepare(state)

    assert state.lifecycle_state is StageState.VARIANCE_READY


def test_psgld_writes_only_a_finalized_checkpoint_that_restores_its_samples(
    variance_ready_run: MakeVarianceReadyRun,
    make_stages: MakeStages,
) -> None:
    store = RecordingStore()
    state = variance_ready_run(checkpoint_store=store).state
    stage = _psgld_stage(
        PreconditionedSGLD(step_size=1e-3, burn_in=1, thinning=2),
        checkpoint_store=store,
    )
    stage.prepare(state)
    stage.train(state)

    assert [key for key, _ in store.saves if key.startswith("posterior/")] == [
        "posterior/best"
    ]
    finalized = store.load("posterior/best")
    assert finalized.metadata == {"stage": "posterior", "stage_complete": True}
    assert finalized.parameters is None
    trained = state.model_components["posterior"]
    assert trained.num_draws == 2

    restored_state = restore_mean_and_variance(make_stages(seed=7), store)
    restored_stage = _psgld_stage(PreconditionedSGLD(step_size=1e-3))
    restored_stage.restore(restored_state, finalized)

    assert restored_stage.validate(restored_state).passed
    inputs = jnp.linspace(-1.0, 1.0, 5)[:, None]
    key = jax.random.key(0)
    np.testing.assert_array_equal(
        restored_state.model_components["posterior"].sample_means(inputs, key),
        trained.sample_means(inputs, key),
    )


def test_the_posterior_stage_skips_validation_until_psgld_retains_a_sample(
    variance_ready_run: MakeVarianceReadyRun,
) -> None:
    state = variance_ready_run().state
    # Two batches per epoch: the first sample is retained in the third epoch.
    method = PreconditionedSGLD(step_size=1e-3, burn_in=4, thinning=1)
    stage = _psgld_stage(method, epochs=4, validation_loader=regression_loader)

    stage.prepare(state)
    stage.train(state)

    history = state.metric_history
    assert len(history["posterior/train/loss"]) == 4
    for name in ("nll", "crps"):
        values = history[f"posterior/validation/{name}"]
        assert len(values) == 2
        assert all(math.isfinite(value) for value in values)
    assert state.model_components["posterior"].num_draws == 4


def test_a_psgld_posterior_stage_compiles_nothing_after_warmup_as_samples_accumulate(
    variance_ready_run: MakeVarianceReadyRun,
    compile_counter: Callable[[], Any],
) -> None:
    state = variance_ready_run().state
    method = PreconditionedSGLD(step_size=1e-3, burn_in=0, thinning=1)
    with compile_counter() as counter:
        stage = _psgld_stage(
            method,
            epochs=5,
            validation_loader=regression_loader,
            event_sinks=(counter,),
        )
        stage.prepare(state)
        stage.train(state)

    assert counter.during_warmup("posterior") > 0
    assert counter.after_warmup("posterior") == 0
    assert len(state.metric_history["posterior/validation/nll"]) == 5
    # Two batches per epoch, every position retained.
    assert state.model_components["posterior"].num_draws == 10


def test_a_psgld_posterior_stage_sizes_its_sample_buffer_from_its_epochs(
    variance_ready_run: MakeVarianceReadyRun,
    compile_counter: Callable[[], Any],
) -> None:
    # Two batches per epoch, every position retained: 80 samples, more than
    # the 64 slots a chain reserves when it knows nothing of its run.
    state = variance_ready_run().state
    method = PreconditionedSGLD(step_size=1e-3, burn_in=0, thinning=1)
    with compile_counter() as counter:
        stage = _psgld_stage(
            method,
            epochs=40,
            validation_loader=regression_loader,
            event_sinks=(counter,),
        )
        stage.prepare(state)
        stage.train(state)

    assert counter.after_warmup("posterior") == 0
    assert state.model_components["posterior"].num_draws == 80


def test_a_full_sample_buffer_grows_and_keeps_every_retained_position(
    linear_regression_problem: PosteriorProblem, every_position: PyTree
) -> None:
    method = PreconditionedSGLD(step_size=0.01, capacity=3)

    _run_chain(method, linear_regression_problem, _MAX_STEPS)

    assert method.posterior().num_draws == _MAX_STEPS
    jax.tree.map(
        np.testing.assert_array_equal, method.posterior_state(), every_position
    )


def test_a_posterior_is_independent_of_later_updates(
    linear_regression_problem: PosteriorProblem, every_position: PyTree
) -> None:
    method = PreconditionedSGLD(step_size=0.01, capacity=4)
    _run_chain(method, linear_regression_problem, 3)
    early = method.posterior()
    inputs = jnp.linspace(-1.0, 1.0, 5)[:, None]
    before = early.sample_means(inputs, jax.random.key(0))

    batch = next(iter(linear_regression_problem.train_loader(split="train", epoch=0)))
    for key in jax.random.split(jax.random.key(9), 6):
        method.update(batch, key)

    assert early.num_draws == 3
    np.testing.assert_array_equal(early.sample_means(inputs, jax.random.key(0)), before)
    jax.tree.map(
        np.testing.assert_array_equal,
        early.samples,
        jax.tree.map(lambda leaf: leaf[:3], every_position),
    )


def test_psgld_refuses_a_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        PreconditionedSGLD(step_size=0.01, capacity=0)


def test_a_buffer_first_due_on_a_ragged_batch_is_sized_from_the_full_batches() -> None:
    # Batches of 31 and 1 rows over 10 epochs: 20 steps, 19 retained. Sizing
    # from the 1-row batch the first sample falls on would plan 320 steps.
    full = _linear_regression_batch()
    batches = [
        Batch(inputs=full.inputs[:31], targets=full.targets[:31]),
        Batch(inputs=full.inputs[31:], targets=full.targets[31:]),
    ]
    problem = dataclasses.replace(
        _linear_regression_problem(full),
        train_loader=lambda *, split, epoch: batches,
        epochs=10,
    )
    method = PreconditionedSGLD(step_size=0.01, burn_in=1, thinning=1)
    method.init(problem)

    for key, batch in zip(jax.random.split(jax.random.key(0), 2), batches):
        method.update(batch, key)

    draw_mask = method.posterior()._traced().draw_mask
    assert draw_mask is not None
    assert draw_mask.shape == (19,)
