"""Tests of the clustered-data example's joint posterior.

The joint run infers every weight of one network with a mean head and a variance head,
from a random initialization, under the network's own predicted variance.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")


class _RecordingMethod:
    """An inference method that records what the hand loop asks of it.

    Attributes:
        problem: The problem it was initialized with, or `None` before `init`.
        updates: Update steps taken so far.
    """

    _method: Any
    problem: Any
    updates: int

    def __init__(self, method: Any) -> None:
        self._method = method
        self.problem = None
        self.updates = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._method, name)

    def init(self, problem: Any) -> None:
        self.problem = problem
        self._method.init(problem)

    def update(self, batch: Any, key: Any) -> Mapping[str, Any]:
        self.updates += 1
        return self._method.update(batch, key)


def test_joint_run_updates_once_per_batch_of_its_epoch_budget(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    method = _RecordingMethod(
        clustered_methods.bayes_by_backprop(tiny_clustered_config)
    )

    clustered_joint.run_joint(clustered_data, tiny_clustered_config, method, epochs=3)

    assert method.updates == 3 * 10  # 1,000 points in batches of 100
    assert method.problem.epochs == 3


def test_joint_run_infers_both_heads_from_a_random_initialization(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_dataset: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    method = _RecordingMethod(
        clustered_methods.bayes_by_backprop(tiny_clustered_config)
    )

    clustered_joint.run_joint(clustered_data, tiny_clustered_config, method, epochs=1)

    parameters = method.problem.initial_parameters
    assert {"trunk", "mean_head", "variance_head"} <= set(parameters)
    other_seed = nnx.state(
        clustered_joint.joint_network(clustered_dataset.ClusteredDataConfig(seed=4)),
        nnx.Param,
    )
    assert not jnp.array_equal(
        parameters["variance_head"]["kernel"][...],
        other_seed["variance_head"]["kernel"][...],
    )


def test_joint_draws_carry_their_own_positive_variances(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    draws = clustered_joint.run_joint(
        clustered_data,
        tiny_clustered_config,
        clustered_methods.bayes_by_backprop(tiny_clustered_config),
        epochs=1,
    )

    grid_size = clustered_data.grid_inputs.shape[0]
    assert draws.means.shape == (4, grid_size)
    assert draws.variances.shape == (4, grid_size)
    assert bool(jnp.all(draws.variances >= 1e-6))
    assert not bool(jnp.all(draws.variances == draws.variances[:1]))


def test_joint_run_is_reproducible(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    first, second = (
        clustered_joint.run_joint(
            clustered_data,
            tiny_clustered_config,
            clustered_methods.bayes_by_backprop(tiny_clustered_config),
            epochs=1,
        )
        for _ in range(2)
    )

    assert jnp.array_equal(first.means, second.means)
    assert jnp.array_equal(first.variances, second.variances)


def test_log_likelihood_is_the_scaled_gaussian_under_the_variance_head(
    clustered_joint: ModuleType,
    clustered_dataset: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    network = clustered_joint.joint_network(tiny_clustered_config)
    problem = clustered_joint.joint_problem(
        network, clustered_data, tiny_clustered_config, epochs=1
    )
    loader = clustered_dataset.make_loader(clustered_data, tiny_clustered_config)
    batch = loader(split="train", epoch=0)[0]
    outputs = network(batch.inputs)
    mean, variance = outputs[:, :1], outputs[:, 1:]

    log_likelihood = problem.log_likelihood(
        problem.initial_parameters, batch, jax.random.key(0)
    )

    # 1,000 training points over a batch of 100 scale the batch sum by 10.
    expected = 10 * jnp.sum(
        jax.scipy.stats.norm.logpdf(batch.targets, mean, jnp.sqrt(variance))
    )
    assert jnp.allclose(log_likelihood, expected, rtol=1e-5)
    assert bool(jnp.all(variance >= 1e-6))


@settings(deadline=None, max_examples=25)
@given(point_epochs=st.integers(min_value=0, max_value=2_000))
def test_joint_psgld_burns_in_the_staged_point_epochs_too(
    clustered_methods: ModuleType, tiny_clustered_config: Any, point_epochs: int
) -> None:
    staged = clustered_methods.METHODS[clustered_methods.PSGLD](tiny_clustered_config)

    joint = clustered_methods.joint_method(
        clustered_methods.PSGLD, tiny_clustered_config, point_epochs=point_epochs
    )

    steps_per_epoch = tiny_clustered_config.steps_per_epoch
    assert joint.burn_in == staged.burn_in + point_epochs * steps_per_epoch
    assert joint.thinning == staged.thinning
    assert joint.step_size == staged.step_size


@pytest.mark.parametrize("name", ["BBB", "MCD"])
def test_joint_bbb_and_mcd_are_built_as_their_staged_counterparts(
    clustered_methods: ModuleType, tiny_clustered_config: Any, name: str
) -> None:
    staged = clustered_methods.METHODS[name](tiny_clustered_config)

    joint = clustered_methods.joint_method(name, tiny_clustered_config, point_epochs=6)

    assert type(joint) is type(staged)
    if name == "BBB":
        assert joint.initial_std == staged.initial_std


def test_joint_psgld_retains_num_draws_over_the_final_posterior_epochs(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    point_epochs, posterior_epochs = 1, 2
    config = tiny_clustered_config.__class__(
        posterior_epochs=posterior_epochs, num_draws=4, seed=3
    )
    method = clustered_methods.joint_method(
        clustered_methods.PSGLD, config, point_epochs=point_epochs
    )

    draws = clustered_joint.run_joint(
        clustered_data, config, method, epochs=point_epochs + posterior_epochs
    )

    assert draws.means.shape == (4, clustered_data.grid_inputs.shape[0])
    total_steps = (point_epochs + posterior_epochs) * config.steps_per_epoch
    first_draw = method.burn_in + method.thinning
    # Every draw falls in the final half of the posterior epochs.
    assert first_draw > (point_epochs + posterior_epochs // 2) * config.steps_per_epoch
    assert method.burn_in + 4 * method.thinning == total_steps


def test_joint_mc_dropout_draws_one_dropout_mask_per_draw(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    network = clustered_joint.joint_network(
        tiny_clustered_config, dropout_rate=tiny_clustered_config.mc_dropout_rate
    )

    draws = clustered_joint.run_joint(
        clustered_data,
        tiny_clustered_config,
        clustered_methods.mc_dropout(tiny_clustered_config),
        epochs=1,
        network=network,
    )

    assert draws.means.shape == (4, clustered_data.grid_inputs.shape[0])
    # Each draw is one function of the input: its own mask over every point.
    assert not bool(jnp.all(draws.means == draws.means[:1]))
    assert not bool(jnp.all(draws.variances == draws.variances[:1]))
