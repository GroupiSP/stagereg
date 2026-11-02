"""Tests of the clustered-data example's joint posterior.

The joint run infers every weight of one network with a mean head and a variance head,
from a random initialization, under the network's own predicted variance.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")


@pytest.fixture(scope="session")
def tiny_joint_config(clustered_dataset: ModuleType) -> Any:
    """A clustered-data configuration whose joint run trains in a second."""
    return clustered_dataset.ClusteredDataConfig(num_draws=4, seed=3)


@pytest.fixture(scope="session")
def clustered_data(clustered_dataset: ModuleType, tiny_joint_config: Any) -> Any:
    """The clustered training set and evaluation grid."""
    return clustered_dataset.make_clustered_data(tiny_joint_config)


class _RecordingMethod:
    """An inference method that records what the hand loop asks of it."""

    def __init__(self, method: Any) -> None:
        self._method = method
        self.problem: Any = None
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
    tiny_joint_config: Any,
) -> None:
    method = _RecordingMethod(clustered_methods.bayes_by_backprop(tiny_joint_config))

    clustered_joint.run_joint(clustered_data, tiny_joint_config, method, epochs=3)

    assert method.updates == 3 * 10  # 1,000 points in batches of 100
    assert method.problem.epochs == 3


def test_joint_run_infers_both_heads_from_a_random_initialization(
    clustered_joint: ModuleType,
    clustered_methods: ModuleType,
    clustered_dataset: ModuleType,
    clustered_data: Any,
    tiny_joint_config: Any,
) -> None:
    method = _RecordingMethod(clustered_methods.bayes_by_backprop(tiny_joint_config))

    clustered_joint.run_joint(clustered_data, tiny_joint_config, method, epochs=1)

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
    tiny_joint_config: Any,
) -> None:
    draws = clustered_joint.run_joint(
        clustered_data,
        tiny_joint_config,
        clustered_methods.bayes_by_backprop(tiny_joint_config),
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
    tiny_joint_config: Any,
) -> None:
    first, second = (
        clustered_joint.run_joint(
            clustered_data,
            tiny_joint_config,
            clustered_methods.bayes_by_backprop(tiny_joint_config),
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
    tiny_joint_config: Any,
) -> None:
    network = clustered_joint.joint_network(tiny_joint_config)
    problem = clustered_joint.joint_problem(
        network, clustered_data, tiny_joint_config, epochs=1
    )
    loader = clustered_dataset.make_loader(clustered_data, tiny_joint_config)
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
