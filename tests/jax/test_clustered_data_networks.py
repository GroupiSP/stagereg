"""Tests of the clustered-data example's shared backbone.

The trunk maps the training-input range onto `[-1, 1]` before its first layer, as the
xsin example does, so every network built on it sees inputs of order one.
"""

from __future__ import annotations

from types import ModuleType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")


@pytest.mark.parametrize(("raw", "normalized"), [(-0.2, -1.0), (0.4, 0.0), (1.0, 1.0)])
def test_normalization_maps_the_training_range_onto_minus_one_to_one(
    clustered_networks: ModuleType, raw: float, normalized: float
) -> None:
    inputs = jnp.asarray([[raw]])

    assert clustered_networks.normalize_inputs(inputs)[0, 0] == pytest.approx(
        normalized, abs=1e-6
    )


@settings(deadline=None, max_examples=25)
@given(inputs=st.lists(st.floats(-0.5, 1.5), min_size=1, max_size=8))
def test_trunk_applies_its_layers_to_the_normalized_inputs(
    clustered_networks: ModuleType, inputs: list[float]
) -> None:
    trunk = clustered_networks.Trunk(rngs=nnx.Rngs(0))
    raw = jnp.asarray(inputs, dtype=jnp.float32)[:, None]

    hidden = clustered_networks.normalize_inputs(raw)
    for layer in trunk.layers:
        hidden = jnp.tanh(layer(hidden))

    assert jnp.allclose(trunk(raw), hidden, atol=1e-6)
