"""Tests for the CMAPSS 1D-CNN mean model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

_MODEL_PATH = Path(__file__).parents[2] / "examples" / "jax" / "cmapss" / "model.py"
_SPEC = importlib.util.spec_from_file_location("cmapss_model", _MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load the CMAPSS model module.")
_MODEL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODEL
_SPEC.loader.exec_module(_MODEL)


def test_cnn1d_mean_model_outputs_one_scalar_per_example() -> None:
    batch_size, window_length, n_sensors = 4, 30, 9
    model = _MODEL.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(
        jax.random.key(1), (batch_size, window_length, n_sensors)
    )

    predictions = model(inputs)

    assert predictions.shape == (batch_size, 1)


def test_cnn1d_mean_model_handles_a_single_example() -> None:
    model = _MODEL.Cnn1DMeanModel(3, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(2), (1, 30, 3))

    predictions = model(inputs)

    assert predictions.shape == (1, 1)


def test_cnn1d_mean_model_varies_hidden_channels_and_kernel_size() -> None:
    model = _MODEL.Cnn1DMeanModel(5, hidden_channels=8, kernel_size=3, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(3), (6, 30, 5))

    predictions = model(inputs)

    assert predictions.shape == (6, 1)


def test_cnn1d_gamma_model_outputs_one_gamma_distribution_per_example() -> None:
    batch_size, window_length, n_sensors = 4, 30, 9
    model = _MODEL.Cnn1DGammaModel(n_sensors, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(
        jax.random.key(1), (batch_size, window_length, n_sensors)
    )

    prediction = model(inputs)

    assert prediction.concentration.shape == (batch_size, 1)
    assert prediction.rate.shape == (batch_size, 1)
    assert bool(jax.numpy.all(prediction.concentration > 0))
    assert bool(jax.numpy.all(prediction.rate > 0))


def test_composite_gaussian_model_has_strictly_positive_variance() -> None:
    batch_size, window_length, n_sensors = 4, 30, 9
    mean_model = _MODEL.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    variance_model = _MODEL.Cnn1DGammaModel(n_sensors, rngs=nnx.Rngs(1))
    composite = _MODEL.CompositeGaussianModel(mean_model, variance_model)
    inputs = jax.random.normal(
        jax.random.key(2), (batch_size, window_length, n_sensors)
    )

    prediction = composite(inputs)

    assert prediction.loc.shape == (batch_size, 1)
    assert prediction.scale.shape == (batch_size, 1)
    assert bool(jax.numpy.all(prediction.variance() > 0))


_DROPOUT_MODELS = ("Cnn1DMeanModel", "Cnn1DGaussianModel")
"""The CMAPSS models that accept a dropout rate."""


def _outputs(model: nnx.Module, inputs: jax.Array, **kwargs: object) -> jax.Array:
    """Return a model's outputs as one array, whatever its output type."""
    prediction = model(inputs, **kwargs)
    if isinstance(prediction, jax.Array):
        return prediction
    return jax.numpy.concatenate([prediction.loc, prediction.scale], axis=-1)


@pytest.mark.parametrize("model_name", _DROPOUT_MODELS)
def test_dropout_rate_zero_leaves_the_model_unchanged(model_name: str) -> None:
    model_type = getattr(_MODEL, model_name)
    default = model_type(4, rngs=nnx.Rngs(0))
    zero_rate = model_type(4, dropout_rate=0.0, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(5), (3, 30, 4))

    default_state = nnx.state(default)
    zero_rate_state = nnx.state(zero_rate)

    assert jax.tree.structure(default_state) == jax.tree.structure(zero_rate_state)
    for default_leaf, zero_rate_leaf in zip(
        jax.tree.leaves(default_state), jax.tree.leaves(zero_rate_state), strict=True
    ):
        assert bool(jax.numpy.array_equal(default_leaf, zero_rate_leaf))
    assert bool(
        jax.numpy.array_equal(
            _outputs(default, inputs),
            _outputs(zero_rate, inputs, dropout_key=jax.random.key(6)),
        )
    )


@pytest.mark.parametrize("model_name", _DROPOUT_MODELS)
def test_active_dropout_shares_one_mask_across_the_batch(model_name: str) -> None:
    model = getattr(_MODEL, model_name)(4, dropout_rate=0.5, rngs=nnx.Rngs(0))
    row = jax.random.normal(jax.random.key(7), (1, 30, 4))
    inputs = jax.numpy.repeat(row, 8, axis=0)

    outputs = _outputs(model, inputs, dropout_key=jax.random.key(8))
    model.eval()
    without_dropout = _outputs(model, inputs)

    assert bool(jax.numpy.all(outputs == outputs[:1]))
    assert not bool(jax.numpy.allclose(outputs, without_dropout))


@pytest.mark.parametrize("model_name", _DROPOUT_MODELS)
def test_active_dropout_differs_between_keys(model_name: str) -> None:
    model = getattr(_MODEL, model_name)(4, dropout_rate=0.5, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(9), (3, 30, 4))

    first = _outputs(model, inputs, dropout_key=jax.random.key(10))
    same = _outputs(model, inputs, dropout_key=jax.random.key(10))
    other = _outputs(model, inputs, dropout_key=jax.random.key(11))

    assert bool(jax.numpy.array_equal(first, same))
    assert not bool(jax.numpy.allclose(first, other))


@pytest.mark.parametrize("model_name", _DROPOUT_MODELS)
def test_active_dropout_draws_fresh_masks_from_its_own_streams(
    model_name: str,
) -> None:
    model = getattr(_MODEL, model_name)(4, dropout_rate=0.5, rngs=nnx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(12), (3, 30, 4))

    assert not bool(
        jax.numpy.allclose(_outputs(model, inputs), _outputs(model, inputs))
    )


@settings(max_examples=10, deadline=None)
@given(dropout_rate=st.floats(min_value=0.0, max_value=0.99))
@pytest.mark.parametrize("model_name", _DROPOUT_MODELS)
def test_dropout_rate_leaves_the_parameter_tree_unchanged(
    model_name: str, dropout_rate: float
) -> None:
    model_type = getattr(_MODEL, model_name)
    without_dropout = model_type(4, rngs=nnx.Rngs(0))
    with_dropout = model_type(4, dropout_rate=dropout_rate, rngs=nnx.Rngs(0))

    without_parameters = nnx.state(without_dropout, nnx.Param)
    with_parameters = nnx.state(with_dropout, nnx.Param)

    assert jax.tree.structure(with_parameters) == jax.tree.structure(without_parameters)
    assert [leaf.shape for leaf in jax.tree.leaves(with_parameters)] == [
        leaf.shape for leaf in jax.tree.leaves(without_parameters)
    ]


def test_gamma_model_takes_no_dropout_rate() -> None:
    with pytest.raises(TypeError, match="dropout_rate"):
        _MODEL.Cnn1DGammaModel(4, dropout_rate=0.1, rngs=nnx.Rngs(0))
