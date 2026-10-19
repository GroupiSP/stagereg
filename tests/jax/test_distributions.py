from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statistics import NormalDist
from flax import nnx
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from stagereg.jax.distributions import (
    Gamma,
    GammaHead,
    Gaussian,
    GaussianHead,
    MomentMatchedPredictive,
    PosteriorPredictive,
)
from stagereg.core import Batch

_FINITE = st.floats(min_value=-10.0, max_value=10.0, width=32)
_POSITIVE = st.floats(min_value=0.0625, max_value=10.0, width=32)


@st.composite
def posterior_predictives(
    draw: st.DrawFn, *, num_draws: st.SearchStrategy[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Draw component means ``[S, B]`` and an aleatoric variance ``[B]``."""
    count = draw(num_draws if num_draws is not None else st.integers(1, 6))
    batch = draw(st.integers(1, 4))
    means = draw(arrays(np.float32, (count, batch), elements=_FINITE))
    variance = draw(arrays(np.float32, (batch,), elements=_POSITIVE))
    return means, variance


def test_gaussian_log_prob_matches_analytic_normal_density() -> None:
    loc = jnp.array([0.0, 1.0])
    scale = jnp.array([1.0, 2.0])
    targets = jnp.array([0.0, 3.0])
    distribution = Gaussian(loc=loc, scale=scale)

    expected = (
        -0.5 * jnp.log(2 * jnp.pi * scale**2) - 0.5 * ((targets - loc) ** 2) / scale**2
    )

    assert jnp.allclose(distribution.log_prob(targets), expected)


def test_gaussian_mean_and_variance() -> None:
    loc = jnp.array([1.0, -2.0])
    scale = jnp.array([0.5, 3.0])
    distribution = Gaussian(loc=loc, scale=scale)

    assert jnp.allclose(distribution.mean(), loc)
    assert jnp.allclose(distribution.variance(), scale**2)


def test_gaussian_batch_and_event_shape() -> None:
    distribution = Gaussian(loc=jnp.zeros((4, 2)), scale=jnp.ones((4, 2)))

    assert distribution.batch_shape == (4, 2)
    assert distribution.event_shape == ()


def test_gaussian_sample_shape_includes_sample_and_batch_shape() -> None:
    distribution = Gaussian(loc=jnp.zeros((3,)), scale=jnp.ones((3,)))

    samples = distribution.sample(jax.random.key(0), sample_shape=(5,))

    assert samples.shape == (5, 3)


def test_gaussian_sample_matches_reparametrized_normal() -> None:
    loc = jnp.array([2.0])
    scale = jnp.array([0.5])
    distribution = Gaussian(loc=loc, scale=scale)
    key = jax.random.key(0)

    sample = distribution.sample(key)
    expected = loc + scale * jax.random.normal(key, (1,))

    assert jnp.allclose(sample, expected)


def test_gaussian_head_produces_positive_scale_under_extreme_inputs() -> None:
    head = GaussianHead(1, 2, rngs=nnx.Rngs(0))
    extreme_features = jnp.array([[1e6], [-1e6]])

    prediction = head(extreme_features)

    assert prediction.loc.shape == (2, 2)
    assert prediction.scale.shape == (2, 2)
    assert bool(jnp.all(prediction.scale > 0.0))
    assert bool(jnp.all(jnp.isfinite(prediction.scale)))
    assert bool(jnp.all(jnp.isfinite(prediction.loc)))


def test_gaussian_head_rejects_non_positive_out_features() -> None:
    with pytest.raises(ValueError, match="out_features"):
        GaussianHead(1, 0, rngs=nnx.Rngs(0))


@given(
    eps=st.one_of(
        st.floats(
            max_value=0.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.sampled_from([float("nan"), float("inf"), float("-inf")]),
    )
)
@example(eps=float("nan"))
def test_gaussian_head_rejects_invalid_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        GaussianHead(1, 1, rngs=nnx.Rngs(0), eps=eps)


def test_gamma_log_prob_matches_analytic_shape_rate_density() -> None:
    concentration = jnp.array([2.0, 3.0])
    rate = jnp.array([4.0, 0.5])
    targets = jnp.array([0.5, 2.0])
    distribution = Gamma(concentration=concentration, rate=rate)

    expected = (
        concentration * jnp.log(rate)
        - jax.scipy.special.gammaln(concentration)
        + (concentration - 1.0) * jnp.log(targets)
        - rate * targets
    )

    assert jnp.allclose(distribution.log_prob(targets), expected)


def test_gamma_mean_and_variance_use_shape_rate_parameterization() -> None:
    concentration = jnp.array([2.0, 8.0])
    rate = jnp.array([4.0, 2.0])
    distribution = Gamma(concentration=concentration, rate=rate)

    assert jnp.allclose(distribution.mean(), concentration / rate)
    assert jnp.allclose(distribution.variance(), concentration / rate**2)


def test_gamma_batch_and_event_shape() -> None:
    distribution = Gamma(
        concentration=jnp.ones((4, 1)),
        rate=jnp.ones((1, 2)),
    )

    assert distribution.batch_shape == (4, 2)
    assert distribution.event_shape == ()


def test_gamma_sample_shape_and_key_are_reproducible() -> None:
    distribution = Gamma(
        concentration=jnp.array([2.0, 3.0]),
        rate=jnp.array([1.0, 2.0]),
    )
    key = jax.random.key(0)

    samples = distribution.sample(key, sample_shape=(5,))
    duplicate = distribution.sample(key, sample_shape=(5,))

    assert samples.shape == (5, 2)
    assert jnp.array_equal(samples, duplicate)
    assert bool(jnp.all(samples > 0.0))


def test_gamma_head_produces_positive_parameters_under_extreme_inputs() -> None:
    head = GammaHead(1, 2, rngs=nnx.Rngs(0))

    prediction = head(jnp.array([[1e6], [-1e6]]))

    assert prediction.concentration.shape == (2, 2)
    assert prediction.rate.shape == (2, 2)
    assert bool(jnp.all(prediction.concentration > 0.0))
    assert bool(jnp.all(prediction.rate > 0.0))
    assert bool(jnp.all(jnp.isfinite(prediction.concentration)))
    assert bool(jnp.all(jnp.isfinite(prediction.rate)))


@pytest.mark.parametrize(
    ("out_features", "eps"),
    [(0, 1e-6), (1, 0.0), (1, -1.0), (1, float("inf"))],
)
def test_gamma_head_rejects_invalid_configuration(
    out_features: int,
    eps: float,
) -> None:
    with pytest.raises(ValueError):
        GammaHead(1, out_features, rngs=nnx.Rngs(0), eps=eps)


@settings(deadline=None, max_examples=25)
@given(
    components=posterior_predictives(num_draws=st.just(1)),
    targets=_FINITE,
    seed=st.integers(0, 2**31 - 1),
)
def test_posterior_predictive_with_one_draw_is_the_gaussian(
    components: tuple[np.ndarray, np.ndarray], targets: float, seed: int
) -> None:
    means, variance = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    gaussian = Gaussian(loc=jnp.asarray(means[0]), scale=jnp.sqrt(variance))
    key = jax.random.key(seed)
    y = jnp.full(variance.shape, targets)

    assert mixture.batch_shape == gaussian.batch_shape
    assert mixture.event_shape == ()
    np.testing.assert_allclose(mixture.log_prob(y), gaussian.log_prob(y), rtol=1e-5)
    np.testing.assert_allclose(mixture.mean(), gaussian.mean(), rtol=1e-6)
    np.testing.assert_allclose(mixture.variance(), gaussian.variance(), rtol=1e-6)
    np.testing.assert_allclose(
        mixture.sample(key, (3,)), gaussian.sample(key, (3,)), rtol=1e-6
    )


def _normal_log_density(y: np.ndarray, loc: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (y - loc) ** 2 / var


@settings(deadline=None, max_examples=25)
@given(components=posterior_predictives(), targets=_FINITE)
def test_posterior_predictive_log_prob_is_logsumexp_of_components_minus_log_s(
    components: tuple[np.ndarray, np.ndarray], targets: float
) -> None:
    means, variance = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    y = np.full(variance.shape, targets, dtype=np.float64)
    component = _normal_log_density(y, means.astype(np.float64), variance)
    peak = component.max(axis=0)
    expected = (
        peak + np.log(np.exp(component - peak).sum(axis=0)) - np.log(means.shape[0])
    )

    np.testing.assert_allclose(mixture.log_prob(jnp.asarray(y)), expected, atol=1e-3)


@settings(deadline=None, max_examples=25)
@given(components=posterior_predictives())
def test_posterior_predictive_moments_obey_the_law_of_total_variance(
    components: tuple[np.ndarray, np.ndarray],
) -> None:
    means, variance = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    draws = means.astype(np.float64)
    expected_mean = draws.mean(axis=0)
    epistemic = ((draws - expected_mean) ** 2).mean(axis=0)
    summary = mixture.moment_matched()

    np.testing.assert_allclose(mixture.mean(), expected_mean, atol=1e-4)
    np.testing.assert_allclose(mixture.variance(), variance + epistemic, atol=1e-3)
    np.testing.assert_allclose(summary.aleatoric_variance, variance, rtol=1e-6)
    np.testing.assert_allclose(summary.epistemic_variance, epistemic, atol=1e-3)
    np.testing.assert_allclose(summary.mean(), mixture.mean(), rtol=1e-6)
    np.testing.assert_allclose(summary.variance(), mixture.variance(), rtol=1e-6)
    assert summary.batch_shape == mixture.batch_shape
    assert summary.event_shape == ()


@settings(deadline=None, max_examples=15)
@given(components=posterior_predictives(), seed=st.integers(0, 2**31 - 1))
def test_posterior_predictive_sample_moments_match_its_mean_and_variance(
    components: tuple[np.ndarray, np.ndarray], seed: int
) -> None:
    means, variance = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    count = 20_000

    samples = np.asarray(mixture.sample(jax.random.key(seed), (count,)), np.float64)

    assert samples.shape == (count, *mixture.batch_shape)
    mean = np.asarray(mixture.mean(), np.float64)
    var = np.asarray(mixture.variance(), np.float64)
    centred = samples - samples.mean(axis=0)
    fourth = (centred**4).mean(axis=0)
    assert np.all(np.abs(samples.mean(axis=0) - mean) <= 6.0 * np.sqrt(var / count))
    assert np.all(
        np.abs(samples.var(axis=0) - var) <= 6.0 * np.sqrt(fourth / count) + 1e-3
    )


def _mixture_cdf(y: np.ndarray, means: np.ndarray, variance: np.ndarray) -> np.ndarray:
    standard = NormalDist()
    z = (y - means.astype(np.float64)) / np.sqrt(variance.astype(np.float64))
    return np.vectorize(standard.cdf)(z).mean(axis=0)


@settings(deadline=None, max_examples=25)
@given(
    components=posterior_predictives(),
    probability=st.floats(min_value=0.01, max_value=0.99),
)
def test_posterior_predictive_quantile_inverts_the_mixture_cdf(
    components: tuple[np.ndarray, np.ndarray], probability: float
) -> None:
    means, variance = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )

    quantile = np.asarray(mixture.quantile(probability), np.float64)

    assert quantile.shape == mixture.batch_shape
    np.testing.assert_allclose(
        _mixture_cdf(quantile, means, variance), probability, atol=1e-4
    )


def test_posterior_predictive_keeps_the_draw_axis_out_of_broadcasting() -> None:
    draws = jnp.array([[0.0, 1.0, 2.0], [-1.0, 0.5, 3.0]])
    variance = jnp.array([[0.5], [1.0], [2.0]])
    lower_rank = PosteriorPredictive(draws=draws, aleatoric_variance=variance)
    aligned = PosteriorPredictive(draws=draws[:, None, :], aleatoric_variance=variance)
    targets = jnp.array([0.3, -0.2, 1.5])

    assert lower_rank.batch_shape == (3, 3)
    for value, expected in [
        (lower_rank.log_prob(targets), aligned.log_prob(targets)),
        (lower_rank.cdf(targets), aligned.cdf(targets)),
        (lower_rank.quantile(0.3), aligned.quantile(0.3)),
    ]:
        assert value.shape == lower_rank.batch_shape
        np.testing.assert_allclose(value, expected, rtol=1e-6)


@pytest.mark.parametrize("draws", [jnp.array(1.0), jnp.zeros((0, 3))])
def test_posterior_predictive_requires_a_non_empty_draw_axis(draws: jax.Array) -> None:
    with pytest.raises(ValueError, match="draw axis"):
        PosteriorPredictive(draws=draws, aleatoric_variance=jnp.ones(3))


@pytest.mark.parametrize("probability", [0.0, 1.0, float("nan")])
def test_posterior_predictive_quantile_rejects_probabilities_outside_unit_interval(
    probability: float,
) -> None:
    mixture = PosteriorPredictive(
        draws=jnp.zeros((2, 3)), aleatoric_variance=jnp.ones(3)
    )

    with pytest.raises(ValueError, match="probability"):
        mixture.quantile(probability)


@st.composite
def distributions(draw: st.DrawFn) -> Any:
    """Draw any of the JAX backend's distributions, with array-valued fields."""
    batch = draw(st.integers(1, 4))

    def field(elements: st.SearchStrategy[float], shape: tuple[int, ...]) -> jax.Array:
        return jnp.asarray(draw(arrays(np.float32, shape, elements=elements)))

    kind = draw(
        st.sampled_from([Gaussian, Gamma, MomentMatchedPredictive, PosteriorPredictive])
    )
    if kind is Gaussian:
        return Gaussian(loc=field(_FINITE, (batch,)), scale=field(_POSITIVE, (batch,)))
    if kind is Gamma:
        return Gamma(
            concentration=field(_POSITIVE, (batch,)), rate=field(_POSITIVE, (batch,))
        )
    if kind is MomentMatchedPredictive:
        return MomentMatchedPredictive(
            loc=field(_FINITE, (batch,)),
            aleatoric_variance=field(_POSITIVE, (batch,)),
            epistemic_variance=field(_POSITIVE, (batch,)),
        )
    num_draws = draw(st.integers(1, 6))
    return PosteriorPredictive(
        draws=field(_FINITE, (num_draws, batch)),
        aleatoric_variance=field(_POSITIVE, (batch,)),
    )


def assert_same_distribution(actual: Any, expected: Any) -> None:
    """Assert both are the same distribution type with equal array fields."""
    assert type(actual) is type(expected)
    for item in dataclasses.fields(expected):
        np.testing.assert_array_equal(
            getattr(actual, item.name), getattr(expected, item.name)
        )


@given(distributions())
def test_distribution_round_trips_through_flatten_and_unflatten(
    distribution: Any,
) -> None:
    leaves, treedef = jax.tree.flatten(distribution)

    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert_same_distribution(jax.tree.unflatten(treedef, leaves), distribution)


@settings(deadline=None)
@given(distributions())
def test_distribution_passes_through_an_identity_jitted_function(
    distribution: Any,
) -> None:
    assert_same_distribution(jax.jit(lambda value: value)(distribution), distribution)


def test_batch_stays_a_single_pytree_leaf() -> None:
    batch = Batch(inputs=jnp.zeros((2, 1)), targets=jnp.ones((2,)))

    assert jax.tree.leaves(batch) == [batch]
