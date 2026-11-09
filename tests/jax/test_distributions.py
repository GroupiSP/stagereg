from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
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
    ResidualScale,
)
from stagereg.core import metrics as core_metrics
from stagereg.core import Batch, ClosedFormCRPS, expected_gaussian_mixture_crps

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


def test_gamma_head_holds_a_unit_residual_scale_by_default() -> None:
    head = GammaHead(1, 1, rngs=nnx.Rngs(0))

    scales = jax.tree.leaves(nnx.state(head, ResidualScale))

    assert [float(scale) for scale in scales] == [1.0]


def test_gamma_head_without_residual_scaling_holds_only_its_linear_layer() -> None:
    # The state of a 0.5.0 `GammaHead`, so its checkpoints restore.
    head = GammaHead(1, 1, rngs=nnx.Rngs(0), residual_scaling=False)

    assert set(nnx.state(head)) == {"linear"}


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
    """The mixture CDF at ``y``, in float64, of ``[S, *batch]`` means and a variance
    shared as ``[*batch]`` or one per draw as ``[S, *batch]``."""
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


def test_posterior_predictive_quantile_compiles_once_per_shape(
    compile_counter: Any,
) -> None:
    rng = np.random.default_rng(0)

    def mixture() -> PosteriorPredictive:
        return PosteriorPredictive(
            draws=jnp.asarray(rng.normal(size=(5, 4)), jnp.float32),
            aleatoric_variance=jnp.asarray(rng.uniform(0.5, 2.0, 4), jnp.float32),
        )

    first, *others = [mixture() for _ in range(4)]
    first.quantile(0.1).block_until_ready()
    with compile_counter() as repeated:
        for mixture_, probability in zip(others, (0.1, 0.5, 0.9), strict=True):
            mixture_.quantile(probability).block_until_ready()

    assert repeated.total == 0


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
def padded_posterior_predictives(
    draw: st.DrawFn,
) -> tuple[PosteriorPredictive, PosteriorPredictive]:
    """Draw a mixture and the same mixture padded with excluded draw slots."""
    means, variance = draw(posterior_predictives())
    padding = draw(
        arrays(np.float32, (draw(st.integers(1, 4)), means.shape[1]), elements=_FINITE)
    )
    padded = np.concatenate([means, padding])
    order = np.asarray(draw(st.permutations(range(len(padded)))))
    mask = order < len(means)
    exact = PosteriorPredictive(
        draws=jnp.asarray(padded[order][mask]), aleatoric_variance=jnp.asarray(variance)
    )
    masked = PosteriorPredictive(
        draws=jnp.asarray(padded[order]),
        aleatoric_variance=jnp.asarray(variance),
        draw_mask=jnp.asarray(mask),
    )
    return masked, exact


@settings(deadline=None, max_examples=25)
@given(mixtures=padded_posterior_predictives(), targets=_FINITE)
def test_a_draw_mask_excludes_its_slots_from_the_mixture(
    mixtures: tuple[PosteriorPredictive, PosteriorPredictive], targets: float
) -> None:
    masked, exact = mixtures

    for name in ("mean", "variance"):
        np.testing.assert_allclose(
            getattr(masked, name)(), getattr(exact, name)(), rtol=1e-5, atol=1e-5
        )
    for name in ("log_prob", "cdf"):
        np.testing.assert_allclose(
            getattr(masked, name)(jnp.float32(targets)),
            getattr(exact, name)(jnp.float32(targets)),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        masked.quantile(0.3), exact.quantile(0.3), rtol=1e-4, atol=1e-4
    )


@settings(deadline=None, max_examples=15)
@given(components=posterior_predictives(), seed=st.integers(0, 2**31 - 1))
def test_a_mixture_padded_after_its_draws_samples_like_the_unpadded_one(
    components: tuple[np.ndarray, np.ndarray], seed: int
) -> None:
    means, variance = components
    padded = np.concatenate([means, np.full_like(means, 1e6)])
    mask = np.arange(len(padded)) < len(means)
    exact = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    masked = PosteriorPredictive(
        draws=jnp.asarray(padded),
        aleatoric_variance=jnp.asarray(variance),
        draw_mask=jnp.asarray(mask),
    )
    key = jax.random.key(seed)

    np.testing.assert_array_equal(masked.sample(key, (16,)), exact.sample(key, (16,)))


def test_a_masked_mixture_passes_through_a_jitted_function_unchanged() -> None:
    masked = PosteriorPredictive(
        draws=jnp.arange(6.0).reshape(3, 2),
        aleatoric_variance=jnp.ones(2),
        draw_mask=jnp.array([True, False, True]),
    )

    passed = jax.jit(lambda mixture: mixture)(masked)

    np.testing.assert_array_equal(passed.draw_mask, masked.draw_mask)
    np.testing.assert_allclose(jax.jit(PosteriorPredictive.mean)(masked), [2.0, 3.0])


def test_a_draw_mask_must_have_one_entry_per_draw() -> None:
    with pytest.raises(ValueError, match="draw_mask"):
        PosteriorPredictive(
            draws=jnp.zeros((3, 2)),
            aleatoric_variance=jnp.ones(2),
            draw_mask=jnp.array([True, False]),
        )


def test_a_draw_mask_must_include_a_draw() -> None:
    with pytest.raises(ValueError, match="at least one draw"):
        PosteriorPredictive(
            draws=jnp.zeros((3, 2)),
            aleatoric_variance=jnp.ones(2),
            draw_mask=jnp.zeros(3, dtype=bool),
        )


def test_a_traced_draw_mask_is_accepted_without_reading_its_values() -> None:
    @jax.jit
    def masked_mean(draw_mask: jax.Array) -> jax.Array:
        return PosteriorPredictive(
            draws=jnp.array([[1.0], [3.0]]),
            aleatoric_variance=jnp.ones(1),
            draw_mask=draw_mask,
        ).mean()

    np.testing.assert_allclose(masked_mean(jnp.array([True, False])), [1.0])


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


@st.composite
def gaussians_and_references(
    draw: st.DrawFn,
) -> tuple[Gaussian, np.ndarray]:
    """Draw a Gaussian of batch rank 1 or 2 and ``[R, *batch]`` references."""
    batch_shape = draw(st.lists(st.integers(1, 3), min_size=1, max_size=2).map(tuple))
    reference_count = draw(st.integers(1, 4))
    loc = draw(arrays(np.float32, batch_shape, elements=_FINITE))
    scale = draw(arrays(np.float32, batch_shape, elements=_POSITIVE))
    references = draw(
        arrays(np.float32, (reference_count, *batch_shape), elements=_FINITE)
    )
    return Gaussian(loc=jnp.asarray(loc), scale=jnp.asarray(scale)), references


def _core_crps(draw_means: Any, variance: Any, references: np.ndarray) -> np.ndarray:
    """Score references with the core function as the oracle, in float64.

    Takes the JAX layout, ``[M, *batch]`` draw means, ``[*batch]`` variances and
    ``[R, *batch]`` references, and returns ``[*batch]`` scores.
    """
    draw_means = np.asarray(draw_means, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    scores = expected_gaussian_mixture_crps(
        draw_means.reshape(draw_means.shape[0], -1).T,
        variance.reshape(-1),
        references.reshape(references.shape[0], -1).T,
    )
    return scores.reshape(variance.shape)


def _core_gaussian_crps(gaussian: Gaussian, references: np.ndarray) -> np.ndarray:
    """Score ``[R, *batch]`` references against ``gaussian`` with the core function."""
    scale = np.asarray(gaussian.scale, dtype=np.float64)
    return _core_crps(np.asarray(gaussian.loc)[None], np.square(scale), references)


@settings(deadline=None)
@given(gaussians_and_references())
def test_gaussian_crps_of_one_target_per_unit_matches_the_core_function(
    case: tuple[Gaussian, np.ndarray],
) -> None:
    gaussian, references = case
    targets = references[0]

    np.testing.assert_allclose(
        gaussian.crps(jnp.asarray(targets)),
        _core_gaussian_crps(gaussian, references[:1]),
        rtol=1e-4,
        atol=1e-4,
    )


@settings(deadline=None)
@given(gaussians_and_references())
def test_gaussian_crps_of_reference_samples_matches_the_core_function(
    case: tuple[Gaussian, np.ndarray],
) -> None:
    gaussian, references = case

    np.testing.assert_allclose(
        gaussian.crps(jnp.asarray(references)),
        _core_gaussian_crps(gaussian, references),
        rtol=1e-4,
        atol=1e-4,
    )


@settings(deadline=None)
@given(gaussians_and_references())
def test_gaussian_crps_is_the_textbook_gaussian_crps(
    case: tuple[Gaussian, np.ndarray],
) -> None:
    # The core function shares the JAX formula, so it cannot catch an error in
    # it; Gneiting and Raftery's (2007) Gaussian CRPS is independent of it.
    gaussian, references = case
    loc = np.asarray(gaussian.loc, dtype=np.float64)
    scale = np.asarray(gaussian.scale, dtype=np.float64)
    standard = NormalDist()
    z = (references.astype(np.float64) - loc) / scale
    cdf = np.vectorize(standard.cdf)(z)
    pdf = np.vectorize(standard.pdf)(z)
    expected = scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))

    np.testing.assert_allclose(
        gaussian.crps(jnp.asarray(references)),
        expected.mean(axis=0),
        rtol=1e-4,
        atol=1e-4,
    )


@settings(deadline=None)
@given(gaussians_and_references())
def test_gaussian_crps_is_the_same_under_jit(
    case: tuple[Gaussian, np.ndarray],
) -> None:
    gaussian, references = case
    targets = jnp.asarray(references)

    np.testing.assert_allclose(
        jax.jit(Gaussian.crps)(gaussian, targets),
        gaussian.crps(targets),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gaussian_crps_rejects_targets_of_another_rank() -> None:
    gaussian = Gaussian(loc=jnp.zeros(3), scale=jnp.ones(3))

    with pytest.raises(ValueError, match="targets"):
        gaussian.crps(jnp.zeros((2, 2, 3)))


def test_only_distributions_with_a_closed_form_offer_the_closed_form_crps() -> None:
    gaussian = Gaussian(loc=jnp.zeros(2), scale=jnp.ones(2))
    gamma = Gamma(concentration=jnp.ones(2), rate=jnp.ones(2))

    assert isinstance(gaussian, ClosedFormCRPS)
    assert not isinstance(gamma, ClosedFormCRPS)


@st.composite
def posterior_predictives_and_references(
    draw: st.DrawFn,
) -> tuple[PosteriorPredictive, np.ndarray]:
    """Draw a mixture over a ``[B]`` batch and ``[R, B]`` reference samples."""
    means, variance = draw(posterior_predictives())
    reference_count = draw(st.integers(1, 4))
    references = draw(
        arrays(np.float32, (reference_count, means.shape[1]), elements=_FINITE)
    )
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    return mixture, references


@settings(deadline=None, max_examples=25)
@given(posterior_predictives_and_references(), st.booleans())
def test_posterior_predictive_crps_matches_the_core_function(
    case: tuple[PosteriorPredictive, np.ndarray], one_target_per_unit: bool
) -> None:
    mixture, references = case
    if one_target_per_unit:
        references = references[:1]
    targets = references[0] if one_target_per_unit else references

    np.testing.assert_allclose(
        mixture.crps(jnp.asarray(targets)),
        _core_crps(mixture.draws, mixture.aleatoric_variance, references),
        rtol=1e-4,
        atol=1e-4,
    )


@settings(deadline=None, max_examples=25)
@given(mixtures=padded_posterior_predictives(), targets=_FINITE)
def test_a_draw_mask_excludes_its_slots_from_the_crps(
    mixtures: tuple[PosteriorPredictive, PosteriorPredictive], targets: float
) -> None:
    masked, exact = mixtures

    np.testing.assert_allclose(
        masked.crps(jnp.full(masked.batch_shape, targets)),
        exact.crps(jnp.full(exact.batch_shape, targets)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("extreme", [3e38, -3e38, float("inf"), float("nan")])
def test_extreme_means_in_excluded_slots_leave_the_crps_finite(extreme: float) -> None:
    included = jnp.array([[0.5, -1.0], [2.0, 0.0]])
    masked = PosteriorPredictive(
        draws=jnp.concatenate([included, jnp.full((1, 2), extreme)]),
        aleatoric_variance=jnp.array([0.5, 2.0]),
        draw_mask=jnp.array([True, True, False]),
    )
    exact = PosteriorPredictive(
        draws=included, aleatoric_variance=jnp.array([0.5, 2.0])
    )
    references = jnp.array([[0.0, 1.0], [3.0, -2.0]])

    np.testing.assert_allclose(
        masked.crps(references), exact.crps(references), rtol=1e-5, atol=1e-6
    )


@settings(deadline=None, max_examples=25)
@given(posterior_predictives_and_references(), st.booleans())
def test_moment_matched_crps_matches_the_core_function(
    case: tuple[PosteriorPredictive, np.ndarray], one_target_per_unit: bool
) -> None:
    mixture, references = case
    if one_target_per_unit:
        references = references[:1]
    targets = references[0] if one_target_per_unit else references
    summary = mixture.moment_matched()

    np.testing.assert_allclose(
        summary.crps(jnp.asarray(targets)),
        _core_crps(summary.loc[None], summary.variance(), references),
        rtol=1e-4,
        atol=1e-4,
    )


@settings(deadline=None, max_examples=15)
@given(mixtures=padded_posterior_predictives(), references=_FINITE)
def test_posterior_crps_is_the_same_under_jit(
    mixtures: tuple[PosteriorPredictive, PosteriorPredictive], references: float
) -> None:
    masked, _ = mixtures
    targets = jnp.full((2, *masked.batch_shape), references)
    summary = masked.moment_matched()

    np.testing.assert_allclose(
        jax.jit(PosteriorPredictive.crps)(masked, targets),
        masked.crps(targets),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        jax.jit(MomentMatchedPredictive.crps)(summary, targets),
        summary.crps(targets),
        rtol=1e-5,
        atol=1e-5,
    )


def test_posterior_crps_is_the_same_in_blocks_of_draws_under_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_means, key_references = jax.random.split(jax.random.key(0))
    masked = PosteriorPredictive(
        draws=jax.random.normal(key_means, (13, 2)),
        aleatoric_variance=jnp.array([0.5, 2.0]),
        draw_mask=jnp.arange(13) < 10,
    )
    references = jax.random.normal(key_references, (3, 2))
    whole = masked.crps(references)

    # Blocks of two draws: seven blocks, the last one padded.
    monkeypatch.setattr(core_metrics, "_CRPS_BLOCK_ELEMENTS", 2 * 13 * 2)

    np.testing.assert_allclose(
        jax.jit(PosteriorPredictive.crps)(masked, references),
        whole,
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_posterior_predictive_and_its_summary_offer_the_closed_form_crps() -> None:
    mixture = PosteriorPredictive(
        draws=jnp.zeros((2, 3)), aleatoric_variance=jnp.ones(3)
    )

    assert isinstance(mixture, ClosedFormCRPS)
    assert isinstance(mixture.moment_matched(), ClosedFormCRPS)


@st.composite
def per_draw_posterior_predictives(
    draw: st.DrawFn, *, num_draws: st.SearchStrategy[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Draw component means ``[S, B]`` and per-draw aleatoric variances ``[S, B]``."""
    count = draw(num_draws if num_draws is not None else st.integers(1, 6))
    batch = draw(st.integers(1, 4))
    means = draw(arrays(np.float32, (count, batch), elements=_FINITE))
    variances = draw(arrays(np.float32, (count, batch), elements=_POSITIVE))
    return means, variances


def _assert_same_predictive(
    actual: PosteriorPredictive,
    expected: Any,
    targets: jax.Array,
    key: jax.Array,
    *,
    tolerance: float = 1e-5,
) -> None:
    """Assert two distributions agree on every method the posterior offers."""
    assert actual.batch_shape == expected.batch_shape
    close = {"rtol": tolerance, "atol": tolerance}
    for name in ("mean", "variance"):
        np.testing.assert_allclose(
            getattr(actual, name)(), getattr(expected, name)(), **close
        )
    for name in ("log_prob", "crps"):
        np.testing.assert_allclose(
            getattr(actual, name)(targets), getattr(expected, name)(targets), **close
        )
    np.testing.assert_allclose(
        actual.sample(key, (4,)), expected.sample(key, (4,)), **close
    )
    if isinstance(expected, PosteriorPredictive):
        np.testing.assert_allclose(actual.cdf(targets), expected.cdf(targets), **close)
        np.testing.assert_allclose(
            actual.quantile(0.3), expected.quantile(0.3), rtol=1e-4, atol=1e-4
        )
        actual_summary = actual.moment_matched()
        expected_summary = expected.moment_matched()
        for name in ("aleatoric_variance", "epistemic_variance"):
            np.testing.assert_allclose(
                getattr(actual_summary, name), getattr(expected_summary, name), **close
            )


@settings(deadline=None, max_examples=25)
@given(
    components=posterior_predictives(),
    targets=_FINITE,
    seed=st.integers(0, 2**31 - 1),
)
def test_a_per_draw_variance_equal_across_draws_is_the_shared_variance(
    components: tuple[np.ndarray, np.ndarray], targets: float, seed: int
) -> None:
    means, variance = components
    shared = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variance)
    )
    per_draw = PosteriorPredictive(
        draws=jnp.asarray(means),
        aleatoric_variance=jnp.asarray(np.broadcast_to(variance, means.shape)),
    )

    _assert_same_predictive(
        per_draw,
        shared,
        jnp.full(shared.batch_shape, targets, jnp.float32),
        jax.random.key(seed),
    )


@settings(deadline=None, max_examples=25)
@given(
    components=per_draw_posterior_predictives(num_draws=st.just(1)),
    targets=_FINITE,
    seed=st.integers(0, 2**31 - 1),
)
def test_one_draw_with_its_own_variance_is_the_gaussian(
    components: tuple[np.ndarray, np.ndarray], targets: float, seed: int
) -> None:
    means, variances = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variances)
    )
    gaussian = Gaussian(loc=jnp.asarray(means[0]), scale=jnp.sqrt(variances[0]))

    _assert_same_predictive(
        mixture,
        gaussian,
        jnp.full(gaussian.batch_shape, targets, jnp.float32),
        jax.random.key(seed),
    )


@pytest.mark.parametrize(
    ("draws_shape", "variance_shape"),
    [((3, 2), (4,)), ((3, 2), (3, 3)), ((3, 2), (2, 2, 3)), ((3, 1, 2), (3, 4))],
)
def test_a_variance_neither_shared_nor_per_draw_is_rejected(
    draws_shape: tuple[int, ...], variance_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="aleatoric_variance"):
        PosteriorPredictive(
            draws=jnp.zeros(draws_shape), aleatoric_variance=jnp.ones(variance_shape)
        )


def test_a_variance_with_the_draw_axis_is_read_per_draw() -> None:
    # Both readings broadcast here; the leading axis of size S makes it per draw.
    mixture = PosteriorPredictive(
        draws=jnp.zeros((2, 2)), aleatoric_variance=jnp.array([[1.0, 1.0], [3.0, 3.0]])
    )

    assert mixture.batch_shape == (2,)
    np.testing.assert_allclose(mixture.moment_matched().aleatoric_variance, [2.0, 2.0])


def test_a_variance_with_a_leading_axis_of_one_is_shared_by_several_draws() -> None:
    # With S > 1 a leading axis of size 1 is not the draw axis: the variance is
    # shared, broadcast against one draw's mean like any other.
    shared = PosteriorPredictive(
        draws=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
        aleatoric_variance=jnp.array([[1.0, 4.0]]),
    )

    assert shared.batch_shape == (1, 2)
    np.testing.assert_allclose(shared.moment_matched().aleatoric_variance, [[1.0, 4.0]])


_GARBAGE_VARIANCE = st.sampled_from([0.0, -1.0, float("inf"), float("nan")])
_GARBAGE_MEAN = st.sampled_from([3e38, -3e38, float("inf"), float("nan"), 0.0])


@st.composite
def padded_per_draw_posterior_predictives(
    draw: st.DrawFn,
) -> tuple[PosteriorPredictive, PosteriorPredictive]:
    """Draw a per-draw-variance mixture and the same one padded with garbage slots."""
    means, variances = draw(per_draw_posterior_predictives())
    padding_shape = (draw(st.integers(1, 3)), means.shape[1])
    padded_means = np.concatenate(
        [means, draw(arrays(np.float32, padding_shape, elements=_GARBAGE_MEAN))]
    )
    padded_variances = np.concatenate(
        [variances, draw(arrays(np.float32, padding_shape, elements=_GARBAGE_VARIANCE))]
    )
    order = np.asarray(draw(st.permutations(range(len(padded_means)))))
    mask = order < len(means)
    exact = PosteriorPredictive(
        draws=jnp.asarray(padded_means[order][mask]),
        aleatoric_variance=jnp.asarray(padded_variances[order][mask]),
    )
    masked = PosteriorPredictive(
        draws=jnp.asarray(padded_means[order]),
        aleatoric_variance=jnp.asarray(padded_variances[order]),
        draw_mask=jnp.asarray(mask),
    )
    return masked, exact


@settings(deadline=None, max_examples=25)
@given(
    mixtures=padded_per_draw_posterior_predictives(),
    targets=_FINITE,
    seed=st.integers(0, 2**31 - 1),
)
def test_excluded_slots_with_invalid_variances_change_nothing(
    mixtures: tuple[PosteriorPredictive, PosteriorPredictive], targets: float, seed: int
) -> None:
    masked, exact = mixtures

    _assert_same_predictive(
        masked,
        exact,
        jnp.full(exact.batch_shape, targets, jnp.float32),
        jax.random.key(seed),
        tolerance=1e-4,
    )


@settings(deadline=None, max_examples=25)
@given(
    components=per_draw_posterior_predictives(),
    probability=st.floats(min_value=0.01, max_value=0.99),
)
def test_a_per_draw_variance_quantile_inverts_its_cdf(
    components: tuple[np.ndarray, np.ndarray], probability: float
) -> None:
    means, variances = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variances)
    )

    quantile = mixture.quantile(probability)

    np.testing.assert_allclose(mixture.cdf(quantile), probability, atol=1e-4)
    np.testing.assert_allclose(
        _mixture_cdf(np.asarray(quantile, np.float64), means, variances),
        probability,
        atol=1e-4,
    )


@settings(deadline=None, max_examples=15)
@given(components=per_draw_posterior_predictives(), targets=_FINITE)
def test_a_per_draw_variance_crps_matches_numerical_integration(
    components: tuple[np.ndarray, np.ndarray],
    targets: float,
    mixture_crps_oracle: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
) -> None:
    means, variances = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variances)
    )

    expected = mixture_crps_oracle(means, variances, targets)

    np.testing.assert_allclose(
        mixture.crps(jnp.full(means.shape[1:], targets, jnp.float32)),
        expected,
        rtol=1e-3,
        atol=1e-3,
    )


@settings(deadline=None, max_examples=15)
@given(components=per_draw_posterior_predictives(), seed=st.integers(0, 2**31 - 1))
def test_the_moment_matched_split_matches_monte_carlo_moments_of_the_samples(
    components: tuple[np.ndarray, np.ndarray], seed: int
) -> None:
    means, variances = components
    mixture = PosteriorPredictive(
        draws=jnp.asarray(means), aleatoric_variance=jnp.asarray(variances)
    )
    summary = mixture.moment_matched()
    count = 20_000

    samples = np.asarray(mixture.sample(jax.random.key(seed), (count,)), np.float64)

    np.testing.assert_allclose(
        summary.aleatoric_variance, variances.astype(np.float64).mean(axis=0), rtol=1e-5
    )
    np.testing.assert_allclose(
        summary.epistemic_variance, means.astype(np.float64).var(axis=0), atol=1e-3
    )
    np.testing.assert_allclose(mixture.variance(), summary.variance(), rtol=1e-6)
    var = np.asarray(summary.variance(), np.float64)
    centred = samples - samples.mean(axis=0)
    fourth = (centred**4).mean(axis=0)
    assert np.all(
        np.abs(samples.mean(axis=0) - np.asarray(summary.mean()))
        <= 6.0 * np.sqrt(var / count)
    )
    assert np.all(
        np.abs(samples.var(axis=0) - var) <= 6.0 * np.sqrt(fourth / count) + 1e-3
    )


def test_a_per_draw_variance_mixture_passes_through_jit_as_a_pytree() -> None:
    mixture = PosteriorPredictive(
        draws=jnp.arange(6.0).reshape(3, 2),
        aleatoric_variance=jnp.array([[1.0, 2.0], [jnp.nan, 0.0], [3.0, 4.0]]),
        draw_mask=jnp.array([True, False, True]),
    )
    targets = jnp.array([0.5, 2.0])

    passed = jax.jit(lambda distribution: distribution)(mixture)

    assert_same_distribution(passed, mixture)
    for method in (PosteriorPredictive.log_prob, PosteriorPredictive.crps):
        np.testing.assert_allclose(
            jax.jit(method)(mixture, targets), method(mixture, targets), rtol=1e-6
        )
    np.testing.assert_allclose(
        jax.jit(PosteriorPredictive.variance)(mixture), mixture.variance(), rtol=1e-6
    )
