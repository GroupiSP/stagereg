"""Tests for the clustered-data example's region masks and shared scoring.

None of these trains a model: they score hand-built per-draw predictions
whose answers are known by construction.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jnp = pytest.importorskip("jax.numpy")

_GRID_INPUTS = st.lists(
    st.floats(min_value=-0.5, max_value=1.5, width=32), min_size=1, max_size=50
)


@settings(deadline=None, max_examples=25)
@given(inputs=_GRID_INPUTS)
def test_region_masks_partition_any_grid(
    clustered_regions: ModuleType, inputs: list[float]
) -> None:
    masks = clustered_regions.region_masks(
        jnp.asarray(inputs, dtype=jnp.float32)[:, None]
    )

    stacked = np.stack([np.asarray(masks[name]) for name in clustered_regions.REGIONS])
    assert stacked.shape == (3, len(inputs))
    assert (stacked.sum(axis=0) == 1).all()


@pytest.mark.parametrize(
    ("point", "region"),
    [
        (-0.5, "extrapolation"),
        (-0.2, "in_distribution"),
        (0.2, "in_distribution"),
        (0.6, "in_distribution"),
        (1.0, "in_distribution"),
        (1.5, "extrapolation"),
        (0.4, "gap"),
    ],
)
def test_region_masks_place_the_boundaries_as_the_glossary_defines(
    clustered_regions: ModuleType, point: float, region: str
) -> None:
    inputs = jnp.asarray([[point]], dtype=jnp.float32)
    above = jnp.nextafter(inputs, jnp.inf)
    below = jnp.nextafter(inputs, -jnp.inf)

    assert bool(clustered_regions.region_masks(inputs)[region][0])
    if point in (0.2, 1.0):
        outside = "gap" if point == 0.2 else "extrapolation"
        assert bool(clustered_regions.region_masks(above)[outside][0])
    if point in (-0.2, 0.6):
        outside = "extrapolation" if point == -0.2 else "gap"
        assert bool(clustered_regions.region_masks(below)[outside][0])


def _mixture_samples(
    rng: np.random.Generator, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Draw one target per point from the equal-weight mixture of the draws."""
    num_draws, num_points = means.shape
    chosen = rng.integers(num_draws, size=num_points)
    columns = np.arange(num_points)
    return rng.normal(means[chosen, columns], np.sqrt(variances[chosen, columns]))


@settings(deadline=None, max_examples=10)
@given(
    level=st.floats(min_value=0.05, max_value=0.95),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_coverage_matches_the_level_when_targets_come_from_the_predictive(
    clustered_scoring: ModuleType, level: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(5, 20_000))
    variances = rng.uniform(0.1, 2.0, size=(5, 20_000))
    targets = _mixture_samples(rng, means, variances)
    draws = clustered_scoring.PredictiveDraws(
        means=jnp.asarray(means), variances=jnp.asarray(variances)
    )

    coverage = clustered_scoring.central_coverage(draws, jnp.asarray(targets), level)

    assert coverage == pytest.approx(level, abs=0.02)


def test_coverage_falls_far_below_the_level_for_a_far_too_narrow_predictive(
    clustered_scoring: ModuleType,
) -> None:
    rng = np.random.default_rng(0)
    targets = rng.normal(size=10_000)
    draws = clustered_scoring.PredictiveDraws(
        means=jnp.zeros((4, 10_000)), variances=jnp.full((4, 10_000), 0.01**2)
    )

    coverage = clustered_scoring.central_coverage(draws, jnp.asarray(targets), 0.95)

    assert coverage < 0.1


_GRID = jnp.linspace(-0.5, 1.5, 201)[:, None]
_NUM_DRAWS = 4


@pytest.fixture(scope="session")
def grid_masks(clustered_regions: ModuleType) -> object:
    """Region masks over a small evaluation grid."""
    return clustered_regions.region_masks(_GRID)


@settings(deadline=None, max_examples=15)
@given(
    std=st.floats(min_value=1e-3, max_value=10.0),
    offset=st.floats(min_value=-5.0, max_value=5.0),
)
def test_score_splits_constant_means_and_equal_variances(
    clustered_scoring: ModuleType,
    clustered_regions: ModuleType,
    grid_masks: object,
    std: float,
    offset: float,
) -> None:
    curve = jnp.sin(3.0 * _GRID[:, 0]) + offset
    draws = clustered_scoring.PredictiveDraws(
        means=jnp.broadcast_to(curve, (_NUM_DRAWS, curve.size)),
        variances=jnp.full((_NUM_DRAWS, curve.size), std**2),
    )

    scores = clustered_scoring.score(draws, curve, grid_masks)

    assert set(scores.regions) == set(clustered_regions.REGIONS)
    for region in scores.regions.values():
        assert region.epistemic_std == pytest.approx(0.0, abs=1e-6)
        assert region.aleatoric_std == pytest.approx(std, rel=1e-5)
    assert scores.rmse == pytest.approx(0.0, abs=1e-6)


@settings(deadline=None, max_examples=15)
@given(spread=st.floats(min_value=1e-3, max_value=10.0))
def test_score_recovers_a_known_spread_of_means_as_epistemic_std(
    clustered_scoring: ModuleType, grid_masks: object, spread: float
) -> None:
    signs = jnp.asarray([1.0, -1.0] * (_NUM_DRAWS // 2))[:, None]
    curve = jnp.cos(2.0 * _GRID[:, 0])
    draws = clustered_scoring.PredictiveDraws(
        means=curve + spread * signs,
        variances=jnp.ones((_NUM_DRAWS, curve.size)),
    )

    scores = clustered_scoring.score(draws, curve, grid_masks)

    for region in scores.regions.values():
        assert region.epistemic_std == pytest.approx(spread, rel=1e-4, abs=1e-6)
        assert region.aleatoric_std == pytest.approx(1.0, rel=1e-5)
        assert 0.0 <= region.coverage <= 1.0
    assert scores.rmse == pytest.approx(0.0, abs=1e-5)


def test_score_reports_rmse_on_the_in_distribution_region_only(
    clustered_scoring: ModuleType, clustered_regions: ModuleType, grid_masks: object
) -> None:
    in_distribution = grid_masks[clustered_regions.IN_DISTRIBUTION]
    targets = jnp.where(in_distribution, 2.0, 100.0)
    draws = clustered_scoring.PredictiveDraws(
        means=jnp.zeros((_NUM_DRAWS, targets.size)),
        variances=jnp.ones((_NUM_DRAWS, targets.size)),
    )

    scores = clustered_scoring.score(draws, targets, grid_masks)

    assert scores.rmse == pytest.approx(2.0)
