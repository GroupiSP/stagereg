"""Tests for the clustered-data example's uncertainty grid and calibration figures.

None of these trains a model: they draw hand-built posterior predictives whose
bands and calibration curves are known by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jnp = pytest.importorskip("jax.numpy")
stagereg_jax = pytest.importorskip("stagereg.jax")
plt = pytest.importorskip("matplotlib.pyplot")

_MEAN = 0.5
_SPREAD = 0.1
_VARIANCE = 0.04


@dataclass(frozen=True)
class _Configuration:
    """A hand-built configuration: a training, a method and its predictive."""

    training: str
    method: str
    predictive: Any


def _spread_draws(
    scoring: ModuleType,
    num_points: int,
    *,
    mean: float = _MEAN,
    spread: float = _SPREAD,
    variance: float = _VARIANCE,
) -> Any:
    """Draws at `mean ± spread` everywhere, with aleatoric variance `variance`."""
    signs = jnp.asarray([1.0, -1.0, 1.0, -1.0])[:, None]
    return stagereg_jax.PosteriorPredictive(
        draws=jnp.full((4, num_points), mean) + spread * signs,
        aleatoric_variance=jnp.full((4, num_points), variance),
    )


@pytest.fixture
def staged_bbb_only(clustered_scoring: ModuleType, clustered_data: Any) -> list[Any]:
    """Only the staged BBB configuration, as before the other tickets land."""
    draws = _spread_draws(clustered_scoring, clustered_data.grid_inputs.shape[0])
    return [_Configuration("staged", "BBB", draws)]


@pytest.fixture
def closed_figures() -> Any:
    """Close every figure a test opens."""
    yield
    plt.close("all")


def _band_extent(axes: Any, label: str) -> tuple[float, float]:
    """The lowest and highest y of the filled band with the given label."""
    (band,) = [c for c in axes.collections if c.get_label() == label]
    vertices = np.concatenate([path.vertices for path in band.get_paths()])
    return float(vertices[:, 1].min()), float(vertices[:, 1].max())


def _is_empty(axes: Any) -> bool:
    return not axes.lines and not axes.collections and not axes.patches


def test_uncertainty_grid_draws_only_the_configurations_that_exist(
    clustered_figures: ModuleType,
    clustered_data: Any,
    staged_bbb_only: list[Any],
    closed_figures: None,
) -> None:
    figure = clustered_figures.plot_uncertainty_grid(staged_bbb_only, clustered_data)

    grid = np.asarray(figure.axes).reshape(2, 3)
    assert not _is_empty(grid[1, 0])
    for row, column in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]:
        assert _is_empty(grid[row, column])
    for axes in figure.axes:
        assert axes.get_xlim() == pytest.approx((-0.5, 1.5))
        assert axes.get_ylim() == pytest.approx((-1.5, 2.5))


@settings(deadline=None, max_examples=15)
@given(
    mean=st.floats(min_value=-1.0, max_value=2.0),
    spread=st.floats(min_value=0.0, max_value=0.5),
    variance=st.floats(min_value=0.0, max_value=0.5),
)
def test_uncertainty_grid_nests_the_epistemic_band_inside_the_total_band(
    clustered_figures: ModuleType,
    clustered_scoring: ModuleType,
    clustered_data: Any,
    mean: float,
    spread: float,
    variance: float,
) -> None:
    draws = _spread_draws(
        clustered_scoring,
        clustered_data.grid_inputs.shape[0],
        mean=mean,
        spread=spread,
        variance=variance,
    )
    figure = clustered_figures.plot_uncertainty_grid(
        [_Configuration("staged", "BBB", draws)], clustered_data
    )
    try:
        axes = np.asarray(figure.axes).reshape(2, 3)[1, 0]
        epistemic = _band_extent(axes, clustered_figures.EPISTEMIC_BAND_LABEL)
        total = _band_extent(axes, clustered_figures.TOTAL_BAND_LABEL)
    finally:
        plt.close(figure)

    # Four draws at mean ± spread have variance spread², around the mean.
    epistemic_half = 1.96 * spread
    total_half = 1.96 * np.sqrt(spread**2 + variance)
    tolerance = 1e-5
    assert epistemic == pytest.approx(
        (mean - epistemic_half, mean + epistemic_half), abs=tolerance
    )
    assert total == pytest.approx((mean - total_half, mean + total_half), abs=tolerance)
    assert total[0] <= epistemic[0] + tolerance
    assert epistemic[1] <= total[1] + tolerance


def test_uncertainty_grid_shades_the_gap_and_extrapolation_regions(
    clustered_figures: ModuleType,
    clustered_data: Any,
    staged_bbb_only: list[Any],
    closed_figures: None,
) -> None:
    figure = clustered_figures.plot_uncertainty_grid(staged_bbb_only, clustered_data)

    axes = np.asarray(figure.axes).reshape(2, 3)[1, 0]
    spans = sorted(
        (round(patch.get_x(), 6), round(patch.get_x() + patch.get_width(), 6))
        for patch in axes.patches
    )
    assert spans == [(-0.5, -0.2), (0.2, 0.6), (1.0, 1.5)]


def test_calibration_figure_draws_one_curve_per_available_method_and_a_diagonal(
    clustered_figures: ModuleType,
    clustered_data: Any,
    staged_bbb_only: list[Any],
    closed_figures: None,
) -> None:
    figure = clustered_figures.plot_extrapolation_calibration(
        staged_bbb_only, clustered_data
    )

    joint, staged = figure.axes
    assert _is_empty(joint)
    curves = {line.get_label(): line for line in staged.lines}
    (diagonal,) = [line for label, line in curves.items() if label != "BBB"]
    assert np.asarray(diagonal.get_xdata()) == pytest.approx([0.0, 1.0])
    assert np.asarray(diagonal.get_ydata()) == pytest.approx([0.0, 1.0])
    assert np.asarray(curves["BBB"].get_xdata()) == pytest.approx(np.arange(1, 20) / 20)


def test_calibration_curves_score_the_extrapolation_region_only(
    clustered_figures: ModuleType,
    clustered_scoring: ModuleType,
    clustered_regions: ModuleType,
    clustered_data: Any,
    closed_figures: None,
) -> None:
    # The draws hit the targets exactly in the extrapolation region and miss
    # them by far elsewhere, so every interval covers every scored target.
    targets = clustered_data.grid_targets
    extrapolation = clustered_regions.region_masks(clustered_data.grid_inputs)[
        clustered_regions.EXTRAPOLATION
    ]
    means = jnp.where(extrapolation, targets, targets + 100.0)
    draws = stagereg_jax.PosteriorPredictive(
        draws=jnp.broadcast_to(means, (4, means.size)),
        aleatoric_variance=jnp.full((4, means.size), 0.01),
    )

    figure = clustered_figures.plot_extrapolation_calibration(
        [_Configuration("joint", "BBB", draws), _Configuration("joint", "MCD", draws)],
        clustered_data,
    )

    joint, staged = figure.axes
    assert _is_empty(staged)
    curves = {line.get_label(): line for line in joint.lines}
    for method in ("BBB", "MCD"):
        assert np.asarray(curves[method].get_ydata()) == pytest.approx(1.0)


def test_figures_reject_a_configuration_without_a_panel(
    clustered_figures: ModuleType,
    clustered_scoring: ModuleType,
    clustered_data: Any,
) -> None:
    draws = _spread_draws(clustered_scoring, clustered_data.grid_inputs.shape[0])

    with pytest.raises(ValueError, match="No panel"):
        clustered_figures.plot_uncertainty_grid(
            [_Configuration("staged", "SWAG", draws)], clustered_data
        )


def test_draw_figures_saves_both_into_the_directory_without_showing(
    clustered_figures: ModuleType,
    clustered_data: Any,
    staged_bbb_only: list[Any],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(clustered_figures.plt, "show", lambda: shown.append(True))
    plot_path = tmp_path / "figures" / "nested"

    clustered_figures.draw_figures(staged_bbb_only, clustered_data, plot_path=plot_path)

    for name in ("uncertainty_grid.png", "extrapolation_calibration.png"):
        assert (plot_path / name).is_file()
        assert (plot_path / name).stat().st_size > 0
    assert shown == []
