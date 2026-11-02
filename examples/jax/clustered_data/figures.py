"""The clustered-data example's uncertainty grid and calibration figures.

The uncertainty grid has one row per training (joint, staged) and one column
per inference method (BBB, MCD, pSGLD). Each panel draws the training points,
the noise-free truth and the predictive mean, an inner band of
`±1.96·√epistemic` and an outer band of `±1.96·√(aleatoric + epistemic)`,
with the gap region and the extrapolation region lightly shaded.

The calibration figure has one panel per training, each with one calibration
curve per method in the extrapolation region and the diagonal for reference.
Its coverages come from `scoring.calibration_curve`, the same scoring path as
the table's.

Both figures draw whichever configurations exist and leave the panels of
missing ones empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from clustered import GRID_BOUNDS, ClusteredData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from methods import METHODS, TRAININGS
from regions import CLUSTERS, EXTRAPOLATION, region_masks
from scoring import CALIBRATION_LEVELS, PredictiveDraws, calibration_curve

METHOD_COLUMNS: tuple[str, ...] = tuple(METHODS)
"""The inference methods, one per column of the uncertainty grid, in table order.

The trainings, `methods.TRAININGS`, give the grid's rows and the calibration
figure's panels.
"""

METHOD_COLORS: dict[str, str] = {
    method: f"C{index}" for index, method in enumerate(METHOD_COLUMNS)
}
"""Each method's fixed color in the calibration figure."""

Y_LIMITS = (-1.5, 2.5)
"""The uncertainty grid's target range, as in the UvA tutorial."""

Z_95 = 1.96
"""Half-width of the central 95% Gaussian interval, in standard deviations."""

EPISTEMIC_BAND_LABEL = "±1.96 σ_epi"
"""Legend label of the inner, epistemic band."""

TOTAL_BAND_LABEL = "±1.96 σ_total"
"""Legend label of the outer, total (aleatoric + epistemic) band."""

UNCERTAINTY_GRID_FILE = "uncertainty_grid.png"
"""File name of the uncertainty grid inside the `--plot-path` directory."""

CALIBRATION_FILE = "extrapolation_calibration.png"
"""File name of the calibration figure inside the `--plot-path` directory."""


class Configuration(Protocol):
    """One trained configuration, as the figures read it.

    Attributes:
        training: The training's name, one of `TRAININGS`.
        method: The inference method's name, one of `METHOD_COLUMNS`.
        draws: The per-draw predictive means and variances on the grid.
    """

    @property
    def training(self) -> str: ...

    @property
    def method(self) -> str: ...

    @property
    def draws(self) -> PredictiveDraws: ...


def _by_cell(
    configurations: Sequence[Configuration],
) -> dict[tuple[str, str], Configuration]:
    """Index the configurations by `(training, method)`.

    Raises:
        ValueError: If a configuration's training or method has no panel.
    """
    cells = {}
    for configuration in configurations:
        if (
            configuration.training not in TRAININGS
            or configuration.method not in METHOD_COLUMNS
        ):
            raise ValueError(
                f"No panel for {configuration.training} {configuration.method}; "
                f"trainings are {TRAININGS} and methods {METHOD_COLUMNS}."
            )
        cells[configuration.training, configuration.method] = configuration
    return cells


def _shade_regions(axes: Axes) -> None:
    """Lightly shade the extrapolation region and the gap region."""
    (first_low, first_high), (second_low, second_high) = CLUSTERS
    low, high = GRID_BOUNDS
    for start, end in ((low, first_low), (second_high, high)):
        axes.axvspan(start, end, color="0.5", alpha=0.12, linewidth=0)
    axes.axvspan(first_high, second_low, color="0.5", alpha=0.06, linewidth=0)


def _draw_panel(axes: Axes, configuration: Configuration, data: ClusteredData) -> None:
    """Draw one configuration's predictive on the grid, with the data and truth."""
    draws = configuration.draws
    inputs = np.asarray(data.grid_inputs)[:, 0]
    mean = np.asarray(jnp.mean(draws.means, axis=0))
    epistemic = np.asarray(jnp.var(draws.means, axis=0))
    aleatoric = np.asarray(jnp.mean(draws.variances, axis=0))
    epistemic_half = Z_95 * np.sqrt(epistemic)
    total_half = Z_95 * np.sqrt(aleatoric + epistemic)

    _shade_regions(axes)
    axes.fill_between(
        inputs,
        mean - total_half,
        mean + total_half,
        color="C0",
        alpha=0.2,
        linewidth=0,
        label=TOTAL_BAND_LABEL,
    )
    axes.fill_between(
        inputs,
        mean - epistemic_half,
        mean + epistemic_half,
        color="C0",
        alpha=0.4,
        linewidth=0,
        label=EPISTEMIC_BAND_LABEL,
    )
    axes.plot(
        np.asarray(data.train_inputs)[:, 0],
        np.asarray(data.train_targets)[:, 0],
        linestyle="none",
        marker=".",
        markersize=2,
        color="0.3",
        label="Training data",
    )
    axes.plot(
        inputs, np.asarray(data.grid_truth), color="k", linewidth=1, label="Truth"
    )
    axes.plot(inputs, mean, color="C3", linewidth=1.5, label="Predictive mean")


def plot_uncertainty_grid(
    configurations: Sequence[Configuration], data: ClusteredData
) -> Figure:
    """Draw the 2×3 grid of predictive bands, one panel per configuration.

    Args:
        configurations: The configurations to draw, e.g. one seed's; a
            configuration missing from them leaves its panel empty.
        data: The training set and the evaluation grid.

    Returns:
        The figure, neither shown nor saved.

    Raises:
        ValueError: If a configuration's training or method has no panel.
    """
    cells = _by_cell(configurations)
    figure, grid = plt.subplots(
        len(TRAININGS),
        len(METHOD_COLUMNS),
        figsize=(13.0, 7.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, training in enumerate(TRAININGS):
        for column, method in enumerate(METHOD_COLUMNS):
            axes = grid[row, column]
            configuration = cells.get((training, method))
            if configuration is not None:
                _draw_panel(axes, configuration, data)
            axes.set_xlim(*GRID_BOUNDS)
            axes.set_ylim(*Y_LIMITS)
            axes.set_title(f"{training} {method}")
            if row == len(TRAININGS) - 1:
                axes.set_xlabel("x")
            if column == 0:
                axes.set_ylabel("y")

    drawn = [axes for axes in grid.flat if axes.lines]
    if drawn:
        handles, labels = drawn[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(labels))
    figure.suptitle("Clustered data: predictive mean with epistemic and total bands")
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return figure


def plot_extrapolation_calibration(
    configurations: Sequence[Configuration], data: ClusteredData
) -> Figure:
    """Draw the 1×2 calibration curves in the extrapolation region.

    Args:
        configurations: The configurations to draw, e.g. one seed's; a
            training with none of them leaves its panel empty.
        data: The training set and the evaluation grid; the curves score
            the grid's noisy targets.

    Returns:
        The figure, neither shown nor saved.

    Raises:
        ValueError: If a configuration's training or method has no panel.
    """
    cells = _by_cell(configurations)
    extrapolation = region_masks(data.grid_inputs)[EXTRAPOLATION]
    levels = np.asarray(CALIBRATION_LEVELS)
    figure, panels = plt.subplots(
        1, len(TRAININGS), figsize=(10.0, 4.8), sharey=True, squeeze=False
    )
    for axes, training in zip(panels[0], TRAININGS, strict=True):
        methods = [method for method in METHOD_COLUMNS if (training, method) in cells]
        if methods:
            axes.plot([0.0, 1.0], [0.0, 1.0], color="0.5", linestyle="--", linewidth=1)
        for method in methods:
            coverage = calibration_curve(
                cells[training, method].draws,
                data.grid_targets,
                CALIBRATION_LEVELS,
                where=extrapolation,
            )
            axes.plot(
                levels,
                coverage,
                marker="o",
                markersize=4,
                linewidth=2,
                color=METHOD_COLORS[method],
                label=method,
            )
        if methods:
            axes.legend(loc="upper left")
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.set_aspect("equal")
        axes.set_title(training)
        axes.set_xlabel("Nominal level")
    panels[0, 0].set_ylabel("Observed coverage")
    figure.suptitle("Calibration curves in the extrapolation region")
    figure.tight_layout()
    return figure


def draw_figures(
    configurations: Sequence[Configuration],
    data: ClusteredData,
    *,
    plot_path: Path | None = None,
) -> tuple[Figure, Figure]:
    """Draw both figures, then save them or show them on screen.

    Args:
        configurations: The configurations to draw, e.g. the first seed's.
        data: The training set and the evaluation grid.
        plot_path: A directory to save `UNCERTAINTY_GRID_FILE` and
            `CALIBRATION_FILE` into, created if needed; nothing is shown
            then. `None` shows both figures instead.

    Returns:
        The uncertainty grid and the calibration figure.
    """
    figures = (
        plot_uncertainty_grid(configurations, data),
        plot_extrapolation_calibration(configurations, data),
    )
    if plot_path is None:
        plt.show()
        return figures
    plot_path.mkdir(parents=True, exist_ok=True)
    for figure, name in zip(
        figures, (UNCERTAINTY_GRID_FILE, CALIBRATION_FILE), strict=True
    ):
        figure.savefig(plot_path / name)
        plt.close(figure)
    return figures
