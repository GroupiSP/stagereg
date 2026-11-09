"""RUL curves with 95% predictive intervals for the CMAPSS example.

Turns one or more trained models, each predicting a Gaussian or a posterior
predictive, and the standardized trajectories
they were trained under into the example's headline figure: one row per
model, and in each row three held-out validation units side by side — the
shortest-, median-, and longest-lifetime one — each showing its true
linear RUL across the whole trajectory, the model's predicted mean, and
the 95% predictive interval around that mean — for a posterior predictive,
a total band and a narrower epistemic band. Spanning the lifetime range
that way shows whether the
model's accuracy and its stated uncertainty behave consistently across
units that fail early and units that survive several times as long.

The predicted curve begins at each unit's first full window, because no
prediction exists before `window_length` cycles of sensor history have
accumulated. The true-RUL line spans the whole trajectory regardless, so
the length of that warm-up stays visible rather than hidden.

This module knows nothing about training, about the archive, or about
which units are worth looking at: it takes already-standardized
trajectories, already-trained labelled models, and the three units to
draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# `preprocessing` is a sibling module, importable because whoever loads
# this module has already put the example's directory on `sys.path`:
# `run.py` when it is run as a script, and the tests when they load it via
# `importlib`. See issue #24 for making this directory a real package.
from preprocessing import LifetimeSpanningUnits, build_unit_rul_curve

_INTERVAL_Z_SCORE = 1.96
_TRUE_RUL_LABEL = "True linear RUL"
_PREDICTED_MEAN_LABEL = "Predicted mean"
_INTERVAL_LABEL = "95% predictive interval"
_TOTAL_INTERVAL_LABEL = "95% total interval"
_EPISTEMIC_INTERVAL_LABEL = "95% epistemic interval"
# Central 95% interval of a posterior predictive and of its draws' means.
_LOWER_PROBABILITY = 0.025
_UPPER_PROBABILITY = 0.975
# The role each column plays in the figure, in column order.
_ROLE_LABELS: tuple[str, ...] = (
    "shortest lifetime",
    "median lifetime",
    "longest lifetime",
)


class PredictiveDistribution(Protocol):
    """A location-scale predictive distribution over a batch of windows."""

    @property
    def loc(self) -> Any:
        """The predictive mean, shape `(n_windows, 1)`."""
        ...

    @property
    def scale(self) -> Any:
        """The predictive standard deviation, shape `(n_windows, 1)`."""
        ...


@runtime_checkable
class PosteriorPredictiveDistribution(Protocol):
    """A posterior predictive over a batch of windows: a mixture over draws.

    The library's `PosteriorPredictive` is one. Its draws are told apart
    from a location-scale Gaussian by having `draws` at all.
    """

    @property
    def draws(self) -> Any:
        """Each draw's mean, shape `(n_draws, n_windows, 1)`."""
        ...

    @property
    def draw_mask(self) -> Any | None:
        """Which draw slots the mixture includes, shape `(n_draws,)`.

        `None` includes every slot. Declared so the runtime `isinstance`
        check still matches the library's `PosteriorPredictive`, which always
        carries the field.
        """
        ...

    def mean(self) -> Any:
        """Return the mixture mean, shape `(n_windows, 1)`."""
        ...

    def quantile(self, probability: float) -> Any:
        """Return the mixture quantile at `probability`, shape `(n_windows, 1)`."""
        ...


class PredictiveModel(Protocol):
    """A model mapping windowed sensor readings to a predictive distribution."""

    def __call__(
        self, inputs: Any
    ) -> PredictiveDistribution | PosteriorPredictiveDistribution:
        """Predict one distribution per windowed example.

        Args:
            inputs: Windowed sensor readings, shape `(n_windows,
                window_length, n_features)`.

        Returns:
            The predictive distribution over those windows: a Gaussian, or a
            posterior predictive each of whose draws is one function applied
            to the whole batch.
        """
        ...


@dataclass(frozen=True)
class LabelledModel:
    """A trained predictive model with the label naming its row in the figure.

    Attributes:
        label: The name of the model, shown as its row's label, e.g.
            `"Staged model"` or `"Joint model"`.
        model: The trained model whose Gaussian or posterior-predictive
            output the row draws.
    """

    label: str
    model: PredictiveModel


def _per_window_array(values: Any) -> np.ndarray:
    """Return per-window values as a flat float array.

    Args:
        values: One value per window, in any array-like shape.

    Returns:
        The values as a 1-D float array, one entry per window.
    """
    return np.asarray(values, dtype=float).reshape(-1)


def _draw_gaussian_band(
    axes: Axes, window_cycles: Any, prediction: PredictiveDistribution
) -> None:
    """Draw a Gaussian's mean and its `loc ± 1.96·scale` band."""
    loc = _per_window_array(prediction.loc)
    half_width = _INTERVAL_Z_SCORE * _per_window_array(prediction.scale)
    axes.plot(window_cycles, loc, label=_PREDICTED_MEAN_LABEL, color="tab:blue")
    axes.fill_between(
        window_cycles,
        loc - half_width,
        loc + half_width,
        alpha=0.25,
        color="tab:blue",
        label=_INTERVAL_LABEL,
    )


def _draw_posterior_bands(
    axes: Axes, window_cycles: Any, prediction: PosteriorPredictiveDistribution
) -> None:
    """Draw a posterior predictive's mean, total band and epistemic band.

    The total band is the mixture's central 95% interval, from its
    quantiles; the epistemic band is the 2.5% and 97.5% quantiles of the
    draws' means at each window, ignoring the aleatoric variance. Draw
    slots a `draw_mask` excludes are left out of the epistemic band, as the
    mixture leaves them out of its quantiles.
    """
    draws = np.asarray(prediction.draws, dtype=float)
    if prediction.draw_mask is not None:
        draws = draws[np.asarray(prediction.draw_mask, dtype=bool)]
    draws = draws.reshape(draws.shape[0], -1)
    epistemic_lower, epistemic_upper = np.quantile(
        draws, (_LOWER_PROBABILITY, _UPPER_PROBABILITY), axis=0
    )
    axes.plot(
        window_cycles,
        _per_window_array(prediction.mean()),
        label=_PREDICTED_MEAN_LABEL,
        color="tab:blue",
    )
    axes.fill_between(
        window_cycles,
        _per_window_array(prediction.quantile(_LOWER_PROBABILITY)),
        _per_window_array(prediction.quantile(_UPPER_PROBABILITY)),
        alpha=0.2,
        color="tab:blue",
        label=_TOTAL_INTERVAL_LABEL,
    )
    axes.fill_between(
        window_cycles,
        epistemic_lower,
        epistemic_upper,
        alpha=0.4,
        color="tab:orange",
        label=_EPISTEMIC_INTERVAL_LABEL,
    )


def _draw_unit_rul_curve(
    axes: Axes,
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    model: PredictiveModel,
    *,
    unit_id: object,
    role: str,
    window_length: int,
) -> float:
    """Draw one unit's RUL curve and predictive band onto one column.

    Args:
        axes: The column to draw onto. Its limits are left to autoscale, so
            each column is scaled to its own unit and the band stays
            readable in the short-lived unit.
        trajectories: The standardized trajectories holding `unit_id`.
        feature_columns: Names of the columns to use as window channels, in
            the order the model was trained with.
        model: The trained predictive model.
        unit_id: Identifier of the unit to draw.
        role: The role this unit plays in the figure, named in the title.
        window_length: Number of cycles per window, as trained with.

    Returns:
        The unit's lifetime, in cycles.
    """
    curve = build_unit_rul_curve(
        trajectories,
        feature_columns,
        unit_id=unit_id,
        window_length=window_length,
    )
    # One evaluation over all of the unit's windows, so that each draw of a
    # posterior predictive is the same function along the whole curve.
    prediction = model(np.asarray(curve.windows, dtype=np.float32))

    axes.plot(
        curve.trajectory_cycles,
        curve.trajectory_linear_rul,
        label=_TRUE_RUL_LABEL,
        color="black",
    )
    if isinstance(prediction, PosteriorPredictiveDistribution):
        _draw_posterior_bands(axes, curve.window_cycles, prediction)
    else:
        _draw_gaussian_band(axes, curve.window_cycles, prediction)
    return curve.lifetime


def plot_validation_rul_curves(
    trajectories: pd.DataFrame,
    feature_columns: Sequence[str],
    models: Sequence[LabelledModel],
    *,
    units: LifetimeSpanningUnits,
    window_length: int = 30,
    save_path: Path | None = None,
) -> Figure:
    """Plot RUL curves for three units spanning a lifetime range, per model.

    The figure is an N×3 grid: one row per labelled model, in the order
    given, and one column per unit. Each column shares its x-axis across
    rows, so every model's curve for a unit sits on the same time axis,
    and each row's leftmost axis is labelled with its model's name. A
    single model yields a single row.

    The units to draw are chosen by the caller, so this module decides how
    a RUL curve looks and never which subjects it looks at. Passing the
    lifetime-spanning units of a held-out validation subset gives the
    example's headline figure: those units are never trained on, so the
    curves are evidence of generalization rather than of memorized fit,
    and a unit's lifetime is only observable at all for the
    run-to-failure train-split trajectories the validation subset is
    carved out of. A Gaussian row's band is `loc ± 1.96 * scale`, taken
    analytically from the model's Gaussian. A posterior-predictive row
    draws the mixture mean, a total band (the mixture's central 95%
    interval, from its quantiles) and a narrower epistemic band (the 2.5%
    and 97.5% quantiles of the draws' means at each window). Each curve
    comes from one evaluation over the unit's windows, so every draw is
    the same function along it. No band is clipped at zero, so that a band
    dipping below zero stays visible as evidence of the distributional
    assumption breaking down near end of life. The one figure legend lists
    the union of the elements drawn across rows.

    Args:
        trajectories: The standardized trajectories holding `units`,
            already scaled with the statistics fitted on the training
            subset only.
        feature_columns: Names of the columns to use as window channels, in
            the order the model was trained with.
        models: The trained models to draw, one row each, top to bottom.
        units: The three units to draw, one per column. Each must have at
            least `window_length` cycles in `trajectories`.
        window_length: Number of cycles per window, as trained with.
        save_path: Where to save the figure. When omitted, the figure is
            displayed instead.

    Returns:
        The figure holding the grid, rows in `models` order and columns in
        shortest / median / longest lifetime order.

    Raises:
        ValueError: If `models` is empty, or if any of `units` is absent
            from `trajectories` or has fewer than `window_length` cycles.
    """
    if not models:
        raise ValueError("At least one labelled model is required.")
    # x-axes shared per column only, so a unit is compared on the same time
    # axis across models while each column keeps its own range; y-axes are
    # left unshared, so each panel's band stays readable on its own scale.
    figure, grid = plt.subplots(
        len(models),
        len(_ROLE_LABELS),
        figsize=(15.0, 4.5 * len(models)),
        sharex="col",
        squeeze=False,
    )
    plotted_units = (units.shortest, units.median, units.longest)
    last_row = len(models) - 1
    for row, labelled in enumerate(models):
        for column, (unit_id, role) in enumerate(
            zip(plotted_units, _ROLE_LABELS, strict=True)
        ):
            axes = grid[row, column]
            lifetime = _draw_unit_rul_curve(
                axes,
                trajectories,
                feature_columns,
                labelled.model,
                unit_id=unit_id,
                role=role,
                window_length=window_length,
            )
            # Unit titles head the top row and time labels sit under the
            # bottom row; the rows in between share both with them.
            if row == 0:
                axes.set_title(f"Unit {unit_id} — {role} ({lifetime:g} cycles)")
            if row == last_row:
                axes.set_xlabel("Time cycles")
            y_label = "RUL (cycles)"
            axes.set_ylabel(f"{labelled.label}\n{y_label}" if column == 0 else y_label)

    # One legend for the whole figure, listing the union of the elements
    # drawn in first-drawn order: the panels of a row draw the same
    # elements, so a legend per panel would say nothing new, while a
    # posterior-predictive row adds its two bands to a Gaussian row's one.
    legend_entries: dict[str, Any] = {}
    for axes in grid.flat:
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            legend_entries.setdefault(label, handle)
    figure.legend(
        list(legend_entries.values()),
        list(legend_entries),
        loc="lower center",
        ncol=len(legend_entries),
    )
    figure.suptitle("CMAPSS FD001 held-out validation RUL curves")
    figure.tight_layout(rect=(0.0, 0.08 / len(models), 1.0, 1.0))

    if save_path is not None:
        figure.savefig(Path(save_path))
    else:
        plt.show()
    return figure
