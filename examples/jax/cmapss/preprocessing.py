"""Pure preprocessing utilities for CMAPSS-style trajectory DataFrames.

Takes care of standardizing the sensor features and preparing fixed-length
time windows. Considers left-padding and handling units with fewer rows than
the window length.

This module has no I/O dependencies and does not import from
``examples/jax/cmapss/data.py``. It operates generically on any DataFrame
with a ``unit_id`` column, a ``time_cycles`` column, and a set of
sensor/feature columns supplied by the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorStandardization:
    """Per-sensor standardization statistics fitted on a training subset.

    Attributes:
        feature_columns: Names of the columns these statistics apply to, in
            the order matching `mean` and `scale`.
        mean: Per-column mean, shape `(len(feature_columns),)`.
        scale: Per-column standard deviation used as the scaling divisor,
            shape `(len(feature_columns),)`. Zero-variance columns are
            guarded to a scale of 1.0 to avoid division by zero.
    """

    feature_columns: list[str]
    mean: np.ndarray
    scale: np.ndarray


def split_by_unit(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trajectories into train/validation subsets grouped by unit ID.

    An entire unit's rows are assigned to exactly one of the two returned
    subsets, so no unit appears in both.

    Args:
        data: A DataFrame with a `unit_id` column identifying each
            trajectory.
        test_size: Fraction of units to assign to the validation subset.
        random_state: Seed controlling the random group assignment, for
            reproducibility.

    Returns:
        A tuple `(train_df, val_df)` of DataFrames, each a subset of the
        input rows, grouped by whole unit ID.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(data, groups=data["unit_id"]))
    train_df = data.iloc[train_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def fit_standardization(
    data: pd.DataFrame, feature_columns: Sequence[str]
) -> SensorStandardization:
    """Fit per-column standardization statistics from a training subset.

    Args:
        data: A DataFrame to compute statistics from. This should be a
            training subset only.
        feature_columns: Names of the columns to standardize.

    Returns:
        The fitted per-column mean and scale, with zero-variance columns
        guarded to a scale of 1.0.
    """
    feature_columns = list(feature_columns)
    mean = data[feature_columns].mean().to_numpy()
    scale = data[feature_columns].std(ddof=0).to_numpy()
    scale = np.where(scale == 0, 1.0, scale)
    return SensorStandardization(
        feature_columns=feature_columns, mean=mean, scale=scale
    )


def apply_standardization(
    data: pd.DataFrame, stats: SensorStandardization
) -> pd.DataFrame:
    """Apply previously fitted standardization statistics to a DataFrame.

    Args:
        data: A DataFrame containing at least `stats.feature_columns`.
        stats: Standardization statistics, typically fitted on a training
            subset via `fit_standardization`.

    Returns:
        A copy of `data` with `stats.feature_columns` replaced by
        `(value - mean) / scale`. All other columns are left untouched.
    """
    result = data.copy()
    result[stats.feature_columns] = (
        data[stats.feature_columns].to_numpy() - stats.mean
    ) / stats.scale
    return result


@dataclass(frozen=True)
class TargetStandardization:
    """RUL-target standardization statistics fitted on a training subset.

    Attributes:
        mean: Mean of the training RUL targets, in cycles.
        scale: Standard deviation of the training RUL targets, in cycles,
            guarded to 1.0 when they are constant.
    """

    mean: float
    scale: float

    def standardize(self, targets: np.ndarray) -> np.ndarray:
        """Map RUL targets in cycles to standardized units.

        Args:
            targets: RUL targets in cycles, any shape.

        Returns:
            `(targets - mean) / scale`, shaped like `targets`.
        """
        return (np.asarray(targets, dtype=float) - self.mean) / self.scale


def fit_target_standardization(targets: np.ndarray) -> TargetStandardization:
    """Fit RUL-target standardization statistics from training targets.

    Args:
        targets: Training RUL targets in cycles, any shape; at least one.

    Returns:
        The targets' mean and population standard deviation (`ddof=0`), the
        latter guarded to 1.0 for constant targets.
    """
    values = np.asarray(targets, dtype=float)
    scale = float(values.std(ddof=0))
    return TargetStandardization(
        mean=float(values.mean()), scale=scale if scale > 0.0 else 1.0
    )


def _pad_unit_features(
    unit_id: object, features: np.ndarray, window_length: int
) -> np.ndarray:
    """Left-pad a unit's feature rows to `window_length` by repeating the first row.

    Args:
        unit_id: Identifier of the unit being padded, used only for logging.
        features: The unit's feature rows, sorted by `time_cycles`, shape
            `(n_cycles, n_features)`.
        window_length: Target number of rows after padding.

    Returns:
        The feature array left-padded to exactly `window_length` rows.
    """
    n_cycles = features.shape[0]
    n_padded = window_length - n_cycles
    padding = np.repeat(features[:1], n_padded, axis=0)
    _LOGGER.info(
        "Unit %s trajectory has %d cycles, fewer than the window length of "
        "%d; left-padding by repeating the first observed cycle %d time(s).",
        unit_id,
        n_cycles,
        window_length,
        n_padded,
    )
    return np.concatenate([padding, features], axis=0)


@dataclass(frozen=True)
class ProxyVarianceSchedule:
    """A fictitious RUL standard deviation that shrinks linearly to end of life.

    The scale, in cycles, is `initial_scale * rul / reference_rul`, clipped
    to `[floor, initial_scale]`: wide early in life, where the time to failure
    is uncertain, and near zero at failure. It depends on the true RUL, so it
    can weight a training objective but not a prediction.

    Attributes:
        initial_scale: The scale at and above `reference_rul`, in cycles; the
            spread of the training units' lifetimes, never below `floor`.
        reference_rul: The RUL, in cycles, at which the scale reaches
            `initial_scale`: the training units' mean lifetime.
        floor: The scale at end of life, in cycles.
        target_standardization: The statistics the targets passed to `scale`
            were standardized with.
    """

    initial_scale: float
    reference_rul: float
    floor: float
    target_standardization: TargetStandardization

    def scale(self, targets: Any) -> Any:
        """Return the schedule's scale at standardized RUL targets.

        Uses only arithmetic and the array's own `clip`, so it works on NumPy
        and JAX arrays alike, inside a traced loss included.

        Args:
            targets: Standardized RUL targets, a NumPy or JAX array of any
                shape.

        Returns:
            The scale in standardized units, i.e. divided by the target
            scale, shaped like `targets`.
        """
        standardization = self.target_standardization
        rul = targets * standardization.scale + standardization.mean
        cycles = (rul * (self.initial_scale / self.reference_rul)).clip(
            self.floor, self.initial_scale
        )
        return cycles / standardization.scale


def fit_proxy_variance_schedule(
    lifetimes: np.ndarray,
    target_standardization: TargetStandardization,
    *,
    floor_cycles: float,
) -> ProxyVarianceSchedule:
    """Fit a `ProxyVarianceSchedule` to the training units' lifetimes.

    Args:
        lifetimes: The training units' lifetimes in cycles, one per unit; at
            least one, all positive.
        target_standardization: The statistics the training targets were
            standardized with.
        floor_cycles: The scale at end of life, in cycles; positive.

    Returns:
        The schedule starting at the lifetimes' population standard
        deviation (`ddof=0`), raised to `floor_cycles` if below it, and
        reaching it at their mean.
    """
    values = np.asarray(lifetimes, dtype=float)
    return ProxyVarianceSchedule(
        initial_scale=max(float(values.std(ddof=0)), floor_cycles),
        reference_rul=float(values.mean()),
        floor=floor_cycles,
        target_standardization=target_standardization,
    )


def unit_lifetimes(trajectories: pd.DataFrame) -> np.ndarray:
    """Return each unit's lifetime: its last observed `time_cycles`.

    For run-to-failure trajectories this is the unit's linear RUL at cycle
    zero, the RUL `build_windows` counts down from.

    Args:
        trajectories: A DataFrame with `unit_id` and `time_cycles` columns.

    Returns:
        One lifetime per unit, in cycles, in ascending `unit_id` order.
    """
    return trajectories.groupby("unit_id")["time_cycles"].max().to_numpy(dtype=float)


def build_windows(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window_length: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window feature arrays and aligned RUL targets.

    Rows are grouped by `unit_id` and sorted by `time_cycles` within each
    group. Trajectories shorter than `window_length` are left-padded by
    repeating their first observed cycle's feature values; this padding
    affects only which feature rows are visible in a window and never
    changes the true remaining-useful-life target, which is always computed
    from the unit's actual (unpadded) `time_cycles` values.

    Args:
        data: A DataFrame with `unit_id`, `time_cycles`, and
            `feature_columns` columns.
        feature_columns: Names of the columns to use as window channels.
        window_length: Number of cycles per window.
        stride: Step size, in cycles, between consecutive windows.

    Returns:
        A tuple `(windows, targets)`:
            windows: Array of shape `(n_windows, window_length,
                len(feature_columns))`.
            targets: Array of shape `(n_windows,)` holding, for each window,
                the RUL at the window's last real cycle: the unit's maximum
                observed `time_cycles` minus the `time_cycles` value at that
                cycle.
    """
    feature_columns = list(feature_columns)
    window_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []

    for unit_id in sorted(data["unit_id"].unique()):
        unit_data = data[data["unit_id"] == unit_id].sort_values("time_cycles")
        time_cycles = unit_data["time_cycles"].to_numpy()
        features = unit_data[feature_columns].to_numpy(dtype=float)
        unit_max_cycle = time_cycles[-1]

        n_cycles = features.shape[0]
        if n_cycles < window_length:
            features = _pad_unit_features(unit_id, features, window_length)
            n_pad = window_length - n_cycles
            # Real cycles occupy the tail of the padded array; the padded
            # rows have no corresponding real time_cycles entry, so extend
            # time_cycles with the first real cycle's value (never used as
            # a window's *last* row here, since the array is now exactly
            # window_length long and the single window's last row is
            # always the true last real cycle).
            time_cycles = np.concatenate([np.full(n_pad, time_cycles[0]), time_cycles])

        n_windows = (features.shape[0] - window_length) // stride + 1
        for start in range(0, n_windows * stride, stride):
            end = start + window_length
            window_arrays.append(features[start:end])
            last_cycle = time_cycles[end - 1]
            target_arrays.append(unit_max_cycle - last_cycle)

    windows = np.stack(window_arrays, axis=0)
    targets = np.asarray(target_arrays, dtype=float)
    return windows, targets


def build_last_windows(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window_length: int = 30,
) -> np.ndarray:
    """Build one trailing window per unit, ending at its last observed cycle.

    Intended for the single-window-per-trajectory test protocol: each
    (possibly truncated) test trajectory contributes exactly one window,
    covering its most recent `window_length` cycles, meant to be paired
    with an externally supplied ground-truth RUL for that unit (e.g. from
    `RUL_FD001.txt`). Units with fewer than `window_length` cycles are
    left-padded the same way as `build_windows`.

    Args:
        data: A DataFrame with `unit_id`, `time_cycles`, and
            `feature_columns` columns.
        feature_columns: Names of the columns to use as window channels.
        window_length: Number of cycles per window.

    Returns:
        Array of shape `(n_units, window_length, len(feature_columns))`,
        one window per unit in ascending `unit_id` order.
    """
    feature_columns = list(feature_columns)
    window_arrays: list[np.ndarray] = []

    for unit_id in sorted(data["unit_id"].unique()):
        unit_data = data[data["unit_id"] == unit_id].sort_values("time_cycles")
        features = unit_data[feature_columns].to_numpy(dtype=float)
        if features.shape[0] < window_length:
            features = _pad_unit_features(unit_id, features, window_length)
        window_arrays.append(features[-window_length:])

    return np.stack(window_arrays, axis=0)


@dataclass(frozen=True)
class UnitRulCurve:
    """One unit's RUL curve: its trajectory truth and its windowed inputs.

    Holds the two time axes a per-unit RUL curve is drawn against, so that
    a caller never has to re-derive either of them. The trajectory axis
    covers every observed cycle and carries the truth the model is judged
    against; the window axis starts at the unit's first full window, since
    no prediction exists before `window_length` cycles of sensor history
    have accumulated. Both axes are named explicitly: an unqualified
    `cycles` would silently mean either one.

    Attributes:
        windows: The unit's sliding windows, shape `(n_windows,
            window_length, n_features)`.
        window_cycles: The `time_cycles` value of each window's last row,
            shape `(n_windows,)`. This is the time axis a predicted curve
            is plotted against.
        window_linear_rul: The true linear RUL at each window's last real
            cycle, shape `(n_windows,)`. Equal to
            `trajectory_linear_rul[window_length - 1:]`, kept as its own
            field because it is what `build_windows` returns as the
            training target and is therefore the array the models are
            actually scored against.
        lifetime: The unit's total number of cycles, i.e. its linear RUL at
            cycle zero.
        trajectory_cycles: Every observed `time_cycles` value of the unit,
            ascending, shape `(n_cycles,)`.
        trajectory_linear_rul: The true linear RUL at every observed cycle,
            `lifetime - trajectory_cycles`, shape `(n_cycles,)`. This is
            the truth axis a RUL curve is plotted against, spanning the
            whole trajectory so that the model's warm-up stays visible
            rather than hidden.
    """

    windows: np.ndarray
    window_cycles: np.ndarray
    window_linear_rul: np.ndarray
    lifetime: float
    trajectory_cycles: np.ndarray
    trajectory_linear_rul: np.ndarray


def build_unit_rul_curve(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    unit_id: object,
    window_length: int = 30,
) -> UnitRulCurve:
    """Build a single unit's RUL curve: its trajectory truth and its windows.

    Windowing itself is delegated to `build_windows` restricted to that
    unit's rows, so the windows and targets are exactly the ones the models
    are trained and scored on. On top of those, the unit's own
    `time_cycles` values supply both time axes: the cycle each window ends
    at, which a predicted curve is plotted against, and the whole
    trajectory, which the true linear RUL is plotted against.

    Args:
        data: A DataFrame with `unit_id`, `time_cycles`, and
            `feature_columns` columns, holding one or more units.
        feature_columns: Names of the columns to use as window channels.
        unit_id: Identifier of the single unit to window.
        window_length: Number of cycles per window.

    Returns:
        The unit's `UnitRulCurve`: its windows with their aligned true
        linear RUL and window-end cycles, its lifetime, and the true linear
        RUL over every observed cycle.

    Raises:
        ValueError: If `unit_id` has no rows in `data`, or if the unit has
            fewer than `window_length` cycles. Such a unit admits no full
            window, and left-padding it would fabricate sensor history and
            manufacture a prediction that has no support in the data. This
            is deliberately stricter than `build_windows`, which does pad
            such a unit; see `docs/adr/0004-cmapss-rul-curves-refuse-padded-windows.md`.
    """
    unit_data = data[data["unit_id"] == unit_id]
    if unit_data.empty:
        raise ValueError(f"unit_id {unit_id!r} has no rows in the given trajectories.")
    trajectory_cycles = unit_data["time_cycles"].sort_values().to_numpy(dtype=float)
    if trajectory_cycles.shape[0] < window_length:
        raise ValueError(
            f"unit_id {unit_id!r} has {trajectory_cycles.shape[0]} cycles, fewer "
            f"than the window length of {window_length}; it admits no full window."
        )

    windows, window_linear_rul = build_windows(
        unit_data, feature_columns, window_length=window_length
    )
    lifetime = float(trajectory_cycles[-1])
    return UnitRulCurve(
        windows=windows,
        window_cycles=trajectory_cycles[window_length - 1 :],
        window_linear_rul=window_linear_rul,
        lifetime=lifetime,
        trajectory_cycles=trajectory_cycles,
        trajectory_linear_rul=lifetime - trajectory_cycles,
    )


@dataclass(frozen=True)
class LifetimeSpanningUnits:
    """Three units spanning the lifetime range of a set of trajectories.

    Plotting these three together shows whether a model's accuracy and its
    stated uncertainty behave consistently across units that fail early and
    units that survive several times as long.

    Attributes:
        shortest: The `unit_id` with the smallest lifetime.
        median: The `unit_id` with the lower-median lifetime.
        longest: The `unit_id` with the largest lifetime.
    """

    shortest: object
    median: object
    longest: object


def select_lifetime_spanning_units(data: pd.DataFrame) -> LifetimeSpanningUnits:
    """Select the shortest-, median-, and longest-lifetime units.

    A unit's lifetime is its maximum observed `time_cycles`, which is its
    RUL at cycle zero and is therefore only meaningful for run-to-failure
    trajectories. Units are ordered ascending by lifetime with the unit ID
    as tie-break, and the median is the lower of the two central entries
    for an even number of units, so the same trajectories always yield the
    same three units.

    Args:
        data: Run-to-failure trajectories with `unit_id` and `time_cycles`
            columns, holding one or more units.

    Returns:
        The shortest-, median-, and longest-lifetime unit IDs.

    Raises:
        ValueError: If `data` holds no rows.
    """
    if data.empty:
        raise ValueError("data must hold at least one unit.")
    lifetimes = data.groupby("unit_id")["time_cycles"].max()
    ordered = sorted(lifetimes.items(), key=lambda item: (item[1], item[0]))
    unit_ids = [unit_id for unit_id, _ in ordered]
    return LifetimeSpanningUnits(
        shortest=unit_ids[0],
        median=unit_ids[(len(unit_ids) - 1) // 2],
        longest=unit_ids[-1],
    )
