"""Tests for the CMAPSS windowing and standardization preprocessing."""

from __future__ import annotations

import dataclasses
import logging
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from cmapss_trajectories import FEATURE_COLUMNS, build_trajectories
from hypothesis import given
from hypothesis import strategies as st


def test_split_by_unit_has_no_cross_unit_leakage(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({1: 5, 2: 4, 3: 6})

    train_df, val_df = cmapss_preprocessing.split_by_unit(
        data, test_size=0.34, random_state=0
    )

    train_units = set(train_df["unit_id"].unique())
    val_units = set(val_df["unit_id"].unique())

    assert train_units.isdisjoint(val_units)
    assert train_units | val_units == set(data["unit_id"].unique())
    assert len(train_df) + len(val_df) == len(data)


def test_fit_and_apply_standardization_uses_train_only_statistics(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({1: 5, 2: 4, 3: 6})
    train_df = data[data["unit_id"] != 3].reset_index(drop=True)
    val_df = data[data["unit_id"] == 3].reset_index(drop=True)
    feature_columns = FEATURE_COLUMNS

    stats = cmapss_preprocessing.fit_standardization(train_df, feature_columns)

    expected_mean = train_df[feature_columns].mean().to_numpy()
    expected_std = train_df[feature_columns].std(ddof=0).to_numpy()
    np.testing.assert_allclose(stats.mean, expected_mean)
    np.testing.assert_allclose(stats.scale, expected_std)
    assert stats.feature_columns == feature_columns

    applied_val = cmapss_preprocessing.apply_standardization(val_df, stats)

    expected_val = (val_df[feature_columns].to_numpy() - expected_mean) / expected_std
    np.testing.assert_allclose(applied_val[feature_columns].to_numpy(), expected_val)
    # Sanity check: val's own statistics differ from train's, so a bug that
    # standardizes with val statistics would fail the assertion above.
    own_mean = val_df[feature_columns].mean().to_numpy()
    assert not np.allclose(own_mean, expected_mean)
    # Non-feature columns remain untouched.
    assert applied_val["unit_id"].tolist() == val_df["unit_id"].tolist()
    assert applied_val["time_cycles"].tolist() == val_df["time_cycles"].tolist()


def test_fit_standardization_guards_zero_variance_columns(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "time_cycles": [1, 2, 3],
            "sensor_a": [5.0, 5.0, 5.0],
        }
    )

    stats = cmapss_preprocessing.fit_standardization(data, ["sensor_a"])

    assert stats.scale[0] == 1.0


def test_fit_target_standardization_maps_training_targets_to_zero_mean_unit_std(
    cmapss_preprocessing: ModuleType,
) -> None:
    targets = np.array([10.0, 40.0, 70.0, 100.0])

    stats = cmapss_preprocessing.fit_target_standardization(targets)

    assert stats.mean == pytest.approx(55.0)
    assert stats.scale == pytest.approx(float(targets.std(ddof=0)))
    standardized = stats.standardize(targets)
    np.testing.assert_allclose(standardized.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(ddof=0), 1.0)


def test_fit_target_standardization_guards_constant_targets(
    cmapss_preprocessing: ModuleType,
) -> None:
    stats = cmapss_preprocessing.fit_target_standardization(np.full(3, 7.0))

    assert stats.scale == 1.0
    np.testing.assert_allclose(stats.standardize(np.array([7.0, 8.0])), [0.0, 1.0])


def test_build_windows_shapes_and_targets_for_long_trajectory(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4, 5],
            "sensor_a": [10.0, 11.0, 12.0, 13.0, 14.0],
            "sensor_b": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    feature_columns = FEATURE_COLUMNS

    windows, targets = cmapss_preprocessing.build_windows(
        data, feature_columns, window_length=3, stride=1
    )

    # 5 cycles, window_length=3, stride=1 -> 3 windows.
    assert windows.shape == (3, 3, 2)
    assert targets.shape == (3,)

    # Unit max cycle is 5. First window covers cycles 1-3, last cycle=3.
    np.testing.assert_allclose(windows[0], [[10.0, 0.1], [11.0, 0.2], [12.0, 0.3]])
    assert targets[0] == 5 - 3

    # Second window covers cycles 2-4, last cycle=4.
    np.testing.assert_allclose(windows[1], [[11.0, 0.2], [12.0, 0.3], [13.0, 0.4]])
    assert targets[1] == 5 - 4

    # Third window covers cycles 3-5, last cycle=5.
    np.testing.assert_allclose(windows[2], [[12.0, 0.3], [13.0, 0.4], [14.0, 0.5]])
    assert targets[2] == 5 - 5


def test_build_windows_unsorted_input_is_sorted_by_time_cycles(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "time_cycles": [3, 1, 2],
            "sensor_a": [30.0, 10.0, 20.0],
        }
    )

    windows, targets = cmapss_preprocessing.build_windows(
        data, ["sensor_a"], window_length=3, stride=1
    )

    assert windows.shape == (1, 3, 1)
    np.testing.assert_allclose(windows[0], [[10.0], [20.0], [30.0]])
    assert targets[0] == 3 - 3


def test_build_windows_pads_short_trajectory_and_logs_info(
    cmapss_preprocessing: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [7, 7],
            "time_cycles": [1, 2],
            "sensor_a": [50.0, 60.0],
        }
    )

    with caplog.at_level(logging.INFO):
        windows, targets = cmapss_preprocessing.build_windows(
            data, ["sensor_a"], window_length=3, stride=1
        )

    # Padded to exactly window_length rows -> exactly one window.
    assert windows.shape == (1, 3, 1)
    assert targets.shape == (1,)

    # Padding repeats the first observed cycle's feature values backward.
    np.testing.assert_allclose(windows[0], [[50.0], [50.0], [60.0]])

    # Padding never changes the true RUL of the real last cycle: unit's own
    # max time_cycles is 2, and the window's last real cycle is 2.
    assert targets[0] == 2 - 2

    padded_cycles = 3 - 2
    assert any(
        "7" in record.message and str(padded_cycles) in record.message
        for record in caplog.records
        if record.levelno == logging.INFO
    )


def test_build_windows_concatenates_multiple_units_deterministically(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({1: 5, 2: 4, 3: 6})
    feature_columns = FEATURE_COLUMNS

    windows, targets = cmapss_preprocessing.build_windows(
        data, feature_columns, window_length=3, stride=1
    )

    # unit 1: 5 cycles -> 3 windows; unit 2: 4 cycles -> 2 windows;
    # unit 3: 6 cycles -> 4 windows.
    assert windows.shape == (9, 3, 2)
    assert targets.shape == (9,)


def test_build_last_windows_takes_trailing_cycles_per_unit(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({1: 5, 2: 4, 3: 6})
    feature_columns = FEATURE_COLUMNS

    windows = cmapss_preprocessing.build_last_windows(
        data, feature_columns, window_length=3
    )

    # 3 units -> one trailing window each.
    assert windows.shape == (3, 3, 2)
    # unit 1 has cycles 1-5; the trailing window covers cycles 3-5.
    np.testing.assert_allclose(
        windows[0], [[1003.0, 9.25], [1004.0, 9.0], [1005.0, 8.75]]
    )
    # unit 3 has cycles 1-6; the trailing window covers cycles 4-6.
    np.testing.assert_allclose(
        windows[2], [[3004.0, 29.0], [3005.0, 28.75], [3006.0, 28.5]]
    )


def test_build_last_windows_pads_short_trajectory(
    cmapss_preprocessing: ModuleType,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = pd.DataFrame(
        {
            "unit_id": [7, 7],
            "time_cycles": [1, 2],
            "sensor_a": [50.0, 60.0],
        }
    )

    with caplog.at_level(logging.INFO):
        windows = cmapss_preprocessing.build_last_windows(
            data, ["sensor_a"], window_length=3
        )

    assert windows.shape == (1, 3, 1)
    np.testing.assert_allclose(windows[0], [[50.0], [50.0], [60.0]])


@given(
    n_cycles=st.integers(min_value=2, max_value=40),
    window_length=st.integers(min_value=1, max_value=40),
    other_n_cycles=st.integers(min_value=1, max_value=40),
)
def test_build_unit_rul_curve_cycle_axis_matches_the_linear_rul(
    cmapss_preprocessing: ModuleType,
    n_cycles: int,
    window_length: int,
    other_n_cycles: int,
) -> None:
    window_length = 1 + (window_length - 1) % n_cycles
    data = build_trajectories({1: other_n_cycles, 2: n_cycles})
    feature_columns = FEATURE_COLUMNS

    curve = cmapss_preprocessing.build_unit_rul_curve(
        data, feature_columns, unit_id=2, window_length=window_length
    )

    assert curve.lifetime == float(n_cycles)
    # The window axis is the window's last real cycle, so it is exactly the
    # unit's lifetime minus the linear RUL at that window.
    np.testing.assert_allclose(
        curve.window_cycles, curve.lifetime - curve.window_linear_rul
    )
    # One window per cycle from the first full window onwards.
    assert curve.windows.shape == (
        n_cycles - window_length + 1,
        window_length,
        len(feature_columns),
    )
    assert curve.window_linear_rul.shape == (n_cycles - window_length + 1,)
    assert curve.window_cycles.shape == (n_cycles - window_length + 1,)
    # The window axis advances by one cycle per window and the linear RUL
    # falls by one, reaching zero at the unit's final cycle.
    assert np.all(np.diff(curve.window_cycles) > 0)
    assert np.all(np.diff(curve.window_linear_rul) < 0)
    assert curve.window_linear_rul[-1] == 0.0
    assert curve.window_cycles[0] == float(window_length)


@given(
    n_cycles=st.integers(min_value=2, max_value=40),
    window_length=st.integers(min_value=1, max_value=40),
)
def test_build_unit_rul_curve_window_axis_is_the_tail_of_the_trajectory_axis(
    cmapss_preprocessing: ModuleType, n_cycles: int, window_length: int
) -> None:
    window_length = 1 + (window_length - 1) % n_cycles
    data = build_trajectories({1: 17, 2: n_cycles})

    curve = cmapss_preprocessing.build_unit_rul_curve(
        data, ["sensor_a", "sensor_b"], unit_id=2, window_length=window_length
    )

    # The truth axis spans every observed cycle, so the window axis, which
    # starts at the first full window, is exactly its tail. Both axes are
    # stored rather than derived, so this agreement is worth asserting.
    assert curve.trajectory_cycles.shape == (n_cycles,)
    assert curve.trajectory_linear_rul.shape == (n_cycles,)
    np.testing.assert_allclose(
        curve.trajectory_cycles[window_length - 1 :], curve.window_cycles
    )
    np.testing.assert_allclose(
        curve.trajectory_linear_rul[window_length - 1 :], curve.window_linear_rul
    )
    np.testing.assert_allclose(
        curve.trajectory_linear_rul, curve.lifetime - curve.trajectory_cycles
    )
    assert curve.trajectory_linear_rul[-1] == 0.0


@given(
    n_cycles=st.integers(min_value=2, max_value=40),
    window_length=st.integers(min_value=1, max_value=40),
)
def test_build_unit_rul_curve_matches_the_shared_window_builder(
    cmapss_preprocessing: ModuleType, n_cycles: int, window_length: int
) -> None:
    window_length = 1 + (window_length - 1) % n_cycles
    data = build_trajectories({1: 17, 2: n_cycles})
    feature_columns = FEATURE_COLUMNS

    curve = cmapss_preprocessing.build_unit_rul_curve(
        data, feature_columns, unit_id=2, window_length=window_length
    )

    expected_windows, expected_targets = cmapss_preprocessing.build_windows(
        data[data["unit_id"] == 2], feature_columns, window_length=window_length
    )
    np.testing.assert_allclose(curve.windows, expected_windows)
    np.testing.assert_allclose(curve.window_linear_rul, expected_targets)


def test_build_unit_rul_curve_rejects_an_absent_unit(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({1: 5})

    with pytest.raises(ValueError, match="unit_id"):
        cmapss_preprocessing.build_unit_rul_curve(
            data, ["sensor_a"], unit_id=99, window_length=3
        )


def test_build_unit_rul_curve_rejects_a_unit_shorter_than_the_window(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({1: 4})

    with pytest.raises(ValueError, match="no full window"):
        cmapss_preprocessing.build_unit_rul_curve(
            data, ["sensor_a"], unit_id=1, window_length=5
        )


@given(
    lifetimes=st.lists(st.integers(min_value=1, max_value=40), min_size=1, max_size=8),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_lifetime_spanning_units_are_invariant_to_row_order(
    cmapss_preprocessing: ModuleType, lifetimes: list[int], seed: int
) -> None:
    by_unit = {unit_id: n_cycles for unit_id, n_cycles in enumerate(lifetimes, start=1)}
    data = build_trajectories(by_unit)
    shuffled = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    selected = cmapss_preprocessing.select_lifetime_spanning_units(data)
    from_shuffled = cmapss_preprocessing.select_lifetime_spanning_units(shuffled)

    assert selected == from_shuffled
    # Repeated calls on the same frame agree too: the selection is a pure
    # function of the lifetimes, never of iteration order.
    assert selected == cmapss_preprocessing.select_lifetime_spanning_units(data)


@given(
    lifetimes=st.lists(
        st.integers(min_value=1, max_value=40), min_size=1, max_size=8, unique=True
    ),
    labels=st.lists(
        st.integers(min_value=1, max_value=999), min_size=8, max_size=8, unique=True
    ),
)
def test_lifetime_spanning_units_are_invariant_to_unit_relabelling(
    cmapss_preprocessing: ModuleType, lifetimes: list[int], labels: list[int]
) -> None:
    by_unit = {unit_id: n_cycles for unit_id, n_cycles in enumerate(lifetimes, start=1)}
    # Relabel every unit through an arbitrary permutation of fresh IDs, so
    # the new labels need not order the same way as the old ones. Lifetimes
    # are unique here, so no tie-break is in play: the same trajectories
    # must be picked out whatever they are called.
    relabelling = {unit_id: labels[unit_id - 1] for unit_id in by_unit}
    relabelled = {
        relabelling[unit_id]: n_cycles for unit_id, n_cycles in by_unit.items()
    }

    selected = cmapss_preprocessing.select_lifetime_spanning_units(
        build_trajectories(by_unit)
    )
    from_relabelled = cmapss_preprocessing.select_lifetime_spanning_units(
        build_trajectories(relabelled)
    )

    assert from_relabelled.shortest == relabelling[selected.shortest]
    assert from_relabelled.median == relabelling[selected.median]
    assert from_relabelled.longest == relabelling[selected.longest]


@given(
    lifetimes=st.lists(st.integers(min_value=1, max_value=40), min_size=1, max_size=8),
)
def test_lifetime_spanning_units_span_and_order_the_lifetimes(
    cmapss_preprocessing: ModuleType,
    lifetimes: list[int],
) -> None:
    by_unit = {unit_id: n_cycles for unit_id, n_cycles in enumerate(lifetimes, start=1)}
    data = build_trajectories(by_unit)

    selected = cmapss_preprocessing.select_lifetime_spanning_units(data)

    assert by_unit[selected.shortest] == min(lifetimes)
    assert by_unit[selected.longest] == max(lifetimes)
    assert (
        by_unit[selected.shortest]
        <= by_unit[selected.median]
        <= by_unit[selected.longest]
    )


@given(
    lifetimes=st.lists(
        st.integers(min_value=1, max_value=80), min_size=2, max_size=8, unique=True
    ),
)
def test_lifetime_spanning_units_take_the_lower_median_for_an_even_count(
    cmapss_preprocessing: ModuleType,
    lifetimes: list[int],
) -> None:
    # An even number of units with distinct lifetimes, so the two central
    # entries genuinely differ and taking the lower one is a real choice.
    ascending = sorted(lifetimes)[: len(lifetimes) // 2 * 2]
    by_unit = {unit_id: n_cycles for unit_id, n_cycles in enumerate(ascending, start=1)}
    data = build_trajectories(by_unit)

    selected = cmapss_preprocessing.select_lifetime_spanning_units(data)

    # Unit IDs ascend with lifetime here, so the two central units are the
    # two middle IDs and the tie-break never comes into play.
    n_units = len(ascending)
    lower_central, upper_central = n_units // 2, n_units // 2 + 1
    assert selected.median == lower_central
    assert by_unit[lower_central] < by_unit[upper_central]


def test_lifetime_spanning_units_are_frozen(cmapss_preprocessing: ModuleType) -> None:
    selected = cmapss_preprocessing.select_lifetime_spanning_units(
        build_trajectories({1: 3, 2: 7})
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        selected.median = selected.shortest  # type: ignore[misc]


def test_lifetime_spanning_units_rejects_empty_trajectories(
    cmapss_preprocessing: ModuleType,
) -> None:
    empty = build_trajectories({1: 1}).iloc[0:0]

    with pytest.raises(ValueError, match="at least one unit"):
        cmapss_preprocessing.select_lifetime_spanning_units(empty)


def test_proxy_variance_schedule_is_linear_in_rul_between_its_floor_and_cap(
    cmapss_preprocessing: ModuleType,
) -> None:
    # Lifetimes 100 and 300 cycles: mean 200, population std 100. So sigma
    # is 100 * rul / 200 cycles, floored at 5 and capped at 100.
    standardization = cmapss_preprocessing.TargetStandardization(mean=50.0, scale=25.0)
    schedule = cmapss_preprocessing.fit_proxy_variance_schedule(
        np.array([100, 300]), standardization, floor_cycles=5.0
    )
    rul = np.array([0.0, 2.0, 100.0, 200.0, 400.0])

    scale = schedule.scale(standardization.standardize(rul))

    # In standardized units: cycles divided by the target scale, 25.
    np.testing.assert_allclose(scale, [0.2, 0.2, 2.0, 4.0, 4.0])


def test_proxy_variance_schedule_of_equal_lifetimes_is_its_floor(
    cmapss_preprocessing: ModuleType,
) -> None:
    standardization = cmapss_preprocessing.TargetStandardization(mean=0.0, scale=10.0)
    schedule = cmapss_preprocessing.fit_proxy_variance_schedule(
        np.array([150, 150, 150]), standardization, floor_cycles=5.0
    )

    scale = schedule.scale(standardization.standardize(np.array([0.0, 75.0, 300.0])))

    np.testing.assert_allclose(scale, [0.5, 0.5, 0.5])


@given(
    lifetimes=st.lists(st.integers(min_value=1, max_value=400), min_size=1),
    rul=st.lists(st.floats(min_value=0.0, max_value=500.0), min_size=2, max_size=20),
    floor_cycles=st.floats(min_value=0.1, max_value=20.0),
)
def test_proxy_variance_schedule_is_monotone_and_bounded(
    cmapss_preprocessing: ModuleType,
    lifetimes: list[int],
    rul: list[float],
    floor_cycles: float,
) -> None:
    standardization = cmapss_preprocessing.TargetStandardization(mean=80.0, scale=40.0)
    schedule = cmapss_preprocessing.fit_proxy_variance_schedule(
        np.array(lifetimes), standardization, floor_cycles=floor_cycles
    )
    ordered = np.sort(np.array(rul))

    scale = schedule.scale(standardization.standardize(ordered)) * (
        standardization.scale
    )

    assert np.all(np.diff(scale) >= -1e-9)
    initial_scale = max(float(np.std(lifetimes)), floor_cycles)
    assert np.all(scale >= floor_cycles - 1e-9)
    assert np.all(scale <= initial_scale + 1e-9)


def test_unit_lifetimes_are_each_units_last_cycle_in_unit_order(
    cmapss_preprocessing: ModuleType,
) -> None:
    data = build_trajectories({3: 7, 1: 5, 2: 9}).sample(frac=1.0, random_state=0)

    lifetimes = cmapss_preprocessing.unit_lifetimes(data)

    np.testing.assert_array_equal(lifetimes, [5.0, 9.0, 7.0])
