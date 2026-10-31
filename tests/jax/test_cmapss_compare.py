"""Tests for the CMAPSS staged-vs-joint comparison entry point."""

from __future__ import annotations

import math
import re
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("jax")
pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_METRIC_NAMES = {"rmse", "coverage", "closed_form_crps", "nll"}


def _synthetic_prepared_data(
    cmapss_run: ModuleType,
    *,
    seed: int = 0,
    window_length: int = 10,
    n_sensors: int = 3,
    max_rul: float = 100.0,
) -> Any:
    """Build small synthetic prepared data; only its windowed arrays are trained on."""
    rng = np.random.default_rng(seed)

    def windows(n_windows: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = rng.normal(size=(n_windows, window_length, n_sensors))
        targets = rng.uniform(low=1.0, high=max_rul, size=n_windows)
        return inputs, targets

    train_windows, train_targets = windows(32)
    validation_windows, validation_targets = windows(8)
    test_windows, test_rul = windows(6)
    sensors = [f"sensor_{i}" for i in range(n_sensors)]
    columns = ["unit_id", "time_cycles", *sensors]
    return cmapss_run.PreparedCmapssData(
        train_windows=train_windows,
        train_targets=train_targets,
        validation_windows=validation_windows,
        validation_targets=validation_targets,
        test_windows=test_windows,
        test_rul=test_rul,
        train_trajectories=pd.DataFrame(columns=columns),
        validation_trajectories=pd.DataFrame(columns=columns),
        standardization=cmapss_run.SensorStandardization(
            feature_columns=sensors,
            mean=np.zeros(n_sensors),
            scale=np.ones(n_sensors),
        ),
    )


@pytest.fixture(scope="module")
def tiny_config(cmapss_compare: ModuleType) -> Any:
    """A configuration small enough to train both models in a second or two."""
    return cmapss_compare.CmapssConfig(
        window_length=10,
        batch_size=16,
        hidden_channels=2,
        kernel_size=3,
        mean_epochs=2,
        variance_epochs=2,
        joint_epochs=2,
        seed=3,
    )


def test_compare_models_scores_both_models_on_the_four_metrics_once_per_seed(
    cmapss_run: ModuleType, cmapss_compare: ModuleType, tiny_config: Any
) -> None:
    prepared = _synthetic_prepared_data(cmapss_run)

    results = cmapss_compare.compare_models(prepared, tiny_config, seeds=2)

    assert [result.seed for result in results] == [3, 4]
    for result in results:
        assert set(result.staged_metrics) == _METRIC_NAMES
        assert set(result.joint_metrics) == _METRIC_NAMES


def test_compare_models_trains_silently_and_reproduces_a_seed(
    cmapss_run: ModuleType,
    cmapss_compare: ModuleType,
    tiny_config: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _synthetic_prepared_data(cmapss_run)

    (first,) = cmapss_compare.compare_models(prepared, tiny_config, seeds=1)
    (second,) = cmapss_compare.compare_models(prepared, tiny_config, seeds=1)

    assert capsys.readouterr().out == ""
    assert first.staged_metrics == second.staged_metrics
    assert first.joint_metrics == second.joint_metrics


def test_compare_models_varies_initialization_across_seeds(
    cmapss_run: ModuleType, cmapss_compare: ModuleType, tiny_config: Any
) -> None:
    prepared = _synthetic_prepared_data(cmapss_run)

    first, second = cmapss_compare.compare_models(prepared, tiny_config, seeds=2)

    assert first.staged_metrics != second.staged_metrics
    assert first.joint_metrics != second.joint_metrics


@settings(max_examples=4, deadline=None)
@given(
    data_seed=st.integers(min_value=0, max_value=2**16),
    max_rul=st.floats(min_value=2.0, max_value=400.0),
)
def test_compare_models_scores_lie_in_their_ranges(
    cmapss_run: ModuleType,
    cmapss_compare: ModuleType,
    tiny_config: Any,
    data_seed: int,
    max_rul: float,
) -> None:
    prepared = _synthetic_prepared_data(cmapss_run, seed=data_seed, max_rul=max_rul)

    (result,) = cmapss_compare.compare_models(prepared, tiny_config, seeds=1)

    for metrics in (result.staged_metrics, result.joint_metrics):
        assert 0.0 <= metrics["coverage"] <= 1.0
        for name in ("rmse", "closed_form_crps", "nll"):
            assert math.isfinite(metrics[name])


_CELL = r"-?\d+\.\d+ ± \d+\.\d+"


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize(("seeds_argv", "seeds"), [([], 5), (["--seeds", "2"], 2)])
def test_main_prints_one_line_per_run_and_a_mean_std_table(
    cmapss_compare: ModuleType,
    tiny_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seeds_argv: list[str],
    seeds: int,
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", *seeds_argv])
    monkeypatch.setattr(cmapss_compare, "CmapssConfig", lambda: tiny_config)
    prepared_with: list[Any] = []
    prepare = cmapss_compare.prepare_cmapss_windows

    def record_prepare(config: Any) -> Any:
        prepared_with.append(config)
        return prepare(config)

    monkeypatch.setattr(cmapss_compare, "prepare_cmapss_windows", record_prepare)

    cmapss_compare.main()

    # The data is prepared once, with the configured seed, for every seed.
    assert prepared_with == [tiny_config]
    lines = capsys.readouterr().out.splitlines()
    for model in ("staged", "joint"):
        run_lines = [line for line in lines if line.startswith(f"[{model} ")]
        expected_seeds = range(tiny_config.seed, tiny_config.seed + seeds)
        assert [line.split()[1] for line in run_lines] == [
            f"seed={seed}]" for seed in expected_seeds
        ]
        for line in run_lines:
            for name in ("rmse", "coverage", "closed_form_crps", "nll"):
                assert f"{name}=" in line
    (header,) = [line for line in lines if re.search(r"staged\s+joint\s*$", line)]
    table = lines[lines.index(header) + 1 :]
    for row in ("RMSE", "95% coverage", "closed-form CRPS", "NLL"):
        (line,) = [line for line in table if line.startswith(row)]
        assert re.fullmatch(rf"{re.escape(row)}\s+{_CELL}\s+{_CELL}", line)
