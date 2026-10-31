"""Tests for the CMAPSS staged-vs-joint comparison entry point."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

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
