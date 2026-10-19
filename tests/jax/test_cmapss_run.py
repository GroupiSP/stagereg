"""Tests for the CMAPSS end-to-end mean-stage training and evaluation."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

# The example module itself arrives through the `cmapss_run` fixture; these
# names are used directly by the test bodies, so they stay module-level.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("optax")


def _synthetic_windows(
    rng: np.random.Generator, *, n_windows: int, window_length: int, n_sensors: int
) -> tuple[np.ndarray, np.ndarray]:
    windows = rng.normal(size=(n_windows, window_length, n_sensors))
    targets = rng.uniform(low=1.0, high=100.0, size=n_windows)
    return windows, targets


def test_train_gamma_model_and_evaluate_composite_metrics_end_to_end(
    cmapss_run: ModuleType,
) -> None:
    rng = np.random.default_rng(0)
    train_windows, train_targets = _synthetic_windows(
        rng, n_windows=64, window_length=30, n_sensors=9
    )
    validation_windows, validation_targets = _synthetic_windows(
        rng, n_windows=16, window_length=30, n_sensors=9
    )
    test_windows, test_rul = _synthetic_windows(
        rng, n_windows=10, window_length=30, n_sensors=9
    )
    config = cmapss_run.CmapssConfig(
        batch_size=16,
        hidden_channels=4,
        kernel_size=3,
        mean_epochs=3,
        variance_epochs=3,
        predictive_sample_count=32,
        seed=1,
    )

    mean_model, state = cmapss_run.train_mean_model(
        train_windows, train_targets, validation_windows, validation_targets, config
    )
    variance_model = cmapss_run.train_gamma_model(
        state, train_windows, train_targets, config
    )
    metrics = cmapss_run.evaluate_composite_metrics(
        mean_model, variance_model, test_windows, test_rul, config
    )

    assert "loss" not in metrics
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["coverage"])
    assert math.isfinite(metrics["point_crps"])
    assert metrics["rmse"] >= 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["point_crps"] >= 0.0


def test_cmapss_config_rejects_invalid_values(cmapss_run: ModuleType) -> None:
    with pytest.raises(ValueError, match="window_length and batch_size"):
        cmapss_run.CmapssConfig(window_length=0)
    with pytest.raises(ValueError, match="validation_fraction"):
        cmapss_run.CmapssConfig(validation_fraction=1.5)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        cmapss_run.CmapssConfig(mean_epochs=0)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        cmapss_run.CmapssConfig(variance_epochs=0)
    with pytest.raises(ValueError, match="predictive_sample_count"):
        cmapss_run.CmapssConfig(predictive_sample_count=0)


@pytest.fixture
def short_window_config(cmapss_run: ModuleType) -> Any:
    """Configure a window short enough for the synthetic trajectories."""
    return cmapss_run.CmapssConfig(window_length=10)


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize("window_length", [5, 10, 30])
def test_prepared_data_windows_and_targets_are_aligned(
    cmapss_run: ModuleType, window_length: int
) -> None:
    config = cmapss_run.CmapssConfig(window_length=window_length)

    prepared = cmapss_run.prepare_cmapss_windows(config)

    n_sensors = len(cmapss_run._SENSOR_NAMES)
    assert prepared.train_windows.shape[1:] == (config.window_length, n_sensors)
    assert prepared.train_windows.shape[0] == prepared.train_targets.shape[0]
    assert (
        prepared.validation_windows.shape[0] == (prepared.validation_targets.shape[0])
    )
    assert prepared.test_windows.shape[0] == prepared.test_rul.shape[0]


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_is_frozen(
    cmapss_run: ModuleType, short_window_config: Any
) -> None:
    prepared = cmapss_run.prepare_cmapss_windows(short_window_config)

    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.train_targets = prepared.train_targets  # type: ignore[misc]


@pytest.mark.usefixtures("offline_fd001")
def test_prepared_data_standardizes_every_subset_with_training_statistics(
    cmapss_run: ModuleType,
    short_window_config: Any,
) -> None:
    prepared = cmapss_run.prepare_cmapss_windows(short_window_config)

    stats = prepared.standardization
    assert stats.feature_columns == cmapss_run._SENSOR_NAMES
    # Statistics fitted on the training subset only: standardizing that
    # subset with its own statistics leaves it at zero mean and unit scale.
    standardized_train = prepared.train_trajectories[stats.feature_columns].to_numpy()
    assert np.allclose(standardized_train.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(standardized_train.std(axis=0, ddof=0), 1.0, atol=1e-8)
    # The validation trajectories carry those same statistics, not their own.
    raw = cmapss_run.load_fd001_data()
    raw_validation = raw[
        raw["unit_id"].isin(prepared.validation_trajectories["unit_id"].unique())
    ]
    expected = cmapss_run.apply_standardization(raw_validation, stats)
    assert np.allclose(
        prepared.validation_trajectories[stats.feature_columns].to_numpy(),
        expected[stats.feature_columns].to_numpy(),
    )


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_build_composite_model_matches_the_source_models_predictions(
    cmapss_run: ModuleType,
    batch_size: int,
) -> None:
    window_length, n_sensors = 30, 9
    mean_model = cmapss_run.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    variance_model = cmapss_run.Cnn1DGammaModel(n_sensors, rngs=nnx.Rngs(1))
    inputs = jax.random.normal(
        jax.random.key(2), (batch_size, window_length, n_sensors)
    )

    composite = cmapss_run.build_composite_model(mean_model, variance_model)
    prediction = composite(inputs)

    assert bool(jnp.allclose(prediction.loc, mean_model(inputs)))
    assert bool(jnp.allclose(prediction.scale, jnp.sqrt(variance_model(inputs).mean())))


def test_build_composite_model_clones_the_source_models(cmapss_run: ModuleType) -> None:
    n_sensors = 9
    mean_model = cmapss_run.Cnn1DMeanModel(n_sensors, rngs=nnx.Rngs(0))
    variance_model = cmapss_run.Cnn1DGammaModel(n_sensors, rngs=nnx.Rngs(1))
    original_bias = jnp.asarray(mean_model.output.bias[...])

    composite = cmapss_run.build_composite_model(mean_model, variance_model)
    composite.mean_model.output.bias[...] = original_bias + 1.0

    assert composite.mean_model is not mean_model
    assert composite.variance_model is not variance_model
    assert bool(jnp.allclose(mean_model.output.bias[...], original_bias))


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize("plot_path_argv", [[], ["--plot-path", "curve.png"]])
def test_main_plots_the_rul_curves_after_printing_metrics(
    cmapss_run: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plot_path_argv: list[str],
) -> None:
    if plot_path_argv:
        plot_path_argv = [plot_path_argv[0], str(tmp_path / plot_path_argv[1])]
    monkeypatch.setattr(sys, "argv", ["run.py", *plot_path_argv])
    fast_config = cmapss_run.CmapssConfig(
        window_length=10,
        batch_size=16,
        hidden_channels=2,
        kernel_size=3,
        mean_epochs=1,
        variance_epochs=1,
        predictive_sample_count=8,
    )
    monkeypatch.setattr(cmapss_run, "CmapssConfig", lambda: fast_config)
    calls: list[dict[str, object]] = []

    def record_plot(
        trajectories: pd.DataFrame,
        feature_columns: list[str],
        model: object,
        *,
        units: object,
        window_length: int,
        save_path: Path | None,
    ) -> None:
        """Record one plotting call and whatever was printed before it."""
        calls.append(
            {
                "trajectories": trajectories,
                "feature_columns": feature_columns,
                "model": model,
                "units": units,
                "window_length": window_length,
                "save_path": save_path,
                "printed": capsys.readouterr().out,
            }
        )

    monkeypatch.setattr(cmapss_run, "plot_validation_rul_curves", record_plot)

    cmapss_run.main()

    (call,) = calls
    assert call["feature_columns"] == cmapss_run._SENSOR_NAMES
    assert call["window_length"] == 10
    assert isinstance(call["model"], cmapss_run.CompositeGaussianModel)
    expected_path = Path(plot_path_argv[1]) if plot_path_argv else None
    assert call["save_path"] == expected_path
    # The unit plotted is a held-out validation unit, and the metrics are
    # already printed by the time the figure is built.
    trajectories = call["trajectories"]
    assert isinstance(trajectories, pd.DataFrame)
    assert not trajectories.empty
    prepared = cmapss_run.prepare_cmapss_windows(fast_config)
    plotted_units = set(trajectories["unit_id"].unique())
    assert plotted_units == set(prepared.validation_trajectories["unit_id"].unique())
    assert plotted_units.isdisjoint(prepared.train_trajectories["unit_id"].unique())
    # The run, not the plotting module, picks the lifetime-spanning trio,
    # and it picks it out of the validation trajectories it hands over.
    units = call["units"]
    assert units == cmapss_run.select_lifetime_spanning_units(trajectories)
    assert {units.shortest, units.median, units.longest} <= plotted_units
    printed = call["printed"]
    assert isinstance(printed, str)
    assert "FD001 test RMSE" in printed
    assert "FD001 test point-CRPS" in printed
