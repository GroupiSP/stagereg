"""Tests for the CMAPSS end-to-end mean-stage training and evaluation."""

from __future__ import annotations

import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

# The example module itself arrives through the `cmapss_run` fixture; these
# names are used directly by the test bodies, so they stay module-level.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("optax")


def test_train_gamma_model_and_evaluate_test_metrics_end_to_end(
    cmapss_run: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    rng = np.random.default_rng(0)
    train_windows, train_targets = synthetic_windows(rng, 64)
    validation_windows, validation_targets = synthetic_windows(rng, 16)
    test_windows, test_rul = synthetic_windows(rng, 10)
    config = tiny_cmapss_config

    mean_model, state = cmapss_run.train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        event_sinks=(),
    )
    variance_model = cmapss_run.train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=()
    )
    metrics = cmapss_run.evaluate_test_metrics(
        cmapss_run.build_composite_model(mean_model, variance_model),
        test_windows,
        test_rul,
    )

    assert set(metrics) == {"rmse", "coverage", "closed_form_crps"}
    assert math.isfinite(metrics["rmse"])
    assert math.isfinite(metrics["closed_form_crps"])
    assert metrics["rmse"] >= 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["closed_form_crps"] >= 0.0


class _RecordingSink:
    """An event sink that keeps every training event it receives."""

    def __init__(self) -> None:
        """Start with no recorded events."""
        self.events: list[Any] = []

    def on_event(self, event: Any) -> None:
        """Record one training event.

        Args:
            event: The training event a stage emitted.
        """
        self.events.append(event)


def test_training_functions_report_only_to_the_event_sinks_they_are_given(
    cmapss_run: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_windows: Callable[..., tuple[np.ndarray, np.ndarray]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rng = np.random.default_rng(3)
    train_windows, train_targets = synthetic_windows(rng, 16)
    validation_windows, validation_targets = synthetic_windows(rng, 8)
    config = dataclasses.replace(
        tiny_cmapss_config, batch_size=8, mean_epochs=2, variance_epochs=3
    )
    mean_sink, gamma_sink = _RecordingSink(), _RecordingSink()

    _, state = cmapss_run.train_mean_model(
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        config,
        event_sinks=[mean_sink],
    )
    cmapss_run.train_gamma_model(
        state, train_windows, train_targets, config, event_sinks=[gamma_sink]
    )

    assert capsys.readouterr().out == ""
    mean_train_steps = [e.step for e in mean_sink.events if e.split == "train"]
    gamma_train_steps = [e.step for e in gamma_sink.events if e.split == "train"]
    assert len(mean_train_steps) == 2
    assert len(gamma_train_steps) == 3
    assert any(e.split == "validation" for e in mean_sink.events)


class _ConstantGaussianModel(nnx.Module):
    """A Gaussian-output model predicting the same distribution for every window."""

    def __init__(self, loc: float, scale: float) -> None:
        """Fix the predicted distribution.

        Args:
            loc: The predicted mean for every window.
            scale: The predicted standard deviation for every window.
        """
        self.loc = loc
        self.scale = scale

    def __call__(self, inputs: Any) -> Any:
        """Predict the fixed Gaussian for every window in the batch.

        Args:
            inputs: Windowed sensor readings, shape `(batch, window_length,
                n_sensors)`.

        Returns:
            A Gaussian with `loc` and `scale` each filled to shape `(batch, 1)`.
        """
        from stagereg.jax import Gaussian

        shape = (inputs.shape[0], 1)
        return Gaussian(
            loc=jnp.full(shape, self.loc), scale=jnp.full(shape, self.scale)
        )


def test_evaluate_test_metrics_scores_any_gaussian_output_model(
    cmapss_run: ModuleType,
) -> None:
    # A standard normal scored at its own mean: RMSE 0, the target inside
    # every interval, and the closed-form CRPS of N(0, 1) at 0, which is
    # 2 * phi(0) - 1 / sqrt(pi) = (sqrt(2) - 1) / sqrt(pi).
    test_windows = np.zeros((4, 30, 9))
    test_rul = np.zeros(4)

    metrics = cmapss_run.evaluate_test_metrics(
        _ConstantGaussianModel(loc=0.0, scale=1.0), test_windows, test_rul
    )

    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["closed_form_crps"] == pytest.approx(
        (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi), rel=1e-5
    )


def test_evaluate_test_metrics_accepts_extra_epoch_metrics(
    cmapss_run: ModuleType,
) -> None:
    from stagereg.core import NegativeLogLikelihood

    test_windows = np.zeros((4, 30, 9))
    test_rul = np.zeros(4)

    metrics = cmapss_run.evaluate_test_metrics(
        _ConstantGaussianModel(loc=0.0, scale=1.0),
        test_windows,
        test_rul,
        epoch_metrics=(*cmapss_run.TEST_METRICS, NegativeLogLikelihood()),
    )

    assert set(metrics) == {"rmse", "coverage", "closed_form_crps", "nll"}
    # NLL of N(0, 1) at its mean: log(2 * pi) / 2.
    assert metrics["nll"] == pytest.approx(0.5 * math.log(2.0 * math.pi), rel=1e-5)


def test_cmapss_config_rejects_invalid_values(cmapss_run: ModuleType) -> None:
    with pytest.raises(ValueError, match="window_length and batch_size"):
        cmapss_run.CmapssConfig(window_length=0)
    with pytest.raises(ValueError, match="validation_fraction"):
        cmapss_run.CmapssConfig(validation_fraction=1.5)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        cmapss_run.CmapssConfig(mean_epochs=0)
    with pytest.raises(ValueError, match="mean_epochs and variance_epochs"):
        cmapss_run.CmapssConfig(variance_epochs=0)


@given(joint_epochs=st.integers(max_value=0))
def test_cmapss_config_rejects_non_positive_joint_epochs(
    cmapss_run: ModuleType, joint_epochs: int
) -> None:
    with pytest.raises(ValueError, match="joint_epochs"):
        cmapss_run.CmapssConfig(joint_epochs=joint_epochs)


def test_cmapss_config_gives_the_joint_model_the_same_default_budget_as_each_stage(
    cmapss_run: ModuleType,
) -> None:
    config = cmapss_run.CmapssConfig()

    assert config.joint_epochs == 20


def test_cmapss_config_has_no_predictive_sample_count(cmapss_run: ModuleType) -> None:
    # The closed-form CRPS draws no predictive samples, so the run has no
    # sample count to configure.
    with pytest.raises(TypeError):
        cmapss_run.CmapssConfig(predictive_sample_count=8)


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
    tiny_cmapss_config: Any,
    plot_recorder: Any,
    plot_path_argv: list[str],
) -> None:
    if plot_path_argv:
        plot_path_argv = [plot_path_argv[0], str(tmp_path / plot_path_argv[1])]
    monkeypatch.setattr(sys, "argv", ["run.py", *plot_path_argv])
    monkeypatch.setattr(cmapss_run, "CmapssConfig", lambda: tiny_cmapss_config)
    monkeypatch.setattr(cmapss_run, "plot_validation_rul_curves", plot_recorder)

    cmapss_run.main()

    (call,) = plot_recorder.calls
    assert call.feature_columns == cmapss_run._SENSOR_NAMES
    assert call.window_length == tiny_cmapss_config.window_length
    # The staged-only run draws its one staged model as a single row.
    (labelled,) = call.models
    assert labelled.label == "Staged model"
    assert isinstance(labelled.model, cmapss_run.CompositeGaussianModel)
    expected_path = Path(plot_path_argv[1]) if plot_path_argv else None
    assert call.save_path == expected_path
    # The unit plotted is a held-out validation unit, and the metrics are
    # already printed by the time the figure is built.
    trajectories = call.trajectories
    assert isinstance(trajectories, pd.DataFrame)
    assert not trajectories.empty
    prepared = cmapss_run.prepare_cmapss_windows(tiny_cmapss_config)
    plotted_units = set(trajectories["unit_id"].unique())
    assert plotted_units == set(prepared.validation_trajectories["unit_id"].unique())
    assert plotted_units.isdisjoint(prepared.train_trajectories["unit_id"].unique())
    # The run, not the plotting module, picks the lifetime-spanning trio,
    # and it picks it out of the validation trajectories it hands over.
    units = call.units
    assert units == cmapss_run.select_lifetime_spanning_units(trajectories)
    assert {units.shortest, units.median, units.longest} <= plotted_units
    printed = call.printed
    assert isinstance(printed, str)
    assert "FD001 test RMSE" in printed
    assert "FD001 test closed-form CRPS" in printed
