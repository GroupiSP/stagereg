"""Tests for the CMAPSS staged-vs-joint comparison entry point."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("jax")
pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_METRIC_NAMES = {"rmse", "coverage", "closed_form_crps", "nll"}


def test_compare_models_scores_both_models_on_the_four_metrics_once_per_seed(
    cmapss_compare: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    prepared = synthetic_prepared_data()

    results = cmapss_compare.compare_models(prepared, tiny_cmapss_config, seeds=2)

    assert [result.seed for result in results] == [3, 4]
    for result in results:
        staged, joint = result.models
        assert (staged.name, staged.label) == ("staged", "Staged model")
        assert (joint.name, joint.label) == ("joint", "Joint model")
        assert set(staged.metrics) == _METRIC_NAMES
        assert set(joint.metrics) == _METRIC_NAMES


def test_compare_models_trains_silently_and_reproduces_a_seed(
    cmapss_compare: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = synthetic_prepared_data()

    (first,) = cmapss_compare.compare_models(prepared, tiny_cmapss_config, seeds=1)
    (second,) = cmapss_compare.compare_models(prepared, tiny_cmapss_config, seeds=1)

    assert capsys.readouterr().out == ""
    for first_model, second_model in zip(first.models, second.models, strict=True):
        assert first_model.metrics == second_model.metrics


def test_compare_models_varies_initialization_across_seeds(
    cmapss_compare: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_prepared_data: Callable[..., Any],
) -> None:
    prepared = synthetic_prepared_data()

    first, second = cmapss_compare.compare_models(prepared, tiny_cmapss_config, seeds=2)

    for first_model, second_model in zip(first.models, second.models, strict=True):
        assert first_model.metrics != second_model.metrics


@settings(max_examples=4, deadline=None)
@given(
    data_seed=st.integers(min_value=0, max_value=2**16),
    max_rul=st.floats(min_value=2.0, max_value=400.0),
)
def test_compare_models_scores_lie_in_their_ranges(
    cmapss_compare: ModuleType,
    tiny_cmapss_config: Any,
    synthetic_prepared_data: Callable[..., Any],
    data_seed: int,
    max_rul: float,
) -> None:
    prepared = synthetic_prepared_data(seed=data_seed, max_rul=max_rul)

    (result,) = cmapss_compare.compare_models(prepared, tiny_cmapss_config, seeds=1)

    for metrics in (model.metrics for model in result.models):
        assert 0.0 <= metrics["coverage"] <= 1.0
        for name in ("rmse", "closed_form_crps", "nll"):
            assert math.isfinite(metrics[name])


_CELL = r"-?\d+\.\d+ ± \d+\.\d+"


def test_format_comparison_table_gives_each_model_a_mean_and_sample_std_column(
    cmapss_compare: ModuleType,
) -> None:
    def seed(staged_rmse: float, joint_rmse: float) -> Any:
        def model(name: str, rmse: float) -> Any:
            metrics = {"rmse": rmse, "coverage": 1.0, "closed_form_crps": 0.0}
            return cmapss_compare.ModelResult(
                name=name, label=name, model=None, metrics={**metrics, "nll": 0.0}
            )

        return cmapss_compare.SeedComparison(
            seed=0, models=(model("staged", staged_rmse), model("joint", joint_rmse))
        )

    table = cmapss_compare.format_comparison_table([seed(1.0, 10.0), seed(3.0, 10.0)])

    header, rmse, *_ = table.splitlines()
    assert header.split() == ["staged", "joint"]
    # Mean 2 and sample std sqrt(2) for the staged model; no spread for the joint.
    assert rmse.split() == ["RMSE", "2.0000", "±", "1.4142", "10.0000", "±", "0.0000"]


@pytest.mark.usefixtures("offline_fd001")
@pytest.mark.parametrize(
    ("seeds_argv", "seeds", "plot_name"),
    [([], 5, None), (["--seeds", "2"], 2, "comparison.png")],
)
def test_main_prints_one_line_per_run_and_a_mean_std_table_then_plots(
    cmapss_run: ModuleType,
    cmapss_compare: ModuleType,
    tiny_cmapss_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plot_recorder: Any,
    seeds_argv: list[str],
    seeds: int,
    plot_name: str | None,
) -> None:
    plot_path = tmp_path / plot_name if plot_name else None
    plot_argv = ["--plot-path", str(plot_path)] if plot_path else []
    monkeypatch.setattr(sys, "argv", ["compare.py", *seeds_argv, *plot_argv])
    monkeypatch.setattr(cmapss_compare, "CmapssConfig", lambda: tiny_cmapss_config)
    prepared_with: list[Any] = []
    prepare = cmapss_compare.prepare_cmapss_windows

    def record_prepare(config: Any) -> Any:
        prepared_with.append(config)
        return prepare(config)

    monkeypatch.setattr(cmapss_compare, "prepare_cmapss_windows", record_prepare)
    compared: list[Any] = []
    compare = cmapss_compare.compare_models

    def record_compare(*args: Any, **kwargs: Any) -> Any:
        results = compare(*args, **kwargs)
        compared.extend(results)
        return results

    monkeypatch.setattr(cmapss_compare, "compare_models", record_compare)
    monkeypatch.setattr(cmapss_compare, "plot_validation_rul_curves", plot_recorder)

    cmapss_compare.main()

    # The data is prepared once, with the configured seed, for every seed.
    assert prepared_with == [tiny_cmapss_config]
    # One figure, drawn from the first seed's models: staged row above joint.
    (plot,) = plot_recorder.calls
    staged, joint = plot.models
    assert staged.label == "Staged model"
    assert joint.label == "Joint model"
    assert staged.model is compared[0].models[0].model
    assert joint.model is compared[0].models[1].model
    assert plot.save_path == plot_path
    assert plot.window_length == tiny_cmapss_config.window_length
    assert plot.feature_columns == cmapss_run._SENSOR_NAMES
    validation = plot.trajectories
    assert plot.units == cmapss_compare.select_lifetime_spanning_units(validation)
    # Every run line and the table are printed before the figure is drawn.
    lines = plot.printed.splitlines()
    for model in ("staged", "joint"):
        run_lines = [line for line in lines if line.startswith(f"[{model} ")]
        expected_seeds = range(tiny_cmapss_config.seed, tiny_cmapss_config.seed + seeds)
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
