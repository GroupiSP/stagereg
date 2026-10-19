"""Smoke test of the benchmark harness in ``benchmarks/harness.py``.

The harness lives outside the package and outside pytest collection; this test only
checks that it runs both reference workloads end to end and reports every measurement
for every stage. It never asserts a timing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_HARNESS_PATH = Path(__file__).parents[2] / "benchmarks" / "harness.py"


@pytest.fixture(scope="module")
def harness(cmapss_run: ModuleType) -> ModuleType:
    """The benchmark harness, loaded from its file path."""
    spec = importlib.util.spec_from_file_location("stagereg_harness", _HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the benchmark harness.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_fully_reported(report: object, stages: tuple[str, ...]) -> None:
    assert tuple(stage.stage for stage in report.stages) == stages
    for stage in report.stages:
        assert stage.epochs >= 2
        assert len(stage.epoch_times) == stage.epochs
        for name in (
            "time_to_first_step",
            "steady_epoch_time",
            "wall_time",
        ):
            value = getattr(stage, name)
            assert math.isfinite(value) and value >= 0.0, name
        assert stage.warmup_compiles >= 0
        assert stage.compiles_after_warmup >= 0
    assert math.isfinite(report.scoring.wall_time)
    assert report.scoring.compiles >= 0
    assert math.isfinite(report.wall_time)

    table = report.format()
    for heading in (
        "time to first step",
        "steady s/epoch",
        "compiles after warm-up",
        "wall",
    ):
        assert heading in table
    for stage in (*stages, "final scoring"):
        assert stage in table


@pytest.mark.usefixtures("offline_fd001")
def test_harness_reports_every_measurement_for_both_reference_workloads(
    harness: ModuleType, cmapss_run: ModuleType
) -> None:
    xsin_config = harness.load_xsin().XSinConfig(
        train_size=40,
        evaluation_size=21,
        batch_size=16,
        hidden_features=8,
        mean_epochs=3,
        variance_epochs=3,
        posterior_epochs=3,
        posterior_num_draws=4,
        psgld_thinning=1,
    )
    cmapss_config = cmapss_run.CmapssConfig(
        window_length=10,
        batch_size=64,
        hidden_channels=4,
        kernel_size=3,
        mean_epochs=3,
        variance_epochs=3,
        predictive_sample_count=16,
    )

    xsin = harness.run_xsin(xsin_config)
    cmapss = harness.run_cmapss(
        cmapss_config, prepared=cmapss_run.prepare_cmapss_windows(cmapss_config)
    )

    _assert_fully_reported(xsin, ("mean", "variance", "posterior"))
    _assert_fully_reported(cmapss, ("mean", "variance"))
