"""Smoke test of the benchmark harness in ``benchmarks/harness.py``.

The harness lives outside the package and outside pytest collection; this test only
checks that it runs the reference workload end to end and reports every measurement for
every stage. It never asserts a timing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_TESTS_DIR = Path(__file__).parent
_HARNESS_PATH = _TESTS_DIR.parent / "benchmarks" / "harness.py"


def _load(name: str, path: Path) -> ModuleType:
    """Load a module that lives outside the package from its file path.

    Args:
        name: Name to register the module under.
        path: The module's file.

    Returns:
        The executed module.

    Raises:
        RuntimeError: If the module could not be loaded from its path.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    """The benchmark harness, loaded from its file path."""
    for dependency in ("jax", "flax.nnx", "optax"):
        pytest.importorskip(dependency)
    return _load("stagereg_harness", _HARNESS_PATH)


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


def test_harness_reports_every_measurement_for_the_reference_workload(
    harness: ModuleType,
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

    xsin = harness.run_xsin(xsin_config)

    _assert_fully_reported(xsin, ("mean", "variance", "posterior"))
