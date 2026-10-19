"""Shared fixtures for the JAX backend's and examples' tests.

The `make_stages` and `variance_ready_run` fixtures build and run the real
mean and variance stages every posterior-stage test starts from; their data,
loader and types live in `posterior_runs.py`. The `compile_counter` fixture
counts XLA compiles per stage, so a test can assert that a stage compiles
nothing after its first epoch.

The examples live under `examples/`, not in the installed package, so
their modules are loaded from their file paths. That loading happened
three times over, once per test module; it happens here once instead,
behind one session-scoped fixture per module. Session scope also keeps
Hypothesis happy: a function-scoped fixture in a `@given` test trips its
`function_scoped_fixture` health check.

Each fixture guards its own optional dependencies, so a test module is
collected even when the extras it needs are absent and only the tests that
actually touch that module skip.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

import numpy as np
import pytest

from stagereg.core.tracking import TrainingEvent

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "jax"
_CMAPSS_DIR = _EXAMPLES_DIR / "cmapss"
_TRACKING_DIR = _EXAMPLES_DIR / "tracking"
if str(_CMAPSS_DIR) not in sys.path:
    sys.path.insert(0, str(_CMAPSS_DIR))


def _load_example_module(
    directory: Path, module_name: str, file_name: str
) -> ModuleType:
    """Load one of the examples' modules from its file path.

    Args:
        directory: The example's directory.
        module_name: Name to register the loaded module under, so that
            anything importing it a second time gets the same object.
        file_name: The module's file name within the example's directory.

    Returns:
        The executed module.

    Raises:
        RuntimeError: If the module could not be loaded from its path.
    """
    spec = importlib.util.spec_from_file_location(module_name, directory / file_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the example module {file_name!r}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cmapss_preprocessing() -> ModuleType:
    """The example's windowing and standardization module."""
    for dependency in ("pandas", "sklearn"):
        pytest.importorskip(dependency)
    return _load_example_module(_CMAPSS_DIR, "cmapss_preprocessing", "preprocessing.py")


@pytest.fixture(scope="session")
def cmapss_plots() -> ModuleType:
    """The example's RUL-curve plotting module."""
    for dependency in ("matplotlib", "pandas", "sklearn"):
        pytest.importorskip(dependency)
    return _load_example_module(_CMAPSS_DIR, "cmapss_plots", "plots.py")


@pytest.fixture(scope="session")
def cmapss_run() -> ModuleType:
    """The example's end-to-end training and evaluation module."""
    for dependency in (
        "jax",
        "jax.numpy",
        "flax.nnx",
        "optax",
        "pandas",
        "matplotlib",
        "sklearn",
    ):
        pytest.importorskip(dependency)
    return _load_example_module(_CMAPSS_DIR, "cmapss_run", "run.py")


@pytest.fixture
def offline_fd001(cmapss_run: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve synthetic FD001 splits in place of the cached NASA archive."""
    from cmapss_trajectories import build_random_trajectories

    rng = np.random.default_rng(7)
    sensor_names = cmapss_run._SENSOR_NAMES
    train_data = build_random_trajectories(
        rng, unit_ids=range(1, 11), n_cycles=60, feature_columns=sensor_names
    )
    test_data = build_random_trajectories(
        rng, unit_ids=range(1, 6), n_cycles=45, feature_columns=sensor_names
    )
    test_rul = rng.uniform(low=1.0, high=100.0, size=5)
    monkeypatch.setattr(cmapss_run, "load_fd001_data", lambda: train_data)
    monkeypatch.setattr(cmapss_run, "load_fd001_test_data", lambda: test_data)
    monkeypatch.setattr(cmapss_run, "load_fd001_test_rul", lambda: test_rul)


@pytest.fixture(scope="session")
def spanning_units(
    cmapss_preprocessing: ModuleType,
) -> Callable[[int, int, int], Any]:
    """Return a factory naming three units in figure column order."""

    def make(shortest: int, median: int, longest: int) -> Any:
        return cmapss_preprocessing.LifetimeSpanningUnits(
            shortest=shortest, median=median, longest=longest
        )

    return make


@pytest.fixture(scope="session")
def tracking_tracker() -> ModuleType:
    """The tracking example's TensorBoard experiment tracker module.

    Deliberately unguarded: the tracker's mapping logic is testable with
    neither `tensorboardX` nor JAX installed, and a skip here would hide
    that property breaking.
    """
    return _load_example_module(
        _TRACKING_DIR, "tracking_tensorboard_tracker", "tensorboard_tracker.py"
    )


@pytest.fixture(scope="session")
def tracking_run() -> ModuleType:
    """The tracking example's end-to-end run module."""
    for dependency in (
        "jax",
        "jax.numpy",
        "flax.nnx",
        "optax",
        "matplotlib",
        "tensorboardX",
    ):
        pytest.importorskip(dependency)
    return _load_example_module(_TRACKING_DIR, "tracking_run", "run.py")


@pytest.fixture(scope="session")
def make_stages(linear_model: type[Any], squared_error: Any) -> Callable[..., Any]:
    """Return a factory of mean and variance stages on `regression_loader`.

    See `posterior_runs.build_stages` for its arguments.
    """
    import posterior_runs

    def make(*sinks: Any, **options: Any) -> Any:
        return posterior_runs.build_stages(
            linear_model, squared_error, *sinks, **options
        )

    return make


@pytest.fixture(scope="session")
def variance_ready_run(make_stages: Callable[..., Any]) -> Callable[..., Any]:
    """Return a driver of a real mean-then-variance run on `regression_loader`.

    It takes the event sinks of both stages, ``mean_only`` to stop after the
    mean stage, and the ``checkpoint_store`` both stages save and finalize
    their best checkpoints into.
    """
    import posterior_runs

    def run(*sinks: Any, mean_only: bool = False, checkpoint_store: Any = None) -> Any:
        stages = make_stages(*sinks, checkpoint_store=checkpoint_store)
        return posterior_runs.run_to_variance_ready(stages, mean_only=mean_only)

    return run


class CompileCounter:
    """Counts XLA backend compiles, per stage, while it is active.

    It is a context manager and an event sink: enter it around a run and
    attach it to every stage's ``event_sinks``. Inside the block it counts one
    compile per XLA backend compile, eager operations included, through
    `jax.monitoring`. As a sink, it notes the running count at every event of
    every stage, which places each stage's warm-up boundary.

    A stage's warm-up is its first epoch: training, the ragged last batch,
    and any validation and early-stopping decision of epoch 0. Its boundary
    is the stage's last event with ``step == 0``. Compiles after warm-up are
    those between that boundary and the stage's last event, which comes after
    its last validation. Warm-up counts from the previous stage's last event,
    or from entering the block for the first stage.

    Example:
        ```python
        with compile_counter() as counter:
            stage = MeanStage(..., options=SupervisedStageOptions(
                ..., event_sinks=(counter,)))
            stage.prepare(state)
            stage.train(state)
        assert counter.during_warmup("mean") > 0
        assert counter.after_warmup("mean") == 0
        ```

    Attributes:
        total: Compiles counted so far while active.
    """

    _EVENT = "/jax/core/compile/backend_compile_duration"

    def __init__(self) -> None:
        self.total = 0
        self._start: dict[str, int] = {}
        self._warmup_end: dict[str, int] = {}
        self._last: dict[str, int] = {}

    def __enter__(self) -> Self:
        import jax.monitoring

        jax.monitoring.register_event_duration_secs_listener(self._on_duration)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        import jax.monitoring

        jax.monitoring.unregister_event_duration_listener(self._on_duration)

    def _on_duration(self, event: str, duration: float, **kwargs: Any) -> None:
        del duration, kwargs
        if event == self._EVENT:
            self.total += 1

    def on_event(self, event: TrainingEvent) -> None:
        """Note the running compile count at a stage's event.

        Args:
            event: The event a stage emitted.
        """
        if event.stage not in self._start:
            self._start[event.stage] = max(self._last.values(), default=0)
        if event.step == 0:
            self._warmup_end[event.stage] = self.total
        self._last[event.stage] = self.total

    def during_warmup(self, stage: str) -> int:
        """Count a stage's compiles up to the end of its first epoch.

        Args:
            stage: The stage's name, as its events carry it.

        Returns:
            The compiles from the stage's start through its warm-up boundary.
        """
        return self._warmup_end[stage] - self._start[stage]

    def after_warmup(self, stage: str) -> int:
        """Count a stage's compiles after its first epoch.

        Args:
            stage: The stage's name, as its events carry it.

        Returns:
            The compiles between its warm-up boundary and its last event.
        """
        return self._last[stage] - self._warmup_end[stage]


@pytest.fixture(scope="session")
def compile_counter() -> type[CompileCounter]:
    """Return `CompileCounter`, to build one fresh counter per run.

    Session-scoped, like every factory fixture here, so `@given` tests may use it; each
    test builds and enters its own counter.
    """
    pytest.importorskip("jax")
    return CompileCounter
