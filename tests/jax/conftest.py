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

The clustered-data example's modules import each other by bare names unique
across the examples (`regions`, `scoring`, `compare`, ...); its fixtures
import them under those same names, so tests and the entry point share one
module object.

Each fixture guards its own optional dependencies, so a test module is
collected even when the extras it needs are absent and only the tests that
actually touch that module skip.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

import pytest

from stagereg.core.tracking import TrainingEvent

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "jax"
_TRACKING_DIR = _EXAMPLES_DIR / "tracking"
_CLUSTERED_DATA_DIR = _EXAMPLES_DIR / "clustered_data"


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


def _import_clustered_data_module(name: str) -> ModuleType:
    """Import one of the clustered-data example's modules by its bare name.

    The example's modules import each other by bare name, so they are
    imported the same way here, with the example's directory on `sys.path`:
    a test and the entry point then share one module object, and one set of
    classes.

    Args:
        name: The module's bare name, e.g. `"scoring"`.

    Returns:
        The imported module.
    """
    if str(_CLUSTERED_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(_CLUSTERED_DATA_DIR))
    return importlib.import_module(name)


@pytest.fixture(scope="session")
def clustered_regions() -> ModuleType:
    """The clustered-data example's region masks."""
    pytest.importorskip("jax")
    return _import_clustered_data_module("regions")


@pytest.fixture(scope="session")
def clustered_scoring() -> ModuleType:
    """The clustered-data example's shared scoring of posterior predictives."""
    pytest.importorskip("jax")
    return _import_clustered_data_module("scoring")


@pytest.fixture(scope="session")
def clustered_compare() -> ModuleType:
    """The clustered-data example's comparison entry point."""
    for dependency in ("jax", "jax.numpy", "flax.nnx", "optax", "matplotlib"):
        pytest.importorskip(dependency)
    return _import_clustered_data_module("compare")


@pytest.fixture(scope="session")
def clustered_figures() -> ModuleType:
    """The clustered-data example's uncertainty grid and calibration figures."""
    for dependency in ("jax", "flax.nnx", "optax", "matplotlib"):
        pytest.importorskip(dependency)
    return _import_clustered_data_module("figures")


@pytest.fixture(scope="session")
def clustered_dataset() -> ModuleType:
    """The clustered-data example's configuration, data and loader."""
    pytest.importorskip("jax")
    return _import_clustered_data_module("clustered")


@pytest.fixture(scope="session")
def tiny_clustered_config(clustered_dataset: ModuleType) -> Any:
    """A clustered-data configuration whose every run trains in a few seconds."""
    return clustered_dataset.ClusteredDataConfig(
        mean_epochs=3,
        variance_epochs=3,
        posterior_epochs=2,
        num_draws=4,
        psgld_chains=2,
        seed=3,
    )


@pytest.fixture(scope="session")
def clustered_data(clustered_dataset: ModuleType, tiny_clustered_config: Any) -> Any:
    """The clustered training set and evaluation grid.

    They depend on the noise level alone, which `tiny_clustered_config`
    keeps at its default.
    """
    return clustered_dataset.make_clustered_data(tiny_clustered_config)


@pytest.fixture(scope="session")
def clustered_staged() -> ModuleType:
    """The clustered-data example's staged model."""
    for dependency in ("jax", "jax.numpy", "flax.nnx", "optax"):
        pytest.importorskip(dependency)
    return _import_clustered_data_module("staged")


@pytest.fixture(scope="session")
def clustered_joint() -> ModuleType:
    """The clustered-data example's joint posterior."""
    for dependency in ("jax", "jax.numpy", "flax.nnx", "optax"):
        pytest.importorskip(dependency)
    return _import_clustered_data_module("joint")


@pytest.fixture(scope="session")
def clustered_methods() -> ModuleType:
    """The clustered-data example's inference methods."""
    for dependency in ("jax", "jax.numpy", "flax.nnx", "optax"):
        pytest.importorskip(dependency)
    return _import_clustered_data_module("methods")


@pytest.fixture(scope="session")
def clustered_networks() -> ModuleType:
    """The clustered-data example's shared backbone and networks."""
    for dependency in ("jax", "jax.numpy", "flax.nnx"):
        pytest.importorskip(dependency)
    return _import_clustered_data_module("networks")
