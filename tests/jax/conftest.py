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

The CMAPSS tests share a tiny configuration (`tiny_cmapss_config`),
synthetic windows and prepared data (`synthetic_windows`,
`synthetic_prepared_data`), and a recording stand-in for the entry points'
plotting (`plot_recorder`).

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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, Self

import numpy as np
import pytest

from stagereg.core.tracking import TrainingEvent

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "jax"
_CMAPSS_DIR = _EXAMPLES_DIR / "cmapss"
_TRACKING_DIR = _EXAMPLES_DIR / "tracking"
_CLUSTERED_DATA_DIR = _EXAMPLES_DIR / "clustered_data"
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


@pytest.fixture(scope="session")
def cmapss_compare(cmapss_run: ModuleType) -> ModuleType:
    """The example's staged-vs-joint comparison module.

    The comparison imports its sibling as `run`. That name is aliased to the
    already loaded `cmapss_run`, so both share one module (and one set of
    monkeypatchable FD001 loaders), and the tracking example's own `run.py`,
    which also puts its directory on `sys.path`, can never shadow it.
    """
    sys.modules["run"] = cmapss_run
    return _load_example_module(_CMAPSS_DIR, "cmapss_compare", "compare.py")


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
def tiny_cmapss_config(cmapss_run: ModuleType) -> Any:
    """A CMAPSS configuration small enough to train every model in a second or two."""
    return cmapss_run.CmapssConfig(
        window_length=10,
        batch_size=16,
        hidden_channels=2,
        kernel_size=3,
        mean_epochs=2,
        variance_epochs=2,
        gamma_hidden_channels=2,
        joint_epochs=2,
        seed=3,
    )


def _synthetic_windows(
    rng: np.random.Generator,
    n_windows: int,
    *,
    window_length: int = 10,
    n_sensors: int = 3,
    max_rul: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw standard-normal windows and uniform RUL targets in `[1, max_rul)`.

    Args:
        rng: The generator to draw from.
        n_windows: Number of windows to draw.
        window_length: Number of cycles per window.
        n_sensors: Number of sensor channels per window.
        max_rul: Upper bound of the uniform RUL targets.

    Returns:
        The windows, shape `(n_windows, window_length, n_sensors)`, and
        their targets, shape `(n_windows,)`.
    """
    windows = rng.normal(size=(n_windows, window_length, n_sensors))
    targets = rng.uniform(low=1.0, high=max_rul, size=n_windows)
    return windows, targets


@pytest.fixture(scope="session")
def synthetic_windows() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Return a factory of random CMAPSS-shaped windows and RUL targets.

    It is called as `synthetic_windows(rng, n_windows, *, window_length=10, n_sensors=3,
    max_rul=100.0)`.
    """
    return _synthetic_windows


@pytest.fixture(scope="session")
def synthetic_prepared_data(cmapss_run: ModuleType) -> Callable[..., Any]:
    """Return a factory of small synthetic `PreparedCmapssData`.

    It is called as `synthetic_prepared_data(*, seed=0, max_rul=100.0)`. Only the
    windowed arrays (32 train, 8 validation and 6 test windows of length 10 over 3
    sensors) are meant to be trained on. The training trajectories hold four units with
    random lifetimes in `[1, max_rul]` and zero sensors, only so that a proxy-variance
    schedule can be fitted to their lifetimes; the validation trajectories are empty. As
    in the real data, the train and validation targets are standardized with statistics
    fitted on the train targets, and the test RUL stays in cycles.
    """
    import pandas as pd

    def make(*, seed: int = 0, max_rul: float = 100.0) -> Any:
        rng = np.random.default_rng(seed)
        train_windows, train_targets = _synthetic_windows(rng, 32, max_rul=max_rul)
        validation_windows, validation_targets = _synthetic_windows(
            rng, 8, max_rul=max_rul
        )
        test_windows, test_rul = _synthetic_windows(rng, 6, max_rul=max_rul)
        target_stats = cmapss_run.fit_target_standardization(train_targets)
        n_sensors = train_windows.shape[-1]
        sensors = [f"sensor_{i}" for i in range(n_sensors)]
        columns = ["unit_id", "time_cycles", *sensors]
        lifetimes = rng.integers(1, int(max_rul), endpoint=True, size=4)
        train_trajectories = pd.DataFrame(
            [
                {"unit_id": unit, "time_cycles": cycle, **dict.fromkeys(sensors, 0.0)}
                for unit, lifetime in enumerate(lifetimes, start=1)
                for cycle in range(1, int(lifetime) + 1)
            ],
            columns=columns,
        )
        return cmapss_run.PreparedCmapssData(
            train_windows=train_windows,
            train_targets=target_stats.standardize(train_targets),
            validation_windows=validation_windows,
            validation_targets=target_stats.standardize(validation_targets),
            test_windows=test_windows,
            test_rul=test_rul,
            train_trajectories=train_trajectories,
            validation_trajectories=pd.DataFrame(columns=columns),
            standardization=cmapss_run.SensorStandardization(
                feature_columns=sensors,
                mean=np.zeros(n_sensors),
                scale=np.ones(n_sensors),
            ),
            target_standardization=target_stats,
        )

    return make


@dataclass(frozen=True)
class RecordedPlot:
    """One call of a CMAPSS entry point's `plot_validation_rul_curves`.

    Attributes:
        trajectories: The validation trajectories handed over.
        feature_columns: The feature columns handed over.
        models: The labelled models, one per figure row.
        units: The three units to draw.
        window_length: The window length handed over.
        save_path: Where the figure would be saved, or `None` to display it.
        printed: Everything printed to stdout before the call.
    """

    trajectories: Any
    feature_columns: list[str]
    models: list[Any]
    units: Any
    window_length: int
    save_path: Path | None
    printed: str


class PlotRecorder:
    """A stand-in for `plot_validation_rul_curves` that records its calls.

    Attributes:
        calls: The calls recorded so far, in order.
    """

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Start with no recorded calls.

        Args:
            capsys: The capture whose output so far each call records.
        """
        self._capsys = capsys
        self.calls: list[RecordedPlot] = []

    def __call__(
        self,
        trajectories: Any,
        feature_columns: list[str],
        models: Sequence[Any],
        *,
        units: Any,
        window_length: int,
        save_path: Path | None,
    ) -> None:
        """Record one plotting call and whatever was printed before it.

        Args:
            trajectories: The validation trajectories to draw from.
            feature_columns: The models' feature columns.
            models: The labelled models, one per figure row.
            units: The three units to draw.
            window_length: Number of cycles per window.
            save_path: Where to save the figure, or `None` to display it.
        """
        self.calls.append(
            RecordedPlot(
                trajectories=trajectories,
                feature_columns=feature_columns,
                models=list(models),
                units=units,
                window_length=window_length,
                save_path=save_path,
                printed=self._capsys.readouterr().out,
            )
        )


@pytest.fixture
def plot_recorder(capsys: pytest.CaptureFixture[str]) -> PlotRecorder:
    """Return a fresh `PlotRecorder` to monkeypatch over an entry point's plotting."""
    return PlotRecorder(capsys)


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
