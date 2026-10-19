"""Benchmark harness for stagereg's two reference workloads.

Runs the XSin posterior example under pSGLD (mean, variance and posterior
stages, then final scoring) and the CMAPSS FD001 run (mean and variance
stages, then final scoring) through their example entry points, and reports
per stage:

* **time to first step**: from the stage's start (its construction and
  ``prepare``, e.g. the variance stage's residual pass, included) to the end
  of its first training step, compile included. The first step's output is
  blocked on, so the step has finished executing, not just been dispatched.
* **steady s/epoch**: the median duration of every epoch after the first. An
  epoch runs from one epoch's training loader request to the next's, so it
  includes validation and any early-stopping work.
* **early/late s/epoch**: the median of the first and of the last ten
  steady epochs, to show whether epoch time grows as training goes on.
* **warm-up compiles** and **compiles after warm-up**: XLA backend compiles,
  counted through ``jax.monitoring``. Warm-up is the stage's start through the
  end of its first epoch, its validation and ragged last batch included.
* **wall**: the stage's start to the end of its epoch loop.

Final scoring is reported as its own row: its compiles and wall time.

Measurement hooks into the epoch loop that every stage runs (it wraps the
training loader and step that the stage hands to it); the workloads
themselves run unchanged. It is a measuring tool, never a CI gate: timings
are noisy and machine-dependent.

Run both workloads at the examples' default configurations on CPU with:

    JAX_PLATFORMS=cpu uv run --extra jax --group example-cmapss \\
        python benchmarks/harness.py

Pass ``--workload xsin`` or ``--workload cmapss`` to run one. CMAPSS reads
the FD001 archive the example downloads and caches.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.monitoring

from stagereg.jax import posterior_stage, supervised

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "jax"
_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_TREND_WINDOW = 10


@dataclass(frozen=True)
class StageReport:
    """Measurements of one stage's training.

    Attributes:
        stage: Name of the stage, e.g. ``mean``.
        epochs: Number of epochs the stage ran.
        time_to_first_step: Seconds from the stage's start to the end of its
            first training step.
        steady_epoch_time: Median seconds per epoch after the first; ``nan``
            when the stage ran a single epoch.
        early_epoch_time: Median seconds of the first ten steady epochs.
        late_epoch_time: Median seconds of the last ten steady epochs.
        warmup_compiles: Compiles from the stage's start through its first
            epoch.
        compiles_after_warmup: Compiles after the first epoch.
        wall_time: Seconds from the stage's start to the end of its epoch
            loop.
        epoch_times: Seconds of every epoch, in order.
    """

    stage: str
    epochs: int
    time_to_first_step: float
    steady_epoch_time: float
    early_epoch_time: float
    late_epoch_time: float
    warmup_compiles: int
    compiles_after_warmup: int
    wall_time: float
    epoch_times: tuple[float, ...]


@dataclass(frozen=True)
class PhaseReport:
    """Measurements of a workload phase that is not a stage.

    Attributes:
        name: Name of the phase, e.g. ``final scoring``.
        compiles: Compiles during the phase.
        wall_time: Seconds the phase took.
    """

    name: str
    compiles: int
    wall_time: float


@dataclass(frozen=True)
class WorkloadReport:
    """Measurements of one reference workload.

    Attributes:
        name: Name of the workload.
        stages: Report of every stage, in training order.
        scoring: Report of the final scoring after training.
        wall_time: Seconds the whole workload took, data preparation
            excluded.
    """

    name: str
    stages: tuple[StageReport, ...]
    scoring: PhaseReport
    wall_time: float

    def format(self) -> str:
        """Render the report as a plain-text table.

        Returns:
            One header line, one line per stage and one for final scoring.
        """
        headings = (
            "stage",
            "epochs",
            "time to first step",
            "steady s/epoch",
            "early/late s/epoch",
            "warm-up compiles",
            "compiles after warm-up",
            "wall",
        )
        rows = [
            (
                stage.stage,
                str(stage.epochs),
                f"{stage.time_to_first_step:.3f}",
                f"{stage.steady_epoch_time:.4f}",
                f"{stage.early_epoch_time:.4f}/{stage.late_epoch_time:.4f}",
                str(stage.warmup_compiles),
                str(stage.compiles_after_warmup),
                f"{stage.wall_time:.2f}",
            )
            for stage in self.stages
        ]
        rows.append(
            (
                self.scoring.name,
                "-",
                "-",
                "-",
                "-",
                "-",
                str(self.scoring.compiles),
                f"{self.scoring.wall_time:.2f}",
            )
        )
        widths = [
            max(len(row[column]) for row in (headings, *rows))
            for column in range(len(headings))
        ]
        lines = [
            " | ".join(cell.ljust(width) for cell, width in zip(row, widths))
            for row in (headings, *rows)
        ]
        lines.insert(1, "-+-".join("-" * width for width in widths))
        title = f"{self.name} (total wall {self.wall_time:.2f}s)"
        return "\n".join((title, *lines))


@dataclass
class _StageTimeline:
    """Raw timestamps of one stage, filled while its epoch loop runs."""

    stage: str
    start: float
    first_step_end: float = float("nan")
    epoch_starts: list[float] = field(default_factory=list)
    end: float = float("nan")


class _Probe:
    """Records compile times and the epoch-loop timeline of every stage.

    While active, it counts XLA compiles and wraps the epoch loop of every
    supervised and posterior stage. ``mark`` sets the start of the next
    stage; a stage that follows another without a mark starts where the
    previous one's epoch loop ended.
    """

    def __init__(self) -> None:
        self.compile_times: list[float] = []
        self.timelines: list[_StageTimeline] = []
        self._next_start = time.perf_counter()

    def mark(self) -> float:
        """Start the next stage now.

        Returns:
            The current time.
        """
        self._next_start = time.perf_counter()
        return self._next_start

    def compiles_between(self, start: float, end: float) -> int:
        """Count the compiles that finished in ``[start, end)``.

        Args:
            start: Start of the window.
            end: End of the window.

        Returns:
            The number of compiles.
        """
        return sum(start <= moment < end for moment in self.compile_times)

    @contextlib.contextmanager
    def active(self) -> Iterator[_Probe]:
        """Instrument compiles and epoch loops until the block exits.

        Yields:
            This probe.
        """

        def on_compile(event: str, duration: float, **kwargs: Any) -> None:
            del duration, kwargs
            if event == _COMPILE_EVENT:
                self.compile_times.append(time.perf_counter())

        modules = (supervised, posterior_stage)
        originals = [module.run_epoch_loop for module in modules]
        jax.monitoring.register_event_duration_secs_listener(on_compile)
        for module, original in zip(modules, originals):
            module.run_epoch_loop = self._wrap(original)
        try:
            yield self
        finally:
            for module, original in zip(modules, originals):
                module.run_epoch_loop = original
            jax.monitoring.unregister_event_duration_listener(on_compile)

    def _wrap(self, run_epoch_loop: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an epoch loop to record its stage's timeline."""

        def timed(**kwargs: Any) -> Any:
            timeline = _StageTimeline(stage=kwargs["stage"], start=self._next_start)
            self.timelines.append(timeline)
            kwargs["step"] = _time_first_step(kwargs["step"], timeline)
            kwargs["train_loader"] = _time_epochs(kwargs["train_loader"], timeline)
            try:
                return run_epoch_loop(**kwargs)
            finally:
                timeline.end = self.mark()

        return timed

    def stage_reports(self) -> tuple[StageReport, ...]:
        """Summarize every recorded stage.

        Returns:
            One report per stage, in the order they ran.
        """
        return tuple(self._stage_report(timeline) for timeline in self.timelines)

    def _stage_report(self, timeline: _StageTimeline) -> StageReport:
        boundaries = [*timeline.epoch_starts, timeline.end]
        epoch_times = tuple(
            later - earlier for earlier, later in zip(boundaries, boundaries[1:])
        )
        steady = epoch_times[1:]
        warmup_end = boundaries[1] if len(boundaries) > 1 else timeline.end
        return StageReport(
            stage=timeline.stage,
            epochs=len(epoch_times),
            time_to_first_step=timeline.first_step_end - timeline.start,
            steady_epoch_time=_median(steady),
            early_epoch_time=_median(steady[:_TREND_WINDOW]),
            late_epoch_time=_median(steady[-_TREND_WINDOW:]),
            warmup_compiles=self.compiles_between(timeline.start, warmup_end),
            compiles_after_warmup=self.compiles_between(warmup_end, timeline.end),
            wall_time=timeline.end - timeline.start,
            epoch_times=epoch_times,
        )


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _time_first_step(
    step: Callable[..., Mapping[str, Any]], timeline: _StageTimeline
) -> Callable[..., Mapping[str, Any]]:
    """Wrap a batch step to record when its first call finished executing."""

    def timed(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        output = step(*args, **kwargs)
        if math.isnan(timeline.first_step_end):
            jax.block_until_ready(output)
            timeline.first_step_end = time.perf_counter()
        return output

    return timed


def _time_epochs(
    train_loader: Callable[..., Any], timeline: _StageTimeline
) -> Callable[..., Any]:
    """Wrap a training loader to record when each epoch starts."""

    def timed(*, split: str, epoch: int) -> Any:
        timeline.epoch_starts.append(time.perf_counter())
        return train_loader(split=split, epoch=epoch)

    return timed


def _load_example_module(name: str, path: Path) -> ModuleType:
    """Load an example module under ``name``, reusing one already loaded.

    The examples import their sibling modules by bare name, so each is
    registered in ``sys.modules`` under the name its siblings use.
    """
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__ or "").resolve() == path:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the example module {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_xsin() -> ModuleType:
    """Load the shared XSin example module.

    Returns:
        ``examples/jax/xsin/benchmark.py``, with ``XSinConfig`` and
        ``make_xsin_data``.
    """
    return _load_example_module("benchmark", _EXAMPLES_DIR / "xsin" / "benchmark.py")


def load_xsin_posterior() -> ModuleType:
    """Load the XSin posterior example module.

    Returns:
        ``examples/jax/xsin/posterior_benchmark.py``.
    """
    load_xsin()
    return _load_example_module(
        "posterior_benchmark", _EXAMPLES_DIR / "xsin" / "posterior_benchmark.py"
    )


def load_cmapss() -> ModuleType:
    """Load the CMAPSS run example module.

    Returns:
        ``examples/jax/cmapss/run.py``.
    """
    return _load_example_module("cmapss_run", _EXAMPLES_DIR / "cmapss" / "run.py")


def run_xsin(config: Any = None) -> WorkloadReport:
    """Run the XSin posterior example under pSGLD and measure it.

    Trains the mean and variance stages, then a pSGLD posterior stage on
    them, and scores the posterior predictive, as the example does.

    Args:
        config: An ``XSinConfig``. Defaults to ``None``, the example's default
            configuration.

    Returns:
        The report of the mean, variance and posterior stages and final
        scoring.
    """
    shared = load_xsin()
    xsin = load_xsin_posterior()
    config = shared.XSinConfig() if config is None else config
    data = shared.make_xsin_data(config)
    with _Probe().active() as probe, _quiet():
        start = probe.mark()
        supervised_stages = xsin.train_xsin_supervised_stages(data, config)
        probe.mark()
        xsin.run_xsin_posterior(
            data, config, xsin.xsin_psgld(config), supervised=supervised_stages
        )
        end = time.perf_counter()
    # The posterior run scores its predictive after its epoch loop returns.
    scoring_start = probe.timelines[-1].end
    return WorkloadReport(
        name="XSin posterior (pSGLD)",
        stages=probe.stage_reports(),
        scoring=PhaseReport(
            name="final scoring",
            compiles=probe.compiles_between(scoring_start, end),
            wall_time=end - scoring_start,
        ),
        wall_time=end - start,
    )


def run_cmapss(config: Any = None, *, prepared: Any = None) -> WorkloadReport:
    """Run the CMAPSS FD001 example and measure it.

    Trains the mean and Gamma variance stages and scores the composite
    Gaussian on the test split, as the example does; it skips the plot.

    Args:
        config: A ``CmapssConfig``. Defaults to ``None``, the example's
            default configuration.
        prepared: The windowed data from ``prepare_cmapss_windows``. Defaults
            to ``None``, which prepares it from the cached FD001 archive.

    Returns:
        The report of the mean and variance stages and final scoring.
    """
    cmapss = load_cmapss()
    config = cmapss.CmapssConfig() if config is None else config
    if prepared is None:
        prepared = cmapss.prepare_cmapss_windows(config)
    with _Probe().active() as probe, _quiet():
        start = probe.mark()
        mean_model, state = cmapss.train_mean_model(
            prepared.train_windows,
            prepared.train_targets,
            prepared.validation_windows,
            prepared.validation_targets,
            config,
        )
        probe.mark()
        variance_model = cmapss.train_gamma_model(
            state, prepared.train_windows, prepared.train_targets, config
        )
        scoring_start = probe.mark()
        cmapss.evaluate_composite_metrics(
            mean_model, variance_model, prepared.test_windows, prepared.test_rul, config
        )
        end = time.perf_counter()
    return WorkloadReport(
        name="CMAPSS FD001",
        stages=probe.stage_reports(),
        scoring=PhaseReport(
            name="final scoring",
            compiles=probe.compiles_between(scoring_start, end),
            wall_time=end - scoring_start,
        ),
        wall_time=end - start,
    )


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Silence the examples' per-epoch printing."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


_WORKLOADS: dict[str, Callable[[], WorkloadReport]] = {
    "xsin": run_xsin,
    "cmapss": run_cmapss,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected reference workloads and print their reports.

    Args:
        argv: Command-line arguments. Defaults to ``None``, which reads
            ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workload",
        choices=(*_WORKLOADS, "all"),
        default="all",
        help="Reference workload to run (default: all).",
    )
    args = parser.parse_args(argv)
    names = list(_WORKLOADS) if args.workload == "all" else [args.workload]
    print(f"jax {jax.__version__} on {jax.default_backend()}")
    for name in names:
        print()
        print(_WORKLOADS[name]().format(), flush=True)


if __name__ == "__main__":
    main()
