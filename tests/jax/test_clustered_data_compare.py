"""End-to-end smoke tests of the clustered-data comparison entry point.

They train at tiny epochs and draws, and assert only what holds for any
training run: the shape of the results and the table, and metrics in range.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_TINY_FLAGS = (
    "--mean-epochs",
    "3",
    "--variance-epochs",
    "3",
    "--posterior-epochs",
    "2",
    "--num-draws",
    "4",
    "--seed",
    "3",
)
_NUMBER = r"\d+\.\d+"
_CONFIGURATION_NAMES = [
    "joint BBB",
    "joint MCD",
    "joint pSGLD",
    "staged BBB",
    "staged MCD",
    "staged pSGLD",
]


@pytest.fixture(autouse=True)
def shown_figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[bool]]:
    """Record `plt.show` calls instead of opening a window."""
    pyplot = pytest.importorskip("matplotlib.pyplot")
    shown: list[bool] = []
    monkeypatch.setattr(pyplot, "show", lambda: shown.append(True))
    yield shown
    pyplot.close("all")


def test_compare_groups_joint_and_staged_results_per_seed(
    clustered_compare: ModuleType,
    clustered_regions: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
) -> None:
    results = clustered_compare.compare(clustered_data, tiny_clustered_config, seeds=2)

    assert [result.seed for result in results] == [3, 4]
    for result in results:
        names = [configuration.name for configuration in result.configurations]
        assert names == _CONFIGURATION_NAMES
        for configuration in result.configurations:
            # pSGLD retains exactly num_draws samples, like the others draw.
            means = configuration.draws.means
            assert means.shape == (4, clustered_data.grid_inputs.shape[0])
            assert configuration.draws.variances.shape == means.shape
            assert set(configuration.scores.regions) == set(clustered_regions.REGIONS)
            for region in configuration.scores.regions.values():
                assert 0.0 <= region.coverage <= 1.0
                assert region.aleatoric_std >= 0.0
                assert region.epistemic_std > 0.0


def test_joint_runs_train_as_many_epochs_as_the_mean_stage(
    clustered_compare: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dataclasses.replace(tiny_clustered_config, mean_epochs=4)
    budgets = []
    run_joint = clustered_compare.run_joint

    def recording_run_joint(*args: Any, epochs: int, **kwargs: Any) -> Any:
        budgets.append(epochs)
        return run_joint(*args, epochs=epochs, **kwargs)

    monkeypatch.setattr(clustered_compare, "run_joint", recording_run_joint)
    clustered_compare.compare(clustered_data, config, seeds=1)

    # Neither the variance stage's 3 epochs nor the posterior stage's 2 add up.
    assert budgets == [4, 4, 4]


def test_joint_runs_share_each_methods_hyperparameters_and_draw_schedule(
    clustered_compare: ModuleType,
    clustered_data: Any,
    tiny_clustered_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_runs: dict[str, Any] = {}
    staged_runs: dict[str, Any] = {}
    run_joint, run_staged = clustered_compare.run_joint, clustered_compare.run_staged

    def recording_run_joint(data: Any, config: Any, method: Any, **kwargs: Any) -> Any:
        draws = run_joint(data, config, method, **kwargs)
        joint_runs[type(method).__name__] = (method, kwargs["network"])
        return draws

    def recording_run_staged(*args: Any, **kwargs: Any) -> Any:
        draws = run_staged(*args, **kwargs)
        staged_runs[type(args[3]).__name__] = (args[3], kwargs["network"])
        return draws

    monkeypatch.setattr(clustered_compare, "run_joint", recording_run_joint)
    monkeypatch.setattr(clustered_compare, "run_staged", recording_run_staged)
    clustered_compare.compare(clustered_data, tiny_clustered_config, seeds=1)

    assert (
        set(joint_runs)
        == set(staged_runs)
        == {
            "BayesByBackprop",
            "MCDropout",
            "PreconditionedSGLD",
        }
    )
    for name, (_, joint_network) in joint_runs.items():
        staged_network = staged_runs[name][1]
        joint_rates = [dropout.rate for dropout in joint_network.trunk.dropouts]
        staged_rates = [dropout.rate for dropout in staged_network.trunk.dropouts]
        assert joint_rates == staged_rates
    assert joint_runs["MCDropout"][1].trunk.dropouts[0].rate == 0.1
    joint_psgld, staged_psgld = (
        joint_runs["PreconditionedSGLD"][0],
        staged_runs["PreconditionedSGLD"][0],
    )
    # The joint run's 3 epochs of 10 steps put 4 draws 5 steps apart over the
    # final 20; the posterior stage's 2 epochs, 2 apart over the final 10.
    assert (joint_psgld.burn_in, joint_psgld.thinning) == (10, 5)
    assert (staged_psgld.burn_in, staged_psgld.thinning) == (12, 2)
    assert joint_psgld.step_size == staged_psgld.step_size


def test_main_prints_a_row_per_configuration_with_every_region(
    clustered_compare: ModuleType,
    clustered_regions: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", *_TINY_FLAGS])

    clustered_compare.main()

    lines = capsys.readouterr().out.strip().splitlines()
    title = next(i for i, line in enumerate(lines) if line.startswith("Clustered"))
    header, *rows = lines[title + 1 :]
    assert header.split()[:3] == ["ID", "RMSE", "ID"]
    assert [row.split("  ")[0] for row in rows] == _CONFIGURATION_NAMES
    for row in rows:
        cells = re.findall(_NUMBER, row)
        assert len(cells) == 1 + 3 * len(clustered_regions.REGIONS)
        coverages = [float(cell) for cell in cells[3::3]]
        assert len(coverages) == 3
        assert all(0.0 <= coverage <= 1.0 for coverage in coverages)


def test_main_aggregates_every_configuration_over_two_seeds(
    clustered_compare: ModuleType,
    clustered_regions: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", *_TINY_FLAGS, "--seeds", "2"])

    clustered_compare.main()

    lines = capsys.readouterr().out.strip().splitlines()
    runs = [re.match(r"\[(.+) seed=(\d+)\]", line) for line in lines]
    runs_per_seed: dict[str, list[str]] = {}
    for run in filter(None, runs):
        runs_per_seed.setdefault(run[2], []).append(run[1])
    assert set(runs_per_seed) == {"3", "4"}
    assert runs_per_seed["3"] == runs_per_seed["4"]
    header_index = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("ID RMSE")
    )
    rows = lines[header_index + 1 :]
    assert [row.split("  ")[0].strip() for row in rows] == runs_per_seed["3"]
    for row in rows:
        cells = re.findall(rf"{_NUMBER} ± {_NUMBER}", row)
        assert len(cells) == 1 + 3 * len(clustered_regions.REGIONS)


def test_main_rejects_a_non_positive_number_of_seeds(
    clustered_compare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", "--seeds", "0"])

    with pytest.raises(SystemExit):
        clustered_compare.main()


@pytest.mark.parametrize(
    "arguments",
    [("--num-draws", "0"), ("--mc-dropout-rate", "1.0")],
)
def test_main_rejects_invalid_settings(
    clustered_compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", *arguments])

    with pytest.raises(SystemExit):
        clustered_compare.main()


def test_supervised_stages_run_their_full_epochs_without_stopping_early(
    clustered_staged: ModuleType, clustered_dataset: ModuleType
) -> None:
    # A learning rate this high makes the training loss oscillate, so an
    # early stopper would stop and restore an early best epoch.
    config = clustered_dataset.ClusteredDataConfig(
        learning_rate=0.3, mean_epochs=20, variance_epochs=15, seed=3
    )
    data = clustered_dataset.make_clustered_data(config)

    supervised = clustered_staged.train_supervised_stages(data, config)

    history = supervised.state.metric_history
    assert len(history["mean/train/loss"]) == 20
    assert len(history["variance/train/loss"]) == 15


def test_main_saves_both_figures_into_the_plot_path_without_showing(
    clustered_compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shown_figures: list[bool],
) -> None:
    plot_path = tmp_path / "figures"
    monkeypatch.setattr(
        sys, "argv", ["compare.py", *_TINY_FLAGS, "--plot-path", str(plot_path)]
    )

    clustered_compare.main()

    for name in ("uncertainty_grid.png", "extrapolation_calibration.png"):
        assert (plot_path / name).is_file()
        assert (plot_path / name).stat().st_size > 0
    assert shown_figures == []


def test_main_draws_every_configuration_run_in_both_figures(
    clustered_compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    drawn: list[Any] = []
    draw_figures = clustered_compare.draw_figures

    def recording_draw_figures(*args: Any, **kwargs: Any) -> Any:
        figures = draw_figures(*args, **kwargs)
        drawn.append(figures)
        return figures

    monkeypatch.setattr(clustered_compare, "draw_figures", recording_draw_figures)
    monkeypatch.setattr(
        sys, "argv", ["compare.py", *_TINY_FLAGS, "--plot-path", str(tmp_path)]
    )

    clustered_compare.main()

    ((grid, calibration),) = drawn
    titles_with_lines = {axes.get_title() for axes in grid.axes if axes.lines}
    assert titles_with_lines == set(_CONFIGURATION_NAMES)
    # One curve per method, plus the diagonal, in both trainings' panels.
    assert [len(axes.lines) for axes in calibration.axes] == [4, 4]


def test_main_shows_the_figures_without_a_plot_path(
    clustered_compare: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    shown_figures: list[bool],
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", *_TINY_FLAGS])

    clustered_compare.main()

    assert shown_figures == [True]
