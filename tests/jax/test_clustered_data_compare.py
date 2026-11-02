"""End-to-end smoke tests of the clustered-data comparison entry point.

They train at tiny epochs and draws, and assert only what holds for any
training run: the shape of the results and the table, and metrics in range.
"""

from __future__ import annotations

import re
import sys
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
_STAGED_NAMES = ["staged BBB", "staged MCD", "staged pSGLD"]


@pytest.fixture(scope="session")
def tiny_clustered_config(clustered_dataset: ModuleType) -> Any:
    """A clustered-data configuration that trains in a few seconds."""
    return clustered_dataset.ClusteredDataConfig(
        mean_epochs=3, variance_epochs=3, posterior_epochs=2, num_draws=4, seed=3
    )


def test_compare_groups_joint_and_staged_results_per_seed(
    clustered_compare: ModuleType,
    clustered_dataset: ModuleType,
    clustered_regions: ModuleType,
    tiny_clustered_config: Any,
) -> None:
    data = clustered_dataset.make_clustered_data(tiny_clustered_config)

    results = clustered_compare.compare(data, tiny_clustered_config, seeds=2)

    assert [result.seed for result in results] == [3, 4]
    for result in results:
        names = [configuration.name for configuration in result.configurations]
        assert names == ["joint BBB", *_STAGED_NAMES]
        for configuration in result.configurations:
            # pSGLD retains exactly num_draws samples, like the others draw.
            means = configuration.draws.means
            assert means.shape == (4, data.grid_inputs.shape[0])
            assert configuration.draws.variances.shape == means.shape
            assert set(configuration.scores.regions) == set(clustered_regions.REGIONS)
            for region in configuration.scores.regions.values():
                assert 0.0 <= region.coverage <= 1.0
                assert region.aleatoric_std >= 0.0
                assert region.epistemic_std > 0.0
        assert result.point_epochs == 6  # patience outlasts both stages


def test_joint_runs_get_the_staged_runs_total_epochs(
    clustered_compare: ModuleType,
    clustered_dataset: ModuleType,
    tiny_clustered_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets = []
    run_joint = clustered_compare.run_joint

    def recording_run_joint(*args: Any, epochs: int, **kwargs: Any) -> Any:
        budgets.append(epochs)
        return run_joint(*args, epochs=epochs, **kwargs)

    monkeypatch.setattr(clustered_compare, "run_joint", recording_run_joint)
    data = clustered_dataset.make_clustered_data(tiny_clustered_config)

    (result,) = clustered_compare.compare(data, tiny_clustered_config, seeds=1)

    assert budgets
    # Mean and variance stages run 3 epochs each, the posterior stage 2.
    assert all(budget == 3 + 3 + 2 for budget in budgets)
    assert result.point_epochs == 6


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
    assert [row.split("  ")[0] for row in rows] == ["joint BBB", *_STAGED_NAMES]
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


def test_point_epochs_count_every_epoch_run_not_only_up_to_the_best(
    clustered_staged: ModuleType, clustered_dataset: ModuleType
) -> None:
    # A learning rate this high makes the training loss oscillate, so the best
    # epoch comes early; the patience still outlasts both stages' maximum.
    config = clustered_dataset.ClusteredDataConfig(
        learning_rate=0.3, mean_epochs=20, variance_epochs=20, seed=3
    )
    data = clustered_dataset.make_clustered_data(config)

    supervised = clustered_staged.train_supervised_stages(data, config)

    assert supervised.point_epochs == 40
