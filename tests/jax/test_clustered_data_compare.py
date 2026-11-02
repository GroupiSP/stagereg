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


@pytest.fixture(scope="session")
def tiny_clustered_config(clustered_dataset: ModuleType) -> Any:
    """A clustered-data configuration that trains in a few seconds."""
    return clustered_dataset.ClusteredDataConfig(
        mean_epochs=3, variance_epochs=3, posterior_epochs=2, num_draws=4, seed=3
    )


def test_compare_groups_staged_bbb_results_per_seed(
    clustered_compare: ModuleType,
    clustered_dataset: ModuleType,
    clustered_regions: ModuleType,
    tiny_clustered_config: Any,
) -> None:
    data = clustered_dataset.make_clustered_data(tiny_clustered_config)

    results = clustered_compare.compare(data, tiny_clustered_config, seeds=2)

    assert [result.seed for result in results] == [3, 4]
    for result in results:
        (configuration,) = result.configurations
        assert configuration.name == "staged BBB"
        assert configuration.draws.means.shape == (4, data.grid_inputs.shape[0])
        assert configuration.draws.variances.shape == configuration.draws.means.shape
        assert set(configuration.scores.regions) == set(clustered_regions.REGIONS)
        for region in configuration.scores.regions.values():
            assert 0.0 <= region.coverage <= 1.0
            assert region.aleatoric_std >= 0.0
            assert region.epistemic_std >= 0.0
        assert result.point_epochs == 6  # patience outlasts both stages


def test_main_prints_a_one_row_table_with_every_region(
    clustered_compare: ModuleType,
    clustered_regions: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", *_TINY_FLAGS])

    clustered_compare.main()

    lines = capsys.readouterr().out.strip().splitlines()
    header, row = lines[-2], lines[-1]
    assert header.split()[:3] == ["ID", "RMSE", "ID"]
    assert row.startswith("staged BBB")
    cells = re.findall(_NUMBER, row)
    assert len(cells) == 1 + 3 * len(clustered_regions.REGIONS)
    coverages = [float(cell) for cell in cells[3::3]]
    assert len(coverages) == 3
    assert all(0.0 <= coverage <= 1.0 for coverage in coverages)


def test_main_rejects_invalid_settings(
    clustered_compare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["compare.py", "--num-draws", "0"])

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
