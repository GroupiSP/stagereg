"""Reference values of the clustered-data comparison, pinned before a refactor.

The example's scoring is about to move from its own per-draw mixture code
onto the library's posterior predictive (#172). These tests run the
comparison and the scoring entry points on one small, fixed configuration
and check the region scores, the extrapolation calibration curves and the
printed table against values recorded from the example's own mixture code,
so that the migration is provably a refactor.

They touch only `compare.compare`, `compare.format_table`,
`scoring.calibration_curve` and the results they return, never the
example's mixture helpers.

The tolerance is tight enough to catch any change of formula (a variance
averaged differently, a different quantile) and loose enough to survive a
reordering of float32 reductions. The table prints four decimals, so its
cells are compared to within one unit of the last printed place.
"""

from __future__ import annotations

import re
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax.nnx")
pytest.importorskip("optax")

_RTOL = 1e-5
_ATOL = 1e-6
_TABLE_ATOL = 1e-4

_REFERENCE_CONFIG = {
    "mean_epochs": 3,
    "variance_epochs": 3,
    "posterior_epochs": 2,
    "num_draws": 4,
    "psgld_chains": 2,
    "seed": 3,
}
"""The pinned configuration; every other setting keeps its default."""

# Each configuration's in-distribution R², then, per region, its aleatoric
# std, epistemic std and 95% coverage.
_REFERENCE_SCORES = {
    "joint BBB": (
        0.444913268,
        {
            "in_distribution": (0.421248198, 0.0120263053, 1.0),
            "gap": (0.369665444, 0.00874218531, 1.0),
            "extrapolation": (0.403597564, 0.0132831912, 0.413333327),
        },
    ),
    "joint MCD": (
        0.0187790394,
        {
            "in_distribution": (0.419267058, 0.0176368542, 0.877499998),
            "gap": (0.420480043, 0.0137857702, 1.0),
            "extrapolation": (0.408382535, 0.0105329249, 0.342500001),
        },
    ),
    "joint pSGLD": (
        0.460824311,
        {
            "in_distribution": (0.39116177, 0.0726053193, 0.999166667),
            "gap": (0.347090095, 0.0965322107, 1.0),
            "extrapolation": (0.381588906, 0.0388715565, 0.402500004),
        },
    ),
    "staged BBB": (
        0.454686761,
        {
            "in_distribution": (0.433861673, 0.0100002866, 1.0),
            "gap": (0.429172039, 0.00933753699, 1.0),
            "extrapolation": (0.420710832, 0.0111898351, 0.425000012),
        },
    ),
    "staged MCD": (
        0.227318048,
        {
            "in_distribution": (0.433861673, 0.0823190734, 1.0),
            "gap": (0.429172039, 0.0480193086, 1.0),
            "extrapolation": (0.420710832, 0.0844262391, 0.409166664),
        },
    ),
    "staged pSGLD": (
        0.45492959,
        {
            "in_distribution": (0.418602169, 0.0512408167, 1.0),
            "gap": (0.403518647, 0.0502165928, 1.0),
            "extrapolation": (0.39909485, 0.0556461513, 0.420833319),
        },
    ),
}

# Each configuration's observed coverage in the extrapolation region at the
# nominal levels 0.05, 0.10, ..., 0.95.
_REFERENCE_CALIBRATION = {
    "joint BBB": [
        *[0.0] * 4,
        0.0258333329,
        0.108333334,
        0.159166664,
        0.205833331,
        0.239166662,
        0.274166673,
        0.299166679,
        0.318333328,
        0.341666669,
        0.364166677,
        0.38166666,
        0.385833323,
        0.390833348,
        0.399166673,
        0.413333327,
    ],
    "joint MCD": [
        *[0.0] * 15,
        0.00416666688,
        0.13666667,
        0.237499997,
        0.342500001,
    ],
    "joint pSGLD": [
        *[0.0] * 5,
        0.0166666675,
        0.0933333337,
        0.155000001,
        0.191666663,
        0.230833337,
        0.262499988,
        0.292499989,
        0.314166665,
        0.334166676,
        0.361666679,
        0.379166663,
        0.383333325,
        0.389166653,
        0.402500004,
    ],
    "staged BBB": [
        *[0.0] * 6,
        0.0466666669,
        0.124166667,
        0.166666672,
        0.211666673,
        0.244166672,
        0.273333341,
        0.300833344,
        0.320833325,
        0.344999999,
        0.377499998,
        0.391666681,
        0.403333336,
        0.425000012,
    ],
    "staged MCD": [
        *[0.0] * 12,
        0.0816666633,
        0.159166664,
        0.223333329,
        0.275833338,
        0.319999993,
        0.37166667,
        0.409166664,
    ],
    "staged pSGLD": [
        *[0.0] * 4,
        0.0441666655,
        0.119166665,
        0.163333327,
        0.204166666,
        0.233333334,
        0.264166653,
        0.290833324,
        0.310000002,
        0.327499986,
        0.352499992,
        0.37833333,
        0.38499999,
        0.390833348,
        0.401666671,
        0.420833319,
    ],
}

_REFERENCE_TABLE = """\
               ID R²  ID σe/σt  ID cov95  gap σe/σt  gap cov95  extrap σe/σt  extrap cov95
joint BBB     0.4449    0.0285    1.0000     0.0236     1.0000        0.0329        0.4133
joint MCD     0.0188    0.0420    0.8775     0.0328     1.0000        0.0258        0.3425
joint pSGLD   0.4608    0.1825    0.9992     0.2679     1.0000        0.1013        0.4025
staged BBB    0.4547    0.0230    1.0000     0.0218     1.0000        0.0266        0.4250
staged MCD    0.2273    0.1864    1.0000     0.1112     1.0000        0.1968        0.4092
staged pSGLD  0.4549    0.1215    1.0000     0.1235     1.0000        0.1381        0.4208"""  # noqa: E501

_NUMBER = re.compile(r"\d+\.\d+")


@pytest.fixture(scope="module")
def reference_data(clustered_dataset: ModuleType) -> Any:
    """The training set and evaluation grid of the pinned configuration."""
    config = clustered_dataset.ClusteredDataConfig(**_REFERENCE_CONFIG)
    return clustered_dataset.make_clustered_data(config)


@pytest.fixture(scope="module")
def reference_results(
    clustered_compare: ModuleType, clustered_dataset: ModuleType, reference_data: Any
) -> list[Any]:
    """The comparison of the pinned configuration, over its one seed."""
    config = clustered_dataset.ClusteredDataConfig(**_REFERENCE_CONFIG)
    return clustered_compare.compare(reference_data, config, seeds=1)


def test_region_scores_match_the_pinned_reference(
    reference_results: list[Any],
) -> None:
    (result,) = reference_results
    configurations = {c.name: c for c in result.configurations}

    assert list(configurations) == list(_REFERENCE_SCORES)
    for name, (r2, regions) in _REFERENCE_SCORES.items():
        scores = configurations[name].scores
        assert scores.r2 == pytest.approx(r2, rel=_RTOL, abs=_ATOL), name
        assert set(scores.regions) == set(regions)
        for region, expected in regions.items():
            observed = scores.regions[region]
            actual = (observed.aleatoric_std, observed.epistemic_std, observed.coverage)
            assert actual == pytest.approx(expected, rel=_RTOL, abs=_ATOL), (
                name,
                region,
            )


def test_extrapolation_calibration_curves_match_the_pinned_reference(
    clustered_scoring: ModuleType,
    clustered_regions: ModuleType,
    reference_data: Any,
    reference_results: list[Any],
) -> None:
    (result,) = reference_results
    extrapolation = clustered_regions.region_masks(reference_data.grid_inputs)[
        clustered_regions.EXTRAPOLATION
    ]

    for configuration in result.configurations:
        curve = clustered_scoring.calibration_curve(
            configuration.draws,
            reference_data.grid_targets,
            clustered_scoring.CALIBRATION_LEVELS,
            where=extrapolation,
        )

        expected = _REFERENCE_CALIBRATION[configuration.name]
        assert list(curve) == pytest.approx(expected, rel=_RTOL, abs=_ATOL), (
            configuration.name
        )


def test_comparison_table_matches_the_pinned_reference(
    clustered_compare: ModuleType, reference_results: list[Any]
) -> None:
    table = clustered_compare.format_table(reference_results)

    lines, expected_lines = table.splitlines(), _REFERENCE_TABLE.splitlines()
    assert lines[0] == expected_lines[0]
    assert len(lines) == len(expected_lines)
    for line, expected in zip(lines[1:], expected_lines[1:], strict=True):
        assert line.split("  ")[0] == expected.split("  ")[0]
        cells = [float(cell) for cell in _NUMBER.findall(line)]
        expected_cells = [float(cell) for cell in _NUMBER.findall(expected)]
        assert cells == pytest.approx(expected_cells, abs=_TABLE_ATOL), line
