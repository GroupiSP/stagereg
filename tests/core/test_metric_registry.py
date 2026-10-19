from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stagereg.core.metric_registry import (
    ContinuousRankedProbabilityScore,
    EpochPredictionData,
    EvaluationGrid,
    IntervalCoverage,
    MetricRequirements,
    NegativeLogLikelihood,
    PointContinuousRankedProbabilityScore,
    PredictionInterval,
    RootMeanSquaredError,
    SampleContinuousRankedProbabilityScore,
    WeightedSpread,
)
from stagereg.core.metrics import coverage, crps, point_crps, rmse, sample_crps, wsu

AdapterType = type[
    RootMeanSquaredError
    | IntervalCoverage
    | WeightedSpread
    | PointContinuousRankedProbabilityScore
    | ContinuousRankedProbabilityScore
    | SampleContinuousRankedProbabilityScore
]


def test_rmse_adapter_matches_core_function() -> None:
    data = EpochPredictionData(
        targets=np.array([1.0, 2.0, 3.0]),
        mean=np.array([1.5, 1.5, 3.0]),
    )

    assert RootMeanSquaredError()(data) == pytest.approx(rmse(data.targets, data.mean))


def test_interval_coverage_selects_explicit_labelled_level() -> None:
    selected = PredictionInterval(
        level=0.8,
        lower=np.array([-1.0, 0.5, 1.5]),
        upper=np.array([0.5, 1.5, 2.5]),
    )
    data = EpochPredictionData(
        targets=np.array([0.0, 1.0, 2.0]),
        mean=np.array([0.1, 1.1, 1.9]),
        variance=np.array([100.0, 100.0, 100.0]),
        intervals=(
            PredictionInterval(0.95, np.full(3, -10.0), np.full(3, 10.0)),
            selected,
        ),
    )

    assert IntervalCoverage(level=0.8)(data) == pytest.approx(
        coverage(data.targets, selected.lower, selected.upper)
    )


def test_weighted_spread_uses_explicit_coordinate_and_interval() -> None:
    interval = PredictionInterval(
        level=0.9,
        lower=np.array([0.0, 1.0, 1.0, 2.0]),
        upper=np.array([2.0, 3.0, 5.0, 6.0]),
    )
    coordinate = np.array([0.0, 1.0, 2.0, 4.0])
    data = EpochPredictionData(
        targets=np.zeros(4),
        mean=np.zeros(4),
        intervals=(interval,),
        coordinate=coordinate,
    )

    assert WeightedSpread(level=0.9)(data) == pytest.approx(
        wsu(interval.lower, interval.upper, coordinate)
    )


def test_point_crps_scores_each_scalar_unit_before_averaging() -> None:
    grid = EvaluationGrid(np.linspace(-2.0, 4.0, 61))
    targets = np.array([0.0, 2.0])
    samples = np.array([[-0.5, 0.0, 0.5], [1.0, 2.0, 3.0]])
    data = EpochPredictionData(
        targets=targets,
        mean=np.mean(samples, axis=1),
        predictive_samples=samples,
        evaluation_grid=grid,
    )

    expected = np.mean(
        [point_crps(target, row, grid.values) for target, row in zip(targets, samples)]
    )
    assert PointContinuousRankedProbabilityScore()(data) == pytest.approx(expected)


def test_sample_crps_scores_each_scalar_unit_without_a_grid() -> None:
    targets = np.array([0.0, 2.0])
    samples = np.array([[-0.5, 0.0, 0.5], [1.0, 2.0, 3.0]])
    data = EpochPredictionData(
        targets=targets,
        mean=np.mean(samples, axis=1),
        predictive_samples=samples,
    )

    expected = np.mean(
        [sample_crps(target, row) for target, row in zip(targets, samples)]
    )
    assert SampleContinuousRankedProbabilityScore()(data) == pytest.approx(expected)
    requirements = SampleContinuousRankedProbabilityScore().requirements
    assert requirements.predictive_samples
    assert not requirements.evaluation_grid


def test_sample_crps_requires_predictive_samples() -> None:
    data = EpochPredictionData(targets=np.zeros(2), mean=np.zeros(2))

    with pytest.raises(ValueError, match="predictive_samples"):
        SampleContinuousRankedProbabilityScore()(data)


# Small integers make samples, targets and grid points collide, which is where
# the ``samples <= grid`` and ``grid >= target`` conventions matter.
_crps_values = st.one_of(
    st.integers(-5, 5).map(float),
    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
)


@st.composite
def _scoring_units(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(units,)`` targets and ``(units, S)`` predictive samples."""
    units = draw(st.integers(1, 6))
    sample_count = draw(st.integers(1, 8))
    targets = draw(st.lists(_crps_values, min_size=units, max_size=units))
    samples = draw(
        st.lists(
            _crps_values,
            min_size=units * sample_count,
            max_size=units * sample_count,
        )
    )
    return np.asarray(targets), np.reshape(samples, (units, sample_count))


_grids = st.lists(_crps_values, min_size=2, max_size=12, unique=True).map(
    lambda values: EvaluationGrid(np.sort(values))
)


@given(units=_scoring_units(), grid=_grids)
def test_point_crps_matches_a_loop_over_scoring_units(
    units: tuple[np.ndarray, np.ndarray], grid: EvaluationGrid
) -> None:
    targets, samples = units
    data = EpochPredictionData(
        targets=targets,
        mean=np.mean(samples, axis=1),
        predictive_samples=samples,
        evaluation_grid=grid,
    )

    expected = np.mean(
        [
            point_crps(target, row, grid.values)
            for target, row in zip(targets, samples, strict=True)
        ]
    )
    assert PointContinuousRankedProbabilityScore()(data) == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )


@given(units=_scoring_units())
def test_sample_crps_matches_a_loop_over_scoring_units(
    units: tuple[np.ndarray, np.ndarray],
) -> None:
    targets, samples = units
    data = EpochPredictionData(
        targets=targets,
        mean=np.mean(samples, axis=1),
        predictive_samples=samples,
    )

    expected = np.mean(
        [sample_crps(target, row) for target, row in zip(targets, samples, strict=True)]
    )
    assert SampleContinuousRankedProbabilityScore()(data) == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )


def test_conditional_crps_scores_paired_distributions_before_averaging() -> None:
    grid = EvaluationGrid(np.linspace(-2.0, 4.0, 61))
    reference = np.array([[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    predictive = np.array([[-0.5, 0.0, 0.5], [2.0, 3.0, 4.0]])
    data = EpochPredictionData(
        targets=np.array([0.0, 2.0]),
        mean=np.mean(predictive, axis=1),
        predictive_samples=predictive,
        reference_samples=reference,
        evaluation_grid=grid,
    )

    expected = np.mean(
        [crps(true, pred, grid.values) for true, pred in zip(reference, predictive)]
    )
    assert ContinuousRankedProbabilityScore()(data) == pytest.approx(expected)


def test_adapters_declare_materialization_requirements() -> None:
    coverage_requirements = IntervalCoverage(level=0.8).requirements
    spread_requirements = WeightedSpread(level=0.9).requirements
    point_requirements = PointContinuousRankedProbabilityScore().requirements
    conditional_requirements = ContinuousRankedProbabilityScore().requirements

    assert coverage_requirements.interval_levels == frozenset({0.8})
    assert spread_requirements.coordinate
    assert spread_requirements.interval_levels == frozenset({0.9})
    assert point_requirements.predictive_samples
    assert point_requirements.evaluation_grid
    assert conditional_requirements.reference_samples


_ADAPTER_DEFAULT_NAMES: list[tuple[AdapterType, str]] = [
    (RootMeanSquaredError, "rmse"),
    (IntervalCoverage, "coverage"),
    (WeightedSpread, "wsu"),
    (PointContinuousRankedProbabilityScore, "point_crps"),
    (ContinuousRankedProbabilityScore, "crps"),
    (SampleContinuousRankedProbabilityScore, "sample_crps"),
]


@pytest.mark.parametrize(("adapter_type", "default_name"), _ADAPTER_DEFAULT_NAMES)
@given(name=st.text(min_size=1))
def test_adapters_report_under_the_name_they_are_constructed_with(
    adapter_type: AdapterType, default_name: str, name: str
) -> None:
    assert adapter_type().name == default_name
    assert adapter_type(name=name).name == name


@pytest.mark.parametrize(("adapter_type", "default_name"), _ADAPTER_DEFAULT_NAMES)
def test_adapters_reject_legacy_metric_name_keyword(
    adapter_type: AdapterType, default_name: str
) -> None:
    del default_name
    with pytest.raises(TypeError, match="metric_name"):
        adapter_type(metric_name="legacy")  # type: ignore[call-arg]


def test_epoch_prediction_data_rejects_non_scalar_axes_and_invalid_fields() -> None:
    with pytest.raises(ValueError, match="one-dimensional scalar scoring units"):
        EpochPredictionData(targets=np.ones((2, 1)), mean=np.ones((2, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        EpochPredictionData(
            targets=np.array([0.0]),
            mean=np.array([0.0]),
            variance=np.array([-1.0]),
        )
    with pytest.raises(ValueError, match="one row"):
        EpochPredictionData(
            targets=np.array([0.0, 1.0]),
            mean=np.array([0.0, 1.0]),
            predictive_samples=np.ones((1, 3)),
        )


def test_missing_required_typed_fields_fail_clearly() -> None:
    data = EpochPredictionData(targets=np.array([0.0]), mean=np.array([0.0]))

    with pytest.raises(ValueError, match="interval at level"):
        IntervalCoverage()(data)
    with pytest.raises(ValueError, match="explicit coordinate"):
        WeightedSpread()(data)
    with pytest.raises(ValueError, match="predictive_samples"):
        PointContinuousRankedProbabilityScore()(data)


def test_grid_and_intervals_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        EvaluationGrid(np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        PredictionInterval(1.0, np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="must not exceed"):
        PredictionInterval(0.9, np.array([2.0]), np.array([1.0]))


@given(st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=30))
def test_rmse_is_zero_for_equal_finite_vectors(values: list[float]) -> None:
    vector = np.asarray(values)
    data = EpochPredictionData(targets=vector, mean=vector)

    assert RootMeanSquaredError()(data) == pytest.approx(0.0)


def test_crps_adapters_are_non_negative_and_match_reference_expectation() -> None:
    grid = EvaluationGrid(np.linspace(-3.0, 3.0, 101))
    samples = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    data = EpochPredictionData(
        targets=np.zeros(2),
        mean=np.zeros(2),
        predictive_samples=samples,
        reference_samples=samples.copy(),
        evaluation_grid=grid,
    )

    expected = np.mean(
        [
            np.mean([point_crps(target, row, grid.values) for target in reference])
            for reference, row in zip(samples, samples, strict=True)
        ]
    )
    assert ContinuousRankedProbabilityScore()(data) == pytest.approx(expected)
    assert PointContinuousRankedProbabilityScore()(data) >= 0.0


@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=30))
def test_interval_coverage_is_bounded(values: list[float]) -> None:
    targets = np.asarray(values)
    data = EpochPredictionData(
        targets=targets,
        mean=targets,
        intervals=(
            PredictionInterval(
                0.95,
                targets - 1.0,
                targets + 1.0,
            ),
        ),
    )

    observed = IntervalCoverage()(data)
    assert 0.0 <= observed <= 1.0


def test_negative_log_likelihood_averages_scoring_units_log_likelihood() -> None:
    data = EpochPredictionData(
        targets=np.zeros(3),
        mean=np.zeros(3),
        log_likelihood=np.array([-1.0, -3.0, 1.0]),
    )

    assert NegativeLogLikelihood()(data) == pytest.approx(1.0)
    assert NegativeLogLikelihood().name == "nll"
    assert NegativeLogLikelihood(name="validation_nll").name == "validation_nll"
    assert NegativeLogLikelihood().requirements.log_likelihood


def test_negative_log_likelihood_requires_the_log_likelihood_field() -> None:
    data = EpochPredictionData(targets=np.array([0.0]), mean=np.array([0.0]))

    with pytest.raises(ValueError, match="log_likelihood"):
        NegativeLogLikelihood()(data)


def test_log_likelihood_must_match_the_scoring_units_and_be_finite() -> None:
    with pytest.raises(ValueError, match="log_likelihood must match"):
        EpochPredictionData(
            targets=np.zeros(2), mean=np.zeros(2), log_likelihood=np.zeros(3)
        )
    with pytest.raises(ValueError, match="finite"):
        EpochPredictionData(
            targets=np.zeros(1), mean=np.zeros(1), log_likelihood=np.array([-np.inf])
        )


def test_log_likelihood_requirement_survives_union() -> None:
    combined = MetricRequirements().union(MetricRequirements(log_likelihood=True))

    assert combined.log_likelihood
    assert not MetricRequirements().log_likelihood
