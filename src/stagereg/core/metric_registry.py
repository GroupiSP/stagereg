"""Typed host-side epoch metric contracts and numerical adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from stagereg.core.metrics import (
    coverage,
    crps,
    expected_sample_crps,
    rmse,
    wsu,
)

FloatArray = NDArray[np.float64]


def _vector(values: object, *, name: str) -> FloatArray:
    """Return a validated, immutable C-contiguous vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional scalar scoring units.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _matrix(values: object, *, name: str) -> FloatArray:
    """Return a validated, immutable C-contiguous scoring-unit matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_scoring_units, n_samples).")
    if not all(array.shape):
        raise ValueError(f"{name} axes must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _point_crps_per_unit(
    targets: FloatArray, samples: FloatArray, grid: FloatArray
) -> FloatArray:
    """Return the point CRPS of every scoring unit at once.

    Each row matches [`point_crps`][stagereg.core.point_crps]. The empirical
    CDF at each grid point is counted by sorting each row's samples together
    with the grid, which needs ``(units, S + G)`` memory instead of the
    ``(units, G, S)`` of a broadcast comparison.
    """
    unit_count, sample_count = samples.shape
    pooled = np.concatenate(
        [samples, np.broadcast_to(grid, (unit_count, grid.size))], axis=1
    )
    # A stable sort keeps samples ahead of equal grid points, so the samples
    # counted before a grid point are exactly those ``<=`` it.
    order = np.argsort(pooled, axis=1, kind="stable")
    is_sample = order < sample_count
    samples_up_to = np.cumsum(is_sample, axis=1)[~is_sample].reshape(
        unit_count, grid.size
    )
    # Grid points are already increasing, so they appear in their own order.
    predicted_cdf = samples_up_to / sample_count
    true_cdf = (grid[None, :] >= targets[:, None]).astype(np.float64)
    return np.trapezoid((true_cdf - predicted_cdf) ** 2, grid, axis=1)


@dataclass(frozen=True, slots=True)
class EvaluationGrid:
    """A shared scalar CRPS integration grid.

    Attributes:
        values: Finite, strictly increasing integration coordinates.
    """

    values: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize grid values.

        Raises:
            ValueError: If the grid has fewer than two points or is not increasing.
        """
        values = _vector(self.values, name="evaluation_grid")
        if values.size < 2:
            raise ValueError("evaluation_grid must contain at least two values.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("evaluation_grid must be strictly increasing.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """Host-resident bounds labelled by their central confidence level.

    Attributes:
        level: Central confidence level of the interval, strictly between 0 and 1.
        lower: Lower bounds, one per scoring unit.
        upper: Upper bounds, one per scoring unit, each no smaller than its lower
            bound.
    """

    level: float
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        """Validate interval level, bounds, and scoring-unit shape.

        Raises:
            ValueError: If the level or bounds are invalid.
        """
        if not np.isfinite(self.level) or not 0.0 < self.level < 1.0:
            raise ValueError("interval level must be finite and between 0 and 1.")
        lower = _vector(self.lower, name="interval.lower")
        upper = _vector(self.upper, name="interval.upper")
        if lower.shape != upper.shape:
            raise ValueError(
                "interval lower and upper bounds must have matching shapes."
            )
        if np.any(lower > upper):
            raise ValueError("interval lower bounds must not exceed upper bounds.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class EpochPredictionData:
    """Typed, host-materialized scalar predictions for one batch or epoch.

    Rows are independent scalar scoring units. Sample matrices use rows for scoring
    units and columns for samples; non-scalar distribution events are intentionally not
    represented by this contract.

    Attributes:
        targets: Observed target, one per scoring unit.
        mean: Predictive mean, one per scoring unit.
        variance: Non-negative predictive variance, one per scoring unit. Defaults
            to ``None`` when no metric requires it.
        predictive_samples: Samples from the predictive distribution, one row per
            scoring unit and one column per sample. Defaults to ``None`` when no
            metric requires them.
        reference_samples: Samples from each scoring unit's reference distribution,
            the empirical distribution the predictive samples are scored against,
            one row per scoring unit and one column per sample. Defaults to ``None``
            when no metric requires them.
        intervals: Prediction intervals, each a
            [`PredictionInterval`][stagereg.core.PredictionInterval] at a distinct
            confidence level. Defaults to no intervals.
        coordinate: Numeric coordinate of each scoring unit, such as time or
            remaining useful life, that a metric orders or weights scoring units
            by. Defaults to ``None`` when no metric requires it.
        evaluation_grid: Shared [`EvaluationGrid`][stagereg.core.EvaluationGrid]
            that distributional scores are integrated over. Defaults to ``None``
            when no metric requires it.
        log_likelihood: Log-density of each scoring unit's target under the
            predictive distribution. Defaults to ``None`` when no metric
            requires it.
        closed_form_crps: Closed-form CRPS of each scoring unit's predictive
            distribution at its target, computed from the distribution's own
            formula (see [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS]).
            Defaults to ``None`` when no metric requires it.
        closed_form_crps_against_reference: Whether ``closed_form_crps`` scored
            each unit's reference samples rather than its target. Defaults to
            ``False``.
    """

    targets: FloatArray
    mean: FloatArray
    variance: FloatArray | None = None
    predictive_samples: FloatArray | None = None
    reference_samples: FloatArray | None = None
    intervals: tuple[PredictionInterval, ...] = ()
    coordinate: FloatArray | None = None
    evaluation_grid: EvaluationGrid | None = None
    log_likelihood: FloatArray | None = None
    closed_form_crps: FloatArray | None = None
    closed_form_crps_against_reference: bool = False

    def __post_init__(self) -> None:
        """Validate all fields against the common scoring-unit axis.

        Raises:
            ValueError: If a field is non-finite, malformed, or shape-incompatible.
        """
        targets = _vector(self.targets, name="targets")
        mean = _vector(self.mean, name="mean")
        if targets.shape != mean.shape:
            raise ValueError("targets and mean must have matching scoring-unit shapes.")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "mean", mean)
        n_units = targets.size

        if self.variance is not None:
            variance = _vector(self.variance, name="variance")
            if variance.shape != targets.shape:
                raise ValueError("variance must match the scoring-unit shape.")
            if np.any(variance < 0.0):
                raise ValueError("variance must be non-negative.")
            object.__setattr__(self, "variance", variance)

        for field_name in ("predictive_samples", "reference_samples"):
            values = getattr(self, field_name)
            if values is None:
                continue
            samples = _matrix(values, name=field_name)
            if samples.shape[0] != n_units:
                raise ValueError(
                    f"{field_name} must have one row per scalar scoring unit."
                )
            object.__setattr__(self, field_name, samples)

        intervals = tuple(self.intervals)
        levels: set[float] = set()
        for interval in intervals:
            if not isinstance(interval, PredictionInterval):
                raise TypeError("intervals must contain PredictionInterval values.")
            if interval.lower.size != n_units:
                raise ValueError("interval bounds must match the scoring-unit shape.")
            if interval.level in levels:
                raise ValueError(f"duplicate interval level {interval.level}.")
            levels.add(interval.level)
        object.__setattr__(self, "intervals", intervals)

        if self.evaluation_grid is not None and not isinstance(
            self.evaluation_grid, EvaluationGrid
        ):
            raise TypeError("evaluation_grid must be an EvaluationGrid.")

        if self.coordinate is not None:
            coordinate = _vector(self.coordinate, name="coordinate")
            if coordinate.shape != targets.shape:
                raise ValueError("coordinate must match the scoring-unit shape.")
            object.__setattr__(self, "coordinate", coordinate)

        for field_name in ("log_likelihood", "closed_form_crps"):
            values = getattr(self, field_name)
            if values is None:
                continue
            per_unit = _vector(values, name=field_name)
            if per_unit.shape != targets.shape:
                raise ValueError(f"{field_name} must match the scoring-unit shape.")
            object.__setattr__(self, field_name, per_unit)

    def interval(self, level: float) -> PredictionInterval:
        """Return the prediction interval registered at exactly ``level``.

        Args:
            level: Required central confidence level.

        Returns:
            The matching labelled interval.

        Raises:
            ValueError: If no interval at ``level`` is present.
        """
        for interval in self.intervals:
            if interval.level == level:
                return interval
        raise ValueError(f"prediction interval at level {level} is required.")


@dataclass(frozen=True, slots=True)
class MetricRequirements:
    """Materialized fields required by one or more epoch metrics.

    Targets and predictive means are always materialized; each attribute asks for one
    more field of [`EpochPredictionData`][stagereg.core.EpochPredictionData].

    Attributes:
        variance: Whether the predictive variance is required. Defaults to
            ``False``.
        predictive_samples: Whether samples from the predictive distribution are
            required. Defaults to ``False``.
        reference_samples: Whether samples from each scoring unit's reference
            distribution are required. Defaults to ``False``.
        interval_levels: Confidence levels, each in ``(0, 1)``, at which
            prediction intervals are required. Defaults to no levels.
        coordinate: Whether each scoring unit's numeric coordinate is required.
            Defaults to ``False``.
        evaluation_grid: Whether the shared evaluation grid is required. Defaults
            to ``False``.
        log_likelihood: Whether each target's predictive log-density is
            required. Defaults to ``False``.
        closed_form_crps: Whether each scoring unit's closed-form CRPS is
            required. Defaults to ``False``.
    """

    variance: bool = False
    predictive_samples: bool = False
    reference_samples: bool = False
    interval_levels: frozenset[float] = frozenset()
    coordinate: bool = False
    evaluation_grid: bool = False
    log_likelihood: bool = False
    closed_form_crps: bool = False

    def __post_init__(self) -> None:
        """Validate requested interval levels.

        Raises:
            ValueError: If an interval level is outside ``(0, 1)``.
        """
        if any(
            not np.isfinite(level) or not 0.0 < level < 1.0
            for level in self.interval_levels
        ):
            raise ValueError(
                "required interval levels must be finite and between 0 and 1."
            )

    def union(self, other: MetricRequirements) -> MetricRequirements:
        """Combine this requirement set with another.

        Args:
            other: Additional requirements.

        Returns:
            A requirement set containing the union of both inputs.
        """
        return MetricRequirements(
            variance=self.variance or other.variance,
            predictive_samples=self.predictive_samples or other.predictive_samples,
            reference_samples=self.reference_samples or other.reference_samples,
            interval_levels=self.interval_levels | other.interval_levels,
            coordinate=self.coordinate or other.coordinate,
            evaluation_grid=self.evaluation_grid or other.evaluation_grid,
            log_likelihood=self.log_likelihood or other.log_likelihood,
            closed_form_crps=self.closed_form_crps or other.closed_form_crps,
        )


class EpochMetric(Protocol):
    """A host-side scalar metric over typed epoch prediction data."""

    @property
    def name(self) -> str:
        """Return the emitted metric name."""
        ...

    @property
    def requirements(self) -> MetricRequirements:
        """Return fields that must be materialized for this metric."""
        ...

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute the scalar metric value."""
        ...


@dataclass(frozen=True, slots=True)
class RootMeanSquaredError:
    """Epoch adapter for root mean squared error.

    Attributes:
        name: Bare metric name the value is reported under. Defaults to ``rmse``.
    """

    name: str = "rmse"

    @property
    def requirements(self) -> MetricRequirements:
        """Return the base target/mean requirements."""
        return MetricRequirements()

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute RMSE from scalar targets and predictive means."""
        return rmse(data.targets, data.mean)


@dataclass(frozen=True, slots=True)
class NegativeLogLikelihood:
    """Mean negative log-density of the targets under the predictive distribution.

    The predictor computes each target's exact log-density, so a non-Gaussian
    predictive (such as a mixture over draws) is scored without a Gaussian
    approximation.

    Attributes:
        name: Bare metric name the value is reported under. Defaults to ``nll``.
    """

    name: str = "nll"

    @property
    def requirements(self) -> MetricRequirements:
        """Require each target's predictive log-density."""
        return MetricRequirements(log_likelihood=True)

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Average the negative log-likelihood over scoring units.

        Raises:
            ValueError: If the log-likelihood is absent.
        """
        if data.log_likelihood is None:
            raise ValueError("NLL requires log_likelihood.")
        return float(-np.mean(data.log_likelihood))


@dataclass(frozen=True, slots=True)
class IntervalCoverage:
    """Observed coverage of an explicitly labelled prediction interval.

    Attributes:
        level: Confidence level of the labelled interval, in ``(0, 1)``.
        name: Bare metric name the value is reported under. Defaults to ``coverage``.
    """

    level: float = 0.95
    name: str = "coverage"

    def __post_init__(self) -> None:
        """Validate the requested confidence level.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.
        """
        MetricRequirements(interval_levels=frozenset({self.level}))

    @property
    def requirements(self) -> MetricRequirements:
        """Require an interval at the configured level."""
        return MetricRequirements(interval_levels=frozenset({self.level}))

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute inclusive interval coverage."""
        interval = data.interval(self.level)
        return coverage(data.targets, interval.lower, interval.upper)


@dataclass(frozen=True, slots=True)
class WeightedSpread:
    """Weighted spread for an interval and explicit numeric coordinate.

    Attributes:
        level: Confidence level of the labelled interval, in ``(0, 1)``.
        name: Bare metric name the value is reported under. Defaults to ``wsu``.
    """

    level: float = 0.95
    name: str = "wsu"

    def __post_init__(self) -> None:
        """Validate the requested confidence level.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.
        """
        MetricRequirements(interval_levels=frozenset({self.level}))

    @property
    def requirements(self) -> MetricRequirements:
        """Require labelled bounds and a coordinate."""
        return MetricRequirements(
            interval_levels=frozenset({self.level}), coordinate=True
        )

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute normalized weighted interval spread.

        Raises:
            ValueError: If the required coordinate is absent.
        """
        if data.coordinate is None:
            raise ValueError("WeightedSpread requires an explicit coordinate.")
        interval = data.interval(self.level)
        order = np.argsort(data.coordinate)
        return wsu(
            interval.lower[order],
            interval.upper[order],
            data.coordinate[order],
        )


@dataclass(frozen=True, slots=True)
class PointContinuousRankedProbabilityScore:
    """Mean point-CRPS across independent scalar scoring units.

    Attributes:
        name: Bare metric name the value is reported under. Defaults to ``point_crps``.
    """

    name: str = "point_crps"

    @property
    def requirements(self) -> MetricRequirements:
        """Require predictive samples and the shared evaluation grid."""
        return MetricRequirements(predictive_samples=True, evaluation_grid=True)

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute one point-CRPS per unit, then average units equally.

        Raises:
            ValueError: If predictive samples or the grid are absent.
        """
        if data.predictive_samples is None:
            raise ValueError("Point CRPS requires predictive_samples.")
        if data.evaluation_grid is None:
            raise ValueError("Point CRPS requires an evaluation_grid.")
        scores = _point_crps_per_unit(
            data.targets, data.predictive_samples, data.evaluation_grid.values
        )
        return float(np.mean(scores))


@dataclass(frozen=True, slots=True)
class SampleContinuousRankedProbabilityScore:
    """Mean exact CRPS of predictive samples across scalar scoring units.

    Each unit's predictive samples are scored against its target with
    [`sample_crps`][stagereg.core.sample_crps], the energy form of the CRPS,
    so unlike [`PointContinuousRankedProbabilityScore`][stagereg.core.PointContinuousRankedProbabilityScore]
    it needs no evaluation grid.

    Attributes:
        name: Bare metric name the value is reported under. Defaults to
            ``sample_crps``.
    """

    name: str = "sample_crps"

    @property
    def requirements(self) -> MetricRequirements:
        """Require predictive samples only."""
        return MetricRequirements(predictive_samples=True)

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute one CRPS per unit, then average units equally.

        Raises:
            ValueError: If predictive samples are absent.
        """
        if data.predictive_samples is None:
            raise ValueError("Sample CRPS requires predictive_samples.")
        scores = expected_sample_crps(data.predictive_samples, data.targets[:, None])
        return float(np.mean(scores))


@dataclass(frozen=True, slots=True)
class ClosedFormContinuousRankedProbabilityScore:
    """Mean closed-form CRPS of the predictive distribution across scoring units.

    Each unit is scored by its predictive distribution's own CRPS formula, so the
    value carries no Monte Carlo noise and needs neither predictive samples nor
    an evaluation grid. With ``against_reference`` set, each unit is scored
    against its reference samples instead of its target, giving the mean
    closed-form CRPS over the reference distribution. The predictor's
    distribution must implement
    [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS]; otherwise the predictor
    raises a ``TypeError`` rather than fall back to a sample estimate (ADR 0015).
    For a distribution without a closed form, use
    [`SampleContinuousRankedProbabilityScore`][stagereg.core.SampleContinuousRankedProbabilityScore].

    Attributes:
        name: Bare metric name the value is reported under. Defaults to
            ``closed_form_crps``.
        against_reference: Whether to score each unit's reference samples
            instead of its observed target. The predictor then needs a
            reference samples extractor. Defaults to ``False``.
    """

    name: str = "closed_form_crps"
    against_reference: bool = False

    @property
    def requirements(self) -> MetricRequirements:
        """Require the closed-form CRPS, and reference samples to score against."""
        return MetricRequirements(
            closed_form_crps=True, reference_samples=self.against_reference
        )

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Average the closed-form CRPS over scoring units.

        Raises:
            ValueError: If the closed-form CRPS is absent, or scored the target
                when this metric asks for the reference samples, or the reverse.
        """
        if data.closed_form_crps is None:
            raise ValueError("Closed-form CRPS requires closed_form_crps.")
        if data.closed_form_crps_against_reference != self.against_reference:
            scored, wanted = (
                ("reference samples", "target")
                if data.closed_form_crps_against_reference
                else ("target", "reference samples")
            )
            raise ValueError(
                f"closed_form_crps scored the {scored}, but this metric scores "
                f"the {wanted}."
            )
        return float(np.mean(data.closed_form_crps))


@dataclass(frozen=True, slots=True)
class ContinuousRankedProbabilityScore:
    """Mean expected CRPS under per-unit empirical reference distributions.

    Attributes:
        name: Bare metric name the value is reported under. Defaults to ``crps``.
    """

    name: str = "crps"

    @property
    def requirements(self) -> MetricRequirements:
        """Require predictive/reference samples and the shared grid."""
        return MetricRequirements(
            predictive_samples=True,
            reference_samples=True,
            evaluation_grid=True,
        )

    def __call__(self, data: EpochPredictionData, /) -> float:
        """Compute expected empirical-reference CRPS and average scoring units.

        Raises:
            ValueError: If either sample matrix or the grid is absent.
        """
        if data.predictive_samples is None:
            raise ValueError("CRPS requires predictive_samples.")
        if data.reference_samples is None:
            raise ValueError("CRPS requires reference_samples.")
        if data.evaluation_grid is None:
            raise ValueError("CRPS requires an evaluation_grid.")
        scores = [
            crps(reference, predictive, data.evaluation_grid.values)
            for reference, predictive in zip(
                data.reference_samples, data.predictive_samples, strict=True
            )
        ]
        return float(np.mean(scores))
