"""Metric registration and prediction materialization for the JAX backend."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike
from numpy.typing import NDArray

from stagereg.core.metric_registry import (
    EpochMetric,
    EpochPredictionData,
    EvaluationGrid,
    MetricRequirements,
    PredictionInterval,
)
from stagereg.core.distributions import ClosedFormCRPS, PredictiveDistribution
from stagereg.core.types import Batch, PyTree
from stagereg.jax.distributions import Gaussian, PosteriorPredictive

_METRIC_RNG_NAMESPACE = 0x4D455452  # "METR" in ASCII

type StepScalar = jax.Array | float
"""A scalar a training or evaluation step outputs, on the device until read."""


def _metric_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of metric values.

    Args:
        values: Metric values to average.

    Returns:
        The arithmetic mean.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("a loader must provide at least one batch.")
    return sum(values) / len(values)


def _finite_float(name: str, value: float) -> float:
    """Return ``value`` as a finite Python float.

    Args:
        name: Metric name for error messages.
        value: Metric value.

    Returns:
        ``value`` cast to a Python float.

    Raises:
        ValueError: If the metric is not finite.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"metric {name!r} must be finite, got {value}.")
    return value


class BatchMetric(Protocol):
    """A JAX-native metric computed per batch."""

    def __call__(
        self,
        model: nnx.Module,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
        training: bool,
        /,
    ) -> jax.Array:
        """Return a scalar batch metric value."""
        ...


@dataclass(frozen=True, slots=True)
class BatchMetricSpec:
    """A named batch metric with a host-side reduction policy.

    Attributes:
        name: Bare metric name the reduced value is reported under. Must not be
            empty.
        metric: The [`BatchMetric`][stagereg.jax.BatchMetric] evaluated on every
            batch.
        reduce: Host-side reduction of the per-batch values to one epoch value.
            Defaults to the arithmetic mean.
    """

    name: str
    metric: BatchMetric
    reduce: Callable[[Sequence[float]], float] = _metric_mean

    def __post_init__(self) -> None:
        """Validate a batch metric registration.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not self.name:
            raise ValueError("batch metric name must not be empty.")


@dataclass(frozen=True, slots=True)
class PredictionRequirements:
    """Combined metric requirements plus explicit materialization configuration.

    Attributes:
        fields: Union of the prediction fields the registered epoch metrics read.
        predictive_sample_count: Number of predictive samples per scoring unit.
            Required, and positive, when ``fields`` asks for predictive samples.
        evaluation_grid: Grid the CRPS metrics integrate over. Required when
            ``fields`` asks for an evaluation grid.
    """

    fields: MetricRequirements
    predictive_sample_count: int | None = None
    evaluation_grid: EvaluationGrid | None = None

    def __post_init__(self) -> None:
        """Validate sample and grid configuration.

        Raises:
            ValueError: If required samples/grid lack explicit configuration.
        """
        if self.fields.predictive_samples:
            if self.predictive_sample_count is None:
                raise ValueError(
                    "sample-based epoch metrics require predictive_sample_count."
                )
            if self.predictive_sample_count <= 0:
                raise ValueError("predictive_sample_count must be positive.")
        elif (
            self.predictive_sample_count is not None
            and self.predictive_sample_count <= 0
        ):
            raise ValueError("predictive_sample_count must be positive.")
        if self.fields.evaluation_grid and self.evaluation_grid is None:
            raise ValueError("CRPS epoch metrics require an explicit evaluation_grid.")


class CoordinateExtractor(Protocol):
    """Extract a numeric scoring-unit coordinate from a domain batch."""

    def __call__(self, batch: Batch, /) -> Any:
        """Return coordinates corresponding to the batch targets."""
        ...


class ReferenceSamplesExtractor(Protocol):
    """Materialize reference samples at the JAX/domain boundary."""

    def __call__(self, batch: Batch, key: jax.Array, /) -> Any:
        """Return samples shaped ``(n_scoring_units, n_samples)``."""
        ...


class Predictor(Protocol):
    """Materialize typed host predictions from a JAX model and batch.

    During training the model it receives runs compiled, so its output must be a JAX
    pytree, as every built-in distribution is.
    """

    def __call__(
        self,
        model: nnx.Module,
        batch: Batch,
        requirements: PredictionRequirements,
        key: jax.Array,
        /,
    ) -> EpochPredictionData:
        """Produce host-resident prediction data for ``batch``."""
        ...


@dataclass(frozen=True, slots=True)
class GaussianPredictor:
    """Materialize scalar epoch data from a model returning ``Gaussian``.

    Attributes:
        coordinate_extractor: Optional explicit adapter for coordinate metadata.
        reference_samples_extractor: Optional explicit adapter for conditional CRPS
            reference samples. It must return ``(n_scoring_units, n_samples)``.
    """

    coordinate_extractor: CoordinateExtractor | None = None
    reference_samples_extractor: ReferenceSamplesExtractor | None = None

    def __call__(
        self,
        model: nnx.Module,
        batch: Batch,
        requirements: PredictionRequirements,
        key: jax.Array,
        /,
    ) -> EpochPredictionData:
        """Materialize only fields requested by registered epoch metrics.

        Args:
            model: Model mapping batch inputs to a scalar-event Gaussian.
            batch: Domain batch containing targets.
            requirements: Combined fields and explicit sample/grid configuration.
            key: Dedicated metric PRNG key.

        Returns:
            Validated host-resident prediction data.

        Raises:
            TypeError: If the model does not return a Gaussian.
            ValueError: If targets, event shape, extractors, or shapes are invalid.
        """
        targets = _require_targets(batch)
        prediction = _require_scalar_gaussian(model(batch.inputs))
        return _materialize_epoch_data(
            prediction,
            targets=targets,
            batch=batch,
            requirements=requirements,
            key=key,
            interval=_gaussian_interval,
            coordinate_extractor=self.coordinate_extractor,
            reference_samples_extractor=self.reference_samples_extractor,
        )


@dataclass(frozen=True, slots=True)
class PosteriorPredictivePredictor:
    """Materialize scalar epoch data from a model returning ``PosteriorPredictive``.

    Scores the exact mixture rather than its moment-matched Gaussian: the
    log-likelihood is the mixture log-density, the closed-form CRPS is the
    mixture's, predictive samples are drawn from the mixture, and prediction
    intervals are central mixture quantiles.

    Attributes:
        coordinate_extractor: Optional explicit adapter for coordinate metadata.
        reference_samples_extractor: Optional explicit adapter for conditional CRPS
            reference samples. It must return ``(n_scoring_units, n_samples)``.
    """

    coordinate_extractor: CoordinateExtractor | None = None
    reference_samples_extractor: ReferenceSamplesExtractor | None = None

    def __call__(
        self,
        model: nnx.Module,
        batch: Batch,
        requirements: PredictionRequirements,
        key: jax.Array,
        /,
    ) -> EpochPredictionData:
        """Materialize only fields requested by registered epoch metrics.

        Args:
            model: Model mapping batch inputs to a
                [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive].
            batch: Domain batch containing targets.
            requirements: Combined fields and explicit sample/grid configuration.
            key: Dedicated metric PRNG key.

        Returns:
            Validated host-resident prediction data.

        Raises:
            TypeError: If the model does not return a posterior predictive.
            ValueError: If targets, extractors, or shapes are invalid.
        """
        targets = _require_targets(batch)
        prediction = model(batch.inputs)
        if not isinstance(prediction, PosteriorPredictive):
            raise TypeError(
                "PosteriorPredictivePredictor requires a model returning "
                "PosteriorPredictive."
            )
        return _materialize_epoch_data(
            prediction,
            targets=targets,
            batch=batch,
            requirements=requirements,
            key=key,
            interval=_mixture_interval,
            coordinate_extractor=self.coordinate_extractor,
            reference_samples_extractor=self.reference_samples_extractor,
        )


def _materialize_epoch_data[Prediction: PredictiveDistribution](
    prediction: Prediction,
    *,
    targets: Any,
    batch: Batch,
    requirements: PredictionRequirements,
    key: jax.Array,
    interval: Callable[[Prediction, float], PredictionInterval],
    coordinate_extractor: CoordinateExtractor | None,
    reference_samples_extractor: ReferenceSamplesExtractor | None,
) -> EpochPredictionData:
    """Materialize the requested fields of one scalar-event prediction.

    The mean, variance, predictive samples, log-likelihood and closed-form CRPS
    are computed by one compiled call and copied to the host together. When
    reference samples are requested they are extracted before that call, so
    the closed-form CRPS can score them on the device.

    Raises:
        TypeError: If the closed-form CRPS is requested and ``prediction`` does
            not implement [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS].
    """
    fields = requirements.fields
    if fields.closed_form_crps:
        _require_closed_form_crps(prediction)
    batch_shape = prediction.batch_shape
    reference_samples = _materialize_reference_samples(
        reference_samples_extractor,
        batch=batch,
        key=key,
        n_scoring_units=math.prod(jnp.shape(targets)),
        required=fields.reference_samples,
    )
    against_reference = fields.closed_form_crps and reference_samples is not None
    sample_count = (
        requirements.predictive_sample_count if fields.predictive_samples else None
    )
    # A subclass not registered as a pytree itself cannot cross a jit boundary.
    compute = (
        _compiled_device_fields
        if not jax.tree_util.treedef_is_leaf(jax.tree.structure(prediction))
        else _device_fields
    )
    device = jax.device_get(
        compute(
            prediction,
            (
                jnp.asarray(targets)
                if fields.log_likelihood
                or (fields.closed_form_crps and not against_reference)
                else None
            ),
            (
                _rows_to_samples(reference_samples, batch_shape)
                if against_reference
                else None
            ),
            key,
            variance=fields.variance,
            sample_count=sample_count,
            log_likelihood=fields.log_likelihood,
            closed_form_crps=fields.closed_form_crps,
        )
    )
    mean, target_values = _host_mean_and_targets(device["mean"], targets, batch_shape)
    return EpochPredictionData(
        targets=target_values,
        mean=mean,
        variance=_host_per_unit(device.get("variance"), batch_shape),
        predictive_samples=(
            None
            if sample_count is None
            else _samples_to_rows(
                device["predictive_samples"],
                prediction_shape=batch_shape,
                name="predictive_samples",
            )
        ),
        reference_samples=reference_samples,
        intervals=tuple(
            interval(prediction, level) for level in sorted(fields.interval_levels)
        ),
        coordinate=_materialize_coordinate(
            coordinate_extractor,
            batch=batch,
            required=fields.coordinate,
        ),
        evaluation_grid=_required_evaluation_grid(requirements),
        log_likelihood=_host_per_unit(device.get("log_likelihood"), batch_shape),
        closed_form_crps=_host_per_unit(device.get("closed_form_crps"), batch_shape),
        closed_form_crps_against_reference=against_reference,
    )


def _require_closed_form_crps(prediction: object) -> ClosedFormCRPS:
    """Return ``prediction`` narrowed to a closed-form CRPS, or refuse it.

    The closed-form CRPS never falls back to a sample estimate (ADR 0015).

    Args:
        prediction: The predictive distribution to be scored.

    Returns:
        ``prediction`` itself, typed as
        [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS].

    Raises:
        TypeError: If ``prediction`` does not implement
            [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS].
    """
    if not isinstance(prediction, ClosedFormCRPS):
        raise TypeError(
            f"{type(prediction).__name__} has no closed-form CRPS: it does not "
            "implement ClosedFormCRPS (a crps method). Score it with "
            "SampleContinuousRankedProbabilityScore instead, or implement crps."
        )
    return prediction


def _device_fields(
    prediction: PredictiveDistribution,
    targets: jax.Array | None,
    references: jax.Array | None,
    key: jax.Array,
    *,
    variance: bool,
    sample_count: int | None,
    log_likelihood: bool,
    closed_form_crps: bool,
) -> dict[str, jax.Array]:
    """Compute a prediction's requested device fields.

    Args:
        prediction: The predictive distribution.
        targets: The observed targets, shaped like the prediction's batch,
            or ``None`` when no field needs them.
        references: Reference samples shaped ``[R, *batch]`` for the
            closed-form CRPS to score instead of ``targets``, or ``None``.
        key: The metric key; predictive samples use ``fold_in(key, 0)``.
        variance: Whether to compute the predictive variance.
        sample_count: The number of predictive samples, or ``None`` for none.
        log_likelihood: Whether to compute the targets' log-density.
        closed_form_crps: Whether to compute the closed-form CRPS of the
            references, or of the targets without references; ``prediction``
            must then implement
            [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS].

    Returns:
        The mean, and each requested field under its ``EpochPredictionData``
        name.

    Raises:
        ValueError: If a requested field has nothing to score: the
            log-likelihood without ``targets``, or the closed-form CRPS
            without ``targets`` or ``references``.
        TypeError: If the closed-form CRPS is requested and ``prediction``
            does not implement [`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS].
    """
    computed = {"mean": prediction.mean()}
    if variance:
        computed["variance"] = prediction.variance()
    if sample_count is not None:
        computed["predictive_samples"] = prediction.sample(
            jax.random.fold_in(key, 0), sample_shape=(sample_count,)
        )
    if log_likelihood:
        if targets is None:
            raise ValueError("the log-likelihood requires targets.")
        computed["log_likelihood"] = prediction.log_prob(targets)
    if closed_form_crps:
        scored = targets if references is None else references
        if scored is None:
            raise ValueError("the closed-form CRPS requires targets or references.")
        computed["closed_form_crps"] = _require_closed_form_crps(prediction).crps(
            scored
        )
    return computed


_compiled_device_fields = jax.jit(
    _device_fields,
    static_argnames=("variance", "sample_count", "log_likelihood", "closed_form_crps"),
)


def _require_targets(batch: Batch) -> Any:
    """Return batch targets required by host-side epoch metrics."""
    if batch.targets is None:
        raise ValueError("batch.targets must be provided for epoch metrics.")
    return batch.targets


def _require_scalar_gaussian(prediction: object) -> Gaussian:
    """Validate and return a scalar-event Gaussian prediction."""
    if not isinstance(prediction, Gaussian):
        raise TypeError("GaussianPredictor requires a model returning Gaussian.")
    if prediction.event_shape != ():
        raise ValueError("epoch metrics support scalar distribution events only.")
    return prediction


def _host_mean_and_targets(
    mean: ArrayLike, targets: ArrayLike, batch_shape: tuple[int, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return shape-compatible host predictive means and targets, flattened."""
    host_mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), batch_shape)
    target_values = np.asarray(jax.device_get(targets), dtype=np.float64)
    if host_mean.shape != target_values.shape:
        raise ValueError("prediction mean and targets must have matching shapes.")
    return host_mean.reshape(-1), target_values.reshape(-1)


def _host_per_unit(
    values: Any, batch_shape: tuple[int, ...]
) -> NDArray[np.float64] | None:
    """Broadcast host values to the prediction's shape and flatten them."""
    if values is None:
        return None
    return np.broadcast_to(np.asarray(values, dtype=np.float64), batch_shape).reshape(
        -1
    )


def _materialize_reference_samples(
    extractor: ReferenceSamplesExtractor | None,
    *,
    batch: Batch,
    key: jax.Array,
    n_scoring_units: int,
    required: bool,
) -> NDArray[np.float64] | None:
    """Materialize and validate empirical reference samples when requested."""
    if not required:
        return None
    if extractor is None:
        raise ValueError("conditional CRPS requires a reference_samples_extractor.")
    samples = np.asarray(
        jax.device_get(extractor(batch, jax.random.fold_in(key, 1))),
        dtype=np.float64,
    )
    if samples.ndim != 2 or samples.shape[0] != n_scoring_units:
        raise ValueError(
            "reference samples must have shape (n_scoring_units, n_samples)."
        )
    return samples


def _materialize_coordinate(
    extractor: CoordinateExtractor | None,
    *,
    batch: Batch,
    required: bool,
) -> NDArray[np.float64] | None:
    """Materialize an explicit scoring-unit coordinate when requested."""
    if not required:
        return None
    if extractor is None:
        raise ValueError("coordinate-based metrics require a coordinate_extractor.")
    return np.asarray(
        jax.device_get(extractor(batch)),
        dtype=np.float64,
    ).reshape(-1)


def _required_evaluation_grid(
    requirements: PredictionRequirements,
) -> EvaluationGrid | None:
    """Return the configured grid only when requested by the metric suite."""
    if not requirements.fields.evaluation_grid:
        return None
    return requirements.evaluation_grid


def _samples_to_rows(
    samples: Any,
    *,
    prediction_shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float64]:
    """Convert JAX sample-first samples to host scoring-unit rows."""
    array = np.asarray(jax.device_get(samples), dtype=np.float64)
    if array.ndim != len(prediction_shape) + 1 or array.shape[1:] != prediction_shape:
        raise ValueError(f"{name} must have shape (n_samples, *prediction_shape).")
    return np.moveaxis(array, 0, -1).reshape(-1, array.shape[0])


def _rows_to_samples(
    rows: NDArray[np.float64], batch_shape: tuple[int, ...]
) -> jax.Array:
    """Convert host scoring-unit rows to JAX sample-first samples.

    ``device_put`` rather than ``jnp.asarray``, which compiles a copy per shape.

    Args:
        rows: Samples per scoring unit, shaped ``(units, R)`` with ``units``
            the product of ``batch_shape``.
        batch_shape: The prediction's batch shape.

    Returns:
        The samples on the device, shaped ``[R, *batch]``.
    """
    return jax.device_put(rows.T.reshape(rows.shape[1], *batch_shape))


def _gaussian_interval(prediction: Gaussian, level: float) -> PredictionInterval:
    """Materialize an exact central Gaussian interval at ``level``."""
    z_value = NormalDist().inv_cdf((1.0 + level) / 2.0)
    lower = prediction.loc - z_value * prediction.scale
    upper = prediction.loc + z_value * prediction.scale
    return PredictionInterval(
        level=level,
        lower=np.asarray(jax.device_get(lower), dtype=np.float64).reshape(-1),
        upper=np.asarray(jax.device_get(upper), dtype=np.float64).reshape(-1),
    )


def _mixture_interval(
    prediction: PosteriorPredictive, level: float
) -> PredictionInterval:
    """Materialize a central interval from posterior predictive quantiles."""
    lower = prediction.quantile((1.0 - level) / 2.0)
    upper = prediction.quantile((1.0 + level) / 2.0)
    return PredictionInterval(
        level=level,
        lower=np.asarray(jax.device_get(lower), dtype=np.float64).reshape(-1),
        upper=np.asarray(jax.device_get(upper), dtype=np.float64).reshape(-1),
    )


@dataclass(frozen=True, slots=True)
class MetricSuite:
    """Complete metric registration and materialization configuration.

    Attributes:
        batch: Batch metrics evaluated on every batch.
        epoch: [`EpochMetric`][stagereg.core.EpochMetric]s evaluated once per epoch
            over the merged predictions.
        predictor: The [`Predictor`][stagereg.jax.Predictor] that turns model
            outputs into host arrays. Required when ``epoch`` is non-empty.
        predictive_sample_count: Number of predictive samples per scoring unit.
            Required by sample-based epoch metrics.
        evaluation_grid: Grid the CRPS metrics integrate over. Required by CRPS
            epoch metrics.
    """

    batch: tuple[BatchMetricSpec, ...] = ()
    epoch: tuple[EpochMetric, ...] = ()
    predictor: Predictor | None = None
    predictive_sample_count: int | None = None
    evaluation_grid: EvaluationGrid | None = None

    def __post_init__(self) -> None:
        """Validate names and explicit epoch materialization configuration.

        Raises:
            ValueError: If names/configuration are invalid, a predictor is
                absent, or a metric scores reference samples and the predictor's
                ``reference_samples_extractor`` is ``None``, or the suite asks
                for the closed-form CRPS of both targets and reference samples.
        """
        registered_names = [spec.name for spec in self.batch] + [
            metric.name for metric in self.epoch
        ]
        if any(not name for name in registered_names):
            raise ValueError("metric names must not be empty.")
        if "loss" in registered_names:
            raise ValueError("'loss' is reserved and cannot be registered as a metric.")
        duplicate_names = {
            name for name, count in Counter(registered_names).items() if count > 1
        }
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"duplicate metric names are not allowed: {duplicates}.")
        if self.epoch and self.predictor is None:
            raise ValueError("epoch metrics require an explicit suite predictor.")
        requirements = self.prediction_requirements
        if (
            requirements.fields.reference_samples
            and getattr(self.predictor, "reference_samples_extractor", ...) is None
        ):
            raise ValueError(
                "metrics that score reference samples require the predictor's "
                "reference_samples_extractor."
            )
        # The predictor scores the closed-form CRPS against the references
        # whenever they are materialised, so no metric may expect the target.
        if requirements.fields.reference_samples and any(
            metric.requirements.closed_form_crps
            and not metric.requirements.reference_samples
            for metric in self.epoch
        ):
            raise ValueError(
                "a suite that materialises reference samples scores the closed-form "
                "CRPS against them, so it cannot also score the closed-form CRPS "
                "against the target."
            )

    @property
    def metric_requirements(self) -> MetricRequirements:
        """Return the union of registered epoch metric requirements."""
        combined = MetricRequirements()
        for metric in self.epoch:
            combined = combined.union(metric.requirements)
        return combined

    @property
    def prediction_requirements(self) -> PredictionRequirements:
        """Return combined fields plus explicit suite configuration."""
        return PredictionRequirements(
            fields=self.metric_requirements,
            predictive_sample_count=self.predictive_sample_count,
            evaluation_grid=self.evaluation_grid,
        )


def metric_key(batch_key: jax.Array) -> jax.Array:
    """Derive a stable metric-only key without advancing runner RNG state."""
    return jax.random.fold_in(batch_key, _METRIC_RNG_NAMESPACE)


def resolve_epoch_prediction_data(
    *,
    suite: MetricSuite,
    model: nnx.Module,
    batch: Batch,
    key: jax.Array,
) -> EpochPredictionData:
    """Materialize one batch using an isolated inference-mode model clone."""
    if suite.predictor is None:
        raise ValueError("epoch metrics require an explicit suite predictor.")
    prediction_model = nnx.clone(model)
    prediction_model.eval()
    return suite.predictor(
        prediction_model,
        batch,
        suite.prediction_requirements,
        key,
    )


class _CompiledInferenceModel(nnx.Module):
    """Run a live model's inference-mode forward pass as one compiled call.

    The inference-mode structure is taken once, from a single clone switched
    to ``eval()``; every call then merges it with the live
    model's current state inside a jitted function. Whatever the forward pass
    writes to its variables stays inside that call, so the live model is
    never mutated, and the forward pass compiles once per input shape.
    """

    def __init__(self, model: nnx.Module) -> None:
        """Take the inference-mode structure of ``model`` and compile its call.

        Args:
            model: The live model, whose current state each call reads.
        """
        inference_model = nnx.clone(model)
        inference_model.eval()
        graphdef, _ = nnx.split(inference_model)

        def forward(state: nnx.State, inputs: PyTree) -> Any:
            return nnx.merge(graphdef, state)(inputs)

        self._forward = jax.jit(forward)
        self._model = model

    def __call__(self, inputs: PyTree) -> Any:
        """Predict on ``inputs`` with the live model's current state.

        Args:
            inputs: Batch inputs.

        Returns:
            The inference-mode model output, a JAX pytree.
        """
        return self._forward(nnx.state(self._model), inputs)


def make_epoch_prediction_resolver(
    *,
    suite: MetricSuite,
    model: nnx.Module,
) -> Callable[[Batch, jax.Array], EpochPredictionData]:
    """Build a per-batch epoch-prediction resolver for a model being trained.

    Unlike ``resolve_epoch_prediction_data``, which clones the model and
    runs it eagerly on every batch, the resolver takes the inference-mode
    structure once and runs every forward pass compiled, against the model's
    state at that moment. The live model is never mutated. The model's
    output must be a JAX pytree, as
    [`Gaussian`][stagereg.jax.Gaussian] and
    [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive] are.

    Args:
        suite: The metric suite whose predictor materializes each batch.
        model: The live model; its parameters may change between calls.

    Returns:
        A callable ``(batch, key) -> EpochPredictionData``.

    Raises:
        ValueError: If the suite has no predictor.
    """
    predictor = suite.predictor
    if predictor is None:
        raise ValueError("epoch metrics require an explicit suite predictor.")
    requirements = suite.prediction_requirements
    inference_model = _CompiledInferenceModel(model)

    def resolve(batch: Batch, key: jax.Array, /) -> EpochPredictionData:
        return predictor(inference_model, batch, requirements, key)

    return resolve


def merge_epoch_prediction_data(
    parts: Sequence[EpochPredictionData],
) -> EpochPredictionData:
    """Merge batches according to each typed field's scoring axes.

    Args:
        parts: Per-batch host-materialized prediction data.

    Returns:
        One validated epoch prediction object.

    Raises:
        ValueError: If parts are empty or optional fields/configuration conflict.
    """
    if not parts:
        raise ValueError("at least one epoch prediction batch is required.")
    interval_levels = {interval.level for interval in parts[0].intervals}
    if any(
        {interval.level for interval in part.intervals} != interval_levels
        for part in parts
    ):
        raise ValueError("prediction interval levels must match across batches.")
    intervals = tuple(
        PredictionInterval(
            level=level,
            lower=np.concatenate([part.interval(level).lower for part in parts]),
            upper=np.concatenate([part.interval(level).upper for part in parts]),
        )
        for level in sorted(interval_levels)
    )
    return EpochPredictionData(
        targets=np.concatenate([part.targets for part in parts]),
        mean=np.concatenate([part.mean for part in parts]),
        variance=_concat_optional_vectors(parts, "variance"),
        predictive_samples=_concat_optional_samples(parts, "predictive_samples"),
        reference_samples=_concat_optional_samples(parts, "reference_samples"),
        intervals=intervals,
        coordinate=_concat_optional_vectors(parts, "coordinate"),
        evaluation_grid=_merge_grid(parts),
        log_likelihood=_concat_optional_vectors(parts, "log_likelihood"),
        closed_form_crps=_concat_optional_vectors(parts, "closed_form_crps"),
        closed_form_crps_against_reference=_merge_closed_form_crps_scoring(parts),
    )


def _merge_closed_form_crps_scoring(parts: Sequence[EpochPredictionData]) -> bool:
    """Return whether every part's closed-form CRPS scored reference samples.

    Args:
        parts: The per-batch prediction data being merged, at least one.

    Returns:
        The parts' shared ``closed_form_crps_against_reference`` flag.

    Raises:
        ValueError: If some parts scored the target and others the references.
    """
    scoring = {part.closed_form_crps_against_reference for part in parts}
    if len(scoring) > 1:
        raise ValueError(
            "closed_form_crps must score the target in all batches or the "
            "reference samples in all batches."
        )
    return scoring.pop()


def _concat_optional_vectors(
    parts: Sequence[EpochPredictionData], name: str
) -> NDArray[np.float64] | None:
    """Concatenate an optional vector field with all-or-none semantics."""
    values = [getattr(part, name) for part in parts]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} must be present for all batches or none.")
    return np.concatenate(values)  # type: ignore[arg-type]


def _concat_optional_samples(
    parts: Sequence[EpochPredictionData], name: str
) -> NDArray[np.float64] | None:
    """Concatenate sample rows while requiring a stable sample count."""
    values = [getattr(part, name) for part in parts]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} must be present for all batches or none.")
    matrices = values  # narrowed by checks above
    sample_counts = {matrix.shape[1] for matrix in matrices if matrix is not None}
    if len(sample_counts) != 1:
        raise ValueError(f"{name} sample counts must match across batches.")
    return np.concatenate(matrices, axis=0)  # type: ignore[arg-type]


def _merge_grid(parts: Sequence[EpochPredictionData]) -> EvaluationGrid | None:
    """Retain one shared grid and reject missing or conflicting grids."""
    grids = [part.evaluation_grid for part in parts]
    if all(grid is None for grid in grids):
        return None
    if any(grid is None for grid in grids):
        raise ValueError("evaluation_grid must be present for all batches or none.")
    first = grids[0]
    assert first is not None
    if any(
        grid is None or not np.array_equal(grid.values, first.values)
        for grid in grids[1:]
    ):
        raise ValueError("evaluation_grid must be shared across all batches.")
    return first


def initialize_batch_metric_values(
    metrics: Sequence[BatchMetricSpec],
) -> dict[str, list[StepScalar]]:
    """Create per-metric accumulation buffers for raw step outputs."""
    return {spec.name: [] for spec in metrics}


def collect_step_metrics(
    step_output: Mapping[str, Any],
    *,
    metrics: Sequence[BatchMetricSpec],
    losses: list[StepScalar] | None,
    batch_metric_values: dict[str, list[StepScalar]],
    context: str,
) -> None:
    """Append one step's loss and registered batch metrics as they are.

    The values are not read on the host, so device arrays stay on the device
    and the next step can be dispatched without waiting for this one;
    ``step_outputs_to_host`` moves
    them once the epoch ends. Passing ``losses=None`` collects only the
    registered batch metrics, for steps that compute no loss at all.
    """
    if losses is not None:
        losses.append(step_output["loss"])
    for spec in metrics:
        if spec.name not in step_output:
            raise ValueError(
                f"{context} did not produce registered metric {spec.name!r}."
            )
        batch_metric_values[spec.name].append(step_output[spec.name])


def step_outputs_to_host[T](values: T) -> T:
    """Move accumulated step outputs to the host in one transfer.

    Args:
        values: A pytree whose leaves are scalar step outputs, either device
            arrays or Python numbers.

    Returns:
        ``values`` with every leaf replaced by a Python float.
    """
    return jax.tree.map(float, jax.device_get(values))


def maybe_collect_epoch_prediction_data(
    parts: list[EpochPredictionData] | None,
    *,
    suite: MetricSuite,
    model: nnx.Module,
    batch: Batch,
    batch_key: jax.Array,
) -> None:
    """Append one batch's typed data when epoch metrics are enabled."""
    if parts is None:
        return
    parts.append(
        resolve_epoch_prediction_data(
            suite=suite,
            model=model,
            batch=batch,
            key=metric_key(batch_key),
        )
    )


def reduce_metric_suite(
    *,
    suite: MetricSuite,
    losses: Sequence[Any] | None,
    batch_metric_values: Mapping[str, Sequence[Any]],
    epoch_metric_parts: Sequence[EpochPredictionData] | None,
) -> dict[str, float]:
    """Reduce accumulated batch and typed epoch metrics into one mapping.

    ``losses`` and ``batch_metric_values`` may hold raw step outputs; they
    are moved to the host together before reducing, so each
    [`BatchMetricSpec.reduce`][stagereg.jax.BatchMetricSpec] still
    receives Python floats. Passing ``losses=None`` omits the ``"loss"`` key
    entirely, for evaluations that compute no loss. This is distinct from an
    empty sequence, which means a loss was requested but no batch produced
    one.
    """
    losses, batch_metric_values = step_outputs_to_host(
        (losses, dict(batch_metric_values))
    )
    reduced: dict[str, float] = {}
    if losses is not None:
        reduced["loss"] = _finite_float("loss", _metric_mean(losses))
    for spec in suite.batch:
        reduced[spec.name] = _finite_float(
            spec.name, spec.reduce(batch_metric_values[spec.name])
        )
    if epoch_metric_parts is None:
        return reduced
    merged = merge_epoch_prediction_data(epoch_metric_parts)
    for epoch_metric in suite.epoch:
        reduced[epoch_metric.name] = _finite_float(
            epoch_metric.name, epoch_metric(merged)
        )
    return reduced
