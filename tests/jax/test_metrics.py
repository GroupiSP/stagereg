from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from statistics import NormalDist

from hypothesis import given, settings
from hypothesis import strategies as st

from stagereg.core.metric_registry import (
    ClosedFormContinuousRankedProbabilityScore,
    ContinuousRankedProbabilityScore,
    EpochPredictionData,
    EvaluationGrid,
    IntervalCoverage,
    MetricRequirements,
    NegativeLogLikelihood,
    PointContinuousRankedProbabilityScore,
    PredictionInterval,
    RootMeanSquaredError,
    WeightedSpread,
)
from stagereg.core.metrics import expected_gaussian_mixture_crps
from stagereg.core.types import Batch
from stagereg.jax.distributions import Gaussian, GaussianHead, PosteriorPredictive
from stagereg.jax.metrics import (
    GaussianPredictor,
    MetricSuite,
    PosteriorPredictivePredictor,
    PredictionRequirements,
    merge_epoch_prediction_data,
    reduce_metric_suite,
    make_epoch_prediction_resolver,
    metric_key,
    resolve_epoch_prediction_data,
)


class GaussianModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> Gaussian:
        return Gaussian(loc=2.0 * inputs, scale=jnp.ones_like(inputs) * 0.5)


class BroadcastScaleGaussianModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> Gaussian:
        return Gaussian(loc=2.0 * inputs, scale=jnp.array(0.5))


class VectorEventGaussian(Gaussian):
    @property
    def event_shape(self) -> tuple[int, ...]:
        return (1,)


class VectorEventModel(nnx.Module):
    def __call__(self, inputs: jax.Array) -> Gaussian:
        return VectorEventGaussian(loc=inputs, scale=jnp.ones_like(inputs))


class StatefulGaussianModel(nnx.Module):
    def __init__(self) -> None:
        self.call_count = nnx.Variable(jnp.array(0))

    def __call__(self, inputs: jax.Array) -> Gaussian:
        self.call_count[...] += 1
        return Gaussian(loc=inputs, scale=jnp.ones_like(inputs))


def coordinate_from_batch(batch: Batch) -> object:
    return batch.metadata["coordinate"]


def reference_from_batch(batch: Batch, key: jax.Array) -> object:
    del key
    return batch.metadata["reference_samples"]


def stateful_predictor(
    model: nnx.Module,
    batch: Batch,
    requirements: PredictionRequirements,
    key: jax.Array,
) -> EpochPredictionData:
    del requirements, key
    prediction = model(batch.inputs)
    assert isinstance(prediction, Gaussian)
    assert batch.targets is not None
    return EpochPredictionData(
        targets=np.asarray(batch.targets).reshape(-1),
        mean=np.asarray(prediction.mean()).reshape(-1),
    )


def test_suite_combines_requirements_and_requires_explicit_sampling_config() -> None:
    predictor = GaussianPredictor(coordinate_extractor=coordinate_from_batch)
    grid = EvaluationGrid(np.linspace(-5.0, 5.0, 51))
    suite = MetricSuite(
        epoch=(
            RootMeanSquaredError(),
            IntervalCoverage(level=0.9),
            WeightedSpread(level=0.8),
            PointContinuousRankedProbabilityScore(),
        ),
        predictor=predictor,
        predictive_sample_count=16,
        evaluation_grid=grid,
    )

    requirements = suite.prediction_requirements
    assert requirements.fields.predictive_samples
    assert requirements.fields.coordinate
    assert requirements.fields.interval_levels == frozenset({0.8, 0.9})
    assert requirements.predictive_sample_count == 16
    assert requirements.evaluation_grid is grid

    with pytest.raises(ValueError, match="predictive_sample_count"):
        MetricSuite(
            epoch=(PointContinuousRankedProbabilityScore(),),
            predictor=GaussianPredictor(),
            evaluation_grid=grid,
        )
    with pytest.raises(ValueError, match="evaluation_grid"):
        MetricSuite(
            epoch=(PointContinuousRankedProbabilityScore(),),
            predictor=GaussianPredictor(),
            predictive_sample_count=4,
        )
    with pytest.raises(ValueError, match="explicit suite predictor"):
        MetricSuite(epoch=(RootMeanSquaredError(),))


def test_gaussian_predictor_rejects_non_scalar_events() -> None:
    batch = Batch(inputs=jnp.ones((2, 1)), targets=jnp.ones((2, 1)))
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=GaussianPredictor())

    with pytest.raises(ValueError, match="scalar distribution events only"):
        resolve_epoch_prediction_data(
            suite=suite,
            model=VectorEventModel(),
            batch=batch,
            key=jax.random.key(0),
        )


def test_gaussian_predictor_materializes_only_requested_fields() -> None:
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0]]),
        targets=jnp.array([[0.25], [1.75]]),
    )
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=GaussianPredictor())

    data = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=jax.random.key(0),
    )

    assert np.array_equal(data.targets, np.array([0.25, 1.75]))
    assert np.array_equal(data.mean, np.array([0.0, 2.0]))
    assert data.variance is None
    assert data.predictive_samples is None
    assert data.intervals == ()


def test_gaussian_predictor_broadcasts_variance_to_prediction_shape() -> None:
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0]]),
        targets=jnp.array([[0.25], [1.75]]),
    )
    data = GaussianPredictor()(
        BroadcastScaleGaussianModel(),
        batch,
        PredictionRequirements(fields=MetricRequirements(variance=True)),
        jax.random.key(0),
    )

    assert data.mean.shape == (2,)
    assert data.variance is not None
    assert data.variance.shape == (2,)
    assert np.array_equal(data.variance, np.full(2, 0.25))


def test_epoch_prediction_resolver_does_not_mutate_the_live_model() -> None:
    model = StatefulGaussianModel()
    batch = Batch(inputs=jnp.ones((2, 1)), targets=jnp.ones((2, 1)))
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=stateful_predictor)

    resolve_epoch_prediction_data(
        suite=suite,
        model=model,
        batch=batch,
        key=jax.random.key(0),
    )

    assert int(model.call_count[...]) == 0


def test_gaussian_predictor_materializes_samples_intervals_and_coordinate() -> None:
    grid = EvaluationGrid(np.linspace(-5.0, 5.0, 51))
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0], [2.0]]),
        targets=jnp.array([[0.0], [2.0], [4.0]]),
        metadata={"coordinate": jnp.array([0.0, 1.0, 2.0])},
    )
    suite = MetricSuite(
        epoch=(
            PointContinuousRankedProbabilityScore(),
            IntervalCoverage(level=0.8),
            WeightedSpread(level=0.8),
        ),
        predictor=GaussianPredictor(coordinate_extractor=coordinate_from_batch),
        predictive_sample_count=7,
        evaluation_grid=grid,
    )

    data = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=jax.random.key(4),
    )

    assert data.predictive_samples is not None
    assert data.predictive_samples.shape == (3, 7)
    assert data.coordinate is not None
    assert np.array_equal(data.coordinate, np.array([0.0, 1.0, 2.0]))
    interval = data.interval(0.8)
    assert np.all(interval.lower < data.mean)
    assert np.all(interval.upper > data.mean)
    assert data.evaluation_grid is grid


def test_gaussian_predictor_is_deterministic_for_metric_key() -> None:
    batch = Batch(inputs=jnp.ones((2, 1)), targets=jnp.ones((2, 1)))
    suite = MetricSuite(
        epoch=(PointContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
        predictive_sample_count=8,
        evaluation_grid=EvaluationGrid(np.linspace(-3.0, 3.0, 31)),
    )

    first = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=metric_key(jax.random.key(1)),
    )
    second = resolve_epoch_prediction_data(
        suite=suite,
        model=GaussianModel(),
        batch=batch,
        key=metric_key(jax.random.key(1)),
    )

    assert np.array_equal(first.predictive_samples, second.predictive_samples)


def test_conditional_crps_reference_samples_use_explicit_extractor() -> None:
    reference = jnp.array([[0.0, 0.1], [2.0, 2.1]])
    batch = Batch(
        inputs=jnp.array([[0.0], [1.0]]),
        targets=jnp.array([[0.0], [2.0]]),
        metadata={"reference_samples": reference},
    )
    suite = MetricSuite(
        epoch=(ContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(reference_samples_extractor=reference_from_batch),
        predictive_sample_count=3,
        evaluation_grid=EvaluationGrid(np.linspace(-3.0, 3.0, 31)),
    )

    data = resolve_epoch_prediction_data(
        suite=suite, model=GaussianModel(), batch=batch, key=jax.random.key(0)
    )

    assert np.array_equal(data.reference_samples, np.asarray(reference))


def _part(
    targets: list[float],
    *,
    samples: int = 3,
    grid: EvaluationGrid | None = None,
) -> EpochPredictionData:
    n_units = len(targets)
    return EpochPredictionData(
        targets=np.asarray(targets),
        mean=np.asarray(targets),
        variance=np.ones(n_units),
        predictive_samples=np.ones((n_units, samples)),
        intervals=(PredictionInterval(0.9, np.zeros(n_units), np.full(n_units, 2.0)),),
        coordinate=np.arange(n_units, dtype=float),
        evaluation_grid=grid,
    )


def test_merge_handles_unequal_batch_sizes_by_scoring_axis() -> None:
    grid = EvaluationGrid(np.linspace(-2.0, 2.0, 9))

    merged = merge_epoch_prediction_data(
        [_part([0.0, 1.0], grid=grid), _part([2.0], grid=grid)]
    )

    assert merged.targets.shape == (3,)
    assert merged.predictive_samples is not None
    assert merged.predictive_samples.shape == (3, 3)
    assert merged.interval(0.9).lower.shape == (3,)
    assert merged.evaluation_grid is grid


@given(
    values=st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=3,
        max_size=30,
    ),
    first_end=st.integers(min_value=1),
    second_end=st.integers(min_value=2),
)
def test_merge_is_associative_across_batch_partitions(
    values: list[float],
    first_end: int,
    second_end: int,
) -> None:
    max_first_end = len(values) - 2
    first_end = 1 + (first_end - 1) % max_first_end
    max_second_span = len(values) - first_end - 1
    second_end = first_end + 1 + (second_end - 2) % max_second_span
    parts = [
        _part(values[:first_end]),
        _part(values[first_end:second_end]),
        _part(values[second_end:]),
    ]

    direct = merge_epoch_prediction_data(parts)
    regrouped = merge_epoch_prediction_data(
        [merge_epoch_prediction_data(parts[:2]), parts[2]]
    )

    assert np.array_equal(direct.targets, regrouped.targets)
    assert np.array_equal(direct.mean, regrouped.mean)
    assert np.array_equal(direct.variance, regrouped.variance)
    assert np.array_equal(direct.predictive_samples, regrouped.predictive_samples)
    assert np.array_equal(direct.coordinate, regrouped.coordinate)
    assert np.array_equal(
        direct.interval(0.9).lower,
        regrouped.interval(0.9).lower,
    )
    assert np.array_equal(
        direct.interval(0.9).upper,
        regrouped.interval(0.9).upper,
    )


def test_merge_rejects_conflicting_sample_counts_and_grids() -> None:
    first_grid = EvaluationGrid(np.array([0.0, 1.0, 2.0]))
    second_grid = EvaluationGrid(np.array([0.0, 1.5, 2.0]))

    with pytest.raises(ValueError, match="sample counts"):
        merge_epoch_prediction_data([_part([0.0], samples=2), _part([1.0], samples=3)])
    with pytest.raises(ValueError, match="shared"):
        merge_epoch_prediction_data(
            [_part([0.0], grid=first_grid), _part([1.0], grid=second_grid)]
        )


def test_weighted_spread_is_independent_of_loader_coordinate_order() -> None:
    first = EpochPredictionData(
        targets=np.zeros(2),
        mean=np.zeros(2),
        intervals=(
            PredictionInterval(0.9, np.array([1.0, 1.0]), np.array([3.0, 5.0])),
        ),
        coordinate=np.array([2.0, 3.0]),
    )
    second = EpochPredictionData(
        targets=np.zeros(2),
        mean=np.zeros(2),
        intervals=(
            PredictionInterval(0.9, np.array([0.0, 1.0]), np.array([2.0, 3.0])),
        ),
        coordinate=np.array([0.0, 1.0]),
    )

    merged = merge_epoch_prediction_data([first, second])

    assert WeightedSpread(level=0.9)(merged) == pytest.approx(
        WeightedSpread(level=0.9)(
            EpochPredictionData(
                targets=np.zeros(4),
                mean=np.zeros(4),
                intervals=(
                    PredictionInterval(
                        0.9,
                        np.array([0.0, 1.0, 1.0, 1.0]),
                        np.array([2.0, 3.0, 3.0, 5.0]),
                    ),
                ),
                coordinate=np.array([0.0, 1.0, 2.0, 3.0]),
            )
        )
    )


class MixtureModel(nnx.Module):
    """A posterior predictive whose draws are the inputs shifted by fixed offsets."""

    def __init__(self, offsets: list[float], variance: float = 0.25) -> None:
        self.offsets = jnp.asarray(offsets)
        self.variance = variance

    def __call__(self, inputs: jax.Array) -> PosteriorPredictive:
        draws = inputs[None] + self.offsets.reshape((-1,) + (1,) * inputs.ndim)
        return PosteriorPredictive(
            draws=draws, aleatoric_variance=jnp.full(inputs.shape, self.variance)
        )


class ShiftedGaussianModel(nnx.Module):
    """The Gaussian a one-draw ``MixtureModel`` reduces to."""

    def __init__(self, offset: float, variance: float = 0.25) -> None:
        self.offset = offset
        self.variance = variance

    def __call__(self, inputs: jax.Array) -> Gaussian:
        return Gaussian(
            loc=inputs + self.offset,
            scale=jnp.full(inputs.shape, self.variance**0.5),
        )


def _mixture_suite(predictor: object) -> MetricSuite:
    return MetricSuite(
        epoch=(
            NegativeLogLikelihood(),
            PointContinuousRankedProbabilityScore(name="crps"),
            ClosedFormContinuousRankedProbabilityScore(),
            IntervalCoverage(level=0.9),
        ),
        predictor=predictor,  # type: ignore[arg-type]
        predictive_sample_count=64,
        evaluation_grid=EvaluationGrid(np.linspace(-8.0, 8.0, 161)),
    )


def _scores(suite: MetricSuite, model: nnx.Module, batch: Batch) -> dict[str, float]:
    data = resolve_epoch_prediction_data(
        suite=suite, model=model, batch=batch, key=metric_key(jax.random.key(3))
    )
    return reduce_metric_suite(
        suite=suite, losses=None, batch_metric_values={}, epoch_metric_parts=[data]
    )


_MIXTURE_BATCH = Batch(
    inputs=jnp.array([[-1.0], [0.0], [0.5], [2.0]]),
    targets=jnp.array([[-0.5], [0.3], [1.5], [1.0]]),
)


@settings(deadline=None, max_examples=10)
@given(offset=st.floats(min_value=-2.0, max_value=2.0, width=32))
def test_posterior_predictive_predictor_with_one_draw_scores_as_the_gaussian(
    offset: float,
) -> None:
    mixture = _scores(
        _mixture_suite(PosteriorPredictivePredictor()),
        MixtureModel([offset]),
        _MIXTURE_BATCH,
    )
    gaussian = _scores(
        _mixture_suite(GaussianPredictor()),
        ShiftedGaussianModel(offset),
        _MIXTURE_BATCH,
    )

    assert mixture.keys() == {"nll", "crps", "closed_form_crps", "coverage"}
    assert mixture == pytest.approx(gaussian, rel=1e-4)


def test_posterior_predictive_predictor_scores_the_exact_mixture() -> None:
    offsets = [-1.5, 0.0, 2.5]
    variance = 0.25
    level = 0.9
    data = resolve_epoch_prediction_data(
        suite=_mixture_suite(PosteriorPredictivePredictor()),
        model=MixtureModel(offsets, variance),
        batch=_MIXTURE_BATCH,
        key=jax.random.key(0),
    )
    inputs = np.asarray(_MIXTURE_BATCH.inputs, np.float64).reshape(-1)
    targets = np.asarray(_MIXTURE_BATCH.targets, np.float64).reshape(-1)
    standard = NormalDist(sigma=variance**0.5)

    def density(y: float, x: float) -> float:
        return float(np.mean([standard.pdf(y - x - o) for o in offsets]))

    def cdf(y: float, x: float) -> float:
        return float(np.mean([standard.cdf(y - x - o) for o in offsets]))

    assert data.log_likelihood is not None
    np.testing.assert_allclose(
        data.log_likelihood,
        [np.log(density(y, x)) for y, x in zip(targets, inputs, strict=True)],
        rtol=1e-4,
    )
    interval = data.interval(level)
    np.testing.assert_allclose(
        [cdf(lo, x) for lo, x in zip(interval.lower, inputs, strict=True)],
        (1.0 - level) / 2.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        [cdf(hi, x) for hi, x in zip(interval.upper, inputs, strict=True)],
        (1.0 + level) / 2.0,
        atol=1e-4,
    )
    assert data.predictive_samples is not None
    assert data.predictive_samples.shape == (4, 64)


def test_posterior_predictive_predictor_is_deterministic_for_metric_key() -> None:
    suite = _mixture_suite(PosteriorPredictivePredictor())

    first, second = (
        resolve_epoch_prediction_data(
            suite=suite,
            model=MixtureModel([-1.0, 1.0]),
            batch=_MIXTURE_BATCH,
            key=metric_key(jax.random.key(1)),
        )
        for _ in range(2)
    )

    assert np.array_equal(first.predictive_samples, second.predictive_samples)
    assert np.array_equal(first.log_likelihood, second.log_likelihood)
    assert np.array_equal(first.interval(0.9).lower, second.interval(0.9).lower)


def test_posterior_predictive_predictor_rejects_other_predictions() -> None:
    suite = MetricSuite(
        epoch=(RootMeanSquaredError(),), predictor=PosteriorPredictivePredictor()
    )

    with pytest.raises(TypeError, match="PosteriorPredictive"):
        resolve_epoch_prediction_data(
            suite=suite,
            model=GaussianModel(),
            batch=_MIXTURE_BATCH,
            key=jax.random.key(0),
        )


def test_merge_concatenates_log_likelihood_across_batches() -> None:
    parts = [
        EpochPredictionData(
            targets=np.zeros(n), mean=np.zeros(n), log_likelihood=np.full(n, -float(n))
        )
        for n in (1, 2)
    ]

    merged = merge_epoch_prediction_data(parts)

    assert np.array_equal(merged.log_likelihood, np.array([-1.0, -2.0, -2.0]))


def test_compiled_resolver_matches_the_cloning_resolver() -> None:
    model = GaussianHead(1, 1, rngs=nnx.Rngs(0))
    batch = Batch(
        inputs=jnp.array([[-1.0], [0.0], [2.0]]),
        targets=jnp.array([[-0.5], [0.25], [1.5]]),
    )
    suite = MetricSuite(
        epoch=(
            RootMeanSquaredError(),
            NegativeLogLikelihood(),
            IntervalCoverage(level=0.8),
            PointContinuousRankedProbabilityScore(),
        ),
        predictor=GaussianPredictor(),
        predictive_sample_count=16,
        evaluation_grid=EvaluationGrid(np.linspace(-5.0, 5.0, 51)),
    )
    key = metric_key(jax.random.key(2))

    expected = resolve_epoch_prediction_data(
        suite=suite, model=model, batch=batch, key=key
    )
    actual = make_epoch_prediction_resolver(suite=suite, model=model)(batch, key)

    np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-6)
    np.testing.assert_allclose(
        actual.log_likelihood, expected.log_likelihood, rtol=1e-6
    )
    np.testing.assert_allclose(
        actual.predictive_samples, expected.predictive_samples, rtol=1e-5
    )
    np.testing.assert_allclose(
        actual.interval(0.8).lower, expected.interval(0.8).lower, rtol=1e-6
    )


def test_compiled_resolver_reads_the_live_model_state_at_each_call() -> None:
    model = GaussianHead(1, 1, rngs=nnx.Rngs(0))
    batch = Batch(inputs=jnp.array([[1.0]]), targets=jnp.array([[0.0]]))
    suite = MetricSuite(epoch=(RootMeanSquaredError(),), predictor=GaussianPredictor())
    resolve = make_epoch_prediction_resolver(suite=suite, model=model)

    before = resolve(batch, jax.random.key(0)).mean
    nnx.update(
        model, jax.tree.map(lambda leaf: leaf + 1.0, nnx.state(model, nnx.Param))
    )
    after = resolve(batch, jax.random.key(0)).mean

    assert not np.allclose(before, after)


class FixedPredictionModel(nnx.Module, pytree=False):
    """Returns one prediction, built beforehand, whatever the inputs."""

    def __init__(self, prediction: object) -> None:
        self.prediction = prediction

    def __call__(self, inputs: jax.Array) -> object:
        del inputs
        return self.prediction


_DEVICE_FIELDS = PredictionRequirements(
    fields=MetricRequirements(
        variance=True,
        predictive_samples=True,
        log_likelihood=True,
        closed_form_crps=True,
    ),
    predictive_sample_count=16,
)


@pytest.mark.parametrize(
    ("predictor", "prediction"),
    [
        (
            GaussianPredictor(),
            Gaussian(loc=jnp.linspace(0.0, 1.0, 11), scale=jnp.full(11, 0.5)),
        ),
        (
            PosteriorPredictivePredictor(),
            PosteriorPredictive(
                draws=jnp.linspace(0.0, 1.0, 33).reshape(3, 11),
                aleatoric_variance=jnp.full(11, 0.25),
                draw_mask=jnp.array([True, False, True]),
            ),
        ),
    ],
    ids=["gaussian", "posterior predictive"],
)
def test_a_predictor_materializes_a_prediction_in_one_compiled_call(
    compile_counter: Callable[[], Any], predictor: object, prediction: object
) -> None:
    batch = Batch(inputs=jnp.zeros((11, 1)), targets=jnp.linspace(0.0, 2.0, 11))
    model = FixedPredictionModel(prediction)
    key = metric_key(jax.random.key(0))

    with compile_counter() as counter:
        data = predictor(model, batch, _DEVICE_FIELDS, key)  # type: ignore[operator]

    assert counter.total == 1
    assert data.variance is not None and data.variance.shape == (11,)
    assert data.log_likelihood is not None and data.log_likelihood.shape == (11,)
    assert data.closed_form_crps is not None and data.closed_form_crps.shape == (11,)
    assert data.predictive_samples is not None
    assert data.predictive_samples.shape == (11, 16)


class GaussianWithoutClosedFormCRPS(Gaussian):
    """A Gaussian that, like a user distribution, offers no ``crps`` method."""

    crps = None  # type: ignore[assignment]


def test_closed_form_crps_scores_the_gaussian_without_samples_or_a_grid() -> None:
    suite = MetricSuite(
        epoch=(ClosedFormContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
    )
    model = ShiftedGaussianModel(0.5, variance=0.25)

    scores = _scores(suite, model, _MIXTURE_BATCH)

    inputs = np.asarray(_MIXTURE_BATCH.inputs, np.float64).reshape(-1)
    targets = np.asarray(_MIXTURE_BATCH.targets, np.float64).reshape(-1, 1)
    expected = expected_gaussian_mixture_crps(
        (inputs + 0.5)[:, None], np.full(inputs.size, 0.25), targets
    )
    assert suite.prediction_requirements.predictive_sample_count is None
    assert suite.prediction_requirements.evaluation_grid is None
    assert scores == {"closed_form_crps": pytest.approx(np.mean(expected), rel=1e-5)}


def test_closed_form_crps_refuses_a_prediction_without_a_closed_form() -> None:
    suite = MetricSuite(
        epoch=(ClosedFormContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
    )
    prediction = GaussianWithoutClosedFormCRPS(loc=jnp.zeros(4), scale=jnp.ones(4))

    with pytest.raises(
        TypeError,
        match="GaussianWithoutClosedFormCRPS.*SampleContinuousRankedProbabilityScore",
    ):
        resolve_epoch_prediction_data(
            suite=suite,
            model=FixedPredictionModel(prediction),
            batch=Batch(inputs=jnp.zeros((4, 1)), targets=jnp.zeros(4)),
            key=jax.random.key(0),
        )


def test_merge_concatenates_closed_form_crps_across_batches() -> None:
    parts = [
        EpochPredictionData(
            targets=np.zeros(n), mean=np.zeros(n), closed_form_crps=np.full(n, float(n))
        )
        for n in (1, 2)
    ]

    merged = merge_epoch_prediction_data(parts)

    assert np.array_equal(merged.closed_form_crps, np.array([1.0, 2.0, 2.0]))


def test_closed_form_crps_scores_the_posterior_predictive_without_samples_or_a_grid() -> (
    None
):
    offsets = [-1.5, 0.0, 2.5]
    suite = MetricSuite(
        epoch=(ClosedFormContinuousRankedProbabilityScore(),),
        predictor=PosteriorPredictivePredictor(),
    )

    scores = _scores(suite, MixtureModel(offsets, 0.25), _MIXTURE_BATCH)

    inputs = np.asarray(_MIXTURE_BATCH.inputs, np.float64).reshape(-1)
    targets = np.asarray(_MIXTURE_BATCH.targets, np.float64).reshape(-1, 1)
    expected = expected_gaussian_mixture_crps(
        inputs[:, None] + np.asarray(offsets)[None, :],
        np.full(inputs.size, 0.25),
        targets,
    )
    assert suite.prediction_requirements.predictive_sample_count is None
    assert suite.prediction_requirements.evaluation_grid is None
    assert scores == {"closed_form_crps": pytest.approx(np.mean(expected), rel=1e-5)}


_REFERENCE_BATCH = Batch(
    inputs=jnp.array([[-1.0], [0.0], [0.5], [2.0]]),
    targets=jnp.array([[-0.5], [0.3], [1.5], [1.0]]),
    metadata={
        "reference_samples": jnp.array(
            [[-3.0, 1.0, 0.0], [2.0, 2.5, 4.0], [0.5, -1.0, 1.5], [3.0, 2.0, 2.2]]
        )
    },
)


def test_closed_form_crps_against_reference_scores_the_extractor_samples() -> None:
    suite = MetricSuite(
        epoch=(ClosedFormContinuousRankedProbabilityScore(against_reference=True),),
        predictor=GaussianPredictor(reference_samples_extractor=reference_from_batch),
    )
    model = ShiftedGaussianModel(0.5, variance=0.25)

    scores = _scores(suite, model, _REFERENCE_BATCH)

    inputs = np.asarray(_REFERENCE_BATCH.inputs, np.float64).reshape(-1)
    references = np.asarray(_REFERENCE_BATCH.metadata["reference_samples"])
    expected = expected_gaussian_mixture_crps(
        (inputs + 0.5)[:, None], np.full(inputs.size, 0.25), references
    )
    assert scores == {"closed_form_crps": pytest.approx(np.mean(expected), rel=1e-5)}


@pytest.mark.parametrize(
    "predictor", [GaussianPredictor(), PosteriorPredictivePredictor()]
)
def test_a_suite_scoring_references_without_an_extractor_fails_when_built(
    predictor: object,
) -> None:
    with pytest.raises(ValueError, match="reference_samples_extractor"):
        MetricSuite(
            epoch=(ClosedFormContinuousRankedProbabilityScore(against_reference=True),),
            predictor=predictor,  # type: ignore[arg-type]
        )


def test_a_suite_cannot_score_the_closed_form_crps_of_targets_and_references() -> None:
    with pytest.raises(ValueError, match="closed-form CRPS.*target"):
        MetricSuite(
            epoch=(
                ClosedFormContinuousRankedProbabilityScore(),
                ContinuousRankedProbabilityScore(),
            ),
            predictor=GaussianPredictor(
                reference_samples_extractor=reference_from_batch
            ),
            predictive_sample_count=3,
            evaluation_grid=EvaluationGrid(np.linspace(-3.0, 3.0, 31)),
        )


def test_closed_form_crps_against_reference_is_computed_in_one_compiled_call(
    compile_counter: Callable[[], Any],
) -> None:
    def model_and_batch(n_units: int) -> tuple[nnx.Module, Batch]:
        model = FixedPredictionModel(
            Gaussian(loc=jnp.linspace(0.0, 1.0, n_units), scale=jnp.full(n_units, 0.5))
        )
        batch = Batch(
            inputs=jnp.zeros((n_units, 1)),
            targets=jnp.zeros(n_units),
            metadata={"reference_samples": jnp.ones((n_units, 3))},
        )
        return model, batch

    predictor = GaussianPredictor(reference_samples_extractor=reference_from_batch)
    suite = MetricSuite(
        epoch=(
            NegativeLogLikelihood(),
            ClosedFormContinuousRankedProbabilityScore(against_reference=True),
        ),
        predictor=predictor,
    )
    key = metric_key(jax.random.key(0))
    # Compile the shape-independent eager key operations on another shape first.
    predictor(*model_and_batch(3), suite.prediction_requirements, key)
    model, batch = model_and_batch(4)

    with compile_counter() as counter:
        data = predictor(model, batch, suite.prediction_requirements, key)

    assert counter.total == 1
    assert data.closed_form_crps is not None and data.closed_form_crps.shape == (4,)
    assert data.closed_form_crps_against_reference


def keyed_reference_samples(batch: Batch, key: jax.Array) -> jax.Array:
    assert batch.targets is not None
    return jax.random.normal(key, (batch.targets.shape[0], 5))


def test_reference_samples_come_from_the_extractor_under_the_folded_key() -> None:
    batch = Batch(inputs=jnp.array([[0.0], [1.0], [2.0]]), targets=jnp.zeros((3, 1)))
    suite = MetricSuite(
        epoch=(
            ContinuousRankedProbabilityScore(),
            ClosedFormContinuousRankedProbabilityScore(
                name="reference_crps", against_reference=True
            ),
        ),
        predictor=GaussianPredictor(
            reference_samples_extractor=keyed_reference_samples
        ),
        predictive_sample_count=4,
        evaluation_grid=EvaluationGrid(np.linspace(-5.0, 5.0, 51)),
    )
    key = metric_key(jax.random.key(7))

    data = resolve_epoch_prediction_data(
        suite=suite, model=GaussianModel(), batch=batch, key=key
    )

    expected = keyed_reference_samples(batch, jax.random.fold_in(key, 1))
    assert np.array_equal(data.reference_samples, np.asarray(expected, np.float64))
