# Epoch metrics

The loss tells you how training is going; it does not tell you whether the
predictive distribution is any good. This guide explains how to register
metrics such as RMSE, interval coverage and CRPS with a run, how predictions
reach them, and why some of them refuse to run until you configure them
explicitly.

## One suite, two kinds of metric

Every metric a run computes beyond the loss is registered on a
[`MetricSuite`][stagereg.jax.MetricSuite], which you pass to the runner for the
train split and to the validation strategy for the validation split,
usually the same suite for both. It holds two kinds of metric:

- **Batch metrics**, each a [`BatchMetricSpec`][stagereg.jax.BatchMetricSpec]:
  a JAX function evaluated on every batch, whose per-batch values are reduced
  on the host (by default with the mean). Cheap, and right for any quantity
  that is an average of per-batch averages.
- **Epoch metrics**, each an [`EpochMetric`][stagereg.core.EpochMetric]: a
  host-side function evaluated once per epoch, over the predictions for every
  batch at once. This is what a quantity that does not decompose into batch
  averages needs: an RMSE is the root of a mean over the whole epoch, and an
  interval's coverage or a CRPS is only meaningful over all scoring units.

Each metric has one bare [metric name](../glossary.md) such as `rmse` or
`point_crps`, the same on every split and in every stage. The name `loss` is
reserved for the loss, and two metrics with the same name are rejected.

The epoch metrics in `stagereg.core` are
[`RootMeanSquaredError`][stagereg.core.RootMeanSquaredError],
[`NegativeLogLikelihood`][stagereg.core.NegativeLogLikelihood],
[`IntervalCoverage`][stagereg.core.IntervalCoverage],
[`WeightedSpread`][stagereg.core.WeightedSpread],
[`PointContinuousRankedProbabilityScore`][stagereg.core.PointContinuousRankedProbabilityScore],
[`SampleContinuousRankedProbabilityScore`][stagereg.core.SampleContinuousRankedProbabilityScore],
[`ClosedFormContinuousRankedProbabilityScore`][stagereg.core.ClosedFormContinuousRankedProbabilityScore]
and
[`ContinuousRankedProbabilityScore`][stagereg.core.ContinuousRankedProbabilityScore].

## The predictor boundary

An epoch metric is backend-neutral: it lives in `stagereg.core` and works on
NumPy arrays. The model, however, returns a
[`PredictiveDistribution`][stagereg.core.PredictiveDistribution] made of JAX
arrays, and that protocol deliberately has no way to concatenate or index
distributions, so a run cannot simply collect an epoch's worth of them.

A [`Predictor`][stagereg.jax.Predictor] bridges the two. For each batch it calls
the model, turns the distribution into host arrays, and returns them as
[`EpochPredictionData`][stagereg.core.EpochPredictionData]. The run then joins
the batches with
[`merge_epoch_prediction_data`][stagereg.jax.merge_epoch_prediction_data] and
hands the result to each epoch metric. The predictor is the only place that
knows about the distribution family; for a model returning a
[`Gaussian`][stagereg.jax.Gaussian], use
[`GaussianPredictor`][stagereg.jax.GaussianPredictor]; for a model returning a
[`PosteriorPredictive`][stagereg.jax.PosteriorPredictive], the mixture over
draws of a posterior, use
[`PosteriorPredictivePredictor`][stagereg.jax.PosteriorPredictivePredictor],
which scores the exact mixture: its log-density, samples from it and its
quantiles.

During training, the model a predictor receives runs as one compiled call, so
the model's output must be a JAX pytree. The built-in distributions are (see
[Distributions inside jitted code](distributions-and-losses.md#distributions-inside-jitted-code));
a model returning your own distribution type must register it with
[`jax.tree_util`](https://docs.jax.dev/en/latest/pytrees.html) before epoch
metrics can score it.

`EpochPredictionData` holds one row per **scoring unit**: a single scalar
target with its predictive mean and, when requested, its variance, predictive
and reference samples (as `(n_scoring_units, n_samples)` matrices), labelled
prediction intervals and a coordinate. The predictor flattens targets of any
shape into scoring units, and every field is validated as finite and
shape-consistent. Each scoring unit is one scalar, so the model's distribution
must have a scalar event: a prediction with a non-empty event shape is
rejected.
Because the data are plain arrays, an epoch metric can be called without JAX
at all:

```python
import numpy as np

from stagereg.core import (
    EpochPredictionData,
    IntervalCoverage,
    PredictionInterval,
    RootMeanSquaredError,
)

data = EpochPredictionData(
    targets=np.array([0.0, 1.0, 2.0, 3.0]),
    mean=np.array([0.0, 1.0, 2.0, 5.0]),
    intervals=(
        PredictionInterval(
            level=0.9,
            lower=np.array([-1.0, 0.0, 1.0, 4.0]),
            upper=np.array([1.0, 2.0, 3.0, 6.0]),
        ),
    ),
)

assert RootMeanSquaredError()(data) == 1.0
assert IntervalCoverage(level=0.9)(data) == 0.75
```

## Materializing only what is required

Sampling a distribution and integrating its CDF are not free, so the predictor
materializes only the fields some registered metric asks for. Each epoch
metric declares its needs as
[`MetricRequirements`][stagereg.core.MetricRequirements], and the suite takes
their union:

| Metric | Requires |
| --- | --- |
| `RootMeanSquaredError` | targets and means only |
| `NegativeLogLikelihood` | each target's predictive log-density |
| `IntervalCoverage(level)` | an interval at exactly `level` |
| `WeightedSpread(level)` | an interval at `level` and a coordinate |
| `PointContinuousRankedProbabilityScore` | predictive samples and an evaluation grid |
| `SampleContinuousRankedProbabilityScore` | predictive samples |
| `ClosedFormContinuousRankedProbabilityScore` | each unit's closed-form CRPS, plus reference samples with `against_reference=True`; no predictive samples, no grid |
| `ContinuousRankedProbabilityScore` | predictive samples, reference samples and an evaluation grid |

Intervals cost nothing to configure: `GaussianPredictor` computes the exact
central Gaussian interval at each requested level, and
`PosteriorPredictivePredictor` the central interval between mixture quantiles. The other requirements come
with choices the library will not make for you.

### CRPS: sample count and evaluation grid

`PointContinuousRankedProbabilityScore` and `ContinuousRankedProbabilityScore`
compare empirical CDFs numerically. They need a number of
predictive samples per scoring unit and an
[`EvaluationGrid`][stagereg.core.EvaluationGrid], the strictly increasing
points the CDFs are integrated over. Both are set once on the suite, so every
batch, split and epoch is scored on the same grid with the same number of
samples, and both have no default. A grid that does not span the targets
silently understates the score, and its range and resolution depend on the
scale of your data; the sample count trades cost against noise in the
estimate. The samples are kept on the host for the whole epoch, so they cost
`O(n_scoring_units * predictive_sample_count)` memory. A suite that registers
a CRPS metric without them fails when it is built:

```python
import numpy as np
import pytest

from stagereg.core import EvaluationGrid, PointContinuousRankedProbabilityScore
from stagereg.jax import GaussianPredictor, MetricSuite

with pytest.raises(ValueError, match="predictive_sample_count"):
    MetricSuite(
        epoch=(PointContinuousRankedProbabilityScore(),),
        predictor=GaussianPredictor(),
    )

suite = MetricSuite(
    epoch=(PointContinuousRankedProbabilityScore(),),
    predictor=GaussianPredictor(),
    predictive_sample_count=64,
    evaluation_grid=EvaluationGrid(np.linspace(-5.0, 5.0, 201)),
)
```

`PointContinuousRankedProbabilityScore` (`point_crps`) scores each predictive
distribution against its observed target, which is what a real dataset
offers. `ContinuousRankedProbabilityScore` (`crps`) scores it against a whole
reference distribution of the target, which only exists when you know how the
data were generated, as in a simulated benchmark. Those reference samples come
from a `reference_samples_extractor` you give the predictor, a
[`ReferenceSamplesExtractor`][stagereg.jax.ReferenceSamplesExtractor] that maps a
batch and a key to a `(n_scoring_units, n_samples)` array. A suite whose
metrics need reference samples fails when it is built if the predictor has no
extractor.

`SampleContinuousRankedProbabilityScore` (`sample_crps`) scores the same
pair as `point_crps` but needs no grid: it uses the energy form
`E|X - y| - E|X - X'| / 2` over the predictive samples, which is the exact
CRPS of their empirical distribution. It still needs a
`predictive_sample_count`:

```python
from stagereg.core import SampleContinuousRankedProbabilityScore
from stagereg.jax import GaussianPredictor, MetricSuite

suite = MetricSuite(
    epoch=(SampleContinuousRankedProbabilityScore(),),
    predictor=GaussianPredictor(),
    predictive_sample_count=64,
)
```

To score predictive samples against reference samples outside a training run,
call [`expected_sample_crps`][stagereg.core.expected_sample_crps]. It takes a
`(units, S)` array of predictive samples and a `(units, R)` array of reference
samples and returns each unit's grid-free expected CRPS, the mean of
`sample_crps` over that unit's reference samples:

```python
import numpy as np

from stagereg.core import expected_sample_crps

scores = expected_sample_crps(np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 3.0]]))
```

### Closed-form CRPS: no samples at all

`SampleContinuousRankedProbabilityScore` is exact for the empirical
distribution of the samples, not for the predictive itself. Its value carries
Monte Carlo noise, and the energy form's plug-in bias favours narrow
predictives, so two runs, or two checkpoints of one run, compare numbers that
moved partly because different samples were drawn. When the predictive
distribution has a CRPS formula of its own, as a Gaussian does,
[`ClosedFormContinuousRankedProbabilityScore`][stagereg.core.ClosedFormContinuousRankedProbabilityScore]
(`closed_form_crps`) scores each target with that formula instead. It needs
neither a sample count nor a grid:

```python
from stagereg.core import ClosedFormContinuousRankedProbabilityScore
from stagereg.jax import GaussianPredictor, MetricSuite

suite = MetricSuite(
    epoch=(ClosedFormContinuousRankedProbabilityScore(),),
    predictor=GaussianPredictor(),
)
```

The predictor computes each scoring unit's CRPS on the device, in the same
compiled call as the mean and the log-likelihood, and the metric averages it
over every scoring unit of the split.

Prefer it whenever your distribution offers it: it is what early stopping,
checkpoint selection and pruning should compare. Keep
`SampleContinuousRankedProbabilityScore` for a predictive that can only be
sampled.

A distribution offers the closed form by implementing the optional
[`ClosedFormCRPS`][stagereg.core.ClosedFormCRPS] protocol: a `crps(targets)`
method that takes one target per scoring unit, shaped `[*batch]`, or `R`
reference samples per scoring unit, shaped `[R, *batch]`, and returns each
unit's expected CRPS, shaped `[*batch]`.
[`Gaussian`][stagereg.jax.Gaussian],
[`PosteriorPredictive`][stagereg.jax.PosteriorPredictive] and
[`MomentMatchedPredictive`][stagereg.jax.MomentMatchedPredictive] implement it;
[`Gamma`][stagereg.jax.Gamma] does not, and
[`PredictiveDistribution`][stagereg.core.PredictiveDistribution] does not
require it. When the model's distribution has no `crps` method, the predictor
raises a `TypeError` that names the distribution's class and points to
`SampleContinuousRankedProbabilityScore`. It never falls back to a sample
estimate, so a value reported as `closed_form_crps` is always the exact score
([ADR 0015](https://github.com/GroupiSP/stagereg/blob/main/docs/adr/0015-the-closed-form-crps-never-falls-back-to-sampling.md)).

For a posterior stage, register the metric on a
[`PosteriorPredictivePredictor`][stagereg.jax.PosteriorPredictivePredictor].
It scores the exact mixture over the draws (Grimit et al. 2006), not its
moment-matched Gaussian, and slots that the
[`draw_mask`][stagereg.jax.PosteriorPredictive] excludes do not contribute, so
the score equals that of the mixture over the included draws alone. One draw
scores exactly as a `GaussianPredictor` does, so a variance stage and a
posterior stage report comparable values. To see what the single-Gaussian
summary loses, compare the mixture's `crps` with that of its
[`moment_matched`][stagereg.jax.PosteriorPredictive.moment_matched] summary:

```python
import jax.numpy as jnp

from stagereg.core import ClosedFormContinuousRankedProbabilityScore
from stagereg.jax import MetricSuite, PosteriorPredictive, PosteriorPredictivePredictor

suite = MetricSuite(
    epoch=(ClosedFormContinuousRankedProbabilityScore(),),
    predictor=PosteriorPredictivePredictor(),
)

bimodal = PosteriorPredictive(
    draws=jnp.array([[-2.0], [2.0]]), aleatoric_variance=jnp.array([0.1])
)
targets = jnp.array([2.0])
mixture_crps = bimodal.crps(targets)
summary_crps = bimodal.moment_matched().crps(targets)
assert mixture_crps < summary_crps
```

In a simulated benchmark you can score each predictive distribution against
the reference distribution of its target instead, as
`ContinuousRankedProbabilityScore` does, but without its grid or predictive
samples. Set `against_reference=True`: the metric then also asks for reference
samples, the predictor draws them from its `reference_samples_extractor` under
the same key as for `ContinuousRankedProbabilityScore`, and the value is the
exact expected CRPS over those samples. Scoring references is a choice made on
the metric, never by configuring an extractor. A suite that asks for it from a
predictor without an extractor fails when it is built, and so does a suite that
would score the closed-form CRPS against both targets and references, since
the predictor computes only one of them:

```python
import jax
import pytest

from stagereg.core import ClosedFormContinuousRankedProbabilityScore
from stagereg.core.types import Batch
from stagereg.jax import GaussianPredictor, MetricSuite


def reference_samples(batch: Batch, key: jax.Array) -> jax.Array:
    """Draw 32 samples per target from the known data-generating noise."""
    return batch.targets[:, None] + 0.1 * jax.random.normal(
        key, (batch.targets.shape[0], 32)
    )


reference_crps = ClosedFormContinuousRankedProbabilityScore(
    name="reference_crps", against_reference=True
)
suite = MetricSuite(
    epoch=(reference_crps,),
    predictor=GaussianPredictor(reference_samples_extractor=reference_samples),
)

with pytest.raises(ValueError, match="reference_samples_extractor"):
    MetricSuite(epoch=(reference_crps,), predictor=GaussianPredictor())
```

Outside a training run,
[`expected_gaussian_mixture_crps`][stagereg.core.expected_gaussian_mixture_crps]
computes the same score with NumPy alone. It takes `(units, M)` draw means, a
`(units,)` variance shared by every draw of a unit and `(units, R)` reference
samples, and returns each unit's expected CRPS of the equally weighted mixture
(Grimit et al. 2006). One draw is a Gaussian, and one reference sample is an
observed target:

```python
import numpy as np

from stagereg.core import expected_gaussian_mixture_crps

# A standard normal scored at its own mean: (sqrt(2) - 1) / sqrt(pi).
scores = expected_gaussian_mixture_crps(
    draw_means=np.array([[0.0]]),
    variance=np.array([1.0]),
    reference_samples=np.array([[0.0]]),
)
assert np.allclose(scores, (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi))
```

### Spread: an explicit coordinate

`WeightedSpread` (`wsu`) summarises how an interval's width evolves along an
ordered coordinate, such as time or cycle number, weighting later points more.
Scoring units carry no notion of order, and the right coordinate is a fact
about your domain, not about the model's inputs, so it is never inferred. You
supply it through the predictor's `coordinate_extractor`, a
[`CoordinateExtractor`][stagereg.jax.CoordinateExtractor] that reads it from a
batch, typically from its `metadata`. The metric sorts the scoring units by
coordinate, so they may arrive in any order, but every scoring unit needs a
coordinate of its own: a tie between two units is rejected, as is an epoch with
fewer than three units. A suite with `WeightedSpread` and no extractor fails at
the first batch it predicts.

## Evaluating a suite

With a model, a loader and a suite, a full evaluation is one call to
[`evaluate_loader`][stagereg.jax.evaluate_loader]. It is also what
[`HeldOutValidation`][stagereg.jax.HeldOutValidation] runs on the validation
split after every epoch. Here every metric of this guide is computed on an
untrained Gaussian head, with the coordinate read from batch metadata:

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from stagereg.core import (
    EvaluationGrid,
    IntervalCoverage,
    PointContinuousRankedProbabilityScore,
    RootMeanSquaredError,
    WeightedSpread,
)
from stagereg.core.types import Batch
from stagereg.jax import GaussianHead, GaussianPredictor, MetricSuite, evaluate_loader

inputs = jnp.linspace(-1.0, 1.0, 16).reshape(-1, 1)
batches = [
    Batch(
        inputs=inputs[start : start + 8],
        targets=2.0 * inputs[start : start + 8],
        metadata={"time": jnp.arange(start, start + 8)},
    )
    for start in (0, 8)
]

suite = MetricSuite(
    epoch=(
        RootMeanSquaredError(),
        IntervalCoverage(level=0.95),
        WeightedSpread(level=0.95),
        PointContinuousRankedProbabilityScore(),
    ),
    predictor=GaussianPredictor(coordinate_extractor=lambda batch: batch.metadata["time"]),
    predictive_sample_count=32,
    evaluation_grid=EvaluationGrid(np.linspace(-6.0, 6.0, 121)),
)

model = GaussianHead(1, 1, rngs=nnx.Rngs(0))
metrics, _ = evaluate_loader(model, batches, key=jax.random.key(0), metrics=suite)

assert set(metrics) == {"rmse", "coverage", "wsu", "point_crps"}
assert 0.0 <= metrics["coverage"] <= 1.0
```

Collecting predictions does not disturb training. The predictor runs on an
inference-mode clone of the model, so stateful layers and the model's internal
RNG streams are left as they were, and sampled metrics draw from keys derived
in a namespace of their own, so registering one does not change the random
keys the loss and batch metrics see.

No `loss` was given, so none is reported. In a training run, pass the same
suite as `metrics=` to
[`run_supervised`][stagereg.jax.run_supervised] and to `HeldOutValidation`; each
value is then recorded under its [metric tag](../glossary.md),
`stage/split/metric`, such as `supervised/validation/point_crps`.

## Writing your own epoch metric

An epoch metric is anything with a `name`, a `requirements` property and a
call on `EpochPredictionData` that returns a float; it does not inherit from
anything. Declare only the fields you read, and the predictor will provide
them:

```python
from dataclasses import dataclass

import numpy as np

from stagereg.core import EpochPredictionData, MetricRequirements


@dataclass(frozen=True)
class MeanPredictedVariance:
    name: str = "mean_variance"

    @property
    def requirements(self):
        return MetricRequirements(variance=True)

    def __call__(self, data: EpochPredictionData, /) -> float:
        return float(np.mean(data.variance))


metric = MeanPredictedVariance()
data = EpochPredictionData(
    targets=np.zeros(3), mean=np.zeros(3), variance=np.array([1.0, 2.0, 3.0])
)
assert metric.requirements.variance
assert metric(data) == 2.0
```

## Where to go next

- [Predictive distributions and losses](distributions-and-losses.md) covers
  the head and the distribution these metrics score.
- The mean-variance regression and CMAPSS examples on the
  [Examples](examples.md) page register epoch metrics in full training runs.
- The [`stagereg.core.metric_registry`](../reference/core.md#stageregcoremetric_registry)
  and [`stagereg.jax.metrics`](../reference/jax.md#stageregjaxmetrics) reference
  sections list every signature.
