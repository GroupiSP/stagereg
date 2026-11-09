# Two-step mean/variance training

Training a mean and a variance jointly, as a Gaussian head under a negative
log-likelihood does ([Nix & Weigend, 1994](https://doi.org/10.1109/ICNN.1994.374138)),
couples their gradients: the variance can grow to excuse a poor mean fit, and
the mean can stall where the variance is large
([Detlefsen et al., 2019](https://doi.org/10.48550/arXiv.1906.03260)). Two-step
training removes the coupling; warm-up schemes that fit the mean first
([Sluijterman et al., 2024](https://doi.org/10.1016/j.neucom.2024.127929)) apply
the same idea for part of training. A first stage fits a deterministic mean model with a squared-error loss. A
second stage freezes it, takes its squared residuals as targets, and fits a
Gamma model to them, whose mean is the aleatoric variance.

The JAX backend ships both stages, [`MeanStage`][stagereg.jax.MeanStage] and
[`GammaVarianceStage`][stagereg.jax.GammaVarianceStage]. They share one
[`TrainingState`][stagereg.core.TrainingState], and its lifecycle is what lets the
second stage check that the first one has finished.

## The lifecycle

A staged workflow moves through the [`StageState`][stagereg.core.StageState]
values in order:

```text
NEW -> INITIALIZED -> MEAN_READY -> VARIANCE_READY
```

Each stage method checks the state it is given and moves it on:

| Call | Requires | Leaves the state at |
| --- | --- | --- |
| `MeanStage.prepare` | `NEW` or `INITIALIZED` | `INITIALIZED` |
| `MeanStage.train` | `INITIALIZED` | `MEAN_READY` |
| `GammaVarianceStage.prepare` | `MEAN_READY` | `MEAN_READY` |
| `GammaVarianceStage.train` | `MEAN_READY`, after `prepare` | `VARIANCE_READY` |

A call made out of order raises `ValueError` before it touches the state, so a
variance stage cannot train on the residuals of a mean model that was never
trained. [`validate_transition`][stagereg.core.validate_transition] states the
rule on its own: a state may only move to the one directly after it.

```python
import pytest

from stagereg.core import StageState, validate_transition

validate_transition(StageState.NEW, StageState.INITIALIZED)
validate_transition(StageState.MEAN_READY, StageState.VARIANCE_READY)

with pytest.raises(ValueError, match="expected 'initialized'"):
    validate_transition(StageState.NEW, StageState.MEAN_READY)
```

`StageState` continues past `VARIANCE_READY` with `POSTERIOR_READY`, reached by
the optional [`PosteriorStage`][stagereg.jax.PosteriorStage], and `COMPLETED`, which
no shipped stage reaches yet. A run that stops at `VARIANCE_READY` is complete.

## A two-step run

The run below is the whole workflow on eight points: a linear mean model, then a
[`GammaHead`][stagereg.jax.GammaHead] on its residuals, each validated after every
epoch. The loader serves the same batch for both splits to keep it short; a real
one returns a different partition of data for `"train"` and `"validation"`.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core import (
    Batch,
    NegativeLogLikelihoodLoss,
    ParameterRole,
    SquaredErrorLoss,
    StageState,
    TrainingState,
    add_epsilon,
)
from stagereg.jax import (
    GammaHead,
    GammaVarianceStage,
    HeldOutValidation,
    MeanStage,
    SupervisedStageOptions,
    create_optimizer,
    make_supervised_loss,
)

inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
targets = 2.0 * inputs + 0.1 * jnp.sin(7.0 * inputs) + 0.01 * jax.random.normal(jax.random.key(0), inputs.shape)


def loader(*, split: str, epoch: int) -> list[Batch]:
    return [Batch(inputs=inputs, targets=targets)]


# The two stages' default losses, spelled out so validation can reuse them.
squared_error = make_supervised_loss(SquaredErrorLoss())
gamma_nll = make_supervised_loss(
    NegativeLogLikelihoodLoss(target_transform=add_epsilon())
)  # add_epsilon() keeps the Gamma's targets positive, as required by its support.

state = TrainingState(rng_state=jax.random.key(0))

# Step 1: a deterministic mean model, trained with squared error.
mean_model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
mean_stage = MeanStage(
    model=mean_model,
    optimizer=create_optimizer(mean_model, optax.adam(0.1)),
    train_loader=loader,
    loss=squared_error,
    options=SupervisedStageOptions(
        epochs=3,
        validation=HeldOutValidation(
            model=mean_model, loader=loader, loss=squared_error
        ),
    ),
)
mean_stage.prepare(state)
mean_stage.train(state)
assert state.lifecycle_state is StageState.MEAN_READY

# Step 2: a Gamma model fitted to the frozen mean model's squared residuals.
variance_model = GammaHead(1, 1, rngs=nnx.Rngs(1))
variance_stage = GammaVarianceStage(
    model=variance_model,
    optimizer=create_optimizer(variance_model, optax.adam(0.1)),
    source_loader=loader,
    loss=gamma_nll,
    options=SupervisedStageOptions(epochs=3),
    # Validate on residual targets, not on the source loader's original ones.
    validation_factory=lambda residuals: HeldOutValidation(
        model=variance_model, loader=residuals, loss=gamma_nll
    ),
)
variance_stage.prepare(state)  # here, the squared residuals are computed and cached
variance_stage.train(state)

assert state.lifecycle_state is StageState.VARIANCE_READY
assert state.frozen_components == {"mean_model"}
assert state.parameter_roles == {
    "mean_model": ParameterRole.MEAN,
    "variance_model": ParameterRole.VARIANCE,
}
assert variance_stage.validate(state).passed
assert sorted(state.metric_history) == [
    "mean/train/loss",
    "mean/validation/loss",
    "variance/train/loss",
    "variance/validation/loss",
]

# The predictive variance is the Gamma mean.
variance = variance_model(inputs).mean()
assert variance.shape == inputs.shape
assert bool(jnp.all(variance > 0.0))
```

The aleatoric variance estimate is the Gamma's mean, `concentration / rate`,
not the Gamma's own variance, `concentration / rate**2`: the latter measures
how uncertain the variance estimate itself is. The mean model takes no part in
the second step's optimization. The variance stage records it as frozen, as
the assertion on `frozen_components` shows, and only the variance model is
handed to the variance optimizer.

The two stages register their components under fixed names: the mean model and
its optimizer as `mean_model` and `mean_optimizer`, the variance model and its
optimizer as `variance_model` and `variance_optimizer`. The variance stage finds
the mean model by that name, so if you rename it on the
[`MeanStage`][stagereg.jax.MeanStage], pass the same name as `mean_model_name` to
the [`GammaVarianceStage`][stagereg.jax.GammaVarianceStage]. Both stages take their
epoch count, validation, early stopping, event sinks and checkpoint store from a
[`SupervisedStageOptions`][stagereg.jax.SupervisedStageOptions], and run the same
epoch loop as [`run_supervised`][stagereg.jax.run_supervised].

A variance stage validates against residuals, not against the original targets.
Give it a `validation_factory`, which receives the materialized residual loader
and returns the validation strategy to use. A `validation` set in its options is
used only when there is no factory, and it sees whatever loader it was built
with.

### Metric tags

Every metric the run records lands in `state.metric_history` under its
[metric tag](../glossary.md), `stage/split/metric`. The stage names are fixed as
well, `mean` and `variance`, so the run above records:

```text
mean/train/loss
mean/validation/loss
variance/train/loss
variance/validation/loss
```

The stage segment is what keeps the two loss curves apart: each stage counts its
own epochs from zero, and a tag without the stage would let the variance stage's
`train/loss` overwrite the mean stage's.

## Residual materialization

[`GammaVarianceStage.prepare`][stagereg.jax.GammaVarianceStage.prepare] turns the
frozen mean model's errors into the variance stage's training data, through
[`materialize_residual_loader`][stagereg.jax.materialize_residual_loader]. For
every split it is given, it reads the source loader once, at `source_epoch`,
replaces each batch's targets with the squared residuals
`(targets - mean_model(inputs)) ** 2`, and keeps the result. The residuals are
computed with an eval-mode copy of the mean model and with gradients stopped, so
nothing the variance stage does can reach the mean model.

```python
import jax.numpy as jnp
from flax import nnx

from stagereg.core import Batch
from stagereg.jax import materialize_residual_loader

inputs = jnp.array([[-1.0], [0.0], [1.0]])
targets = jnp.array([[-2.0], [0.5], [2.0]])
reads = []


def loader(*, split: str, epoch: int) -> list[Batch]:
    reads.append((split, epoch))
    return [Batch(inputs=inputs, targets=targets)]


mean_model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
residuals = materialize_residual_loader(mean_model, loader, splits=("train",))

(batch,) = residuals(split="train", epoch=0)
assert jnp.allclose(batch.targets, (targets - mean_model(inputs)) ** 2)

# Every later epoch replays the same cached batches; the source is not read again.
assert residuals(split="train", epoch=5) is residuals(split="train", epoch=0)
assert reads == [("train", 0)]
```

The splits default to `("train", "validation")`, so the source loader has to
serve both, or the stage has to be told otherwise with `splits=("train",)`.

### Residual scale

Squared residuals are often far from one: a mean model that fits to within
0.03 leaves residuals near 1e-3. To reach such a variance, a Gamma head would
need a rate near a thousand times its concentration, and so huge raw outputs
and weights. Away from the data, a small drift in the features then swings
those outputs through zero, and the predicted variance blows up by orders of
magnitude. `prepare` therefore also sets every
[`ResidualScale`][stagereg.jax.ResidualScale] in the variance model to the
mean squared residual of the training split. A
[`GammaHead`][stagereg.jax.GammaHead] divides its rate by that scale, so its
network predicts at unit scale and its mean stays in the targets' units.
Scaling the targets by `k` scales the trained variance by `k**2`. A custom
variance model gets the same behavior by holding a `ResidualScale` and
dividing its rate by it; one without a `ResidualScale` trains on the raw
residuals, and so does `GammaHead(..., residual_scaling=False)`, which holds
none. Without a residual scale, `prepare` neither reads the training split
nor checks its residuals.

!!! warning "Residuals are held in memory"

    Materialization caches every batch of every configured split, inputs and
    residual targets both, for as long as the variance stage lives. A dataset
    that only fits when streamed from disk does not fit here. The cache is also
    a fixed snapshot: each variance epoch replays the same batches in the same
    order, with no reshuffling, and it reflects the mean model as it was when
    `prepare` ran. Recomputing residuals as the mean model changes, as an
    iterative scheme alternating the two steps would need, is not supported.

## Larger runs

The XSin benchmark runs the same two stages on a harder one-dimensional problem
and plots the result next to joint mean-variance estimation. Its entry point is
short because the benchmark module holds the models, data and plotting:

```{.python notest}
config = XSinConfig()
data = make_xsin_data(config)
result = run_xsin_two_step(data, config, event_sinks=(StagePrintingEventSink(),))
print_xsin_metrics("mean plus Gamma variance", result)
```

Run it with `uv run --extra jax --extra plot python examples/jax/xsin/two_steps.py`;
the source is
[`examples/jax/xsin/two_steps.py`](https://github.com/GroupiSP/stagereg/blob/main/examples/jax/xsin/two_steps.py).
The [CMAPSS example](examples.md#cmapss-remaining-useful-life) uses the same
stages with CNN models on real sensor data. Given a checkpoint store, each stage
saves a finalized checkpoint after `train`: its best model when it also has an
early stopper, which it leaves live, otherwise its last; to use those checkpoints, or to run the variance
stage later from a saved mean model, see [Checkpoints](checkpoints.md).
