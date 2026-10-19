# Predictive distributions and losses

A probabilistic regression model in `stagereg` does not return a number per
input; it returns a distribution over the target. This guide explains how that
distribution is produced, how a loss scores it, and how to choose between plain
negative log-likelihood and its beta-weighted variant.

## The model returns a distribution

Two protocols in `stagereg.core` meet at the model's output:

- a [`DistributionHead`][stagereg.core.DistributionHead] maps features to a
  prediction;
- that prediction is a
  [`PredictiveDistribution`][stagereg.core.PredictiveDistribution]: something
  with a `log_prob`, a `mean`, a `variance` and a keyed `sample`.

The head is the last layer of the model, and the distribution is what the model
hands to everything downstream: the loss, the epoch metrics and your own
plotting code. Nothing downstream knows which distribution family it is
looking at, only that it satisfies the protocol. That is the seam that keeps
losses and metrics in `stagereg.core` free of any backend.

The JAX backend supplies two heads, each with its distribution:
[`GaussianHead`][stagereg.jax.GaussianHead] produces a
[`Gaussian`][stagereg.jax.Gaussian] (location and scale), and
[`GammaHead`][stagereg.jax.GammaHead] produces a
[`Gamma`][stagereg.jax.Gamma] (concentration and rate) for strictly positive
targets. Both keep their parameters positive with a `softplus` and a small
offset, so the network's raw outputs are unconstrained.

To give a head something to work with, put it after a feature extractor in an
ordinary Flax NNX module. The model's call then returns the distribution:

```python
import jax.numpy as jnp
from flax import nnx

from stagereg.jax import GaussianHead


class MeanVarianceModel(nnx.Module):
    def __init__(self, *, rngs):
        self.hidden = nnx.Linear(1, 16, rngs=rngs)
        self.head = GaussianHead(16, 1, rngs=rngs)

    def __call__(self, inputs):
        return self.head(nnx.relu(self.hidden(inputs)))


model = MeanVarianceModel(rngs=nnx.Rngs(0))
prediction = model(jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1))

assert prediction.mean().shape == (8, 1)
assert bool(jnp.all(prediction.variance() > 0.0))
```

### Distributions inside jitted code

The JAX backend's distributions ([`Gaussian`][stagereg.jax.Gaussian],
[`Gamma`][stagereg.jax.Gamma],
[`PosteriorPredictive`][stagereg.jax.PosteriorPredictive] and
[`MomentMatchedPredictive`][stagereg.jax.MomentMatchedPredictive]) are JAX
pytrees whose leaves are their parameter arrays. A distribution can therefore be
passed into, and returned from, a `jax.jit` function without unpacking its
fields by hand:

```python
import jax
import jax.numpy as jnp

from stagereg.jax import Gaussian


@jax.jit
def widen(distribution):
    return Gaussian(loc=distribution.loc, scale=2.0 * distribution.scale)


wider = widen(Gaussian(loc=jnp.zeros(3), scale=jnp.ones(3)))

assert bool(jnp.all(wider.variance() == 4.0))
```

A [`Batch`][stagereg.core.Batch] is not a pytree: pass its `inputs` and
`targets` to jitted code directly.

## Scoring the distribution: negative log-likelihood

[`NegativeLogLikelihoodLoss`][stagereg.core.NegativeLogLikelihoodLoss] is the
loss for a predictive distribution. Its `per_example` asks the prediction for
`log_prob(targets)` and negates it, returning one unreduced value per element,
so reduction and sample weighting stay with the caller. It never reads the
distribution's parameters, so the same loss trains a Gaussian head, a Gamma
head or any distribution of your own.

[`GaussianNLLLoss`][stagereg.core.GaussianNLLLoss] and
[`BetaNLLLoss`][stagereg.core.BetaNLLLoss] are not separate classes. Both are names
for `NegativeLogLikelihoodLoss`, kept from before the two were consolidated, so
the choice between them is really the choice of its `beta` argument.

```python
import jax.numpy as jnp

from stagereg.core import BetaNLLLoss, GaussianNLLLoss, NegativeLogLikelihoodLoss
from stagereg.core.types import Batch
from stagereg.jax import Gaussian

assert GaussianNLLLoss is NegativeLogLikelihoodLoss
assert BetaNLLLoss is NegativeLogLikelihoodLoss

prediction = Gaussian(loc=jnp.zeros((3, 1)), scale=jnp.ones((3, 1)))
batch = Batch(inputs=None, targets=jnp.array([[0.0], [1.0], [2.0]]))

nll = GaussianNLLLoss().per_example(prediction, batch)
assert nll.shape == (3, 1)
# The further a target sits from the mean, the larger its loss.
assert bool(nll[0, 0] < nll[1, 0] < nll[2, 0])
```

## Choosing `beta`

With plain NLL (`beta=0`, the default), the gradient that moves the mean of a
Gaussian is scaled by one over the predicted variance. A model can therefore
"explain away" a hard region by predicting a large variance there, after which
that region barely contributes to learning the mean. On data whose noise
varies with the input, this tends to leave the mean underfit exactly where the
noise is high.

The beta-weighted NLL multiplies each example's NLL by its predicted variance
raised to `beta`, which cancels part of that `1 / variance` scaling:

| `beta` | Behaviour of the mean's gradient |
| --- | --- |
| `0` | Plain NLL; low-variance examples dominate. |
| `0.5` | A common compromise between NLL and MSE. |
| `1` | Matches the gradient of mean squared error. |

The weight must be treated as a constant, or the model could lower the loss by
shrinking the weight rather than by fitting the data. The loss lives in the
backend-neutral core, so it cannot stop gradients itself: its `stop_gradient`
argument defaults to the identity, and with JAX you pass
`jax.lax.stop_gradient`. Leaving it out with a nonzero `beta` silently trains
a different objective.

```python
import jax
import jax.numpy as jnp

from stagereg.core import BetaNLLLoss, GaussianNLLLoss
from stagereg.core.types import Batch
from stagereg.jax import Gaussian

nll = GaussianNLLLoss()
beta_nll = BetaNLLLoss(beta=0.5, stop_gradient=jax.lax.stop_gradient)
batch = Batch(inputs=None, targets=jnp.array([[1.0]]))


def total(objective, loc, scale):
    prediction = Gaussian(loc=loc, scale=scale)
    return jnp.sum(objective.per_example(prediction, batch))


loc, scale = jnp.zeros((1, 1)), jnp.full((1, 1), 2.0)
weight = (scale**2) ** 0.5  # variance ** beta

# The variance weight scales the gradient but is not differentiated itself.
for argnum in (1, 2):
    weighted = jax.grad(total, argnums=argnum)(beta_nll, loc, scale)
    plain = jax.grad(total, argnums=argnum)(nll, loc, scale)
    assert bool(jnp.allclose(weighted, weight * plain))
```

`beta` is validated to lie in `[0, 1]`.

### Positive targets

A [`Gamma`][stagereg.jax.Gamma] has support only on positive values, and its
`log_prob` at an exact zero is not finite. When targets can be zero, pass a
`target_transform` that shifts them away from it before scoring;
[`add_epsilon`][stagereg.core.add_epsilon] builds one:

```python
import jax.numpy as jnp

from stagereg.core import NegativeLogLikelihoodLoss, add_epsilon
from stagereg.core.types import Batch
from stagereg.jax import Gamma

loss = NegativeLogLikelihoodLoss(target_transform=add_epsilon(1e-6))
prediction = Gamma(concentration=jnp.full((2,), 2.0), rate=jnp.ones((2,)))
batch = Batch(inputs=None, targets=jnp.array([0.0, 1.0]))

assert bool(jnp.all(jnp.isfinite(loss.per_example(prediction, batch))))
```

The variance stage of two-step training uses exactly this transform on its
squared residuals.

## Training with a distribution loss

A runner such as [`run_supervised`][stagereg.jax.run_supervised] expects a
[`SupervisedLoss`][stagereg.jax.SupervisedLoss]: one scalar from a model, a
batch's arrays, a key and a training flag. A per-example objective is not
that, so [`make_supervised_loss`][stagereg.jax.make_supervised_loss] adapts
it. The adapted loss calls the model, hands the returned distribution to
`per_example`, applies any sample weights and reduces with the mean. For
evaluation it predicts with an inference-mode clone, so validation never
changes the model being trained.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core import BetaNLLLoss
from stagereg.core.types import Batch
from stagereg.jax import (
    GaussianHead,
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

inputs = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
targets = 2.0 * inputs + 0.1 * jnp.sin(20.0 * inputs)


def loader(*, split, epoch):
    return [Batch(inputs=inputs, targets=targets)]


model = GaussianHead(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.adam(learning_rate=0.05))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))
loss = make_supervised_loss(
    BetaNLLLoss(beta=0.5, stop_gradient=jax.lax.stop_gradient)
)

result = run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=loader,
    loss=loss,
    state=state,
    epochs=3,
)
print(f"final training loss: {result.loss:.4f}")
```

The same adapter accepts a deterministic objective such as
[`SquaredErrorLoss`][stagereg.core.SquaredErrorLoss], for a model that returns
plain arrays.

## Where to go next

- [Epoch metrics](epoch-metrics.md) scores the predictive distribution with
  RMSE, interval coverage and CRPS during training.
- The mean-variance regression example on the [Examples](examples.md) page
  trains a Gaussian head end to end with early stopping.
- The [`stagereg.core.distributions`](../reference/core.md#stageregcoredistributions)
  and [`stagereg.core.losses`](../reference/core.md#stageregcorelosses) reference
  sections list every signature.
