# Training a supervised model

[`run_supervised`][stagereg.jax.run_supervised] trains one Flax NNX model with
one Optax optimizer. It is the single-stage runner of the JAX backend: you hand
it your data as a loader, your objective as a loss, and the training state it
mutates, and it runs epochs. This guide covers what each of those pieces is and
why the runner asks for it in that form. For the full signature, see the
[`stagereg.jax` reference](../reference/jax.md).

## A complete run

The example fits `y = 3x + 2` with mini-batches that are reshuffled every
epoch:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core import Batch, SquaredErrorLoss
from stagereg.jax import (
    create_optimizer,
    initialize_training_state,
    make_supervised_loss,
    run_supervised,
)

inputs = jnp.linspace(-1.0, 1.0, 64).reshape(-1, 1)
targets = 3.0 * inputs + 2.0


def train_loader(*, split, epoch):
    order = jax.random.permutation(jax.random.key(epoch), inputs.shape[0])
    return [
        Batch(inputs=inputs[order[i : i + 16]], targets=targets[order[i : i + 16]])
        for i in range(0, inputs.shape[0], 16)
    ]


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
optimizer = create_optimizer(model, optax.sgd(learning_rate=0.1))
state = initialize_training_state(model, optimizer, rng_key=jax.random.key(0))

result = run_supervised(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    loss=make_supervised_loss(SquaredErrorLoss()),
    state=state,
    epochs=3,
)

assert set(result.metrics) == {"loss"}
assert len(state.metric_history["supervised/train/loss"]) == 3
```

## The model, the optimizer and the training state

[`create_optimizer`][stagereg.jax.create_optimizer] wraps an Optax gradient
transformation in an `nnx.Optimizer` bound to the model's parameters.
[`initialize_training_state`][stagereg.jax.initialize_training_state] registers
both in a [`TrainingState`][stagereg.core.TrainingState], together with the root
PRNG key of the run.

The runner mutates all three in place: the model's parameters, the optimizer's
state, and the training state's RNG key and metric history. Nothing is returned
that you have to thread back in. That is what lets a later stage, a checkpoint
or an [event sink](../glossary.md) see the same live state the runner trained.

## Loaders

`train_loader` is a [`LoaderFactory`][stagereg.core.LoaderFactory]: a callable
taking keyword-only `split` and `epoch` and returning an iterable of
[`Batch`][stagereg.core.Batch] objects. The runner calls it once at the start of
every epoch, with `split="train"`, and iterates what it returns.

A factory rather than a fixed iterable is what makes per-epoch behaviour
explicit. The runner never shuffles, batches or resamples on your behalf;
whatever ordering an epoch needs is decided in the factory, from the `epoch`
it is given. Seeding the shuffle with the epoch index, as above, makes a run
reproducible without the loader holding any state of its own. Since it is
called afresh each epoch, the factory may also return a generator that reads
from disk or from another data pipeline.

A `Batch` carries `inputs`, `targets`, an optional per-example
`sample_weight`, and free-form `metadata`. `inputs` may be any PyTree your
model accepts.

## Losses

The `loss` is a [`SupervisedLoss`][stagereg.jax.SupervisedLoss]: a callable
`loss(model, inputs, targets, sample_weight, key, training)` that returns a
scalar. The runner differentiates it with respect to the model's parameters to
take one optimizer step per batch.

The usual way to obtain one is
[`make_supervised_loss`][stagereg.jax.make_supervised_loss], which adapts a
backend-neutral per-example objective such as
[`SquaredErrorLoss`][stagereg.core.SquaredErrorLoss] or
[`NegativeLogLikelihoodLoss`][stagereg.core.NegativeLogLikelihoodLoss]. It calls
the model, applies `sample_weight` when the batch has one, and reduces to a
mean. It also puts the model in training mode for training calls and evaluates
an inference-mode clone otherwise, so that validation cannot disturb the live
model.

You can also write the callable yourself. The `key` is a fresh PRNG key split
from the training state for every batch, for losses that sample (dropout, a
Monte Carlo estimate); `training` is `True` during the update step:

```python
import jax
import jax.numpy as jnp
from flax import nnx


def mean_absolute_error(model, inputs, targets, sample_weight, key, training):
    errors = jnp.abs(model(inputs) - targets)
    if sample_weight is not None:
        errors = errors * sample_weight
    return jnp.mean(errors)


model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
inputs = jnp.ones((4, 1))
value = mean_absolute_error(model, inputs, inputs, None, jax.random.key(0), True)
assert value.shape == ()
```

## Fixed epochs by default

Without a validation strategy or an early stopper, `run_supervised` runs
exactly `epochs` epochs and stops. There is no implicit convergence check, no
held-out split carved from the training data, and no "best" model chosen behind
your back: the model you get back is the model after the last epoch.

`epochs` is always the maximum. Adding an early stopper can end the run sooner
but never later; see [Validation and early stopping](validation.md).

## The compiled training step

`run_supervised` compiles one training step for the run and binds `model` and
`optimizer` to it once, so a step does not walk the NNX module graph on every
call. The step updates the model's and optimizer's variables in place, and
changes you make to them between steps, such as restoring a checkpoint, are
seen by the next step. The module structure and plain Python attributes are
fixed when the run starts: a loss that sets an attribute, say
`model.flag = ...`, inside the step does not change `model`. Change such
attributes outside the run, or use
[`make_train_step`][stagereg.jax.make_train_step], which writes them back at
the cost of walking the graph on every call.

## What comes back

The runner returns a [`StageResult`][stagereg.core.StageResult]:

- `result.metrics` holds the last epoch's training metrics under their bare
  metric names, such as `loss`;
- `result.loss` is the last epoch's mean training loss;
- `result.state` is the same training state you passed in.

Every epoch is also appended to `state.metric_history`, keyed by
[metric tag](../glossary.md) `stage/split/metric`. The stage segment comes from
the `stage` argument, `"supervised"` unless you name it, and is present even in
a single-stage run, so the loss above is recorded under
`supervised/train/loss`. The bare name in `result.metrics` and the tag in the
history name the same quantity; only the history says which stage and split it
came from.

To compute more than the loss on each epoch, pass a
[`MetricSuite`][stagereg.jax.MetricSuite] as `metrics`. To watch a run while it
trains, pass [event sinks][stagereg.core.EventSink] as `event_sinks`.
