# Posterior stage

The [`PosteriorStage`][stagereg.jax.PosteriorStage] is an optional third stage
after [two-step mean/variance training](two-step-training.md). It replaces the
single trained mean function with a posterior over mean functions, starting
from the trained mean and holding the aleatoric variance fixed, so predictions
also carry the epistemic variance: the uncertainty about the mean that more
data would reduce.

## When to add it

When a run that stops after the variance stage is complete, its predictive is a Gaussian with the trained mean and the aleatoric variance, which is all you need when the inputs you predict at look like the training inputs.

The posterior stage allows to quantify the epistemic uncertainty of the model predictions. This is useful in at least two situations:

1. When you want to know the uncertainty of the model given the data, which is important for decision making and risk assessment.
2. When you seek the uncertainty in regions where the data is scarse or absent, such as extrapolation or rare operating conditions. In these cases, the trained mean is only one of many mean functions that fit the data equally well, and the aleatoric variance alone understates the uncertainty.

## Terms

The stage and its inference methods share a small vocabulary, defined in the
[glossary](../glossary.md):

| Term | Meaning |
| --- | --- |
| Inference method | The interchangeable strategy that turns the stage's posterior problem into a posterior, such as [`BayesByBackprop`][stagereg.jax.BayesByBackprop], [`PreconditionedSGLD`][stagereg.jax.PreconditionedSGLD] or [`MCDropout`][stagereg.jax.MCDropout]. The stage decides what is inferred; the method decides how. |
| Posterior | What the inference method produces: an approximate posterior over mean functions, from which draws can be taken. |
| Inferred subnetwork | The part of the network's parameters the stage infers, selected with `inferred_subnetwork`; by default the whole network. |
| Frozen remainder | Every other parameter, held at its trained-mean value. |
| Draw | One mean function taken from the posterior, the same function at every input: a parameter sample, or for MC-Dropout one dropout mask set. A finite posterior, such as pSGLD's retained samples, always uses all its draws. |
| Posterior predictive | The [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive]: the equally weighted mixture, over the draws, of Gaussians centred on each draw's mean with the aleatoric variance. |
| Moment-matched predictive | The [`MomentMatchedPredictive`][stagereg.jax.MomentMatchedPredictive]: the single Gaussian with the posterior predictive's mean and variance, the aleatoric plus the epistemic variance. A summary, not a substitute. |

A draw is a mean function, not a target value; a value drawn from a predictive
distribution, as the CRPS uses, is a predictive sample.

## A three-stage run

The run below trains a linear mean model and a
[`GammaHead`][stagereg.jax.GammaHead] variance model as in
[the two-step guide](two-step-training.md), then a posterior stage with Bayes by
Backprop. The stage needs the number of training examples, `dataset_size`, to
scale each batch's log-likelihood to the whole data set. It validates every
epoch on the current posterior predictive and reports the NLL as
`posterior/validation/nll`.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from stagereg.core import Batch, ParameterRole, StageState, TrainingState
from stagereg.jax import (
    BayesByBackprop,
    GammaHead,
    GammaVarianceStage,
    MeanStage,
    PosteriorPredictive,
    PosteriorStage,
    PosteriorStageOptions,
    SupervisedStageOptions,
    create_optimizer,
)

inputs = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
targets = 2.0 * inputs + 0.1 * jax.random.normal(jax.random.key(0), inputs.shape)


def loader(*, split: str, epoch: int) -> list[Batch]:
    return [Batch(inputs=inputs, targets=targets)]


state = TrainingState(rng_state=jax.random.key(0))

mean_model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
mean_stage = MeanStage(
    model=mean_model,
    optimizer=create_optimizer(mean_model, optax.adam(0.1)),
    train_loader=loader,
    options=SupervisedStageOptions(epochs=3),
)
mean_stage.prepare(state)
mean_stage.train(state)

variance_model = GammaHead(1, 1, rngs=nnx.Rngs(1))
variance_stage = GammaVarianceStage(
    model=variance_model,
    optimizer=create_optimizer(variance_model, optax.adam(0.1)),
    source_loader=loader,
    options=SupervisedStageOptions(epochs=3),
)
variance_stage.prepare(state)
variance_stage.train(state)

# Step 3: a posterior over mean functions, warm-started from the trained mean.
posterior_stage = PosteriorStage(
    inference_method=BayesByBackprop(optimizer=optax.adam(0.01)),
    train_loader=loader,
    dataset_size=8,
    options=PosteriorStageOptions(epochs=3, num_draws=16, validation_loader=loader),
)
posterior_stage.prepare(state)
posterior_stage.train(state)

assert state.lifecycle_state is StageState.POSTERIOR_READY
assert state.parameter_roles["posterior"] is ParameterRole.POSTERIOR
assert state.frozen_components == {"mean_model", "variance_model"}
assert len(state.metric_history["posterior/validation/nll"]) == 3
assert len(state.metric_history["posterior/validation/crps"]) == 3
```

The mean network is copied for the posterior network, so `mean_model` itself is
untouched. To use a different module with the same parameter tree, pass it as
`model`; a copy of it is warm-started, so it is untouched as well. The prior defaults to an
[`IsotropicGaussianPrior`][stagereg.jax.IsotropicGaussianPrior] with precision 1;
pass `prior` to change it.

Validation reports two metrics by default: `nll`, the exact negative
log-likelihood of the posterior predictive, and `crps`, its closed-form CRPS,
scored by
[`ClosedFormContinuousRankedProbabilityScore`][stagereg.core.ClosedFormContinuousRankedProbabilityScore].
The closed-form CRPS evaluates the mixture's own formula rather than predictive
samples, so the default suite needs neither a predictive sample count nor an
evaluation grid, and early stopping, checkpoint selection and HPO pruning
compare an exact score with no Monte Carlo noise and no bias toward narrow
posteriors. To score other metrics, give the options a
`validation_metrics` suite with a
[`PosteriorPredictivePredictor`][stagereg.jax.PosteriorPredictivePredictor];
[Epoch metrics](epoch-metrics.md) explains how.

Every validation is scored under one fixed key, the options' `validation_key`,
which defaults to `jax.random.key(0)`. It is split once into a draw key, which
selects the posterior's draws, and a metric key, into which each validation
batch's index is folded to draw its predictive samples when a metric needs
them; the default suite needs none, so only the draws depend on the key. Every
epoch, and every run or HPO trial, therefore scores the same draws and the same
predictive samples, so early stopping, checkpoint selection and a pruner compare
posteriors rather than re-drawing noise. The same posterior then scores
identical metrics only if the validation loader yields the same batches in the
same order every epoch; with a shuffling loader the draws stay fixed and only
the predictive samples' keys move with the batch order. Validation never reads
or advances the run's `rng_state`, so turning it on or off leaves training
unchanged. Pass another key to see how sensitive the metrics are to the draws:

```{.python notest}
options = PosteriorStageOptions(
    epochs=3, validation_loader=loader, validation_key=jax.random.key(7)
)
```

## Predicting with the posterior

Once trained, or restored, the stage predicts with its posterior:
[`predict`][stagereg.jax.PosteriorStage.predict] returns the
[`PosteriorPredictive`][stagereg.jax.PosteriorPredictive] at the inputs, the
mixture over the posterior's draws under the key, each with the frozen
aleatoric variance. It takes `options.num_draws` draws of an unlimited
posterior, such as Bayes by Backprop's, and all draws of a finite one, such as
pSGLD's. Continuing the run above:

```{.python continuation}
predictive = posterior_stage.predict(inputs, jax.random.key(1))
assert predictive.draws.shape == (16, 8, 1)

nll = -predictive.log_prob(targets).mean()
summary = predictive.moment_matched()
assert bool(jnp.all(summary.epistemic_variance >= 0.0))
```

For every built-in posterior, `predict` draws the means, evaluates the
aleatoric variance and forms the predictive in one compiled call, compiled once
per input shape. Score with the posterior predictive
itself: it is exact for the draws taken and may be skewed or multimodal. The
moment-matched predictive is for reporting the variance split or drawing a
band.

The trained posterior is also registered as
`state.model_components["posterior"]`, and is consumed only through draws:
`sample_means` returns each draw's mean at the inputs, shaped `[S, *batch]`.
An unlimited posterior needs the number of draws; a finite one refuses it and
always uses all its draws, so call `sample_means(inputs, key)` without it.
Combined with the frozen variance model's aleatoric variance, the draws form
the same posterior predictive by hand:

```{.python continuation}
posterior = state.model_components["posterior"]
draws = posterior.sample_means(inputs, jax.random.key(1), 16)
by_hand = PosteriorPredictive(
    draws=draws, aleatoric_variance=variance_model(inputs).mean()
)
assert bool(jnp.allclose(by_hand.mean(), predictive.mean(), atol=1e-5))
```

## Inferring a subnetwork

By default the stage infers every parameter of the network. To infer only part
of it, such as one convolutional layer of a larger network, pass an NNX filter
as `inferred_subnetwork`; continuing the run above, this stage would infer the
linear layer's kernel alone:

```{.python continuation}
kernel_stage = PosteriorStage(
    inference_method=BayesByBackprop(optimizer=optax.adam(0.01)),
    train_loader=loader,
    dataset_size=8,
    options=PosteriorStageOptions(epochs=3, num_draws=16),
    inferred_subnetwork=nnx.PathContains("kernel"),
)
```

The filter selects from the warm-started posterior network, and only its
`nnx.Param` leaves are inferred, so random-number or batch-statistic variables
never are. Every other parameter is the frozen remainder: it keeps its trained-mean
value in every draw. The inference method sees the inferred subnetwork alone:
the posterior problem's initial parameters, its mean function and the prior
all cover that subtree, so a method needs no change to infer a subnetwork.
`prepare` refuses a filter that selects no parameter, such as a mistyped path.

The checkpoint does not record the filter. Restore with the filter the posterior
was trained with: Bayes by Backprop and pSGLD refuse a saved posterior whose
tree differs from the configured subnetwork's.

## MC-Dropout

[`MCDropout`][stagereg.jax.MCDropout] needs a posterior network with dropout
layers. Pass it as `model`: here it is a mean network with a hidden layer plus
dropout on that layer, so it has the mean network's parameter tree, since
dropout has no parameters, and infers the same subnetwork as any other method.
Each mask then drops some of the hidden units. A training state holds one
posterior, so this trains such a mean network and the variance model in a
fresh state:

```{.python continuation}
from stagereg.jax import MCDropout


class MLP(nnx.Module):
    """A mean network with one hidden layer of 16 units."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(1, 16, rngs=rngs)
        self.output = nnx.Linear(16, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(nnx.relu(self.hidden(inputs)))


class DropoutMLP(MLP):
    """The mean network with a hidden dropout mask shared by the whole batch."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(rngs=rngs)
        self.dropout = nnx.Dropout(0.1, broadcast_dims=(0,), rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(self.dropout(nnx.relu(self.hidden(inputs))))


mlp = MLP(rngs=nnx.Rngs(0))
mlp_stage = MeanStage(
    model=mlp,
    optimizer=create_optimizer(mlp, optax.adam(0.1)),
    train_loader=loader,
    options=SupervisedStageOptions(epochs=3),
)
dropout_state = TrainingState(rng_state=jax.random.key(0))
for stage in (mlp_stage, variance_stage):
    stage.prepare(dropout_state)
    stage.train(dropout_state)

dropout_stage = PosteriorStage(
    inference_method=MCDropout(optimizer=optax.adam(0.01)),
    train_loader=loader,
    dataset_size=8,
    options=PosteriorStageOptions(epochs=3, num_draws=16, validation_loader=loader),
    model=DropoutMLP(rngs=nnx.Rngs(2)),
)
dropout_stage.prepare(dropout_state)
dropout_stage.train(dropout_state)

mask_draws = dropout_state.model_components["posterior"].sample_means(
    inputs, jax.random.key(3), 16
)
assert mask_draws.shape == (16, 8, 1)
assert bool(jnp.any(mask_draws.std(axis=0) > 0.0))
```

The stage fine-tunes the inferred subnetwork with dropout active, under fresh
masks for every batch, maximizing the log-likelihood plus the log-prior. Each
draw of the posterior is the fine-tuned network under one mask set, drawn from
one key, and there are as many draws as you ask for. `broadcast_dims=(0,)`
shares one mask across the batch axis, so a draw is the same function at every
input; without it, every row would get its own mask and a draw would not be one
mean function. MC-Dropout supports early stopping, and its finalized checkpoint
holds the fine-tuned subnetwork alone; restore it with the same `model`.
[The XSin example](#the-xsin-example) runs MC-Dropout on a mean network with
one such dropout layer before its output layer.

## Restoring three stages

The posterior stage's finalized checkpoint holds the posterior alone, not the
mean and variance weights, so a later process restores the three stages in
order: mean, then variance, then posterior. Each stage's `restore` refuses,
before changing anything, while the stages it depends on are missing.
[Checkpoints](checkpoints.md#the-posterior-stage) shows the calls and what each
checkpoint holds.

## Choosing an inference method

| | [`BayesByBackprop`][stagereg.jax.BayesByBackprop] | [`PreconditionedSGLD`][stagereg.jax.PreconditionedSGLD] | [`MCDropout`][stagereg.jax.MCDropout] |
| --- | --- | --- | --- |
| Kind | Mean-field Gaussian variational inference | SG-MCMC: a preconditioned Langevin chain | MAP fine-tuning with dropout active; draws are dropout mask sets (Gal and Ghahramani, 2016) |
| Posterior | Unlimited: takes `num_draws` draws per prediction | Finite: the retained samples, all used | Unlimited: takes `num_draws` draws per prediction |
| Early stopping | Supported, with a best checkpoint | Refused; only a finalized checkpoint | Supported, with a best checkpoint |
| Cost of a prediction | `num_draws` forward passes, chosen freely | One forward pass per retained sample | `num_draws` forward passes, chosen freely |
| Checkpoint size | Twice the network's parameters | The network's parameters times the retained samples | The network's parameters |
| Approximation | Independent Gaussians ignore correlations between parameters, so the spread of the draws can have the wrong shape | The retained samples approach the posterior as the chain runs, if the step size is small enough | The spread of the draws comes from the dropout rate, a fixed choice, not from the data, so its size follows the rate rather than the amount of data |
| Tuning | Learning rate and initial standard deviation | Step size, burn-in and thinning; too large a step inflates the epistemic variance everywhere | Dropout rate and learning rate; needs a posterior network with batch-shared dropout passed as `model` |

Start with Bayes by Backprop when you want early stopping, small checkpoints
and cheap predictions. Choose pSGLD when the shape of the epistemic variance
matters, as it does for extrapolation, and you can afford tuning the step size
and storing samples. Choose MC-Dropout for the cheapest step from a trained
mean network to a posterior, accepting that the size of its spread follows the
chosen dropout rate rather than the data. In the XSin example below, all three
epistemic variances grow away from the training data: pSGLD's grows much more
relative to its variance inside the training domain, where Bayes by Backprop's
is spread, and MC-Dropout's grows the most in absolute terms, which lowers the
NLL outside the training domain far more than the other two.

pSGLD counts `burn_in` and `thinning` in update steps, that is training
batches. It has no posterior before its first retained sample, so the stage
skips validation for the epochs that end before then: they record no
validation metrics. The chain starts from the trained mean, which already fits
the data, so a short burn-in is usually enough.

pSGLD keeps its retained samples in a buffer whose slots it reserves up front,
and validation predicts from every slot with the spare ones masked out, so the
posterior keeps one shape as samples accumulate and validation compiles nothing
after the first epoch. The buffer is sized for the samples the run retains,
from the stage's `epochs`, the `dataset_size` and the size of the first batch,
assuming `ceil(dataset_size / batch size)` batches an epoch. A loader that
yields more batches than that retains more samples than planned: a full buffer
doubles, and each doubling compiles validation once more. Set `capacity` to the
number of samples the run retains to size it yourself. The posterior's
`samples` hold the retained samples alone, however many slots are reserved.

```python
from stagereg.jax import PreconditionedSGLD

# Eight batches per epoch: retain the position at the end of every epoch.
method = PreconditionedSGLD(step_size=3e-3, burn_in=0, thinning=8)
```

## Writing your own inference method

Any object satisfying [`InferenceMethod`][stagereg.jax.InferenceMethod] can
drive the stage. Three parts of the seam serve methods beyond Bayes by Backprop,
pSGLD and MC-Dropout:

- The problem's `mean_function(parameters, inputs, key)` and
  `log_likelihood(parameters, batch, key)` take a PRNG key. The stage reseeds
  the posterior network's random-number streams from it, so a network with
  `nnx.Dropout` draws its masks from the key, purely under `jax.jit` and
  `jax.grad`; a deterministic network ignores it. Pass the key `update`
  receives to the log-likelihood, and one key per draw to the mean function in
  `sample_means`, so that a draw is the same function at every input.

- The problem's `aleatoric_variance` maps inputs to the fixed aleatoric
  variance, without gradient, the same variance its `log_likelihood` uses. An
  implicit method, such as a generator trained against a discriminator, adds
  Gaussian noise with this variance to a draw's means to get predictive samples.
- A method lists its own training signals, such as a discriminator loss, in
  [`diagnostics`][stagereg.jax.InferenceMethod.diagnostics]. The stage records
  each declared key of `update`'s output as `posterior/train/<key>`, its mean
  over the epoch's batches, and passes it to event sinks and trackers like any
  other metric. Undeclared keys are ignored, and `prepare` refuses a key whose
  tag the stage already records, such as `loss`. Bayes by Backprop, pSGLD and
  MC-Dropout declare none.
- Validation and [`predict`][stagereg.jax.PosteriorStage.predict] form a
  built-in posterior's predictive in one compiled call per batch. They call
  your posterior's `sample_means` from Python, with the aleatoric variance
  compiled, which is slower unless `sample_means` is compiled itself, e.g. by
  dispatching to a `jax.jit` function that takes the problem's `mean_function`
  as a static argument and the posterior's arrays as traced ones. The problem's
  `epochs` says how many epochs the stage trains for, should the method size
  anything up front.

## The XSin example

`examples/jax/xsin/posterior.py` runs mean, variance and posterior stages on the
XSin benchmark with each method, prints the NLL and closed-form CRPS of the
variance stage and of the posterior predictive, and plots the moment-matched predictive band
against the variance stage's band. [Examples](examples.md#xsin-benchmark) lists
the run command.
