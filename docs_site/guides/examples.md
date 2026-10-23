# Examples

The runnable examples live under
[`examples/jax/`](https://github.com/GroupiSP/stagereg/tree/main/examples/jax) in
the repository, so run the commands below from a clone of it. They are ordered
from the gentlest upward: start with the first and move down as you need more
of the library. Each one brings its own dependencies, either through a project
extra (`jax`, `plot`) or through an example-only dependency group.

Only the commands shown here are entry points. The other modules beside them,
`xsin/benchmark.py`, `xsin/posterior_benchmark.py`, `cmapss/model.py`,
`cmapss/preprocessing.py`, `cmapss/plots.py` and
`tracking/tensorboard_tracker.py`, hold shared code the scripts import.

## Simple regression

`run_supervised` end to end on a toy linear dataset: one model, held-out
validation, early stopping on the validation loss, an in-memory checkpoint
store keeping the best model, and an event sink that prints each epoch.

```bash
uv run --extra jax python examples/jax/simple_regression.py
```

Source:
[`examples/jax/simple_regression.py`](https://github.com/GroupiSP/stagereg/blob/main/examples/jax/simple_regression.py)

## Mean-variance regression

The same loop with a predictive distribution: a Gaussian head trained with a
negative log-likelihood loss on data whose noise grows with the input, scored
with RMSE and point-CRPS epoch metrics, and a plot of the predictive interval.

```bash
uv run --extra jax --extra plot python examples/jax/mve_regression.py
```

Source:
[`examples/jax/mve_regression.py`](https://github.com/GroupiSP/stagereg/blob/main/examples/jax/mve_regression.py)

## XSin benchmark

Two paired scripts on an XSin-inspired benchmark that compare joint Gaussian
mean-variance estimation with two-step training: a mean stage followed by a
Gamma variance stage fitted on its residuals. Both train only on `x` in
`(0, 10)` and are evaluated on the wider `(-5, 15)`, so each run shows
interpolation and extrapolation side by side.

```bash
uv run --extra jax --extra plot python examples/jax/xsin/mve.py
uv run --extra jax --extra plot python examples/jax/xsin/two_steps.py
```

Each script prints its mean and variance errors against the known
data-generating functions, overall, inside the training domain and outside
it, and plots the result with the training-domain boundaries marked. Every
model shares a SiLU backbone that maps the training domain onto `[-1, 1]`.
Inside the training domain, which method has the lower mean or variance error
depends on the seed.
Outside it, both mean errors grow large; they are reported, not assumed to
improve. The benchmark
reproduces the qualitative comparison motivated by Yi and Bessa (2025) with
compact settings; it does not claim to reproduce the paper's architectures,
runtime or reported values.

A third script adds the optional [posterior stage](posterior-stage.md) after
the two stages, once each with Bayes by Backprop, pSGLD and MC-Dropout:

```bash
uv run --extra jax --extra plot python examples/jax/xsin/posterior.py
```

For each inference method it prints the NLL and the closed-form CRPS of the
variance stage's Gaussian and of the posterior predictive, scored on one noisy
observation per evaluation point, overall, inside the training domain and
outside it. Both stages use the same closed-form CRPS, computed from each predictive's
own formula, so their difference reflects the predictives rather than the
estimator. It plots the moment-matched predictive's 95% band over the variance
stage's band, with the aleatoric and epistemic variance below.

Under the seeded configuration the scores are:

| Predictive | NLL | CRPS | Interpolation NLL | Interpolation CRPS | Extrapolation NLL | Extrapolation CRPS |
| --- | --- | --- | --- | --- | --- | --- |
| Variance stage | 188.38 | 5.512 | 1.005 | 0.376 | 374.52 | 10.614 |
| Bayes by Backprop | 170.91 | 5.539 | 1.048 | 0.388 | 339.65 | 10.655 |
| pSGLD | 184.26 | 5.926 | 0.948 | 0.345 | 366.36 | 11.471 |
| MC-Dropout | 23.60 | 4.338 | 1.089 | 0.393 | 45.97 | 8.256 |

Bayes by Backprop and pSGLD lower the overall NLL, in the extrapolation region,
where the variance stage's band is far too narrow. Bayes by Backprop leaves the
overall CRPS about unchanged and pSGLD raises it, through a worse extrapolation
CRPS. Their epistemic variances both grow away from the training data. pSGLD's
grows the most relative to its variance inside the training domain, where both
its scores improve; Bayes by Backprop's is also spread through the training
domain and worsens both scores there. MC-Dropout, with one dropout layer before
the output layer of the mean network, lowers the overall NLL far more, by about
a factor of eight, and lowers the overall CRPS too: its epistemic variance grows
steeply outside the training domain, widening its band there, while inside it
both scores worsen slightly. All three bands remain too narrow far into
extrapolation, where the trained mean continues the oscillation poorly.

Source:
[`examples/jax/xsin/`](https://github.com/GroupiSP/stagereg/tree/main/examples/jax/xsin)

## CMAPSS remaining useful life

Two-step probabilistic remaining-useful-life estimation on NASA CMAPSS FD001: a
CNN mean stage, a Gamma variance stage on its squared residuals, and a
composite Gaussian scored against the official test split.

```bash
uv run --group example-cmapss python examples/jax/cmapss/run.py
```

To fetch the archive ahead of time, or only to plot the raw sensor
trajectories, run the data module on its own:

```bash
uv run --group example-cmapss python examples/jax/cmapss/data.py --fetch
uv run --group example-cmapss python examples/jax/cmapss/data.py
```

The first run downloads the CMAPSS archive from NASA and caches it on disk, so
it needs network access once. The README explains how to point the example at
a copy you already have.

README:
[`examples/jax/cmapss/README.md`](https://github.com/GroupiSP/stagereg/blob/main/examples/jax/cmapss/README.md)

## Tracking a run

The mean-variance regression again, with the whole run recorded to TensorBoard
through `TrackerEventSink`: per-epoch metrics under their metric tags, the
hyperparameters under their parameter paths, and a final figure.

```bash
uv run --group example-tracking python examples/jax/tracking/run.py --logdir runs/
uv run --group example-tracking tensorboard --logdir runs/
```

README:
[`examples/jax/tracking/README.md`](https://github.com/GroupiSP/stagereg/blob/main/examples/jax/tracking/README.md)
