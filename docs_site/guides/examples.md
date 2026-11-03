# Examples

The runnable examples live under
[`examples/jax/`](https://github.com/GroupiSP/stagereg/tree/main/examples/jax) in
the repository, so run the commands below from a clone of it. They are ordered
from the gentlest upward: start with the first and move down as you need more
of the library. Each one brings its own dependencies, either through a project
extra (`jax`, `plot`) or through an example-only dependency group.

Only the commands shown here are entry points. The other modules beside them,
`xsin/benchmark.py`, `xsin/posterior_benchmark.py`, `cmapss/model.py`,
`cmapss/preprocessing.py`, `cmapss/plots.py`,
`tracking/tensorboard_tracker.py` and every `clustered_data/` module but
`compare.py`, hold shared code the scripts import.

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
composite Gaussian scored against the official test split by RMSE, 95%
interval coverage and closed-form CRPS.

```bash
uv run --group example-cmapss python examples/jax/cmapss/run.py
```

To fetch the archive ahead of time, or only to plot the raw sensor
trajectories, run the data module on its own:

```bash
uv run --group example-cmapss python examples/jax/cmapss/data.py --fetch
uv run --group example-cmapss python examples/jax/cmapss/data.py
```

To compare that staged model with a joint model (one CNN backbone with a
Gaussian head, trained on the Gaussian negative log-likelihood) on the same
prepared data, run the comparison. It trains both models from `--seeds` seeds
(default 5) on a fixed train/validation split and prints a `staged` vs `joint`
table of RMSE, 95% interval coverage, closed-form CRPS and NLL as
`mean ± std` over seeds. It then plots the first seed's validation RUL curves,
the staged model's row above the joint model's; pass `--plot-path` to save the
figure instead of displaying it. The joint model has roughly half the staged
model's parameters.

```bash
uv run --group example-cmapss python examples/jax/cmapss/compare.py --seeds 5
```

The first run downloads the CMAPSS archive from NASA and caches it on disk, so
it needs network access once. The README explains how to point the example at
a copy you already have.

README:
[`examples/jax/cmapss/README.md`](https://github.com/GroupiSP/stagereg/blob/main/examples/jax/cmapss/README.md)

## Clustered data

How confident the joint and staged models are inside the training data and
away from it, on the "Simulate data" set of the UvA DL2 Bayesian neural
network tutorial 1: two clusters of 500 points on `[-0.2, 0.2]` and `[0.6, 1.0]` (seed 42) with a
sum-of-sines target and noise σ = 0.02. The noise also enters the sines'
arguments, so the targets scatter about the noise-free curve by about 0.075
(RMS) inside the clusters: no mean can reach an in-distribution RMSE much
below that. Every model shares one Tanh MLP backbone of 5 hidden layers of 10
units, which first maps the training inputs' range `[-0.2, 1.0]` onto
`[-1, 1]` with a fixed affine map.

```bash
uv run --extra jax --extra plot python examples/jax/clustered_data/compare.py
```

The comparison trains the staged model: a mean stage and a Gamma variance
stage, each early-stopped on its training loss (at most 3,000 epochs, patience
100), then the [posterior stage](posterior-stage.md) for a fixed 2,000 epochs
under an isotropic Gaussian prior of precision 1, once for each inference
method on the same trained stages. Both supervised stages use Adam at learning
rate 1e-3 and batches of 100.

| Method | Posterior-stage setting | Backbone dropout | Draws |
| --- | --- | --- | --- |
| Bayes by Backprop (`BBB`) | Adam at 1e-2, initial σ 1e-2 | 0 | 100 sampled at prediction |
| MC dropout (`MCD`) | Adam at 1e-3 | 0.1 (`--mc-dropout-rate`) | 100 sampled at prediction |
| pSGLD (`pSGLD`) | step size 3e-3 | 0 | 100 retained, evenly thinned over the final 1,000 epochs |

The three runs differ only in the inference method and the dropout rate; the
architecture, prior, batch size and epochs are shared. pSGLD's burn-in and
thinning count update steps: at the defaults it burns in the first 10,000 of
its 20,000 steps and keeps every 100th step after them, the last one included.

It also infers the joint posterior, the baseline: one network shares the
backbone between a mean head and a variance head (a softplus plus a 1e-6
floor), and each of the three methods infers every weight, variance head
included, under the network's own Gaussian likelihood and the same prior. Each
joint run uses its staged counterpart's settings, dropout rate and number of
draws from the table above. It starts from a random initialization with no warm
start and no early stopping, and runs exactly as many epochs as the staged
model of the same seed: its mean-stage, variance-stage and posterior epochs
together. Joint pSGLD burns in the mean-stage and variance-stage epochs as
well, so its 100 draws fall in the same final 1,000 epochs as the staged
run's. Each joint draw carries its own mean and variance. The six
configurations take a few minutes at the defaults on a laptop CPU.

It scores 100 posterior draws on a grid of 3,000 points over `[-0.5, 1.5]`,
with fresh noisy targets drawn from a different seed, split into the glossary's
[in-distribution, gap and extrapolation regions](../glossary.md). Every
configuration is reduced to per-draw predictive means and variances on the
grid, and its predictive is the equal-weight mixture of those Gaussians. The
table prints one row per configuration: the in-distribution RMSE of the
predictive mean, then for each region the aleatoric σ (square root of the
region average of the draw-averaged variance), the epistemic σ (square root of
the region average of the variance of the means over draws) and the 95%
coverage of the mixture's exact central interval.

The squared residuals here are about 1e-3, far from unit scale. The
variance stage divides the Gamma head's rate by their mean (see
[Residual scale](two-step-training.md#residual-scale)), which keeps the
staged extrapolation σ_ale near 0.13 at the default seed. Without that
division, the head's raw rate output had to reach about +100 to fit the
data. Its drift past `x = 1.2` then pushed the rate onto its 1e-6 floor, and
the extrapolation σ_ale grew to 331.

`--seeds N` (default 1) repeats the whole comparison over seeds `--seed` to
`--seed + N - 1`. The training data and the grid stay fixed, so only the
initialization, the batch order and the draws vary. With more than one seed,
every table cell reports `mean ± std` over seeds:

```bash
uv run --extra jax --extra plot python examples/jax/clustered_data/compare.py --seeds 5
```

It then draws the first seed's configurations in two figures. The uncertainty
grid has one row per training and one column per inference method; each panel
shows the training points, the noise-free truth, the predictive mean, an inner
band of ±1.96 epistemic σ and an outer band of ±1.96 total σ, with the gap and
extrapolation regions shaded. The second figure plots one
[calibration curve](../glossary.md) per method in the extrapolation region, at
nominal levels 0.05 to 0.95, from the same mixture scoring as the table's
coverage. Both
figures are shown on screen, or saved as `uncertainty_grid.png` and
`extrapolation_calibration.png` into the directory `--plot-path` names, with
no window opened.

Every setting is a flag, `--noise-std` among them; `--help` lists them with
their defaults. A quick run at small settings, saving the figures:

```bash
uv run --extra jax --extra plot python examples/jax/clustered_data/compare.py \
    --mean-epochs 50 --variance-epochs 50 --posterior-epochs 20 --num-draws 10 \
    --plot-path figures/
```

Source:
[`examples/jax/clustered_data/`](https://github.com/GroupiSP/stagereg/tree/main/examples/jax/clustered_data)

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
