# NASA CMAPSS Dataset

The example in this directory focuses on predicting the remaining useful life (RUL) of the simulated jet engine units that compose the NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset, available from the [NASA CMAPSS dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8).

In particular, the scope is restricted to the FD001 dataset, which consists of 100 train and 100 test time trajectories. The train trajectories are run-to-failure, while the test trajectories are partial histories for which the RUL labels are provided in a separate file.

## Description of the files

### `data.py`

Fetches the CMAPSS archive and loads the FD001 splits into pandas
DataFrames. The train/test DataFrames retain only the unit ID, time cycles,
and sensors 11, 12, 4, 7, 15, 20, 21, 2, and 17; the operational settings
and other sensors are discarded.

The archive is kept in a persistent on-disk cache, so it is downloaded from
NASA at most once rather than once per run. Two environment variables
override the defaults:

- `STAGEREG_CMAPSS_CACHE_DIR` — the directory holding the cached archive.
  Defaults to a per-user cache location (`XDG_CACHE_HOME` if set, otherwise
  `~/.cache/stagereg-cmapss`).
- `STAGEREG_CMAPSS_ARCHIVE` — the path to a manually obtained copy of the
  CMAPSS zip. When set, it seeds the cache and no download is attempted,
  which is useful when the network is unavailable or untrusted.

See `docs/adr/0001-cmapss-archive-caching-and-override.md` for the rationale.

Run the module with:

```shell
uv run --group example-cmapss python examples/jax/cmapss/data.py
```

The `main` function displays all engine trajectories in seaborn line plots,
with one facet per selected sensor and independent sensor scales.

### `preprocessing.py`

Turns the loaded DataFrames into the arrays the models consume:
`split_by_unit` holds out whole units for validation, `fit_standardization`
and `apply_standardization` standardize the sensor columns with statistics
fitted on the training subset only, `build_windows` produces sliding windows
with their aligned RUL targets, and `build_last_windows` produces the single
trailing window per test unit required by the FD001 test protocol. Units
shorter than the window length are left-padded by repeating their first row.
`build_unit_rul_curve` returns a single unit's `UnitRulCurve`: its
windows with their aligned linear RUL and window-end cycles, its lifetime,
and the true linear RUL over every observed cycle. The two time axes are
named apart — `window_cycles` is what a predicted curve is drawn against,
`trajectory_cycles` what the truth is drawn against — so a caller never
re-derives either one. A unit shorter than the window length is rejected
rather than padded; see
`docs/adr/0004-cmapss-rul-curves-refuse-padded-windows.md`.
`select_lifetime_spanning_units` picks the shortest-, median-, and
longest-lifetime units of a set of trajectories, a unit's lifetime being
its maximum observed time cycle. Units are ordered ascending by lifetime
with the unit ID as tie-break and the median is the lower of the two
central entries, so the same data always yields the same three units and
the figure is reproducible.

### `model.py`

Contains the classes defining the two-stage predictive model: a Stage-1
deterministic mean model (`Cnn1DMeanModel`), a Stage-2 Gamma residual model
(`Cnn1DGammaModel`), and a `CompositeGaussianModel` combining the frozen
Stage-1 point prediction with the Stage-2 Gamma mean into a single Gaussian
predictive RUL distribution. It also holds the joint model the comparison
trains against it, `Cnn1DGaussianModel`: one backbone of the same kind,
followed by a Gaussian head predicting mean and scale together.

### `plots.py`

Builds the example's RUL curves. `plot_validation_rul_curves` takes the
standardized validation trajectories, the feature columns, a sequence of
`LabelledModel`s (a label and a trained model each, predicting a Gaussian
or a posterior predictive), the three
units to draw, and the window length, and draws an N×3 grid: one row per
model, labelled with its name, and one column per unit. `run.py` passes
its one staged model, so its figure is a single row, and its first panel's
y-label now names that model ("Staged model") above "RUL (cycles)". It also
passes the
lifetime-spanning units of the validation subset, so the columns are its
shortest-, median-, and longest-lifetime unit in that order; the module
itself decides how a RUL curve looks and never which units it looks at. Spanning the lifetime range that way lets a reader
tell whether the model's accuracy and its stated uncertainty behave
consistently across units that fail early and units that survive several
times as long.

Each column shows one unit's true linear RUL across its whole trajectory,
the model's predicted mean, and the 95% predictive interval
`loc ± 1.96 * scale` taken analytically from the model's Gaussian. A
posterior-predictive row instead draws the mixture mean, a total band
(the mixture's central 95% interval, from its quantiles) and a narrower
epistemic band (the 2.5% and 97.5% quantiles of the draws' means at each
window); each curve comes from one evaluation over the unit's windows, so
every draw is the same function along it. The
columns have independent x limits, so the band stays readable in the
short-lived unit instead of being squashed by the long-lived one's axis
range, while each column shares its x-axis across rows so that every
model's curve for a unit sits on the same time axis; y limits are
independent per panel. Each column is titled with its unit ID, the role it
plays, and that unit's lifetime in cycles, and one shared figure-level
legend names the
union of the elements drawn across rows. The predicted mean and its band start at each unit's
first full window, since no prediction exists before `window_length`
cycles of history have accumulated; the band is left unclipped at zero,
so a band dipping below zero stays visible as evidence of the Gaussian
assumption breaking down near end of life. With a save path the figure is
written there, otherwise it is displayed.

The plotted units are drawn from the **held-out validation subset**, never
from the units the model trained on: those units are never fitted, so the
curves are evidence of generalization rather than of memorized fit. They
also cannot come from the FD001 test split, whose trajectories are
truncated rather than run to failure — a unit's lifetime, and hence its
true RUL at every cycle, is only observable for the run-to-failure train
split the validation subset is carved out of.

### `run.py`

Contains the script to train and evaluate the two-stage predictive model on
the CMAPSS dataset end to end: loading the FD001 train/test splits,
standardizing and windowing them, training the Stage-1 mean model, training
the Stage-2 Gamma model on the frozen Stage-1 model's squared residuals, and
scoring the composite Gaussian model against the official FD001 test split
(`RUL_FD001.txt`), one window per test unit ending at its last observed
cycle. `prepare_cmapss_windows` returns a frozen `PreparedCmapssData`
carrying the windowed arrays together with the standardized trajectories
they were built from and the standardization statistics fitted on the
training subset, so a downstream consumer can re-window a unit under
exactly the feature scaling the models were trained with.
`build_composite_model` is the one factory for the composite predictive
model: it clones both trained stage models and returns the composite in
eval mode, so every consumer scores the same model in the same mode.
`evaluate_test_metrics` scores any model whose output is a Gaussian, the
staged model included. The mean and Gamma training functions take their
event sinks as a parameter, so each entry point decides how verbose
training is. The Gamma network is narrower than the mean network
(`gamma_hidden_channels`, default 4 against `hidden_channels` 16) and trains
with AdamW weight decay (`gamma_weight_decay`, default 0.1): its targets are
the mean's in-sample squared residuals, which understate the held-out error,
and a full-width, unregularized Gamma network overfits them into a spiky,
overconfident variance (see [the variance network](#the-variance-network)).
`train_mean_model` takes an optional `dropout_rate` for the staged
MC-Dropout model's mean stage. Next to them sit the MC-Dropout helpers
`compare.py` uses: the predictive wrapper `McDropoutPredictive` (built by
`build_mc_dropout_predictive`), the inference-only posterior-stage runner
`build_staged_mcd_posterior`, which returns a `PosteriorStagePredictive`, and
`refit_gamma_to_posterior_mean`, which re-fits the Gamma network to that
predictive's mean. The script itself does not use them.

Run it with:

```shell
uv run --group example-cmapss python examples/jax/cmapss/run.py
```

The script prints per-epoch training metrics for both stages, followed by
the final test-set metrics: RMSE, 95% predictive-interval coverage, and
closed-form CRPS. The closed-form CRPS comes from each predictive
Gaussian's own formula, so it carries no sampling noise. It then plots the RUL curves of the shortest-, median-, and
longest-lifetime held-out validation units. Pass `--plot-path` to save
that figure instead of displaying it:

```shell
uv run --group example-cmapss python examples/jax/cmapss/run.py --plot-path rul_curves.png
```

### `compare.py`

Compares the **staged model** of `run.py` with a **joint model** on the same
prepared FD001 data. The joint model (`Cnn1DGaussianModel` in `model.py`) is
one 1D-CNN backbone, with the same architecture and hyperparameters as each
staged model's backbone, followed by `stagereg`'s `GaussianHead`. It is
trained with `run_supervised` on the Gaussian negative log-likelihood for
`joint_epochs` epochs (default 20, the same budget as each stage), and
validated on the held-out validation windows after every epoch. Both models
train on the raw linear-RUL targets, with no target standardization, so any
instability of joint training on unscaled targets shows up in the results.

The data is prepared once, with the configured seed, so the train/validation
split is the same for every seed; seed index `i` initializes both models
from `seed + i`. `compare_models` returns one `SeedComparison` per seed,
holding both trained models and their test metrics: RMSE, 95% interval
coverage, closed-form CRPS and NLL. NLL is included because it is the joint
model's own training objective.

Run it with:

```shell
uv run --group example-cmapss python examples/jax/cmapss/compare.py
```

Training is silent. The script prints one line per (seed, model) run as it
finishes, then a table with one row per metric and a `staged` and a `joint`
column, each cell the `mean ± std` over seeds. The `±` is the sample
standard deviation over seeds (`ddof=1`), and 0 for a single seed. Pass
`--seeds` to trade
runtime for confidence (default 5):

```shell
uv run --group example-cmapss python examples/jax/cmapss/compare.py --seeds 10
```

Finally it plots the first seed's RUL curves for the same shortest-,
median-, and longest-lifetime held-out validation units as `run.py`: the
staged model's row above the joint model's, with a shared time axis in each
column. Using the first seed keeps the figure reproducible. Pass
`--plot-path` to save that figure instead of displaying it:

```shell
uv run --group example-cmapss python examples/jax/cmapss/compare.py --plot-path comparison.png
```

Note that the joint model has one backbone where the staged model has two
(one per stage), so it has roughly half the staged model's parameters. The
comparison holds the architecture and the per-model epoch budget fixed, not
the parameter count.

#### `--posterior-stage`: MC-Dropout posteriors

Pass `--posterior-stage` to add the **joint MC-Dropout model** (`joint+mcd`)
and the **staged MC-Dropout model** (`staged+mcd`). The comparison is then of
a single training (joint) against separate trainings (staged), each with and
without an MC-Dropout posterior. The four models per seed, in column and
figure-row order:

| Column | Model | How it is trained | Predictive distribution |
| --- | --- | --- | --- |
| `joint` | joint model | as without the flag, no dropout | Gaussian |
| `joint+mcd` | joint MC-Dropout | as `joint`, with dropout active | posterior predictive, each draw with its own aleatoric variance |
| `staged` | staged model | as without the flag | Gaussian |
| `staged+mcd` | staged MC-Dropout | as `staged`, with dropout active in the mean stage; an inference-only posterior stage (`MCDropout`); then the Gamma network re-fitted to the posterior predictive's mean | posterior predictive, all draws sharing the re-fitted aleatoric variance |

`joint` and `staged` are the same models, with the same scores, as without
the flag. All four are scored on RMSE, 95% coverage, closed-form CRPS and
NLL on the FD001 test split; the two MC-Dropout models through
`PosteriorPredictivePredictor`, on their exact mixture (mixture
log-density, coverage from mixture quantiles, the mixture's closed-form
CRPS).

Run the comparison without the flag (two models, `staged` and `joint`) or
with it (four models):

```shell
uv run --group example-cmapss python examples/jax/cmapss/compare.py --seeds 5
uv run --group example-cmapss python examples/jax/cmapss/compare.py --seeds 5 --posterior-stage
```

`--seeds` and `--plot-path` work the same either way.

##### How the MC-Dropout models are trained

Each MC-Dropout model is trained once, exactly like its baseline but with
dropout active at `dropout_rate`, and predicts with dropout active. Dropout
is the only regularizer of the joint network and of the staged mean network:
neither has a prior, weight decay or early stopping. The joint and staged
paths then differ only in single against separate training.

`joint+mcd` is the joint model trained with dropout: one Gaussian-head CNN,
trained with `run_supervised` on the Gaussian NLL for `joint_epochs` epochs
at `learning_rate`. Its prediction evaluates the network under `num_draws`
dropout-mask sets, each one applied to the whole batch, and keeps each
draw's own mean and aleatoric variance: a posterior predictive scored on its
exact mixture. The masks come from a fixed key derived from the seed, so
test scoring and the figure use the same draws.

`staged+mcd` trains its own mean network with dropout, for `mean_epochs`
epochs at `learning_rate`, then a Gamma network on its squared residuals, as
`staged` does. A `PosteriorStage` with the `MCDropout` inference method then
does inference only: its optimizer is `optax.set_to_zero()`, so the
posterior is the trained mean network, unchanged, under random dropout
masks. `PosteriorStage` needs at least one epoch, so it runs one, which only
evaluates the training loss. The posterior predictive's mean, the average
over the draws, is not the mean network with dropout off that the Gamma
network was fitted to, so `refit_gamma_to_posterior_mean` trains a fresh
Gamma network, from the same initialization and with the same width and
weight decay, on the squared residuals of the posterior predictive's mean.
The prediction is the stage's posterior predictive with `num_draws` draws
under the same fixed, seed-derived key, every draw sharing the re-fitted
aleatoric variance.

##### Reading the total and epistemic bands

In the figure the Gaussian rows (`joint`, `staged`) keep their single
`loc ± 1.96 * scale` band. Each MC-Dropout row draws the mixture mean and
two bands around it:

- the **total band**, the posterior predictive's central 95% interval
  (from the mixture's quantiles): where the next observed RUL should fall,
  counting both the aleatoric noise and the spread of the draws;
- the narrower **epistemic band**, the 2.5% and 97.5% quantiles of the
  draws' means at each window: how much the dropout draws disagree about
  the mean, i.e. the model's uncertainty about the function itself.

The gap between the two is the aleatoric part. An epistemic band that is
narrow next to the total band says that most of the uncertainty is noise
in the data rather than lack of knowledge. Each curve comes from one
evaluation over the unit's windows, so every draw is the same function
along the whole curve.

##### Defaults

The MC-Dropout settings live in `CmapssConfig` (`run.py`):

| Field | Default | Source |
| --- | --- | --- |
| `dropout_rate` | 0.1 | fixed by the spec (#173) |
| `num_draws` | 100 | fixed by the spec |
| `gamma_hidden_channels` | 4 | chosen, [below](#the-variance-network) |
| `gamma_weight_decay` | 0.1 | chosen, [below](#the-variance-network) |

Every other setting, epochs and learning rate included, is the baseline's.
The current defaults give, with `--seeds 5 --posterior-stage`:

```text
FD001 test metrics over 5 seeds (mean ± std):
                              joint          joint+mcd             staged         staged+mcd
RMSE              61.0066 ± 19.9944  49.1285 ± 12.0127   25.3996 ± 1.2857   27.3938 ± 2.0976
95% coverage        0.9740 ± 0.0219    0.9960 ± 0.0055    0.9880 ± 0.0084    0.9780 ± 0.0110
closed-form CRPS   33.2752 ± 9.4750   26.8821 ± 6.4763   14.2778 ± 0.3020   15.0790 ± 1.0761
NLL                 5.4207 ± 0.2507    5.2065 ± 0.2426    4.6213 ± 0.0223    4.6531 ± 0.0632
```

Both staged models beat both joint models on every metric. Dropout
improves the joint model on average but does not cure the instability of
its training on unscaled targets: `joint+mcd` varies as much across seeds,
and in the figure its mean rises again near the end of life. `staged+mcd`
is close to `staged`, slightly worse on average, with a larger spread
across seeds. Every model over-covers the test split, whose one window per
unit sits late in life, where the variance is smallest; the variance is
calibrated over whole lifetimes instead, see below.

#### The variance network

The test split holds only each unit's last window, so it says little about
early-life uncertainty. The held-out validation units span whole lifetimes,
and on them a Gamma network as wide as the mean network (16 channels,
plain Adam) was overconfident: its 95% coverage was 0.96 on the training
windows it was fitted on but 0.82 on the validation windows, whose mean
squared residual is 1.31 times the training one. In-sample residuals
understate the held-out error, and a flexible Gamma network fits them
closely, giving a spiky, overconfident variance.

Shrinking it fixes most of that. Over 5 seeds, on the validation windows,
for an earlier `staged+mcd` whose posterior stage fine-tuned a mean network
trained without dropout (full-width Gamma network and no re-fit, against the
current Gamma defaults):

| Gamma network | val NLL | val 95% coverage | test NLL |
| --- | --- | --- | --- |
| 16 channels, Adam, no re-fit | 5.433 ± 0.182 | 0.851 | 4.569 ± 0.053 |
| 4 channels, AdamW 0.1, re-fit | 4.939 ± 0.047 | 0.916 | 4.579 ± 0.021 |

With weight decay alone, at 16 channels, the validation NLL of the `staged`
Gamma variance went from 5.62 (no decay) to 5.57, 5.35 and 5.04 at decays
0.01, 0.1 and 1, and its log-variance roughness along a unit's lifetime
from 0.17 to 0.02; at decay 1 the band is almost constant and the test
coverage 0.99. The defaults narrow the network and keep the decay at 0.1.

What no variance network fixes is the mean's bias on long lifetimes: past a
true RUL of 200 the mean levels off around 150, and the validation coverage
there was 0.38 for that earlier `staged+mcd` (0.50 for a joint MC-Dropout
model trained with dropout from scratch). That error is absent from the training residuals the variance is
fitted to.
