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
Next to them sit the MC-Dropout helpers `compare.py` uses:
`train_joint_mcd_model` and the predictive wrapper `McDropoutPredictive`
(built by `build_mc_dropout_predictive`), the staged posterior-stage runner
`train_staged_mcd_posterior`, which returns a `PosteriorStagePredictive`,
and `refit_gamma_to_posterior_mean`, which re-fits the Gamma network to that
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
| `joint+mcd` | joint MC-Dropout | the `joint` network, fine-tuned with dropout active on the Gaussian NLL under the posterior stage's prior; early-stopped on its MC mixture's validation NLL | posterior predictive, each draw with its own aleatoric variance |
| `staged` | staged model | as without the flag | Gaussian |
| `staged+mcd` | staged MC-Dropout | posterior stage (`MCDropout`) fine-tuning the `staged` mean network with dropout active; early-stopped on the validation mixture NLL; then the Gamma network re-fitted to the posterior predictive's mean | posterior predictive, all draws sharing the re-fitted aleatoric variance |

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

`--seeds` and `--plot-path` work the same either way. With the flag a seed
takes about a minute on a laptop CPU, against about 20 seconds without.

##### How the MC-Dropout models are trained

Both MC-Dropout models fine-tune a network first trained with dropout off,
so the joint and staged paths differ only in single against separate
training.

`joint+mcd` reuses the `joint` column's trained network from the same seed:
a Gaussian-head CNN at `dropout_rate` is warm-started from it and fine-tuned
on the Gaussian NLL with dropout active, at `mcd_learning_rate`, under the
posterior stage's isotropic Gaussian prior at `prior_precision`. The prior
enters as the L2 penalty's gradient `prior_precision / n_train * w`, added
before Adam: the NLL is averaged over a batch where the posterior stage sums
the log-likelihood over the training set. It is validated after every epoch
on the NLL of its MC mixture and stops early after `early_stopping_patience`
epochs without improvement, with `joint_mcd_epochs` (the same 20 as
`posterior_epochs`) as the maximum; the best epoch's parameters are kept. Its
prediction evaluates the network under `num_draws` dropout-mask sets, each
one applied to the whole batch, and keeps each draw's own mean and aleatoric
variance: a posterior predictive scored on its exact mixture. The masks come
from a fixed key derived from the seed, so validation, test scoring and the
figure use the same draws.

`staged+mcd` reuses the `staged` column's trained mean and Gamma variance
networks from the same seed, without retraining, so the difference between
`staged` and `staged+mcd` is the posterior stage and the variance re-fit
alone. A `PosteriorStage`
with the `MCDropout` inference method fine-tunes the whole mean network,
rebuilt with dropout active at `dropout_rate`, at `mcd_learning_rate` under
an isotropic Gaussian prior at `prior_precision`; the Gamma stage's
aleatoric variance is held fixed. It runs at most `posterior_epochs` epochs,
stops early on the validation mixture NLL after `early_stopping_patience`
epochs without improvement, and keeps the best epoch. The fine-tune moves the
mean, so the Gamma variance it held fixed was fitted to the residuals of a
mean that no longer exists: `refit_gamma_to_posterior_mean` then trains a
fresh Gamma network, from the same initialization and with the same width
and weight decay, on the squared residuals of the posterior predictive's
mean. The prediction is the stage's posterior predictive with `num_draws`
draws under the same fixed, seed-derived key it was validated under, every
draw sharing the re-fitted aleatoric variance.

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

##### Defaults and how they were chosen

The MC-Dropout settings live in `CmapssConfig` (`run.py`):

| Field | Default | Source |
| --- | --- | --- |
| `dropout_rate` | 0.1 | fixed by the spec (#173) |
| `joint_mcd_epochs` | 20 | the same fine-tune budget as `posterior_epochs` |
| `mcd_learning_rate` | 1e-3 | tuned, below; shared by both MC-Dropout models |
| `posterior_epochs` | 20 | tuned, below |
| `prior_precision` | 1.0 | tuned, below |
| `num_draws` | 100 | fixed by the spec |
| `early_stopping_patience` | 5 | fixed by the spec |
| `gamma_hidden_channels` | 4 | chosen, [below](#the-variance-network) |
| `gamma_weight_decay` | 0.1 | chosen, [below](#the-variance-network) |

The three tuned values come from a grid on real FD001, seeds 0 and 1, with
every other setting at its default. The grid predates the variance-network
changes and the matched warm starts: `joint+mcd` then trained from a random
initialization for up to 60 epochs with no prior, and `staged+mcd` held a
full-width, unregularized Gamma variance with no re-fit.

- `joint+mcd`: `mcd_learning_rate` in {1e-4, 3e-4, 1e-3, 3e-3}.
- `staged+mcd`: `mcd_learning_rate` in {1e-4, 3e-4, 1e-3, 3e-3} ×
  `prior_precision` in {1, 100, 10⁴}, with `posterior_epochs` raised to 40
  so the cap never cut a run short. Each grid point fine-tunes the same
  seed's `staged` networks.

Values were selected on the **validation mixture NLL** at the best epoch,
never on the test split. Means over the two seeds; "stop (best)" is the
epoch count run and the best epoch (0-based) per seed:

| Model | `mcd_learning_rate` | `prior_precision` | stop (best) | val NLL | test RMSE | test coverage | test CRPS | test NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `joint+mcd` | 1e-4 | | 60 (56), 60 (59) | 5.061 | 32.39 | 0.990 | 17.91 | 4.824 |
| `joint+mcd` | 3e-4 | | 60 (56), 60 (57) | 4.947 | 30.43 | 0.980 | 16.70 | 4.747 |
| `joint+mcd` | **1e-3** | | 60 (56), 60 (56) | **4.834** | 29.50 | 0.965 | 15.54 | 4.604 |
| `joint+mcd` | 3e-3 | | 50 (43), 46 (39) | 4.864 | 27.13 | 0.970 | 14.32 | 4.529 |
| `staged+mcd` | 1e-4 | 1 | 10 (3), 7 (0) | 5.326 | 23.91 | 0.950 | 13.29 | 4.546 |
| `staged+mcd` | 1e-4 | 100 | 7 (0), 7 (0) | 5.339 | 23.83 | 0.950 | 13.29 | 4.549 |
| `staged+mcd` | 1e-4 | 10⁴ | 7 (0), 7 (0) | 5.522 | 20.79 | 0.970 | 11.92 | 4.476 |
| `staged+mcd` | 3e-4 | 1 | 10 (3), 9 (2) | 5.348 | 23.57 | 0.955 | 13.11 | 4.533 |
| `staged+mcd` | 3e-4 | 100 | 7 (0), 7 (0) | 5.399 | 22.95 | 0.955 | 12.98 | 4.536 |
| `staged+mcd` | 3e-4 | 10⁴ | 7 (0), 7 (0) | 5.863 | 20.83 | 0.945 | 12.01 | 4.498 |
| `staged+mcd` | **1e-3** | **1** | 7 (0), 7 (0) | **5.413** | 23.49 | 0.950 | 13.31 | 4.559 |
| `staged+mcd` | 1e-3 | 100 | 7 (0), 7 (0) | 5.469 | 22.90 | 0.935 | 13.01 | 4.544 |
| `staged+mcd` | 1e-3 | 10⁴ | 7 (0), 7 (0) | 7.033 | 29.12 | 0.860 | 16.62 | 4.937 |
| `staged+mcd` | 3e-3 | 1 | 7 (0), 8 (1) | 5.445 | 25.39 | 0.920 | 14.11 | 4.618 |
| `staged+mcd` | 3e-3 | 100 | 7 (0), 8 (1) | 5.604 | 24.06 | 0.900 | 13.53 | 4.620 |
| `staged+mcd` | 3e-3 | 10⁴ | 7 (0), 7 (0) | 10.009 | 50.94 | 0.610 | 30.81 | 6.638 |

For reference, the `staged` model without a posterior stage scores a
validation NLL of 5.58 (5.52, 5.63) and a test NLL of 4.668.

How the values were chosen:

- **`mcd_learning_rate = 1e-3`.** It is shared by both models, which pull
  opposite ways: `joint+mcd` reaches its best validation NLL at 1e-3,
  while `staged+mcd` does slightly better lower (5.33 at 1e-4). 1e-3
  minimizes the two models' summed validation NLL (10.25, against 10.29 at
  3e-4, 10.31 at 3e-3 and 10.39 at 1e-4).
- **`prior_precision = 1.0`.** The best validation NLL at every learning
  rate. The likelihood is scaled to the full training set, so a precision
  of 1 or 100 barely regularizes; 10⁴ pulls the network towards zero and
  degrades validation NLL at every learning rate, and destroys the model at
  3e-3. It does improve the test scores at the lowest learning rates, but
  the test split was not used for selection.
- **`posterior_epochs = 20`.** Under every setting the posterior stage
  reached its best validation NLL within its first four epochs and stopped
  after at most ten, so 20 is a cap that early stopping always reaches
  first.
- **Spec-fixed values.** None is broken, so none was changed. In the grid,
  at `mcd_learning_rate = 1e-3`, `joint+mcd` used its whole 60-epoch budget
  from a random initialization (best epoch 56 of 60). It now fine-tunes the
  `joint` network instead, like the posterior stage fine-tunes the `staged`
  mean, so it shares the posterior stage's 20-epoch cap.

Reproduce the grid by calling `train_joint_mcd_model` and
`train_staged_mcd_posterior` (`run.py`) with `dataclasses.replace`-d
configurations and an event sink that records the `validation_end`
events. The current defaults give, with `--seeds 5 --posterior-stage`:

```text
FD001 test metrics over 5 seeds (mean ± std):
                              joint          joint+mcd             staged         staged+mcd
RMSE              61.0066 ± 19.9944   31.1243 ± 1.5233   25.3996 ± 1.2857   27.1698 ± 1.2162
95% coverage        0.9740 ± 0.0219    0.9900 ± 0.0000    0.9880 ± 0.0084    0.9860 ± 0.0055
closed-form CRPS   33.2752 ± 9.4750   17.0551 ± 0.7183   14.2778 ± 0.3020   14.5206 ± 0.5125
NLL                 5.4207 ± 0.2507    4.7619 ± 0.0295    4.6213 ± 0.0223    4.5785 ± 0.0233
```

Both staged models beat `joint+mcd` on every metric, and `staged+mcd` has
the best NLL. `joint+mcd` is far better than `joint`, whose
unscaled-target training is unstable across seeds. Every model over-covers
the test split, whose one window per unit sits late in life, where the
variance is smallest; the variance is calibrated over whole lifetimes
instead, see below.

#### The variance network

The test split holds only each unit's last window, so it says little about
early-life uncertainty. The held-out validation units span whole lifetimes,
and on them a Gamma network as wide as the mean network (16 channels,
plain Adam) was overconfident: its 95% coverage was 0.96 on the training
windows it was fitted on but 0.82 on the validation windows, whose mean
squared residual is 1.31 times the training one. In-sample residuals
understate the held-out error, and a flexible Gamma network fits them
closely, giving a spiky, overconfident variance.

Shrinking it fixes most of that. Over 5 seeds, on the validation windows
(`staged+mcd` with the full-width Gamma network and no re-fit, against the
defaults):

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
there is 0.38 for `staged+mcd` (0.50 for a from-scratch joint MC-Dropout
model). That error is absent from the training residuals the variance is
fitted to.
