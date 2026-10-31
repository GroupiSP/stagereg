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
predictive RUL distribution.

### `plots.py`

Builds the example's RUL curves. `plot_validation_rul_curves` takes the
standardized validation trajectories, the feature columns, a trained
composite model, the three units to draw, and the window length, and
draws one row of three columns, one unit per column. `run.py` passes the
lifetime-spanning units of the validation subset, so the columns are its
shortest-, median-, and longest-lifetime unit in that order; the module
itself decides how a RUL curve looks and never which units it looks at. Spanning the lifetime range that way lets a reader
tell whether the model's accuracy and its stated uncertainty behave
consistently across units that fail early and units that survive several
times as long.

Each column shows one unit's true linear RUL across its whole trajectory,
the model's predicted mean, and the 95% predictive interval
`loc ± 1.96 * scale` taken analytically from the composite Gaussian. The
columns have independent x and y limits, so the band stays readable in
the short-lived unit instead of being squashed by the long-lived one's
axis range; each is titled with its unit ID, the role it plays, and that
unit's lifetime in cycles, and one shared figure-level legend names the
three elements. The predicted mean and its band start at each unit's
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
training is.

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
