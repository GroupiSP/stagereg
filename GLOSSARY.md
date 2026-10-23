# stagereg

The shared vocabulary of this repository: the language of a training run and of recording one with an experiment tracker, which the library itself declares, and the language of the NASA CMAPSS FD001 remaining-useful-life example — obtaining the dataset, and describing the targets the models predict and the curves that visualize them.

One section per area. A term belongs in the training-and-tracking section when the library uses it, and in the CMAPSS section when only that example does.

## Training and tracking

**Stage**:
One named part of a run (mean, variance, posterior) that requires the lifecycle state the previous stage left, trains and validates, and hands on a finalized checkpoint. A run is a sequence of stages rather than one opaque fit.
_Avoid_: Phase, training loop

**Joint training**:
Fitting a mean and a variance together under one Gaussian negative log-likelihood, as classic mean-variance estimation does. Training in stages replaces it, because joint training couples the two fits: the variance can grow to excuse a poor mean, and the mean stalls where the variance is large.
_Avoid_: Combined training, MVE training, single-step training

**Training event**:
A structured observation emitted at a named point in a stage's lifecycle, carrying the split it concerns, any metrics measured at that point, and the live training state. Not a log line: it has no format, and what it means is decided by whoever consumes it.
_Avoid_: Log record, message

**Event sink**:
A consumer notified of every training event. Passive: it observes a run and cannot alter it, so a runner never waits on what a sink returns.
_Avoid_: Callback (implies the runner awaits a decision), logger, hook

**Experiment tracker**:
A destination that records parameters, metrics and artifacts for one run. Ignorant of stagereg's event model: it is told what to record, never when a run reaches a point of interest.
_Avoid_: Logger, writer, backend

**Run**:
One execution of a training script, and the unit a tracker groups its records under. One tracked run maps to one TensorBoard run directory.
_Avoid_: Experiment (a set of runs), trial

**Decision event**:
A training event that reports a judgement about a measurement already emitted (a new best model, an early stop) rather than a new measurement. It carries no metrics of its own, so recording a run never logs the same point twice.
_Avoid_: Metric event

**Decision**:
The measurement a decision event judged: the monitored metric's bare name and its value at that step. It is what the judgement was made on, not the judgement itself, which the event's name carries (`best_model`, `early_stop`).
_Avoid_: Payload, verdict (the early stopper's own result)

**Metric name**:
The one name a metric has, chosen by whoever computes it: bare `snake_case` such as `loss` or `rmse`. No other layer adds to it, so the same quantity has the same name on training and validation, in every stage.
_Avoid_: Prefixed name, `training_loss`, `validation_loss`

**Split**:
The partition of data a metric was measured on: `train` or `validation`. A run knows the split of every metric it reports, so a metric's split is never inferred from the name of the event or of the metric.
_Avoid_: Training (as a split), source, phase

**Outer iteration**:
One pass through a run's sequence of stages, counted from zero, for workflows that return to their first stage after the last (e.g. alternating mean and variance stages). Each stage counts its own epochs within an outer iteration. Not yet implemented: no run returns to its first stage, so neither the training state nor training events record an outer iteration.
_Avoid_: Cycle, round, outer epoch

**Stage segment**:
The leading part of a metric tag, naming the stage that produced the metric. Always present, including for a single-stage run, because two stages report against their own epoch counters and a tag without the stage lets one stage's curve overwrite the other's.

**Metric tag**:
The namespaced identity of a recorded metric, `stage/split/metric` (e.g. `mean/validation/loss`), with `/` as the only separator and no segment allowed to contain it. Built only where a run is observed, and the same string in a run's own metric history and in the experiment tracker.
_Avoid_: Metric key, prefixed metric, `mean_training_loss`

**Parameter path**:
The identity of a recorded hyperparameter: its key path through a nested parameter mapping, joined with the same `/` as a metric tag. Each leaf key is a bare `snake_case` name, and a split is a path segment of its own, never part of a leaf key (`data/train/samples`, not `data/train_samples`).
_Avoid_: Flat parameter name, hparam key

**Scoring unit**:
One scalar target together with everything predicted for it (mean, variance, samples, intervals, coordinate), the row an epoch metric scores. A target of any shape is flattened into as many scoring units as it has elements, and an epoch metric aggregates over all scoring units of a split, not over batches.
_Avoid_: Sample (clashes with predictive samples), data point, observation

**Best checkpoint**:
The checkpoint a stage saves whenever its early stopper reports an improvement. A snapshot of training in progress: its lifecycle state is still the one the stage started from, so restoring it alone does not leave the stage ready.
_Avoid_: Hand-off checkpoint

**Finalized checkpoint**:
The checkpoint a stage saves once it completes, with the stage's ready lifecycle state and marked complete: its best checkpoint when an early stopper is configured, otherwise its last epoch's. The only kind of checkpoint a stage restores from, so a restore always leaves the stage ready.
_Avoid_: Hand-off checkpoint, final checkpoint (with an early stopper it holds the best epoch, not the last)

**Checkpoint key**:
The name a checkpoint is stored under in a checkpoint store, `stage/best` by default, so that stages sharing one store never overwrite each other's checkpoints.
_Avoid_: Checkpoint name, checkpoint ID

**Aleatoric variance**:
The variance of the noise in the targets around their true mean, as estimated by the variance stage. It is a property of the data and does not shrink with more data.
_Avoid_: Noise variance (ambiguous with the noise of an inference method), data uncertainty

**Epistemic variance**:
The spread of the mean function across the draws of a posterior, i.e. the uncertainty about the mean that more data would reduce.
_Avoid_: Model uncertainty, parameter variance

**Posterior stage**:
The stage that follows the variance stage and replaces the single trained mean function with a posterior over mean functions, starting from the trained mean and holding the aleatoric variance fixed. Optional: a run that stops after the variance stage is complete without it.
_Avoid_: Bayesian stage, BNN stage, Step 3

**Inference method**:
The interchangeable strategy a posterior stage uses to turn a posterior problem into a posterior: Markov chain Monte Carlo, variational inference, ensembles, MC-Dropout. The posterior stage decides what is inferred; the inference method decides how.
_Avoid_: Sampler (excludes variational methods), inference engine, backend

**Inferred subnetwork**:
The part of the mean network's parameters a posterior stage infers, such as one convolutional layer; by default the whole network. It is selected from the posterior network's parameters only, never its random-number or batch-statistic variables, and the posterior problem and its prior cover it alone. MC-Dropout fine-tunes the inferred subnetwork with dropout active; the dropout masks are not part of the subnetwork.
_Avoid_: Bayesian layer, stochastic part, Bayesian subset

**Frozen remainder**:
The mean network's parameters outside the inferred subnetwork, held at their trained-mean values throughout the posterior stage.
_Avoid_: Backbone, fixed weights

**Posterior problem**:
Everything a posterior stage hands its inference method: the starting point taken from the trained mean, the likelihood of the targets under the fixed aleatoric variance, that aleatoric variance itself (so an implicit method can form predictive samples), the prior, the training data and its size. Every inference method honors the prior; methods without an explicit prior use it as their regularizer.
_Avoid_: Log-posterior (only one part of it), target density

**Posterior**:
What an inference method produces: an approximate posterior over mean functions, from which draws of the mean at given inputs can be taken. The members of an ensemble count as equally weighted draws.
_Avoid_: BNN, weight samples, variational distribution (one kind of posterior)

**Draw**:
One mean function taken from a posterior: the same function at every input and in every batch, so predictions compared across inputs (such as a RUL curve) come from one consistent function. A posterior with a finite set of draws (retained samples, ensemble members) always uses all of them. For MC-Dropout a draw is the fine-tuned network under one set of dropout masks, drawn from one key; it is one function only if no mask varies across the inputs, so its dropout layers share each mask across the batch axis.
_Avoid_: Sample (clashes with predictive samples), particle, predictive draw

**Predictive sample**:
One value of the target drawn from a predictive distribution, as used to estimate the CRPS. Not a draw: a draw is a mean function, a predictive sample is a target value.
_Avoid_: Draw, predictive draw

**Posterior predictive**:
The predictive distribution of a posterior stage: the equally weighted mixture, over the draws, of Gaussians centred on each draw's mean with the aleatoric variance. Exact for the draws taken, so it may be skewed or multimodal.
_Avoid_: BMA Gaussian, predictive Gaussian

**Closed-form CRPS**:
The CRPS of a predictive distribution computed from the distribution's own formula rather than estimated from predictive samples, so it carries no sampling noise. Defined only for distributions that offer such a formula, such as a Gaussian or the posterior predictive, and never replaced by a sample estimate where none exists.
_Avoid_: Exact CRPS (the sample CRPS is also exact, for the empirical distribution of its samples), mixture CRPS, analytic CRPS

**Sample CRPS**:
The CRPS of the empirical distribution of a set of predictive samples, an estimate of the predictive distribution's CRPS that varies with the samples drawn.
_Avoid_: Exact CRPS, closed-form CRPS

**Moment-matched predictive**:
The single Gaussian with the posterior predictive's mean and variance (the aleatoric plus the epistemic variance). A summary of the posterior predictive, not a substitute for it.
_Avoid_: Posterior predictive (that is the mixture)

## CMAPSS example

**Archive cache**:
The persistent, on-disk copy of the downloaded CMAPSS zip, kept across process runs so the dataset is fetched from NASA at most once rather than once per loader call or per run.
_Avoid_: Temp directory, download folder

**Cache directory**:
The directory holding the archive cache, resolved from `STAGEREG_CMAPSS_CACHE_DIR` if set, otherwise a stdlib-computed per-user cache location (honoring `XDG_CACHE_HOME`, else `~/.cache/stagereg-cmapss`).

**Archive override**:
A user-supplied path to a manually obtained copy of the CMAPSS zip, provided via the `STAGEREG_CMAPSS_ARCHIVE` environment variable or an optional loader parameter. When present, it seeds the archive cache and no download is attempted.
_Avoid_: Manual download, local copy

**Fetching**:
Obtaining the CMAPSS archive and placing it in the archive cache — via network download (with retries) or via an archive override. Distinct from *loading*, which reads sensor columns out of an archive already in the cache.
_Avoid_: Downloading (too narrow — excludes the override path)

**Loading**:
Reading a specific split (train, test, or ground-truth RUL) out of the archive cache into a DataFrame/array. Assumes fetching has already happened, and triggers it transparently if it hasn't.

**Lifetime**:
A run-to-failure unit's total number of cycles, i.e. its remaining useful life at cycle 0 — `max(time_cycles)`. Defined only for units from the train split, whose trajectories run to failure; a truncated test trajectory has no observable lifetime, only a ground-truth remaining life at its last observed cycle.
_Avoid_: Duration, age, total RUL

**Linear RUL**:
The unclipped remaining-useful-life target `max(time_cycles) - t`, decreasing by exactly one per cycle. This is what the example's models are trained on, as opposed to the widely used piecewise-constant convention that caps RUL at a constant (typically 125 or 130) early in life. Naming it explicitly keeps a reader from assuming the capped convention and misreading the targets and curves.
_Avoid_: RUL (ambiguous between the two conventions), piecewise RUL

**Full window**:
A window whose every row is an observed cycle of the unit, as opposed to a *padded window*, whose leading rows repeat the unit's first observed reading to make a short trajectory reach the window length. A padded window's remaining-useful-life target is still computed from real cycles, so it is a usable training sample; but its sensor history is fabricated, so a prediction made from it has no support in the data.
_Avoid_: Complete window, valid window, real window

**RUL curve**:
For a single unit, its linear RUL and the model's predictive RUL plotted against time cycles. The predicted curve begins at the unit's first full window, since no prediction exists before `window_length` cycles of history have accumulated.
_Avoid_: RUL plot, degradation curve

**Lifetime-spanning units**:
The shortest-, lower-median-, and longest-lifetime units of a set of run-to-failure trajectories, drawn side by side so that a reader can tell whether the model's accuracy and its stated uncertainty behave consistently across units that fail early and units that survive several times as long. Ordered ascending by lifetime with the unit ID as tie-break, and the median is the lower of the two central entries, so the same data always yields the same three units.
_Avoid_: Representative units, extreme units, best/worst units (none of the three is chosen for how well the model does on it)

**Predictive interval**:
The `loc ± 1.96 * scale` band of the composite Gaussian predictive distribution, covering 95% of its mass and left unclipped at zero, so that a band extending below zero remains visible as evidence of the Gaussian assumption breaking down near end of life.
_Avoid_: Confidence interval (this is an interval over a predicted value, not over a parameter estimate)
