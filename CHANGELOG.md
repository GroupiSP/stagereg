# Changelog

All notable changes to `stagereg` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.4.0] - 2026-10-21

### Breaking changes

- Posterior validation scores every epoch under one fixed key,
  `PosteriorStageOptions.validation_key`, so every epoch, trial and seed
  scores the same posterior draws and the same predictive samples. Drawing a
  new key every epoch is removed, so default posterior validation metrics
  differ from earlier versions. Validation no longer reads or advances the
  run's RNG state, so turning it on or off leaves training randomness
  unchanged (ADR 0014, #138).

### Added

- `PosteriorStageOptions.validation_key`: the typed PRNG key posterior
  validation is scored under, defaulting to `jax.random.key(0)` (#138).

## [0.3.0] - 2026-10-19

Performance release for the Tier 1 reference workloads (#120): training,
validation and posterior prediction compile once and stay on the device.

### Breaking changes

- `run_supervised` binds the model and optimizer to its compiled training step
  once. Updates to their variables between steps are still seen, but the
  module structure and plain Python attributes are fixed when the run starts:
  an attribute set inside the step is not written back. `make_train_step` is
  unchanged (#130).
- Training epoch metrics run the model's forward pass compiled, so the model's
  output must be a JAX pytree; a custom distribution type must be registered
  with `jax.tree_util` (#129).

### Added

- `PosteriorStage.predict(inputs, key)`: the posterior predictive of a trained
  or restored posterior stage. A built-in posterior is predicted in one
  compiled call, compiled once per input shape (ADR 0013, #125).
- `PosteriorPredictive.draw_mask`: an optional boolean mask over the draw
  slots; every statistic is then the equally weighted mixture over the
  included draws alone. A concrete mask that includes no draw is refused.
- `PosteriorProblem.epochs`: the number of epochs the stage trains for, set by
  `PosteriorStage`, so a method can plan its work. Optional for hand-built
  problems.
- `PreconditionedSGLD.capacity`: the number of slots first reserved for
  retained samples. By default, `None`, the buffer is sized from the planned
  run when the first sample is retained and doubles if the run retains more
  (#123).
- The distributions `Gaussian`, `Gamma`, `MomentMatchedPredictive` and
  `PosteriorPredictive` are JAX pytrees, so they pass into and out of jitted
  functions. `Batch` is not (ADR 0012, #124).

### Changed

- Faster, with no change in results:
  - `HeldOutValidation` compiles its evaluation step once per strategy (#122).
  - Step losses, batch metrics and diagnostics stay on the device within an
    epoch and reach the host once per epoch (#127).
  - `PointContinuousRankedProbabilityScore` and
    `SampleContinuousRankedProbabilityScore` are vectorised across scoring
    units (#128).
  - `PosteriorPredictive.quantile` compiles once per mixture shape (#126).
  - `PreconditionedSGLD` keeps its retained samples in a preallocated buffer
    instead of restacking them every epoch, so posterior validation compiles
    nothing after the first epoch (#94, #123).
  - The XSin posterior example trains its supervised stages once for all
    posterior methods (#132).

## [0.2.0] - 2026-10-15

### Breaking changes

- `TrainingState.active_stage` is now a dataclass field. The `TrainingState.stage`
  field, which the `active_stage` property used to wrap, is removed; pass and read
  `active_stage` instead.
- `Checkpoint.rng_state` is removed. A checkpoint's random key lives only in
  `Checkpoint.state.rng_state`, and restores read it from there.
- `TrainingState.outer_iteration` and `TrainingState.data_fingerprint` are removed.
  Nothing in the library populated them.
- `TrainingEvent.iteration` is removed. It always reported `0`, as no run has outer
  iterations yet.
- The posterior seam is keyed: `PosteriorProblem.mean_function` takes
  `(parameters, inputs, key)` and `PosteriorProblem.log_likelihood` takes
  `(parameters, batch, key)`. `PosteriorStage` reseeds the posterior network's
  RNG streams from the key and turns on the `nnx.Dropout` layers of its copy of
  the network, so a network with `nnx.Dropout` draws its masks from it even if it
  was left in eval mode; a deterministic network ignores it. A custom inference method passes a key
  to both, and a hand-built `PosteriorProblem` accepts one. `BayesByBackprop` and
  `PreconditionedSGLD` give the same results as before for a deterministic
  network (#92).
- `MeanFieldGaussianPosterior.mean_function` and
  `RetainedSamplesPosterior.mean_function` take `(parameters, inputs, key)` as
  well; a hand-built posterior passes a three-argument mean function (#92).
- `RetainedSamplesPosterior.sample_means` now uses its key: draw `s` is
  evaluated with `jax.random.fold_in(key, s)`. A deterministic posterior
  network still ignores it (#92).
- `InferenceMethod` requires a `diagnostics` property; a custom method
  declares `()` to keep recording only its loss (#106).
- `PosteriorProblem.initial_parameters`, `mean_function` and `prior` cover the
  inferred subnetwork only, so a custom `Prior` sees that subtree alone (#105).
- `BayesByBackprop.load_posterior` and, once initialized,
  `PreconditionedSGLD.load_posterior` refuse a posterior whose tree differs
  from the inferred subnetwork's (#105).

### Added

- `MCDropout`, an MC-Dropout inference method, and its unlimited posterior
  `MCDropoutPosterior`. It fine-tunes the inferred subnetwork of a posterior
  network with dropout active by maximizing the keyed log-likelihood plus the
  log-prior, draws one dropout mask set per key, and supports early stopping and
  checkpoints. Its dropout layers must broadcast their masks over the batch axis
  (`broadcast_dims=(0,)`) so that each draw is one mean function (#92).
- `PosteriorProblem.aleatoric_variance`: the fixed, stop-gradient aleatoric
  variance a `PosteriorStage` uses for its likelihood, so an implicit method
  can turn draws into predictive samples. Optional for hand-built problems (#106).
- `InferenceMethod.diagnostics`: the keys of `update`'s output that
  `PosteriorStage` records as `posterior/train/<key>` epoch means. Undeclared
  keys are ignored; a key colliding with a recorded tag is refused at `prepare`.
  `BayesByBackprop` and `PreconditionedSGLD` declare `()` (#106).
- `PosteriorStage.inferred_subnetwork`, an NNX filter selecting the inferred
  subnetwork; the frozen remainder keeps its trained-mean values. The default,
  `nnx.Param`, infers the whole network as in 0.1.0 (#105).
- `stagereg.core.expected_sample_crps(predictive_samples, reference_samples)`: the
  exact, grid-free expected CRPS of each scoring unit's empirical predictive under
  its reference samples, batched over units.
- `stagereg.core.CachedCheckpointStore(persistent, persist_when=finalized_only)`: a
  checkpoint store that caches every save in memory and forwards to `persistent`
  only the saves its predicate selects, reading the cache first and falling back to
  `persistent`. Ready-made predicates `finalized_only`, `every_nth_save(n)` and
  `always`, typed as `PersistencePredicate`.
- `stagereg.core.STAGE_COMPLETE_METADATA_KEY` and `stagereg.core.is_finalized`: the
  checkpoint-metadata marker of a finalized checkpoint, now public. The stages write
  and check it as before.

### Changed

- `MeanStage` and `GammaVarianceStage` with a checkpoint store but no early stopper
  now save their last epoch as their finalized checkpoint under the stage's
  checkpoint key, so the stage can be restored in a later process. Previously they
  saved nothing. With an early stopper the best epoch is still finalized
  (ADR 0011).

## [0.1.0]

First release under the `stagereg` name.
