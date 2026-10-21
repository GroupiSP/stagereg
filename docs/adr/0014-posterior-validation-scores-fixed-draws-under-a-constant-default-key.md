# Posterior validation scores fixed draws under a constant default key

The posterior stage's validation drove early stopping, checkpoint selection and HPO pruning, yet every epoch scored the posterior predictive under fresh keys taken from the run's RNG state: new draws from the posterior and new predictive samples for the CRPS. Re-drawing noise entered the very signal those decisions compare, and pinning the draws from outside meant wrapping the inference method's posterior, which pushed validation off the compiled path (ADR 0013). Taking the keys from the run's RNG state also made the training randomness depend on whether validation ran. Spec #138 (from #137).

`PosteriorStageOptions.validation_key`, a JAX PRNG key defaulting to `jax.random.key(0)`, is the only source of validation randomness. It is split once into a draw key, which selects the posterior's draws, and a metric key; batch `i` of the validation loader draws its predictive samples under the metric key with `i` folded in. Every epoch derives the same keys, so an unchanged posterior scored on the same batches in the same order gives identical metrics (common random numbers), and every trial and seed scores the same draws by default. Drawing a new key every epoch is removed, not kept as a mode. Validation never reads or advances the run's RNG state.

The key is configuration, not state, so a restored run scores the same draws without any change to checkpoints. Built-in posteriors stay on the compiled path; no wrapper is needed to pin the key. `predict` still takes its key explicitly.

## Considered Options

- **An optional key, with a fresh key per epoch as the default** (#137's proposal). Rejected: comparable epochs and trials would be opt-in, and the default would keep mixing re-drawing noise into early stopping and pruning.
- **Keep both modes behind a flag.** Rejected: two behaviours to document and test for one decision, and the per-epoch mode serves no decision the stage makes.
- **Derive the validation key from the run's RNG state once, at `prepare`.** Rejected: validation would still change the training randomness, and the key would vary with the seed without the user choosing it.
- **Check at runtime that the validation loader is deterministic.** Rejected: it would mean fingerprinting every batch. The requirement is documented instead.

## Consequences

Epochs and trials are compared on equal footing, at the cost of independent validation noise across seeds: runs with different seeds score the same draws, so their validation metrics share the same draw error. A user who wants independent validation noise per seed, or to check how sensitive the metrics are to the draws, passes their own key.

Identical metrics need a validation loader that yields the same batches in the same order every epoch. With a shuffling loader the draws stay fixed and only the predictive samples' keys follow the batch order.

Default posterior validation metrics differ from earlier versions, and turning validation on or off no longer changes what is trained.
