# The posterior stage predicts, and built-in posteriors are pytrees

Posterior predictive validation ran eagerly, one dispatched operation at a time, every validation batch of every epoch, and there was no library function to predict with a trained posterior: users built a `PosteriorPredictive` by hand from `sample_means` and the variance model, also eagerly. #125 (spec #120) asks that both be compiled. We made two public commitments to get there.

First, `MeanFieldGaussianPosterior`, `RetainedSamplesPosterior` and `MCDropoutPosterior` are registered as JAX pytrees. Their arrays (`means`/`stds`, `samples`/`draw_mask`, `parameters`) are children and their `mean_function` is static auxiliary data, compared by identity, as ADR 0012's rule for pytree fields requires. A built-in posterior therefore crosses a `jax.jit` boundary whole, and the stage draws its means, evaluates the aleatoric variance and forms the posterior predictive in one compiled call. The posterior is still consumed only through `sample_means` (or the unexported padded variant for pSGLD's fixed-capacity buffer), now traced, so ADR 0009's seam is unchanged.

Second, `PosteriorStage.predict(inputs, key)` returns the trained (or restored) posterior's `PosteriorPredictive`: the mixture over its draws under the key (`options.num_draws` of an unlimited posterior, all of a finite one), each with the frozen aleatoric variance, the mean of the variance model's prediction. It is the same computation validation runs, so prediction after training and validation cannot drift apart.

A posterior that is not a pytree, such as a user's own or a test double recording its calls, still works: the stage calls its `sample_means` from Python and compiles only the aleatoric variance. The `Posterior` protocol does not require pytree registration.

## Considered Options

- **Jit each built-in posterior's `sample_means` internally and leave the rest eager.** Rejected: the aleatoric variance, the mixture and its metrics would still run op by op, and every method would carry its own jit wrapper.
- **Jit the stage's whole predictive with the posterior as a static argument.** Rejected: posteriors hold arrays, which cannot be hashed as static data, and baking them into the compiled function as constants would recompile every epoch.
- **A free function `posterior_predictive(posterior, variance_model, inputs, key, num_draws)`.** Rejected: the caller would have to repeat what the stage already knows (the variance model in inference mode, the draw count, the padded-posterior handling), and the variance model, an NNX module, would have to be split on every call.
- **Make the padded-posterior protocol public.** Not needed: `predict` handles pSGLD's padded posterior internally, so the protocol stays unexported.

## Consequences

`mean_function` is static and compared by identity, so a compiled predictive is reused only while the posterior's mean function is the same object. The stage builds it once per `prepare` or `restore`, so each compiles once per input shape (and draw count). A hand-built posterior given a new closure on every call would recompile on every call.

A built-in posterior is data, not static, to Flax NNX: wrapping it in `nnx.static(...)` as an attribute of an NNX module now raises. Hold it as plain data, or hold `stage.predict` instead, as the XSin example does.

Epoch metrics' device-side fields (mean, variance, predictive samples, log-likelihood) are likewise computed by one compiled call per batch for any prediction that is a pytree, which every built-in distribution is (ADR 0012).
