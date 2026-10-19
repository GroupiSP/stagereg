# The posterior stage predicts, in one compiled call

Posterior predictive validation ran eagerly, one dispatched operation at a time, every validation batch of every epoch, and there was no library function to predict with a trained posterior: users built a `PosteriorPredictive` by hand from `sample_means` and the variance model, also eagerly. #125 (spec #120) asks that both be compiled.

`PosteriorStage.predict(inputs, key)` returns the trained (or restored) posterior's `PosteriorPredictive`: the mixture over its draws under the key (`options.num_draws` of an unlimited posterior, all of a finite one), each with the frozen aleatoric variance, the mean of the variance model's prediction. It is the same computation validation runs, so prediction after training and validation cannot drift apart.

For a built-in posterior (`MeanFieldGaussianPosterior`, `RetainedSamplesPosterior`, `MCDropoutPosterior`) the stage draws the means, evaluates the aleatoric variance and forms the predictive in one compiled call. The posterior's arrays are traced arguments, and its class and `mean_function` static ones, compared by identity: each built-in posterior takes itself apart into an unexported `TracedPosterior` pytree, and the compiled function rebuilds the posterior from its arrays and calls its own `sample_means`, so ADR 0009's seam is unchanged. pSGLD's posterior hands over its whole sample buffer with a mask of the retained slots, so its shapes stay fixed as samples accumulate; its public `samples` still hold the retained samples alone.

The posteriors themselves are not pytrees. They stay plain data, as before #125, so an NNX module can still hold one as `nnx.static(...)`, a pattern the examples documented: NNX treats a pytree holding arrays as data, and `nnx.static` refuses it.

A posterior that is not built in, such as a user's own or a test double recording its calls, still works: the stage calls its `sample_means` from Python and compiles only the aleatoric variance. The `Posterior` protocol requires nothing more.

## Considered Options

- **Register the built-in posteriors as JAX pytrees, `mean_function` static.** Implemented first, then rejected in review: it broke `nnx.static(posterior)` in NNX modules, which the spec's no-breaking-change rule protects.
- **Jit each built-in posterior's `sample_means` internally and leave the rest eager.** Rejected: the aleatoric variance, the mixture and its metrics would still run op by op, and every method would carry its own jit wrapper.
- **Jit the stage's whole predictive with the posterior as a static argument.** Rejected: posteriors hold arrays, which cannot be hashed as static data, and baking them into the compiled function as constants would recompile every epoch.
- **A free function `posterior_predictive(posterior, variance_model, inputs, key, num_draws)`.** Rejected: the caller would have to repeat what the stage already knows (the variance model in inference mode, the draw count, pSGLD's buffer), and the variance model, an NNX module, would have to be split on every call.
- **Make the take-apart protocol public.** Not needed: the stage handles built-in posteriors internally, and a user's posterior can compile its own `sample_means`.

## Consequences

`mean_function` is static and compared by identity, so a compiled predictive is reused only while the posterior's mean function is the same object. The stage builds it once per `prepare` or `restore`, so each compiles once per input shape (and draw count).

Epoch metrics' device-side fields (mean, variance, predictive samples, log-likelihood) are likewise computed by one compiled call per batch for any prediction that is a pytree, which every built-in distribution is (ADR 0012).
