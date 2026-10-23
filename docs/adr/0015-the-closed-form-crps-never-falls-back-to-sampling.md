# The closed-form CRPS never falls back to sampling

The CRPS of a predictive distribution that has a closed form, such as a Gaussian or the posterior predictive (a mixture of Gaussians sharing the aleatoric variance, Grimit et al. 2006), is computed from that formula. It is not estimated from predictive samples, so the score carries no Monte Carlo noise and none of the energy form's plug-in bias toward narrow predictives. `ClosedFormContinuousRankedProbabilityScore` reports only this value. A distribution offers it by implementing a separate, optional core protocol (a `crps` method over targets or reference samples). `PredictiveDistribution` does not require it. A distribution without the method makes the metric raise an error naming its class and pointing to `SampleContinuousRankedProbabilityScore`; the metric never substitutes a sample estimate. The posterior stage's default `crps` is this score. Spec #142.

## Considered Options

- **A required `crps` method on `PredictiveDistribution`, falling back to the sample CRPS for distributions without a closed form.** Rejected for two reasons. First, the fallback would report a noisy, biased estimate under the closed-form metric's name for some models, so two runs reporting the same metric could not be compared, which is the problem this metric exists to remove. Second, a suite's requirements, including the predictive sample count, are fixed when the suite is built, before any distribution is seen. The fallback would therefore either always demand a sample count or fail at the first validation epoch.
- **A required method with no fallback.** Rejected: every user distribution that conforms to the protocol today would stop conforming until it gains a closed form it may not have. An optional protocol can still be made required later; the reverse would be a breaking change.

## Consequences

`Gamma` and user distributions without a `crps` method cannot be scored by this metric until they implement it. Today only `GaussianPredictor` and `PosteriorPredictivePredictor` exist, so a generic predictor would also be needed for them to reach it.
