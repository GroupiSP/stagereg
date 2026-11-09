# The clustered-data example trains on standardized targets under L2

The clustered-data example follows [VeBNN](https://github.com/bessagroup/VeBNN)'s data and regularization choices, short of its outer iterations. Its loader serves the training targets standardized by their mean and population standard deviation, and every configuration, joint or staged, maps its predictive means and variances back to the targets' units before it is scored, so the comparison table and the figures stay in those units. The mean and variance stages optimize their mean loss plus the posterior stage's isotropic Gaussian prior: `optax.add_decayed_weights(prior_precision / train_size)` chained before Adam, the MAP estimate under the prior the posterior stage then uses. The prior's `prior_precision / 2 * ||w||**2` goes with a loss summed over the training set, and the stages average theirs over a batch, hence the `1 / train_size`. The posterior stage and the joint posterior already carried that prior, so their objectives do not change.

Standardization does not replace ADR 0016's residual scale; the example keeps both. Standardizing makes the targets unit scale, but the Gamma head needs the squared residuals at unit scale, and after standardization their mean is about `1 - R²`. On this smooth, low-noise curve (noise σ 0.02, target σ 0.48) that is still about 4e-3, so the head without the residual scale still blew up past the data: the staged extrapolation σ_ale exceeded 1 at 8 of 10 seeds and reached 193, 227 and 283 at seeds 2, 7 and 5. With the residual scale it stays between 0.11 and 0.37 at all 10 seeds, against an in-distribution σ_ale of 0.07 to 0.14; before this change, without standardization or L2, seeds 0 and 8 reached 1.25 and 2.33.

## Considered Options

- **Drop the residual scale and rely on standardization, as VeBNN does.** Rejected for the measurements above. Standardization puts the residuals at unit scale only when the noise is a large share of the targets' spread; here it is not, and on many regression datasets it is not.
- **Standardize the squared residuals in the example's variance stage.** Rejected: it is the residual scale reimplemented in user code, which ADR 0016 already rejected.
- **No L2 on the supervised stages.** Rejected: VeBNN regularizes both its mean warm-up and its variance network, and the staged model's supervised stages would then fit a different objective from the posterior stage they hand over to.
- **A stronger L2 to tame the variance network.** Rejected: at `prior_precision = 100` the variance network collapsed to a constant σ of 0.59 everywhere, far above the in-distribution noise.

## Consequences

- Every configuration trains in standardized units. A new configuration must map its posterior predictive back with the example's `scoring.in_target_units` before scoring, or its scores are in the wrong units.
- The `prior_precision` flag now sets the supervised stages' L2 as well as the posterior prior, so changing it changes the trained mean and variance networks too.
- The networks now predict standardized targets, so the same `prior_precision` is a different prior over the function than before, for the joint baseline as well as the staged model.
- VeBNN's outer iterations, and the early stopping and best-iteration selection across them, are not reproduced: the staged model runs one outer pass.
