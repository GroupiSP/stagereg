# The Gamma head predicts at unit residual scale

`GammaHead` divides its rate by a `ResidualScale`, a non-trainable `nnx.Variable` that starts at one. `GammaVarianceStage.prepare` sets every `ResidualScale` in its model to the mean of the materialized training squared residuals. The head's linear layer therefore predicts the rate as if the squared residuals had unit mean, and `variance_model(x).mean()` stays in the targets' units, so the posterior stage, the examples and user code read the variance as before. Because the scale is a variable of the model, checkpoints, `restore` and `nnx.clone` carry it. Optimizers built with `create_optimizer` update only `nnx.Param`, so training leaves it alone.

The cause was the clustered-data example's staged extrapolation σ_ale, which reached 331 at seed 0 and 699 at seed 2. Its squared residuals are about 1e-3, and with a concentration near 0.3 that forces a rate near 100. A softplus rate needs a raw logit of about +100 to reach that, so the head's weights grow large. Past the data, the tanh trunk's features drift a little, the large weights turn that drift into a logit swing of tens, and once the logit crosses zero the variance, concentration / rate, grows like e^{-logit} until the rate hits its `eps` floor. With the targets at unit scale the logits stay O(1), and the extrapolation σ_ale at both seeds dropped to about 0.13 with the in-distribution fit unchanged.

## Considered Options

- **Normalize the targets inside the stage only, through its loss.** Rejected: the stage trains whatever model it is given, and that model's predictions would then be in normalized units. Every consumer (`PosteriorStage`, the examples, user code) would need the scale to read a variance, and nothing would carry it with a checkpoint.
- **Parametrize the rate in log space (`rate = exp(raw)`).** Rejected: it shrinks the logit to `log(100) ≈ 4.6`, but the variance still grows exponentially as the logit drifts, and the logit's offset still depends on the data's units.
- **Scale the targets in the clustered-data example only.** Rejected: any data whose residuals are far from unit scale hits the same blowup. The example is only where it was seen first.

## Consequences

- Scaling the targets by `k` scales the trained variance by `k**2`, inside and outside the data. `tests/jax/test_supervised_staged.py` tests this as a property.
- The `eps` floor on the rate now sits in normalized units, so the largest variance a head can predict is about concentration × scale / `eps` instead of concentration / `eps`.
- A model without a `ResidualScale`, such as a custom Gamma head, trains exactly as before. To get the same behavior it must hold a `ResidualScale` and divide its rate by it.
- The scale is one unweighted scalar over every target element. A multi-output head whose outputs have very different residual sizes still leaves the smaller ones far from unit scale.
- A model's state now holds one more variable, so checkpoints of a `GammaHead` saved before this change do not restore into the new head.
