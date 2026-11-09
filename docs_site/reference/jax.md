# `stagereg.jax`

Every symbol exported from `stagereg.jax`, the JAX/Flax NNX training backend, grouped by the
module that defines it. Using it needs the `jax` extra.

## `stagereg.jax.bayes_by_backprop`

::: stagereg.jax.BayesByBackprop

::: stagereg.jax.MeanFieldGaussianPosterior

## `stagereg.jax.distributions`

::: stagereg.jax.Gaussian

::: stagereg.jax.Gamma

::: stagereg.jax.PosteriorPredictive

::: stagereg.jax.MomentMatchedPredictive

::: stagereg.jax.GaussianHead

::: stagereg.jax.GammaHead

::: stagereg.jax.ResidualScale

## `stagereg.jax.evaluation`

::: stagereg.jax.SupervisedLoss

::: stagereg.jax.make_evaluation_step

::: stagereg.jax.evaluate_loader

## `stagereg.jax.losses`

::: stagereg.jax.make_supervised_loss

## `stagereg.jax.mc_dropout`

::: stagereg.jax.MCDropout

::: stagereg.jax.MCDropoutPosterior

## `stagereg.jax.metrics`

::: stagereg.jax.BatchMetric

::: stagereg.jax.BatchMetricSpec

::: stagereg.jax.PredictionRequirements

::: stagereg.jax.CoordinateExtractor

::: stagereg.jax.ReferenceSamplesExtractor

::: stagereg.jax.Predictor

::: stagereg.jax.GaussianPredictor

::: stagereg.jax.PosteriorPredictivePredictor

::: stagereg.jax.MetricSuite

::: stagereg.jax.merge_epoch_prediction_data

## `stagereg.jax.posterior`

::: stagereg.jax.PosteriorProblem

::: stagereg.jax.Prior

::: stagereg.jax.IsotropicGaussianPrior

::: stagereg.jax.InferenceMethod

::: stagereg.jax.Posterior

## `stagereg.jax.posterior_stage`

::: stagereg.jax.PosteriorStageOptions

::: stagereg.jax.PosteriorStage

## `stagereg.jax.psgld`

::: stagereg.jax.PreconditionedSGLD

::: stagereg.jax.RetainedSamplesPosterior

## `stagereg.jax.rng`

::: stagereg.jax.split_key

## `stagereg.jax.state`

::: stagereg.jax.NnxSnapshot

::: stagereg.jax.create_optimizer

::: stagereg.jax.initialize_training_state

::: stagereg.jax.freeze_training_state

::: stagereg.jax.snapshot

::: stagereg.jax.restore_checkpoint

## `stagereg.jax.supervised`

::: stagereg.jax.make_train_step

::: stagereg.jax.run_supervised

## `stagereg.jax.supervised_staged`

::: stagereg.jax.materialize_residual_loader

::: stagereg.jax.SupervisedStageOptions

::: stagereg.jax.MeanStage

::: stagereg.jax.GammaVarianceStage

## `stagereg.jax.validation`

::: stagereg.jax.HeldOutValidation
