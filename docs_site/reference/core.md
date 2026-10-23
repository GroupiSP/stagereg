# `stagereg.core`

Every symbol exported from `stagereg.core`, the backend-neutral contracts and utilities,
grouped by the module that defines it.

## `stagereg.core.checkpoints`

::: stagereg.core.Checkpoint

::: stagereg.core.CheckpointStore

::: stagereg.core.InMemoryCheckpointStore

::: stagereg.core.CachedCheckpointStore

::: stagereg.core.PersistencePredicate

::: stagereg.core.finalized_only

::: stagereg.core.every_nth_save

::: stagereg.core.always

::: stagereg.core.is_finalized

::: stagereg.core.STAGE_COMPLETE_METADATA_KEY

## `stagereg.core.distributions`

::: stagereg.core.PredictiveDistribution

::: stagereg.core.ClosedFormCRPS

::: stagereg.core.DistributionHead

::: stagereg.core.Likelihood

::: stagereg.core.DistributionLoss

::: stagereg.core.PredictionLoss

::: stagereg.core.Loss

## `stagereg.core.early_stopping`

::: stagereg.core.OptimizationMode

::: stagereg.core.EarlyStoppingState

::: stagereg.core.EarlyStoppingDecision

::: stagereg.core.EarlyStopper

## `stagereg.core.losses`

::: stagereg.core.NegativeLogLikelihoodLoss

::: stagereg.core.SquaredErrorLoss

::: stagereg.core.add_epsilon

::: stagereg.core.GaussianNLLLoss

::: stagereg.core.BetaNLLLoss

## `stagereg.core.metric_registry`

::: stagereg.core.EvaluationGrid

::: stagereg.core.PredictionInterval

::: stagereg.core.EpochPredictionData

::: stagereg.core.MetricRequirements

::: stagereg.core.EpochMetric

::: stagereg.core.RootMeanSquaredError

::: stagereg.core.NegativeLogLikelihood

::: stagereg.core.IntervalCoverage

::: stagereg.core.WeightedSpread

::: stagereg.core.PointContinuousRankedProbabilityScore

::: stagereg.core.SampleContinuousRankedProbabilityScore

::: stagereg.core.ContinuousRankedProbabilityScore

## `stagereg.core.metrics`

::: stagereg.core.cdf

::: stagereg.core.crps

::: stagereg.core.point_crps

::: stagereg.core.sample_crps

::: stagereg.core.expected_sample_crps

::: stagereg.core.expected_gaussian_mixture_crps

::: stagereg.core.rmse

::: stagereg.core.coverage

::: stagereg.core.wsu

## `stagereg.core.naming`

::: stagereg.core.Split

::: stagereg.core.MetricTag

::: stagereg.core.metric_tag

::: stagereg.core.parse_metric_tag

::: stagereg.core.flatten_parameters

## `stagereg.core.protocols`

::: stagereg.core.Dataset

::: stagereg.core.LoaderFactory

::: stagereg.core.Optimizer

::: stagereg.core.Step

::: stagereg.core.ValidationStrategy

## `stagereg.core.stages`

::: stagereg.core.validate_transition

::: stagereg.core.TrainingStage

## `stagereg.core.tracking`

::: stagereg.core.Decision

::: stagereg.core.TrainingEvent

::: stagereg.core.EventSink

::: stagereg.core.ExperimentTracker

::: stagereg.core.TrackerEventSink

## `stagereg.core.types`

::: stagereg.core.Array

::: stagereg.core.PyTree

::: stagereg.core.ParameterRole

::: stagereg.core.StageState

::: stagereg.core.Batch

::: stagereg.core.CheckpointRef

::: stagereg.core.TrainingState

::: stagereg.core.StageResult

::: stagereg.core.ValidationResult
