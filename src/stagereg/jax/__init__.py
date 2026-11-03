"""Optional JAX/Flax NNX training backend."""

from stagereg.jax.bayes_by_backprop import BayesByBackprop, MeanFieldGaussianPosterior
from stagereg.jax.distributions import (
    Gamma,
    GammaHead,
    Gaussian,
    GaussianHead,
    MomentMatchedPredictive,
    PosteriorPredictive,
    ResidualScale,
)
from stagereg.jax.evaluation import (
    SupervisedLoss,
    evaluate_loader,
    make_evaluation_step,
)
from stagereg.jax.losses import make_supervised_loss
from stagereg.jax.mc_dropout import MCDropout, MCDropoutPosterior
from stagereg.jax.metrics import (
    BatchMetric,
    BatchMetricSpec,
    CoordinateExtractor,
    GaussianPredictor,
    MetricSuite,
    PosteriorPredictivePredictor,
    PredictionRequirements,
    Predictor,
    ReferenceSamplesExtractor,
    merge_epoch_prediction_data,
)
from stagereg.jax.posterior import (
    InferenceMethod,
    IsotropicGaussianPrior,
    Posterior,
    PosteriorProblem,
    Prior,
)
from stagereg.jax.posterior_stage import PosteriorStage, PosteriorStageOptions
from stagereg.jax.psgld import PreconditionedSGLD, RetainedSamplesPosterior
from stagereg.jax.rng import split_key
from stagereg.jax.state import (
    NnxSnapshot,
    create_optimizer,
    freeze_training_state,
    initialize_training_state,
    restore_checkpoint,
    snapshot,
)
from stagereg.jax.supervised import make_train_step, run_supervised
from stagereg.jax.supervised_staged import (
    GammaVarianceStage,
    MeanStage,
    SupervisedStageOptions,
    materialize_residual_loader,
)
from stagereg.jax.validation import HeldOutValidation

__all__ = [
    "BatchMetric",
    "BayesByBackprop",
    "BatchMetricSpec",
    "CoordinateExtractor",
    "Gamma",
    "GammaHead",
    "GammaVarianceStage",
    "Gaussian",
    "GaussianHead",
    "GaussianPredictor",
    "HeldOutValidation",
    "InferenceMethod",
    "IsotropicGaussianPrior",
    "MCDropout",
    "MCDropoutPosterior",
    "MeanFieldGaussianPosterior",
    "MeanStage",
    "MetricSuite",
    "MomentMatchedPredictive",
    "NnxSnapshot",
    "Posterior",
    "PosteriorPredictive",
    "PosteriorPredictivePredictor",
    "PosteriorProblem",
    "PosteriorStage",
    "PosteriorStageOptions",
    "PreconditionedSGLD",
    "PredictionRequirements",
    "Predictor",
    "Prior",
    "ReferenceSamplesExtractor",
    "ResidualScale",
    "RetainedSamplesPosterior",
    "SupervisedLoss",
    "SupervisedStageOptions",
    "create_optimizer",
    "evaluate_loader",
    "freeze_training_state",
    "initialize_training_state",
    "make_evaluation_step",
    "make_supervised_loss",
    "make_train_step",
    "materialize_residual_loader",
    "merge_epoch_prediction_data",
    "restore_checkpoint",
    "run_supervised",
    "snapshot",
    "split_key",
]
