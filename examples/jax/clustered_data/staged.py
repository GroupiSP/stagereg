"""The clustered-data example's staged model.

A mean stage, then a Gamma variance stage on its squared residuals, then the library's
posterior stage over the mean network: one outer pass, as in the xsin example. The mean
and variance stages are trained once per seed and shared by every inference method's
posterior stage. Every stage runs a fixed number of epochs, without early stopping: the
mean stage as many as a joint run, the variance and posterior stages their own budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import optax
from clustered import (
    STAGED_DRAW_STREAM,
    STAGED_POSTERIOR_INIT_STREAM,
    SUPERVISED_STREAM,
    ClusteredData,
    ClusteredDataConfig,
    make_loader,
    seed_key,
)
from flax import nnx
from methods import draw_on_grid
from networks import GammaNetwork, MeanNetwork
from scoring import PredictiveDraws

from stagereg.core.types import TrainingState
from stagereg.jax import (
    GammaVarianceStage,
    InferenceMethod,
    IsotropicGaussianPrior,
    MeanStage,
    PosteriorStage,
    PosteriorStageOptions,
    SupervisedStageOptions,
    create_optimizer,
)


@dataclass(frozen=True)
class SupervisedStages:
    """The trained mean and variance stages of one seed.

    Attributes:
        state: The training state after the variance stage, ready for a
            posterior stage. Fork it before running one.
        mean_model: The trained mean network.
        variance_model: The trained Gamma variance network.
    """

    state: TrainingState
    mean_model: MeanNetwork
    variance_model: GammaNetwork


def train_supervised_stages(
    data: ClusteredData, config: ClusteredDataConfig
) -> SupervisedStages:
    """Train the mean stage, then the Gamma variance stage, silently.

    Each runs its full `mean_epochs` or `variance_epochs`.

    Args:
        data: The training set to fit.
        config: The comparison's settings; `seed` sets the initialization
            and the batch order.

    Returns:
        The trained stages.
    """
    mean_key, variance_key, train_key = jax.random.split(
        seed_key(config, SUPERVISED_STREAM), 3
    )
    loader = make_loader(data, config)
    state = TrainingState(rng_state=train_key)

    mean_model = MeanNetwork(rngs=nnx.Rngs(mean_key))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.adam(config.learning_rate)),
        train_loader=loader,
        options=SupervisedStageOptions(epochs=config.mean_epochs),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)

    variance_model = GammaNetwork(rngs=nnx.Rngs(variance_key))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.adam(config.learning_rate)),
        source_loader=loader,
        options=SupervisedStageOptions(epochs=config.variance_epochs),
        splits=("train",),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)

    return SupervisedStages(
        state=state,
        mean_model=mean_model,
        variance_model=variance_model,
    )


def _fork(state: TrainingState) -> TrainingState:
    """Copy the state's containers so a posterior stage leaves `state` as is.

    The trained models stay shared: the posterior stage only reads them.
    """
    return replace(
        state,
        model_components=dict(state.model_components),
        parameter_roles=dict(state.parameter_roles),
        optimizer_states=dict(state.optimizer_states),
        checkpoint_registry=dict(state.checkpoint_registry),
        metric_history={
            tag: list(values) for tag, values in state.metric_history.items()
        },
    )


def posterior_network(config: ClusteredDataConfig, dropout_rate: float) -> MeanNetwork:
    """Build the posterior stage's network: the mean network's architecture.

    The posterior stage copies the trained mean network's parameters into
    it, so only its dropout rate matters.

    Args:
        config: The comparison's settings; `seed` is read.
        dropout_rate: The backbone's dropout rate.

    Returns:
        An untrained mean network with that dropout rate.
    """
    key = seed_key(config, STAGED_POSTERIOR_INIT_STREAM)
    return MeanNetwork(dropout_rate=dropout_rate, rngs=nnx.Rngs(key))


def run_staged(
    data: ClusteredData,
    config: ClusteredDataConfig,
    supervised: SupervisedStages,
    inference_method: InferenceMethod,
    *,
    network: nnx.Module | None = None,
) -> PredictiveDraws:
    """Run the posterior stage on the trained stages and draw on the grid.

    Args:
        data: The training set and the evaluation grid.
        config: The comparison's settings.
        supervised: The seed's trained mean and variance stages; they are
            left unchanged.
        inference_method: A fresh inference method.
        network: The posterior network, with the mean network's parameter
            tree, or `None` to infer over a copy of the mean network.

    Returns:
        `config.num_draws` draws of the predictive mean on the grid (all
        retained samples for a finite posterior), each paired with the
        variance stage's aleatoric variance.
    """
    state = _fork(supervised.state)
    stage = PosteriorStage(
        inference_method=inference_method,
        train_loader=make_loader(data, config),
        dataset_size=config.train_size,
        prior=IsotropicGaussianPrior(precision=config.prior_precision),
        model=network,
        options=PosteriorStageOptions(
            epochs=config.posterior_epochs, num_draws=config.num_draws
        ),
    )
    stage.prepare(state)
    stage.train(state)

    means = draw_on_grid(inference_method, data, config, stream=STAGED_DRAW_STREAM)[
        ..., 0
    ]
    aleatoric = supervised.variance_model(data.grid_inputs).mean()[..., 0]
    return PredictiveDraws(
        means=means, variances=jnp.broadcast_to(aleatoric, means.shape)
    )
