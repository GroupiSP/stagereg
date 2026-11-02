"""The clustered-data example's staged model.

A mean stage, then a Gamma variance stage on its squared residuals, then the library's
posterior stage over the mean network: one outer pass, as in the xsin example. The mean
and variance stages are trained once per seed, early-stopped on their training loss, and
shared by every inference method's posterior stage; the posterior stage runs a fixed
number of epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import optax
from clustered import ClusteredData, ClusteredDataConfig, make_loader
from flax import nnx
from networks import GammaNetwork, MeanNetwork
from scoring import PredictiveDraws

from stagereg.core import EarlyStopper, InMemoryCheckpointStore
from stagereg.core.tracking import TrainingEvent
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


class _EpochCounter:
    """An event sink counting the training epochs a stage runs.

    A stage that restores its best epoch also rolls its metric history back
    to that epoch, so the history undercounts the epochs actually trained.

    Attributes:
        epochs: Training epochs completed so far.
    """

    def __init__(self) -> None:
        """Start counting from zero."""
        self.epochs = 0

    def on_event(self, event: TrainingEvent) -> None:
        """Count one epoch at the end of each training epoch.

        Args:
            event: The training event.
        """
        if event.name == "epoch_end" and event.split == "train":
            self.epochs += 1


@dataclass(frozen=True)
class SupervisedStages:
    """The trained mean and variance stages of one seed.

    Attributes:
        state: The training state after the variance stage, ready for a
            posterior stage. Fork it before running one.
        mean_model: The trained mean network.
        variance_model: The trained Gamma variance network.
        point_epochs: Epochs the mean and variance stages actually ran,
            together. A joint run of the same seed adds them to its
            posterior epochs, so both trainings get the same budget.
    """

    state: TrainingState
    mean_model: MeanNetwork
    variance_model: GammaNetwork
    point_epochs: int


def _early_stopped(
    epochs: int, patience: int, counter: _EpochCounter
) -> SupervisedStageOptions:
    """Options that stop on a stalled training loss and restore its best epoch."""
    return SupervisedStageOptions(
        epochs=epochs,
        event_sinks=(counter,),
        early_stopper=EarlyStopper(
            metric="loss", mode="min", patience=patience, source="train"
        ),
        checkpoint_store=InMemoryCheckpointStore(),
    )


def train_supervised_stages(
    data: ClusteredData, config: ClusteredDataConfig
) -> SupervisedStages:
    """Train the mean stage, then the Gamma variance stage, silently.

    Args:
        data: The training set to fit.
        config: The comparison's settings; `seed` sets the initialization
            and the batch order.

    Returns:
        The trained stages and the epochs they ran.
    """
    mean_key, variance_key, train_key = jax.random.split(jax.random.key(config.seed), 3)
    loader = make_loader(data, config)
    state = TrainingState(rng_state=train_key)
    counter = _EpochCounter()

    mean_model = MeanNetwork(rngs=nnx.Rngs(mean_key))
    mean_stage = MeanStage(
        model=mean_model,
        optimizer=create_optimizer(mean_model, optax.adam(config.learning_rate)),
        train_loader=loader,
        options=_early_stopped(config.mean_epochs, config.mean_patience, counter),
    )
    mean_stage.prepare(state)
    mean_stage.train(state)

    variance_model = GammaNetwork(rngs=nnx.Rngs(variance_key))
    variance_stage = GammaVarianceStage(
        model=variance_model,
        optimizer=create_optimizer(variance_model, optax.adam(config.learning_rate)),
        source_loader=loader,
        options=_early_stopped(
            config.variance_epochs, config.variance_patience, counter
        ),
        splits=("train",),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)

    return SupervisedStages(
        state=state,
        mean_model=mean_model,
        variance_model=variance_model,
        point_epochs=counter.epochs,
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
    key = jax.random.fold_in(jax.random.key(config.seed), 5)
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

    posterior = inference_method.posterior()
    num_samples = None if posterior.num_draws is not None else config.num_draws
    draw_key = jax.random.fold_in(jax.random.key(config.seed), 3)
    means = posterior.sample_means(data.grid_inputs, draw_key, num_samples)[..., 0]
    aleatoric = supervised.variance_model(data.grid_inputs).mean()[..., 0]
    return PredictiveDraws(
        means=means, variances=jnp.broadcast_to(aleatoric, means.shape)
    )
