"""The clustered-data example's joint posterior.

One network shares the trunk between a mean head and a variance head, and
an inference method infers every weight, variance head included, under the
network's own predicted variance. The example builds the library's
[`PosteriorProblem`][stagereg.jax.PosteriorProblem] by hand and drives the
method with its own loop: a single run from a random initialization, with
no point-estimate warm start and no early stopping. Any
[`InferenceMethod`][stagereg.jax.InferenceMethod] runs in it unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from clustered import (
    JOINT_DRAW_STREAM,
    JOINT_INIT_STREAM,
    JOINT_TRAIN_STREAM,
    ClusteredData,
    ClusteredDataConfig,
    make_loader,
    seed_key,
)
from flax import nnx
from jax.scipy.stats import norm
from methods import draw_on_grid
from networks import JointNetwork
from scoring import PredictiveDraws

from stagereg.core.types import Batch, PyTree
from stagereg.jax import InferenceMethod, IsotropicGaussianPrior, PosteriorProblem


def joint_network(
    config: ClusteredDataConfig, *, dropout_rate: float = 0.0
) -> JointNetwork:
    """Initialize the joint network at random from `config.seed`.

    Args:
        config: The comparison's settings; `seed` sets the initialization.
        dropout_rate: The trunk's dropout rate.

    Returns:
        A freshly initialized `JointNetwork`.
    """
    return JointNetwork(
        dropout_rate=dropout_rate, rngs=nnx.Rngs(seed_key(config, JOINT_INIT_STREAM))
    )


def joint_problem(
    network: JointNetwork,
    data: ClusteredData,
    config: ClusteredDataConfig,
    *,
    epochs: int,
) -> PosteriorProblem:
    """Build the posterior problem over every weight of the joint network.

    Args:
        network: The joint network; its current weights are where inference
            starts.
        data: The training set to condition on.
        config: The comparison's settings; `prior_precision`, `batch_size`
            and `seed` are read.
        epochs: Epochs the inference method will run, so pSGLD can size its
            sample buffer.

    Returns:
        A problem whose mean function returns each input's mean and variance
        stacked on the last axis, whose log-likelihood is the Gaussian one
        under the network's own variance scaled by `N / B`, and whose prior
        is the shared isotropic Gaussian.
    """
    graph, parameters, rest = nnx.split(network, nnx.Param, ...)
    train_size = config.train_size

    def mean_function(
        parameters: PyTree, inputs: jax.Array, key: jax.Array
    ) -> jax.Array:
        model = nnx.merge(graph, parameters, rest)
        return model(inputs, dropout_key=key)

    def log_likelihood(parameters: PyTree, batch: Batch, key: jax.Array) -> jax.Array:
        outputs = mean_function(parameters, batch.inputs, key)
        mean, variance = outputs[..., :1], outputs[..., 1:]
        log_density = norm.logpdf(batch.targets, mean, jnp.sqrt(variance))
        return train_size / batch.targets.shape[0] * jnp.sum(log_density)

    return PosteriorProblem(
        initial_parameters=parameters,
        mean_function=mean_function,
        log_likelihood=log_likelihood,
        prior=IsotropicGaussianPrior(precision=config.prior_precision),
        train_loader=make_loader(data, config),
        dataset_size=train_size,
        epochs=epochs,
    )


def run_joint(
    data: ClusteredData,
    config: ClusteredDataConfig,
    inference_method: InferenceMethod,
    *,
    epochs: int,
    network: JointNetwork | None = None,
) -> PredictiveDraws:
    """Infer the joint posterior from a random initialization and draw on the grid.

    Args:
        data: The training set and the evaluation grid.
        config: The comparison's settings.
        inference_method: A fresh inference method.
        epochs: Epochs to run; the comparison runs `config.mean_epochs`,
            as the staged model's mean stage does.
        network: The joint network to infer over, or `None` for a fresh one
            without dropout.

    Returns:
        `config.num_draws` draws (all retained samples for a finite
        posterior), each with its own predictive mean and aleatoric variance
        on the grid, in the targets' units.
    """
    if network is None:
        network = joint_network(config)
    problem = joint_problem(network, data, config, epochs=epochs)
    inference_method.init(problem)
    key = seed_key(config, JOINT_TRAIN_STREAM)
    for epoch in range(epochs):
        for batch in problem.train_loader(split="train", epoch=epoch):
            key, step_key = jax.random.split(key)
            inference_method.update(batch, step_key)

    outputs = draw_on_grid(inference_method, data, config, stream=JOINT_DRAW_STREAM)
    draws = PredictiveDraws(means=outputs[..., 0], variances=outputs[..., 1])
    return draws.in_target_units(data.target_mean, data.target_std)
