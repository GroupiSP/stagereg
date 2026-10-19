"""Bayes by Backprop: mean-field Gaussian variational inference.

[`BayesByBackprop`][stagereg.jax.BayesByBackprop] is an
[`InferenceMethod`][stagereg.jax.InferenceMethod] that fits an independent
Gaussian to every parameter of the posterior network by maximizing the
minibatch evidence lower bound with the reparameterization trick (Blundell et
al., 2015). Its posterior,
[`MeanFieldGaussianPosterior`][stagereg.jax.MeanFieldGaussianPosterior], is
unlimited: it yields as many draws as its caller asks for.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import optax

from stagereg.core.types import Batch, PyTree
from stagereg.jax.posterior import (
    IsotropicGaussianPrior,
    PosteriorProblem,
    Prior,
    TracedPosterior,
    default_optimizer,
    draw_unlimited,
    require_inferred_tree,
)
from stagereg.jax.rng import network_key


@dataclass(frozen=True)
class MeanFieldGaussianPosterior:
    """An independent Gaussian on every parameter of the posterior network.

    Draw ``s`` under ``key`` samples its parameters from
    ``draw_key = jax.random.fold_in(key, s)`` and evaluates its mean function
    with a key derived from ``draw_key``, so it is the same mean function at every input, in every
    call, and whatever the number of draws asked for.

    Attributes:
        mean_function: Maps ``(parameters, inputs, key)`` to the posterior
            network's mean predictions.
        means: The variational means, in the posterior network's parameter
            tree.
        stds: The variational standard deviations, positive, in the same
            tree.
    """

    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    means: PyTree
    stds: PyTree

    @property
    def num_draws(self) -> int | None:
        """Always ``None``: a variational posterior has unlimited draws."""
        return None

    def sample_means(
        self,
        inputs: PyTree,
        key: jax.Array,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Draw mean predictions at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key selecting the draws.
            num_samples: The number of draws ``S``, positive.

        Returns:
            The draws' mean predictions, shaped ``[S, *batch]``.

        Raises:
            ValueError: If ``num_samples`` is omitted or not positive.
        """

        def draw(index: jax.Array) -> jax.Array:
            draw_key = jax.random.fold_in(key, index)
            parameters = _sample(self.means, self.stds, draw_key)
            return self.mean_function(parameters, inputs, network_key(draw_key))

        return draw_unlimited(draw, num_samples)

    def _traced(self) -> TracedPosterior:
        """Return the posterior taken apart for compiled code.

        Returns:
            Its class and mean function, static, and its arrays, traced.
        """
        return TracedPosterior(
            type(self), self.mean_function, {"means": self.means, "stds": self.stds}
        )


@dataclass
class BayesByBackprop:
    """Mean-field Gaussian variational inference by Bayes by Backprop.

    Every update takes one reparameterized parameter sample and steps the
    variational parameters along the gradient of the negative minibatch
    ELBO: the KL divergence from the prior minus the problem's
    log-likelihood, which is already scaled by ``N / B``. The KL is in closed
    form for an [`IsotropicGaussianPrior`][stagereg.jax.IsotropicGaussianPrior]
    and a one-sample Monte Carlo estimate, ``log q(w) - log p(w)``, for any
    other prior. Standard deviations are parameterized as the softplus of an
    unconstrained ``rho``.

    The variational means are warm-started from the problem's initial
    parameters, the trained mean network, and every standard deviation from
    ``initial_std``. Its default of ``1e-3`` keeps the first draws close to
    the trained mean, so inference starts from a good fit and widens the
    posterior from there; raise it when the parameters' posterior spread is
    expected to be far larger, since a softplus scale grows only about
    linearly in ``rho``.

    It supports early stopping. Its best checkpoint holds the variational
    parameters and optimizer state; its finalized checkpoint holds the
    variational means and standard deviations alone.

    Attributes:
        optimizer: The Optax transformation stepping the variational
            parameters. Defaults to Adam with a learning rate of ``1e-3``.
        initial_std: The standard deviation every parameter starts with,
            positive and finite. Defaults to ``1e-3``.
    """

    optimizer: optax.GradientTransformation = field(default_factory=default_optimizer)
    initial_std: float = 1e-3
    _problem: PosteriorProblem | None = field(default=None, init=False, repr=False)
    _variational: PyTree = field(default=None, init=False, repr=False)
    _optimizer_state: PyTree = field(default=None, init=False, repr=False)
    _step: Callable[..., tuple[PyTree, PyTree, jax.Array]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the initial standard deviation.

        Raises:
            ValueError: If ``initial_std`` is not positive and finite.
        """
        if not math.isfinite(self.initial_std) or self.initial_std <= 0.0:
            raise ValueError("initial_std must be positive and finite.")

    @property
    def supports_early_stopping(self) -> bool:
        """Always ``True``: the variational parameters can stop anywhere."""
        return True

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """``()``: the stage records only the loss."""
        return ()

    @property
    def has_posterior(self) -> bool:
        """Whether the method has been initialized."""
        return self._problem is not None

    def init(self, problem: PosteriorProblem) -> None:
        """Warm-start the variational parameters from ``problem``.

        Args:
            problem: The posterior problem.
        """
        rho = _inverse_softplus(self.initial_std)
        self._problem = problem
        self._variational = {
            "means": jax.tree.map(jnp.copy, problem.initial_parameters),
            "rhos": jax.tree.map(
                lambda leaf: jnp.full_like(leaf, rho), problem.initial_parameters
            ),
        }
        self._optimizer_state = self.optimizer.init(self._variational)
        self._step = _make_step(problem, self.optimizer)

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Step the variational parameters on one training batch.

        Only the batch's ``inputs``, ``targets`` and ``sample_weight`` reach
        the log-likelihood; its ``metadata`` does not.

        Args:
            batch: The training batch.
            key: A fresh PRNG key for this batch's parameter sample; the
                log-likelihood is evaluated with a key derived from it.

        Returns:
            ``{"loss": negative_elbo}``, the minibatch estimate of the
            negative ELBO.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_problem()
        assert self._step is not None
        self._variational, self._optimizer_state, loss = self._step(
            self._variational,
            self._optimizer_state,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            key,
        )
        return {"loss": loss}

    def posterior(self) -> MeanFieldGaussianPosterior:
        """Return the current variational posterior.

        Returns:
            A posterior independent of later updates.

        Raises:
            ValueError: If the method has not been initialized.
        """
        problem = self._require_problem()
        return MeanFieldGaussianPosterior(
            mean_function=problem.mean_function,
            means=self._variational["means"],
            stds=jax.tree.map(jax.nn.softplus, self._variational["rhos"]),
        )

    def state(self) -> PyTree:
        """Return the variational parameters and the optimizer state.

        Returns:
            ``{"variational": {"means", "rhos"}, "optimizer_state"}``.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_problem()
        return {
            "variational": self._variational,
            "optimizer_state": self._optimizer_state,
        }

    def load_state(self, state: PyTree) -> None:
        """Resume inference from a state returned by ``state()``.

        Args:
            state: A state previously returned by
                [`state`][stagereg.jax.BayesByBackprop.state], after
                [`init`][stagereg.jax.BayesByBackprop.init] on the same
                problem.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_problem()
        self._variational = state["variational"]
        self._optimizer_state = state["optimizer_state"]

    def posterior_state(self) -> PyTree:
        """Return the variational means and standard deviations.

        Returns:
            ``{"means", "stds"}``, each in the posterior network's parameter
            tree.

        Raises:
            ValueError: If the method has not been initialized.
        """
        posterior = self.posterior()
        return {"means": posterior.means, "stds": posterior.stds}

    def load_posterior(self, state: PyTree) -> None:
        """Rebuild the posterior from a state returned by ``posterior_state()``.

        Args:
            state: A state previously returned by
                [`posterior_state`][stagereg.jax.BayesByBackprop.posterior_state].

        Raises:
            ValueError: If the method has not been initialized, or if the
                means' or standard deviations' tree differs from the inferred
                subnetwork's.
        """
        problem = self._require_problem()
        for tree in (state["means"], state["stds"]):
            require_inferred_tree(tree, problem.initial_parameters)
        self._variational = {
            "means": state["means"],
            "rhos": jax.tree.map(_inverse_softplus, state["stds"]),
        }

    def _require_problem(self) -> PosteriorProblem:
        """Return the posterior problem, refusing an uninitialized method."""
        if self._problem is None:
            raise ValueError("BayesByBackprop must be initialized with init().")
        return self._problem


def _make_step(
    problem: PosteriorProblem, optimizer: optax.GradientTransformation
) -> Callable[..., tuple[PyTree, PyTree, jax.Array]]:
    """Return the jitted variational update on one batch's arrays."""
    kl = _kl_divergence(problem.prior)

    def negative_elbo(
        variational: PyTree,
        inputs: PyTree,
        targets: PyTree,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        means = variational["means"]
        stds = jax.tree.map(jax.nn.softplus, variational["rhos"])
        parameters = _sample(means, stds, key)
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        return kl(means, stds, parameters) - problem.log_likelihood(
            parameters, batch, network_key(key)
        )

    @jax.jit
    def step(
        variational: PyTree,
        optimizer_state: PyTree,
        inputs: PyTree,
        targets: PyTree,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> tuple[PyTree, PyTree, jax.Array]:
        loss, grads = jax.value_and_grad(negative_elbo)(
            variational, inputs, targets, sample_weight, key
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, variational)
        return optax.apply_updates(variational, updates), optimizer_state, loss

    return step


def _kl_divergence(prior: Prior) -> Callable[[PyTree, PyTree, PyTree], jax.Array]:
    """Return ``KL(q || prior)`` as a function of ``(means, stds, sample)``.

    Closed form for an isotropic Gaussian prior; otherwise the one-sample
    Monte Carlo estimate ``log q(sample) - log prior(sample)``.
    """
    if isinstance(prior, IsotropicGaussianPrior):
        precision = prior.precision

        def closed_form(means: PyTree, stds: PyTree, sample: PyTree) -> jax.Array:
            del sample

            def leaf_kl(mean: jax.Array, std: jax.Array) -> jax.Array:
                variance = precision * jnp.square(std)
                return 0.5 * jnp.sum(
                    variance + precision * jnp.square(mean) - 1.0 - jnp.log(variance)
                )

            return jnp.asarray(sum(jax.tree.leaves(jax.tree.map(leaf_kl, means, stds))))

        return closed_form

    def monte_carlo(means: PyTree, stds: PyTree, sample: PyTree) -> jax.Array:
        log_q = jax.tree.map(
            lambda value, mean, std: jnp.sum(jstats.norm.logpdf(value, mean, std)),
            sample,
            means,
            stds,
        )
        return jnp.asarray(sum(jax.tree.leaves(log_q))) - prior.log_prob(sample)

    return monte_carlo


def _sample(means: PyTree, stds: PyTree, key: jax.Array) -> PyTree:
    """Draw one parameter tree from the mean-field Gaussian."""
    leaves, treedef = jax.tree.flatten(means)
    keys = jax.random.split(key, len(leaves))
    noise = jax.tree.unflatten(
        treedef,
        [
            jax.random.normal(leaf_key, jnp.shape(leaf), jnp.result_type(leaf))
            for leaf_key, leaf in zip(keys, leaves, strict=True)
        ],
    )
    return jax.tree.map(lambda mean, std, eps: mean + std * eps, means, stds, noise)


def _inverse_softplus(value: jax.Array | float) -> jax.Array:
    """Return the ``rho`` whose softplus is ``value``, stably for small values."""
    value = jnp.asarray(value)
    return value + jnp.log(-jnp.expm1(-value))
