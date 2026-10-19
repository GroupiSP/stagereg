"""The seam between the posterior stage and its inference methods.

A [`PosteriorStage`][stagereg.jax.PosteriorStage] hands every
[`InferenceMethod`][stagereg.jax.InferenceMethod] the same
[`PosteriorProblem`][stagereg.jax.PosteriorProblem], steps it once per batch,
and reads back a [`Posterior`][stagereg.jax.Posterior] it consumes only through
draws of the mean function at given inputs (ADR-0009).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

from stagereg.core.protocols import LoaderFactory
from stagereg.core.types import Batch, PyTree


class Prior(Protocol):
    """A log-density over the parameters of the inferred subnetwork.

    The frozen remainder of the posterior network is not inferred, so the prior does not
    cover it.
    """

    def log_prob(self, parameters: PyTree) -> jax.Array:
        """Return the scalar log-density of ``parameters``.

        Args:
            parameters: A tree with the inferred subnetwork's parameter tree.

        Returns:
            The scalar prior log-density.
        """
        ...


class Posterior(Protocol):
    """An approximate posterior over mean functions, consumed through draws.

    A draw is a whole mean function: under the same key, draw ``s`` is the
    same function at every input and in every batch. A finite posterior
    (retained samples, ensemble members) has a fixed set of draws and always
    uses all of them; an unlimited one (a variational distribution,
    MC-Dropout) takes the number of draws from its caller.
    """

    @property
    def num_draws(self) -> int | None:
        """The number of draws of a finite posterior, or ``None`` if unlimited."""
        ...

    def sample_means(
        self,
        inputs: PyTree,
        key: jax.Array,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Draw mean predictions at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key selecting the draws. A finite posterior may
                ignore it.
            num_samples: The number of draws ``S``. Must be ``None`` for a
                finite posterior and given for an unlimited one.

        Returns:
            The draws' mean predictions, shaped ``[S, *batch]``.

        Raises:
            ValueError: If ``num_samples`` is given to a finite posterior, or
                omitted for an unlimited one.
        """
        ...


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["arrays", "draw_mask"],
    meta_fields=["posterior_type", "mean_function"],
)
@dataclass(frozen=True)
class TracedPosterior:
    """A posterior taken apart for compiled code, which takes it as an argument.

    It is a JAX pytree whose children are the posterior's arrays and whose
    type and mean function are static, compared by identity, so a compiled
    function taking it compiles once per posterior lineage and input shape.
    The posterior itself stays plain data, which Flax NNX can hold as
    ``nnx.static``. Inside the compiled function the posterior is rebuilt
    from its arrays and drawn from through its own ``sample_means``
    (ADR-0009).

    Attributes:
        posterior_type: The posterior's class, built as
            ``posterior_type(mean_function=..., **arrays)``.
        mean_function: The posterior's mean function.
        arrays: The posterior's other fields, all arrays, by name.
        draw_mask: Which of the rebuilt posterior's draw slots hold draws,
            or ``None`` when every one does.
    """

    posterior_type: type[Posterior]
    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    arrays: Mapping[str, PyTree]
    draw_mask: jax.Array | None = None

    def sample_means(
        self, inputs: PyTree, key: jax.Array, num_samples: int | None
    ) -> jax.Array:
        """Draw the rebuilt posterior's mean predictions at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key selecting the draws.
            num_samples: The number of draws of an unlimited posterior, or
                ``None`` for a finite one.

        Returns:
            The draws' mean predictions, shaped ``[S, *batch]``, including
            any slots ``draw_mask`` excludes.
        """
        posterior = self.posterior_type(mean_function=self.mean_function, **self.arrays)
        return posterior.sample_means(inputs, key, num_samples)


@runtime_checkable
class TraceablePosterior(Protocol):
    """A posterior the posterior stage can predict from in one compiled call.

    Every built-in posterior is one. Any other posterior is drawn from in
    Python through its ``sample_means``.
    """

    def _traced(self) -> TracedPosterior:
        """Return the posterior taken apart for compiled code."""
        ...


@dataclass(frozen=True)
class PosteriorProblem:
    """Everything a posterior stage hands its inference method.

    Attributes:
        initial_parameters: The warm start: a copy of the trained mean
            network's parameters in the inferred subnetwork, in the posterior
            network's parameter tree restricted to that subnetwork.
        mean_function: Maps ``(parameters, inputs, key)`` to the posterior
            network's mean predictions at ``inputs``, where ``parameters``
            holds the inferred subnetwork only and the frozen remainder keeps
            its trained-mean values. ``key`` seeds the network's RNG streams,
            so a stochastic network, e.g. one with ``nnx.Dropout``, draws its
            masks from it, purely under ``jit`` and ``grad``; the same key
            gives the same function. A deterministic network ignores it.
        log_likelihood: Maps ``(parameters, batch, key)`` to the scalar
            Gaussian log-likelihood of the batch targets around the mean
            predictions under ``key``, under the fixed aleatoric variance,
            scaled by ``N / B`` so that a minibatch estimates the full-data
            log-likelihood. Differentiable in ``parameters``; the aleatoric
            variance carries no gradient.
        prior: The prior over the inferred subnetwork's parameter tree.
        train_loader: Factory producing the training batches of each epoch.
        dataset_size: The number of training examples ``N``.
        aleatoric_variance: Maps inputs to the fixed aleatoric variance at
            them, without gradient: the same variance ``log_likelihood`` uses.
            An implicit method uses it to turn draws into predictive samples,
            e.g. for a discriminator. A
            [`PosteriorStage`][stagereg.jax.PosteriorStage] always sets it.
            Defaults to ``None``, for problems built by hand.
        epochs: The number of epochs the stage trains for, so a method can
            size what it keeps up front, e.g. a sample buffer. A
            [`PosteriorStage`][stagereg.jax.PosteriorStage] always sets it.
            Defaults to ``None``: unknown.
    """

    initial_parameters: PyTree
    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    log_likelihood: Callable[[PyTree, Batch, jax.Array], jax.Array]
    prior: Prior
    train_loader: LoaderFactory
    dataset_size: int
    aleatoric_variance: Callable[[PyTree], jax.Array] | None = None
    epochs: int | None = None


class InferenceMethod(Protocol):
    """How a posterior stage turns a posterior problem into a posterior.

    The stage owns the epoch loop: it calls
    [`init`][stagereg.jax.InferenceMethod.init] once, then
    [`update`][stagereg.jax.InferenceMethod.update] once per training batch,
    and reads the current [`posterior`][stagereg.jax.InferenceMethod.posterior]
    to validate each epoch that ends with
    [`has_posterior`][stagereg.jax.InferenceMethod.has_posterior] true. Every
    method honors the problem's prior.
    """

    @property
    def supports_early_stopping(self) -> bool:
        """Whether stopping on a validation improvement is meaningful.

        ``False`` for SG-MCMC, whose chain an early stopper would truncate.
        """
        ...

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """The keys of ``update``'s output the posterior stage records.

        The stage records each as ``posterior/train/<key>``, the mean of its
        scalar values over the epoch's batches, and ignores every other key
        but ``"loss"``. A key whose tag the stage already records, such as
        ``"loss"``, is refused when the stage is prepared. ``()`` records
        none.
        """
        ...

    @property
    def has_posterior(self) -> bool:
        """Whether [`posterior`][stagereg.jax.InferenceMethod.posterior] can be read.

        ``False`` before ``init`` and, for SG-MCMC, until the first sample is
        retained. The posterior stage skips validation for an epoch that ends
        without a posterior, so a method that supports early stopping must
        have one after every epoch.
        """
        ...

    def init(self, problem: PosteriorProblem) -> None:
        """Start inference from ``problem``, discarding any previous state.

        Args:
            problem: The posterior problem.
        """
        ...

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Advance inference by one training batch.

        Args:
            batch: The training batch.
            key: A fresh PRNG key for this batch.

        Returns:
            A mapping holding the scalar ``"loss"`` the method minimizes and
            a scalar for every declared
            [`diagnostics`][stagereg.jax.InferenceMethod.diagnostics] key.
            Other keys are ignored.
        """
        ...

    def posterior(self) -> Posterior:
        """Return the current posterior, independent of later updates.

        Returns:
            The posterior inferred so far.
        """
        ...

    def state(self) -> PyTree:
        """Return the method's full state, independent of later updates.

        This is what a best checkpoint stores, so that inference can resume
        from it, e.g. variational parameters plus optimizer state.

        Returns:
            A tree of arrays.
        """
        ...

    def load_state(self, state: PyTree) -> None:
        """Resume inference from a state returned by ``state()``.

        Args:
            state: A state previously returned by
                [`state`][stagereg.jax.InferenceMethod.state].
        """
        ...

    def posterior_state(self) -> PyTree:
        """Return the current posterior alone, independent of later updates.

        This is what a finalized checkpoint stores: only what
        [`load_posterior`][stagereg.jax.InferenceMethod.load_posterior] needs
        to rebuild the posterior, e.g. the variational parameters of a
        variational method or the retained samples of an SG-MCMC method, but
        no optimizer state.

        Returns:
            A tree of arrays.
        """
        ...

    def load_posterior(self, state: PyTree) -> None:
        """Rebuild a finalized posterior after
        [`init`][stagereg.jax.InferenceMethod.init].

        Afterwards [`posterior`][stagereg.jax.InferenceMethod.posterior]
        returns the posterior that ``state`` describes. The method need not be
        able to resume inference from it.

        Args:
            state: A state previously returned by
                [`posterior_state`][stagereg.jax.InferenceMethod.posterior_state].
        """
        ...


def require_inferred_tree(parameters: PyTree, inferred: PyTree) -> None:
    """Refuse parameters whose tree is not the inferred subnetwork's.

    Args:
        parameters: A tree meant to hold the inferred subnetwork's parameters,
            such as a saved posterior's.
        inferred: The inferred subnetwork's parameter tree, e.g. a posterior
            problem's ``initial_parameters``.

    Raises:
        ValueError: If the two trees' structures differ, as they do for a
            posterior saved under another ``inferred_subnetwork`` filter.
    """
    if jax.tree.structure(parameters) != jax.tree.structure(inferred):
        raise ValueError(
            "the parameter tree differs from the inferred subnetwork's; was the "
            "posterior saved with another inferred_subnetwork filter?"
        )


def default_optimizer() -> optax.GradientTransformation:
    """Return the default optimizer of the gradient-based inference methods.

    Returns:
        Adam with a learning rate of ``1e-3``.
    """
    return optax.adam(1e-3)


def draw_unlimited(
    draw: Callable[[jax.Array], jax.Array], num_samples: int | None
) -> jax.Array:
    """Stack the draws of an unlimited posterior.

    Args:
        draw: Maps a draw's index to its mean predictions.
        num_samples: The number of draws ``S``, positive.

    Returns:
        The draws ``0, ..., S - 1``, stacked along a leading axis.

    Raises:
        ValueError: If ``num_samples`` is omitted or not positive.
    """
    if num_samples is None:
        raise ValueError("an unlimited posterior needs num_samples.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    return jax.vmap(draw)(jnp.arange(num_samples))


@dataclass(frozen=True)
class IsotropicGaussianPrior:
    """An independent zero-mean Gaussian on every parameter.

    The default precision of 1 is VeBNN's κ = 1.

    Attributes:
        precision: The inverse variance of every parameter, positive and
            finite. Defaults to ``1.0``.
    """

    precision: float = 1.0

    def __post_init__(self) -> None:
        """Validate the precision.

        Raises:
            ValueError: If ``precision`` is not positive and finite.
        """
        if not math.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be positive and finite.")

    def log_prob(self, parameters: PyTree) -> jax.Array:
        """Return the normalized log-density of a parameter tree.

        Args:
            parameters: A tree of parameter arrays.

        Returns:
            The scalar sum, over every parameter, of its Gaussian log-density.
        """
        leaves = jax.tree.leaves(parameters)
        count = sum(jnp.size(leaf) for leaf in leaves)
        squared_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        normalizer = 0.5 * count * math.log(self.precision / (2.0 * math.pi))
        return normalizer - 0.5 * self.precision * jnp.asarray(squared_norm)
