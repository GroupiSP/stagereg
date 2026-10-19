"""MC-Dropout: fine-tune with dropout active, draw by sampling dropout masks.

[`MCDropout`][stagereg.jax.MCDropout] is an
[`InferenceMethod`][stagereg.jax.InferenceMethod] that fine-tunes the inferred
subnetwork of a posterior network with ``nnx.Dropout`` layers, and reads its
posterior as the network under random dropout masks (Gal and Ghahramani, 2016).
Its posterior, [`MCDropoutPosterior`][stagereg.jax.MCDropoutPosterior], is
unlimited: one draw is one mask set, drawn from one key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax

from stagereg.core.types import Batch, PyTree
from stagereg.jax.posterior import (
    PosteriorProblem,
    default_optimizer,
    draw_unlimited,
    require_inferred_tree,
)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["parameters"],
    meta_fields=["mean_function"],
)
@dataclass(frozen=True)
class MCDropoutPosterior:
    """The fine-tuned network under random dropout masks.

    Draw ``s`` under ``key`` evaluates the mean function with the key
    ``jax.random.fold_in(key, s)``, which selects its dropout masks. It is the
    same function at every input only if no mask varies across batch rows;
    see [`MCDropout`][stagereg.jax.MCDropout].

    It is a JAX pytree whose child is ``parameters`` and whose
    ``mean_function`` is static, compared by identity, so it passes into
    jitted code whole, as [`PosteriorStage.predict`][stagereg.jax.PosteriorStage.predict]
    passes it.

    Attributes:
        mean_function: Maps ``(parameters, inputs, key)`` to the posterior
            network's mean predictions, with dropout masks drawn from ``key``.
        parameters: The fine-tuned inferred subnetwork.
    """

    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    parameters: PyTree

    @property
    def num_draws(self) -> int | None:
        """Always ``None``: there are as many mask sets as keys."""
        return None

    def sample_means(
        self,
        inputs: PyTree,
        key: jax.Array,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Draw mean predictions at ``inputs``, one mask set per draw.

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
            return self.mean_function(
                self.parameters, inputs, jax.random.fold_in(key, index)
            )

        return draw_unlimited(draw, num_samples)


@dataclass
class MCDropout:
    """MC-Dropout: maximum a posteriori fine-tuning with dropout active.

    Every update steps the inferred subnetwork along the gradient of the
    negative log-posterior under one dropout mask set: minus the problem's
    log-likelihood, which is already scaled by ``N / B``, minus the prior's
    log-density, as in
    [`BayesByBackprop`][stagereg.jax.BayesByBackprop]. The batch's key selects
    the masks, so every batch trains under fresh ones. Fine-tuning starts from
    the problem's initial parameters, the trained mean network.

    A draw of its posterior is the fine-tuned network under one mask set, so a
    draw is the same function at every input only if no dropout mask varies
    across the rows of a batch. Give every dropout layer of the posterior
    network ``broadcast_dims=(0,)``, so that one mask is shared by the whole
    batch axis, e.g. ``nnx.Dropout(0.1, broadcast_dims=(0,), rngs=rngs)``. A
    posterior network that is the mean network plus such a dropout layer
    without parameters has the mean network's parameter tree, so it infers the
    same subnetwork as the other methods.

    It supports early stopping. Its best checkpoint holds the fine-tuned
    parameters and optimizer state; its finalized checkpoint holds the
    fine-tuned inferred subnetwork alone.

    Attributes:
        optimizer: The Optax transformation stepping the inferred subnetwork.
            Defaults to Adam with a learning rate of ``1e-3``.
    """

    optimizer: optax.GradientTransformation = field(default_factory=default_optimizer)
    _problem: PosteriorProblem | None = field(default=None, init=False, repr=False)
    _parameters: PyTree = field(default=None, init=False, repr=False)
    _optimizer_state: PyTree = field(default=None, init=False, repr=False)
    _step: Callable[..., tuple[PyTree, PyTree, jax.Array]] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def supports_early_stopping(self) -> bool:
        """Always ``True``: fine-tuning can stop anywhere."""
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
        """Start fine-tuning from ``problem``'s initial parameters.

        Args:
            problem: The posterior problem.
        """
        self._problem = problem
        self._parameters = jax.tree.map(jnp.copy, problem.initial_parameters)
        self._optimizer_state = self.optimizer.init(self._parameters)
        self._step = _make_step(problem, self.optimizer)

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Step the inferred subnetwork on one training batch.

        Only the batch's ``inputs``, ``targets`` and ``sample_weight`` reach
        the log-likelihood; its ``metadata`` does not.

        Args:
            batch: The training batch.
            key: A fresh PRNG key selecting this batch's dropout masks.

        Returns:
            ``{"loss": negative_log_posterior}``, the minibatch estimate of
            the negative log-posterior under this batch's masks.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_problem()
        assert self._step is not None
        self._parameters, self._optimizer_state, loss = self._step(
            self._parameters,
            self._optimizer_state,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            key,
        )
        return {"loss": loss}

    def posterior(self) -> MCDropoutPosterior:
        """Return the current fine-tuned network under random masks.

        Returns:
            A posterior independent of later updates.

        Raises:
            ValueError: If the method has not been initialized.
        """
        problem = self._require_problem()
        return MCDropoutPosterior(
            mean_function=problem.mean_function, parameters=self._parameters
        )

    def state(self) -> PyTree:
        """Return the fine-tuned parameters and the optimizer state.

        Returns:
            ``{"parameters", "optimizer_state"}``.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_problem()
        return {
            "parameters": self._parameters,
            "optimizer_state": self._optimizer_state,
        }

    def load_state(self, state: PyTree) -> None:
        """Resume fine-tuning from a state returned by ``state()``.

        Args:
            state: A state previously returned by
                [`state`][stagereg.jax.MCDropout.state], after
                [`init`][stagereg.jax.MCDropout.init] on the same problem.

        Raises:
            ValueError: If the method has not been initialized.
        """
        self._require_problem()
        self._parameters = state["parameters"]
        self._optimizer_state = state["optimizer_state"]

    def posterior_state(self) -> PyTree:
        """Return the fine-tuned inferred subnetwork.

        Returns:
            The fine-tuned parameters, in the inferred subnetwork's tree.

        Raises:
            ValueError: If the method has not been initialized.
        """
        return self.posterior().parameters

    def load_posterior(self, state: PyTree) -> None:
        """Rebuild the posterior from a state returned by ``posterior_state()``.

        Args:
            state: A state previously returned by
                [`posterior_state`][stagereg.jax.MCDropout.posterior_state].

        Raises:
            ValueError: If the method has not been initialized, or if the
                state's tree differs from the inferred subnetwork's.
        """
        problem = self._require_problem()
        require_inferred_tree(state, problem.initial_parameters)
        self._parameters = state

    def _require_problem(self) -> PosteriorProblem:
        """Return the posterior problem, refusing an uninitialized method."""
        if self._problem is None:
            raise ValueError("MCDropout must be initialized with init().")
        return self._problem


def _make_step(
    problem: PosteriorProblem, optimizer: optax.GradientTransformation
) -> Callable[..., tuple[PyTree, PyTree, jax.Array]]:
    """Return the jitted fine-tuning update on one batch's arrays."""

    def negative_log_posterior(
        parameters: PyTree,
        inputs: PyTree,
        targets: PyTree,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> jax.Array:
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        return -problem.log_likelihood(parameters, batch, key) - problem.prior.log_prob(
            parameters
        )

    @jax.jit
    def step(
        parameters: PyTree,
        optimizer_state: PyTree,
        inputs: PyTree,
        targets: PyTree,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> tuple[PyTree, PyTree, jax.Array]:
        loss, grads = jax.value_and_grad(negative_log_posterior)(
            parameters, inputs, targets, sample_weight, key
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), optimizer_state, loss

    return step
