"""Preconditioned stochastic-gradient Langevin dynamics (pSGLD).

An SG-MCMC [`InferenceMethod`][stagereg.jax.InferenceMethod]: a Langevin chain
over the posterior network's parameters, preconditioned by an RMSprop estimate
of the gradient's scale (Li et al., 2016). The chain's positions retained
after burn-in, at the thinning interval, are the draws of a finite
[`RetainedSamplesPosterior`][stagereg.jax.RetainedSamplesPosterior].
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from stagereg.core.types import Batch, PyTree
from stagereg.jax.posterior import PosteriorProblem, require_inferred_tree
from stagereg.jax.rng import network_key


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["samples", "draw_mask"],
    meta_fields=["mean_function"],
)
@dataclass(frozen=True)
class RetainedSamplesPosterior:
    """A finite posterior whose draws are retained parameter samples.

    Draw ``s`` under ``key`` is the posterior network with the ``s``-th
    retained parameters, evaluated with the key ``jax.random.fold_in(key,
    s)``, so it is the same mean function at every input, and every call uses
    all the draws. For a deterministic posterior network the key changes
    nothing.

    The samples may sit in a fixed number of slots, some of them spare, with a
    ``draw_mask`` marking the slots that hold retained samples: a chain that
    keeps retaining samples then keeps the same array shapes, so code that
    consumes it compiles once.
    [`padded_sample_means`][stagereg.jax.RetainedSamplesPosterior.padded_sample_means]
    evaluates every slot, for such code; a draw keeps its slot's index ``s``
    for its key.

    It is a JAX pytree whose children are ``samples`` and ``draw_mask`` and
    whose ``mean_function`` is static, compared by identity, so it passes into
    jitted code whole, as [`PosteriorStage.predict`][stagereg.jax.PosteriorStage.predict]
    passes it.

    Attributes:
        mean_function: Maps ``(parameters, inputs, key)`` to the posterior
            network's mean predictions at ``inputs``.
        samples: The parameter trees of the slots, stacked along a leading
            axis; without a ``draw_mask``, each slot holds a retained sample.
        draw_mask: Which slots hold retained samples, a boolean array shaped
            like the leading axis of ``samples``. Defaults to ``None``: every
            slot does.
    """

    mean_function: Callable[[PyTree, PyTree, jax.Array], jax.Array]
    samples: PyTree
    draw_mask: jax.Array | None = None

    @property
    def num_draws(self) -> int:
        """The number of retained samples ``S``."""
        if self.draw_mask is None:
            return _num_slots(self.samples)
        return int(np.count_nonzero(jax.device_get(self.draw_mask)))

    def sample_means(
        self,
        inputs: PyTree,
        key: jax.Array,
        num_samples: int | None = None,
    ) -> jax.Array:
        """Evaluate every retained sample's mean function at ``inputs``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key each draw's mean function is evaluated with;
                a deterministic posterior network ignores it.
            num_samples: Must be ``None``; all retained samples are used.

        Returns:
            The mean predictions, shaped ``[S, *batch]``.

        Raises:
            ValueError: If ``num_samples`` is given.
        """
        if num_samples is not None:
            raise ValueError(
                "a finite posterior uses all its retained samples; "
                "num_samples must be None."
            )
        if self.draw_mask is None:
            return self._evaluate(
                self.samples, inputs, key, jnp.arange(_num_slots(self.samples))
            )
        slots = np.flatnonzero(jax.device_get(self.draw_mask))
        retained = jax.tree.map(lambda leaf: leaf[slots], self.samples)
        return self._evaluate(retained, inputs, key, jnp.asarray(slots))

    def padded_sample_means(
        self, inputs: PyTree, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate every slot's mean function at ``inputs``, spare ones included.

        The result keeps one shape however many samples are retained, so a
        consumer compiles once; it excludes the spare slots by the returned
        mask, e.g. as a
        [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive]'s
        ``draw_mask``.

        Args:
            inputs: Model inputs with a leading batch dimension.
            key: The PRNG key each slot's mean function is evaluated with.

        Returns:
            The mean predictions of every slot, shaped ``[slots, *batch]``,
            and the boolean mask of the slots holding retained samples.
        """
        count = _num_slots(self.samples)
        means = self._evaluate(self.samples, inputs, key, jnp.arange(count))
        if self.draw_mask is None:
            return means, jnp.ones(count, dtype=bool)
        return means, self.draw_mask

    def _evaluate(
        self, samples: PyTree, inputs: PyTree, key: jax.Array, slots: jax.Array
    ) -> jax.Array:
        """Evaluate stacked samples, each under the key folded with its slot."""
        draw_keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(slots)
        return jax.vmap(self.mean_function, in_axes=(0, None, 0))(
            samples, inputs, draw_keys
        )


@dataclass
class PreconditionedSGLD:
    """RMSprop-preconditioned stochastic-gradient Langevin dynamics.

    Each [`update`][stagereg.jax.PreconditionedSGLD.update] takes one step on
    the batch's estimate ``U`` of the negative log posterior, the problem's
    log-likelihood (already scaled to the full data set) plus its prior.
    With gradient ``g = ∇U``, the preconditioner tracks
    ``V ← decay · V + (1 − decay) · g²`` and ``G = 1 / (stability + √V)``,
    and the parameters move by ``−(step_size / 2) · G · g`` plus Gaussian
    noise of variance ``step_size · G``. The preconditioner's own drift
    correction is neglected, as is usual.

    The step is compiled once per [`init`][stagereg.jax.PreconditionedSGLD.init];
    the log-likelihood sees each batch's inputs, targets and sample weights but
    not its metadata.

    Burn-in and thinning count update steps, i.e. training batches, not
    epochs. Steps are numbered from 1; the position after step ``t`` is
    retained when ``t = burn_in + k · thinning`` for some ``k ≥ 1``. Until
    the first position is retained,
    [`has_posterior`][stagereg.jax.PreconditionedSGLD.has_posterior] is
    ``False`` and [`posterior`][stagereg.jax.PreconditionedSGLD.posterior]
    refuses, so a posterior stage skips validation for the epochs of a
    burn-in.

    Retained samples are written into a buffer of ``capacity`` slots, and the
    [`posterior`][stagereg.jax.PreconditionedSGLD.posterior] holds every
    slot with a mask of the retained ones, so the posterior keeps one shape
    while samples accumulate and validating it compiles nothing new. A full
    buffer doubles its capacity, which compiles the posterior's consumers once
    more; a ``capacity`` of at least the number of samples a run retains
    avoids that.

    SG-MCMC declares no early-stopping support, so a
    [`PosteriorStage`][stagereg.jax.PosteriorStage] refuses an early stopper
    for it and writes only a finalized checkpoint, holding the stacked
    retained samples.

    Attributes:
        step_size: The Langevin step size ``ε``, positive and finite.
        burn_in: The number of initial update steps whose positions are
            discarded. Defaults to ``0``.
        thinning: Retain one position every ``thinning`` steps after burn-in.
            Defaults to ``1``, which retains every position.
        decay: The RMSprop decay of the squared-gradient average, in
            ``[0, 1)``. Defaults to ``0.99``.
        stability: Added to ``√V`` before inverting it, positive. Defaults to
            ``1e-5``.
        capacity: The number of slots first reserved for retained samples,
            positive; the buffer doubles whenever it is full. Defaults to
            ``64``.
    """

    step_size: float
    burn_in: int = 0
    thinning: int = 1
    decay: float = 0.99
    stability: float = 1e-5
    capacity: int = 64
    _problem: PosteriorProblem | None = field(default=None, init=False, repr=False)
    _step: Callable[..., Any] | None = field(default=None, init=False, repr=False)
    _parameters: PyTree = field(default=None, init=False, repr=False)
    _second_moment: PyTree = field(default=None, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)
    _buffer: PyTree = field(default=None, init=False, repr=False)
    _retained: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If ``step_size`` or ``stability`` is not positive and
                finite, ``burn_in`` is negative, ``thinning`` or ``capacity``
                is not positive, or ``decay`` is outside ``[0, 1)``.
        """
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite.")
        if not math.isfinite(self.stability) or self.stability <= 0.0:
            raise ValueError("stability must be positive and finite.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive.")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1).")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive.")

    @property
    def supports_early_stopping(self) -> bool:
        """``False``: an early stopper would truncate the chain."""
        return False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """``()``: the stage records only the loss."""
        return ()

    @property
    def has_posterior(self) -> bool:
        """Whether at least one sample has been retained."""
        return self._retained > 0

    def init(self, problem: PosteriorProblem) -> None:
        """Start a chain at the problem's warm start, discarding any samples.

        Args:
            problem: The posterior problem.
        """
        self._problem = problem
        self._step = _make_step(
            problem,
            step_size=self.step_size,
            decay=self.decay,
            stability=self.stability,
        )
        self._parameters = problem.initial_parameters
        self._second_moment = jax.tree.map(jnp.zeros_like, problem.initial_parameters)
        self._steps = 0
        self._buffer = _empty_buffer(problem.initial_parameters, self.capacity)
        self._retained = 0

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Take one Langevin step, retaining the new position if it is due.

        Args:
            batch: The training batch.
            key: The PRNG key of this step's injected noise; the
                log-likelihood is evaluated with a key derived from it.

        Returns:
            ``{"loss": U}``, the negative log posterior estimate at the
            position before the step.

        Raises:
            ValueError: If the method was not initialized.
        """
        if self._step is None:
            raise ValueError("call init before update.")
        self._parameters, self._second_moment, loss = self._step(
            self._parameters,
            self._second_moment,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            key,
        )
        self._steps += 1
        retained_after = self._steps - self.burn_in
        if retained_after > 0 and retained_after % self.thinning == 0:
            if self._retained == _num_slots(self._buffer):
                self._buffer = _doubled(self._buffer)
            self._buffer = _write(self._buffer, self._parameters, self._retained)
            self._retained += 1
        return {"loss": loss}

    def posterior(self) -> RetainedSamplesPosterior:
        """Return the posterior of the samples retained so far.

        Returns:
            A finite posterior with one draw per retained sample, whose
            samples are a copy of every buffer slot with a mask of the
            retained ones.

        Raises:
            ValueError: If no sample has been retained yet, e.g. during
                burn-in.
        """
        self._require_retained()
        samples, draw_mask = _snapshot(self._buffer, self._retained)
        return RetainedSamplesPosterior(
            mean_function=self._require_problem().mean_function,
            samples=samples,
            draw_mask=draw_mask,
        )

    def state(self) -> PyTree:
        """Return the chain's full state, from which it can resume.

        Returns:
            The position, the preconditioner's squared-gradient average, the
            step count and the retained samples (stacked, possibly none).
        """
        return {
            "parameters": self._parameters,
            "second_moment": self._second_moment,
            "steps": jnp.asarray(self._steps),
            "samples": _retained_samples(self._buffer, self._retained),
        }

    def load_state(self, state: PyTree) -> None:
        """Resume the chain from a state returned by ``state()``.

        Args:
            state: A state previously returned by
                [`state`][stagereg.jax.PreconditionedSGLD.state].
        """
        self._parameters = state["parameters"]
        self._second_moment = state["second_moment"]
        self._steps = int(state["steps"])
        self._buffer, self._retained = _buffer_of(state["samples"], self.capacity)

    def posterior_state(self) -> PyTree:
        """Return the retained samples, stacked along a leading axis.

        Returns:
            The parameter tree with a leading sample axis.

        Raises:
            ValueError: If no sample has been retained yet.
        """
        self._require_retained()
        return _retained_samples(self._buffer, self._retained)

    def load_posterior(self, state: PyTree) -> None:
        """Rebuild the posterior of the given retained samples after ``init``.

        The samples' tree is checked against the inferred subnetwork's once
        the method is initialized. Samples loaded before ``init`` cannot reach
        a posterior, as ``init`` discards them.

        Args:
            state: Stacked samples previously returned by
                [`posterior_state`][stagereg.jax.PreconditionedSGLD.posterior_state].

        Raises:
            ValueError: If the method has been initialized and the samples'
                tree differs from the inferred subnetwork's.
        """
        if self._problem is not None:
            require_inferred_tree(state, self._problem.initial_parameters)
        self._buffer, self._retained = _buffer_of(state, self.capacity)

    def _require_retained(self) -> None:
        """Refuse to read a posterior before the first sample is retained."""
        if not self.has_posterior:
            raise ValueError(
                "no sample has been retained yet; train beyond burn_in "
                f"({self.burn_in} steps) plus thinning ({self.thinning} steps)."
            )

    def _require_problem(self) -> PosteriorProblem:
        """Return the posterior problem, refusing an uninitialized method."""
        if self._problem is None:
            raise ValueError("call init before reading the posterior.")
        return self._problem


def _make_step(
    problem: PosteriorProblem, *, step_size: float, decay: float, stability: float
) -> Callable[..., Any]:
    """Compile one pSGLD step for ``problem``."""

    def negative_log_posterior(
        parameters: PyTree, batch: Batch, key: jax.Array
    ) -> jax.Array:
        return -(
            problem.log_likelihood(parameters, batch, key)
            + problem.prior.log_prob(parameters)
        )

    @jax.jit
    def step(
        parameters: PyTree,
        second_moment: PyTree,
        inputs: PyTree,
        targets: PyTree | None,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> tuple[PyTree, PyTree, jax.Array]:
        batch = Batch(inputs=inputs, targets=targets, sample_weight=sample_weight)
        loss, gradient = jax.value_and_grad(negative_log_posterior)(
            parameters, batch, network_key(key)
        )
        second_moment = jax.tree.map(
            lambda v, g: decay * v + (1.0 - decay) * jnp.square(g),
            second_moment,
            gradient,
        )
        leaves, treedef = jax.tree.flatten(parameters)
        keys = jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))

        def move(
            theta: jax.Array, g: jax.Array, v: jax.Array, leaf_key: jax.Array
        ) -> jax.Array:
            preconditioner = 1.0 / (stability + jnp.sqrt(v))
            noise = jax.random.normal(
                leaf_key, jnp.shape(theta), jnp.result_type(theta)
            )
            return (
                theta
                - 0.5 * step_size * preconditioner * g
                + jnp.sqrt(step_size * preconditioner) * noise
            )

        parameters = jax.tree.map(move, parameters, gradient, second_moment, keys)
        return parameters, second_moment, loss

    return step


# The sample buffer: every leaf of the parameter tree gains a leading axis of
# slots, of which the first ``retained`` hold the retained samples in order.
# Writes donate the buffer, so retaining a sample costs one parameter tree,
# not a copy of the buffer; whatever leaves the method (a posterior, a state)
# is therefore a copy, since the next write invalidates the buffer it read.


@partial(jax.jit, donate_argnums=0)
def _write(buffer: PyTree, sample: PyTree, slot: int) -> PyTree:
    """Write ``sample`` into the buffer's ``slot``; the slot is traced, not static."""
    return jax.tree.map(lambda leaf, value: leaf.at[slot].set(value), buffer, sample)


@jax.jit
def _snapshot(buffer: PyTree, retained: int) -> tuple[PyTree, jax.Array]:
    """Copy every slot of the buffer, with the mask of its first ``retained``."""
    mask = jnp.arange(_num_slots(buffer)) < retained
    return jax.tree.map(jnp.copy, buffer), mask


def _doubled(buffer: PyTree) -> PyTree:
    """Double the buffer's slots; the new ones repeat the old, so stay finite."""
    return jax.tree.map(lambda leaf: jnp.concatenate([leaf, leaf]), buffer)


def _empty_buffer(like: PyTree, capacity: int) -> PyTree:
    """Return a buffer of ``capacity`` slots, each a copy of ``like``."""
    return jax.tree.map(
        lambda leaf: jnp.asarray(np.repeat(np.asarray(leaf)[None], capacity, axis=0)),
        like,
    )


def _retained_samples(buffer: PyTree, retained: int) -> PyTree:
    """Copy the retained samples out of the buffer, stacked; possibly none.

    The slice is taken on the host, so a new count compiles nothing.
    """
    return jax.tree.map(
        lambda leaf: jnp.asarray(np.asarray(leaf)[:retained].copy()), buffer
    )


def _buffer_of(stacked: PyTree, capacity: int) -> tuple[PyTree, int]:
    """Return a buffer holding stacked samples, and their number.

    The capacity doubles from ``capacity`` until the samples fit; spare slots
    repeat the last sample, or hold zeros when there is none.
    """
    retained = _num_slots(stacked)
    while capacity < retained:
        capacity *= 2

    def fill(leaf: jax.Array) -> jax.Array:
        samples = np.asarray(leaf)
        spare = capacity - retained
        if retained == 0:
            padding = np.zeros((spare, *samples.shape[1:]), dtype=samples.dtype)
        else:
            padding = np.repeat(samples[-1:], spare, axis=0)
        return jnp.asarray(np.concatenate([samples, padding]))

    return jax.tree.map(fill, stacked), retained


def _num_slots(stacked: PyTree) -> int:
    """Return the length of the leading axis shared by a stacked tree's leaves."""
    return int(jax.tree.leaves(stacked)[0].shape[0])
