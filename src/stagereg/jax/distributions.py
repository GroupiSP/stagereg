"""A concrete JAX-backed Gaussian predictive distribution and head.

This module binds
[`PredictiveDistribution`][stagereg.core.PredictiveDistribution] and
[`DistributionHead`][stagereg.core.DistributionHead] to JAX arrays and an NNX
linear head, following the same Tier 2 (JAX backend) placement as
[`stagereg.jax.state`][stageregjaxstate] and
[`stagereg.jax.evaluation`][stageregjaxevaluation].

Every distribution here is a JAX pytree whose leaves are its array fields, so it
can be passed into and returned from jitted functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from statistics import NormalDist
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.stats as jstats
from flax import nnx

# Halvings of the quantile bracket; enough to reach float32 resolution.
_BISECTION_STEPS = 64

_Distribution = TypeVar("_Distribution")


def _register_pytree(cls: type[_Distribution]) -> type[_Distribution]:
    """Register a frozen distribution dataclass as a JAX pytree.

    Every dataclass field is a child array; none is static (ADR 0012).
    Unflattening bypasses ``__init__`` and ``__post_init__``, because JAX
    rebuilds pytrees with placeholder leaves (tracers, ``None``, axis specs)
    that validation must not reject.

    Args:
        cls: The frozen dataclass to register.

    Returns:
        ``cls`` itself, so the function works as a class decorator.
    """
    names = tuple(item.name for item in fields(cls))  # type: ignore[arg-type]

    def flatten(
        distribution: _Distribution,
    ) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, Any], ...], None]:
        children = tuple(
            (jax.tree_util.GetAttrKey(name), getattr(distribution, name))
            for name in names
        )
        return children, None

    def unflatten(_: None, children: Any) -> _Distribution:
        distribution = object.__new__(cls)
        for name, child in zip(names, children, strict=True):
            object.__setattr__(distribution, name, child)
        return distribution

    jax.tree_util.register_pytree_with_keys(cls, flatten, unflatten)
    return cls


@_register_pytree
@dataclass(frozen=True)
class Gaussian:
    """A JAX-backed Gaussian predictive distribution parametrized by scale.

    Attributes:
        loc: The distribution mean, shaped like the model's output.
        scale: The distribution's positive standard deviation, broadcastable
            against ``loc``. Callers (e.g.
            [`GaussianHead`][stagereg.jax.GaussianHead]) are responsible for
            ensuring positivity, e.g. via a ``softplus`` transform of an
            unconstrained raw output.
    """

    loc: jax.Array
    scale: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The broadcast shape of independent Gaussian components."""
        return jnp.broadcast_shapes(self.loc.shape, self.scale.shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of a single Gaussian event, always scalar."""
        return ()

    def log_prob(self, targets: jax.Array) -> jax.Array:
        """Compute the elementwise Gaussian log-density of ``targets``.

        Args:
            targets: Target values broadcastable against ``loc``/``scale``.

        Returns:
            The elementwise log-density, shaped like the broadcast of
            ``targets`` against ``loc`` and ``scale``.
        """
        return jstats.norm.logpdf(targets, loc=self.loc, scale=self.scale)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw reparametrized samples from this distribution.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading sample dimensions prepended to
                ``batch_shape``.

        Returns:
            Samples shaped ``sample_shape + batch_shape``.
        """
        shape = sample_shape + self.batch_shape
        noise = jax.random.normal(key, shape)
        return self.loc + self.scale * noise

    def mean(self) -> jax.Array:
        """Return the distribution mean, i.e. ``loc``."""
        return self.loc

    def variance(self) -> jax.Array:
        """Return the elementwise variance, i.e. ``scale ** 2``."""
        return jnp.square(self.scale)


@_register_pytree
@dataclass(frozen=True)
class Gamma:
    """A JAX-backed Gamma distribution parametrized by shape and rate.

    Attributes:
        concentration: Positive Gamma shape parameter.
        rate: Positive Gamma rate parameter, i.e. the inverse scale.
    """

    concentration: jax.Array
    rate: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Return the broadcast shape of independent Gamma components."""
        return jnp.broadcast_shapes(self.concentration.shape, self.rate.shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Return the shape of a single Gamma event, always scalar."""
        return ()

    def log_prob(self, targets: jax.Array) -> jax.Array:
        """Compute the elementwise Gamma log-density of ``targets``.

        Args:
            targets: Positive target values broadcastable against the
                concentration and rate.

        Returns:
            Elementwise log-density values.
        """
        return (
            self.concentration * jnp.log(self.rate)
            - jsp.gammaln(self.concentration)
            + (self.concentration - 1.0) * jnp.log(targets)
            - self.rate * targets
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw keyed samples from the Gamma distribution.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading sample dimensions prepended to
                ``batch_shape``.

        Returns:
            Samples shaped ``sample_shape + batch_shape``.
        """
        shape = sample_shape + self.batch_shape
        concentration = jnp.broadcast_to(self.concentration, self.batch_shape)
        rate = jnp.broadcast_to(self.rate, self.batch_shape)
        return jax.random.gamma(key, concentration, shape=shape) / rate

    def mean(self) -> jax.Array:
        """Return the Gamma mean, i.e. ``concentration / rate``."""
        return self.concentration / self.rate

    def variance(self) -> jax.Array:
        """Return the Gamma variance, i.e. ``concentration / rate ** 2``."""
        return self.concentration / jnp.square(self.rate)


@_register_pytree
@dataclass(frozen=True)
class MomentMatchedPredictive:
    """The Gaussian with a posterior predictive's mean and variance.

    A named summary of a [`PosteriorPredictive`][stagereg.jax.PosteriorPredictive],
    not a substitute for it: its variance is split into the aleatoric and the
    epistemic part. Obtain one from
    [`PosteriorPredictive.moment_matched`][stagereg.jax.PosteriorPredictive.moment_matched].

    Attributes:
        loc: The predictive mean, the average of the draws' means.
        aleatoric_variance: The aleatoric variance shared by every draw.
        epistemic_variance: The variance of the draws' means around ``loc``.
    """

    loc: jax.Array
    aleatoric_variance: jax.Array
    epistemic_variance: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The broadcast shape of the mean and both variance parts."""
        return jnp.broadcast_shapes(
            self.loc.shape,
            self.aleatoric_variance.shape,
            self.epistemic_variance.shape,
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of a single predictive sample, always scalar."""
        return ()

    def _gaussian(self) -> Gaussian:
        """Return the equivalent scale-parametrized Gaussian."""
        return Gaussian(loc=self.loc, scale=jnp.sqrt(self.variance()))

    def log_prob(self, targets: jax.Array) -> jax.Array:
        """Compute the elementwise Gaussian log-density of ``targets``.

        Args:
            targets: Target values broadcastable against ``batch_shape``.

        Returns:
            The elementwise log-density under the moment-matched Gaussian.
        """
        return self._gaussian().log_prob(targets)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw predictive samples from the moment-matched Gaussian.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading sample dimensions prepended to
                ``batch_shape``.

        Returns:
            Predictive samples shaped ``sample_shape + batch_shape``.
        """
        return jnp.broadcast_to(
            self._gaussian().sample(key, sample_shape),
            sample_shape + self.batch_shape,
        )

    def mean(self) -> jax.Array:
        """Return the predictive mean, i.e. ``loc``."""
        return self.loc

    def variance(self) -> jax.Array:
        """Return the aleatoric plus the epistemic variance."""
        return self.aleatoric_variance + self.epistemic_variance


@_register_pytree
@dataclass(frozen=True)
class PosteriorPredictive:
    """The posterior predictive: an equally weighted mixture of Gaussians over draws.

    Each component is a Gaussian centred on one draw's mean with the aleatoric
    variance, so the mixture is exact for the draws taken and may be skewed or
    multimodal. Its single-Gaussian summary is
    [`moment_matched`][stagereg.jax.PosteriorPredictive.moment_matched].

    Attributes:
        draws: The mean of each draw at the predicted inputs, shaped
            ``[S, *batch]`` with ``S >= 1`` draws on the leading axis.
        aleatoric_variance: The positive aleatoric variance, broadcastable
            against ``draws[0]`` and shared by every draw.
    """

    draws: jax.Array
    aleatoric_variance: jax.Array

    def __post_init__(self) -> None:
        """Validate the leading draw axis.

        Raises:
            ValueError: If ``draws`` has no leading draw axis or no draws.
        """
        if jnp.ndim(self.draws) == 0 or jnp.shape(self.draws)[0] == 0:
            raise ValueError("draws must have a non-empty leading draw axis.")

    @property
    def num_draws(self) -> int:
        """The number of draws ``S`` the mixture averages over."""
        return self.draws.shape[0]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The broadcast shape of the draws' means and the aleatoric variance."""
        return jnp.broadcast_shapes(self.draws.shape[1:], self.aleatoric_variance.shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The shape of a single predictive sample, always scalar."""
        return ()

    def _components(self) -> tuple[jax.Array, jax.Array]:
        """Return the draws' means and the shared scale, aligned on ``batch_shape``.

        The means are shaped ``(S,) + batch_shape`` and the scale ``batch_shape``, so
        broadcasting against them never crosses the leading draw axis.
        """
        missing = len(self.batch_shape) - (self.draws.ndim - 1)
        draws = jnp.expand_dims(self.draws, tuple(range(1, 1 + missing)))
        loc = jnp.broadcast_to(draws, (self.num_draws, *self.batch_shape))
        scale = jnp.broadcast_to(jnp.sqrt(self.aleatoric_variance), self.batch_shape)
        return loc, scale

    def log_prob(self, targets: jax.Array) -> jax.Array:
        """Compute the exact elementwise mixture log-density of ``targets``.

        Args:
            targets: Target values broadcastable against ``batch_shape``.

        Returns:
            ``logsumexp`` over draws of the component log-densities, minus
            ``log S``.
        """
        loc, scale = self._components()
        component = jstats.norm.logpdf(targets, loc=loc, scale=scale)
        return jsp.logsumexp(component, axis=0) - jnp.log(self.num_draws)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Draw predictive samples, each from a uniformly chosen draw's Gaussian.

        The Gaussian noise uses ``key`` exactly as
        [`Gaussian.sample`][stagereg.jax.Gaussian.sample] does, so with one draw
        the samples equal that Gaussian's.

        Args:
            key: A JAX PRNG key.
            sample_shape: Leading sample dimensions prepended to
                ``batch_shape``.

        Returns:
            Predictive samples shaped ``sample_shape + batch_shape``.
        """
        shape = sample_shape + self.batch_shape
        noise = jax.random.normal(key, shape)
        component = jax.random.randint(
            jax.random.fold_in(key, 1), shape, 0, self.num_draws
        )
        means = jnp.broadcast_to(
            jnp.moveaxis(self.draws, 0, -1), shape + (self.num_draws,)
        )
        loc = jnp.take_along_axis(means, component[..., None], axis=-1)[..., 0]
        return loc + jnp.sqrt(self.aleatoric_variance) * noise

    def mean(self) -> jax.Array:
        """Return the mixture mean, the average of the draws' means."""
        return jnp.broadcast_to(jnp.mean(self.draws, axis=0), self.batch_shape)

    def variance(self) -> jax.Array:
        """Return the mixture variance, aleatoric plus epistemic."""
        return self.moment_matched().variance()

    def cdf(self, targets: jax.Array) -> jax.Array:
        """Compute the elementwise mixture cumulative distribution of ``targets``.

        Args:
            targets: Target values broadcastable against ``batch_shape``.

        Returns:
            The average over draws of the component Gaussian CDFs.
        """
        loc, scale = self._components()
        component = jstats.norm.cdf(targets, loc=loc, scale=scale)
        return jnp.mean(component, axis=0)

    def quantile(self, probability: float) -> jax.Array:
        """Compute the elementwise mixture quantile at ``probability``.

        The mixture CDF has no closed-form inverse, so the quantile is found
        by bisection between the smallest and the largest of the draws'
        Gaussian quantiles, which bracket it.

        Args:
            probability: The cumulative probability, strictly between 0 and 1.

        Returns:
            The quantile, shaped like ``batch_shape``.

        Raises:
            ValueError: If ``probability`` is not strictly between 0 and 1.
        """
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be strictly between 0 and 1.")
        return _mixture_quantile(self, probability, NormalDist().inv_cdf(probability))

    def moment_matched(self) -> MomentMatchedPredictive:
        """Summarize the mixture by the Gaussian with its mean and variance.

        By the law of total variance, the mixture variance is the aleatoric
        variance plus the epistemic variance of the draws' means.

        Returns:
            The [`MomentMatchedPredictive`][stagereg.jax.MomentMatchedPredictive],
            with both variance parts broadcast to ``batch_shape``.
        """
        return MomentMatchedPredictive(
            loc=self.mean(),
            aleatoric_variance=jnp.broadcast_to(
                self.aleatoric_variance, self.batch_shape
            ),
            epistemic_variance=jnp.broadcast_to(
                jnp.var(self.draws, axis=0), self.batch_shape
            ),
        )


@jax.jit
def _mixture_quantile(
    predictive: PosteriorPredictive, probability: jax.Array, standard: jax.Array
) -> jax.Array:
    """Invert the mixture CDF at ``probability`` by bisection, compiled once per shape.

    The draws' Gaussian quantiles bracket the mixture quantile, and the bracket is
    then halved ``_BISECTION_STEPS`` times. ``probability`` and its standard normal
    quantile ``standard`` are traced, so every probability level shares one compile.

    Args:
        predictive: The mixture to invert.
        probability: The cumulative probability, strictly between 0 and 1.
        standard: The standard normal quantile at ``probability``.

    Returns:
        The quantile, shaped like the mixture's ``batch_shape``.
    """
    loc, scale = predictive._components()
    component = loc + standard * scale
    bracket = (jnp.min(component, axis=0), jnp.max(component, axis=0))

    def bisect(
        _: int, bracket: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bracket
        middle = 0.5 * (low + high)
        below = predictive.cdf(middle) < probability
        return jnp.where(below, middle, low), jnp.where(below, high, middle)

    lower, upper = jax.lax.fori_loop(0, _BISECTION_STEPS, bisect, bracket)
    return 0.5 * (lower + upper)


class GaussianHead(nnx.Module):
    """An NNX head producing a [`Gaussian`][stagereg.jax.Gaussian] from model features.

    A single linear layer maps ``in_features`` to ``2 * out_features``
    outputs, split into an unconstrained location and an unconstrained scale
    that is passed through ``softplus`` (plus ``eps``) to guarantee a
    strictly positive standard deviation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the head's linear layer.

        Args:
            in_features: Number of input feature dimensions.
            out_features: Number of predicted target dimensions.
            rngs: NNX RNG collection used to initialize parameters.
            eps: A small positive constant added to the ``softplus``-mapped
                scale to keep it strictly positive.

        Raises:
            ValueError: If ``out_features`` or ``eps`` is invalid.
        """
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite.")
        self.out_features = out_features
        self.eps = eps
        self.linear = nnx.Linear(in_features, 2 * out_features, rngs=rngs)

    def __call__(self, features: jax.Array) -> Gaussian:
        """Produce a [`Gaussian`][stagereg.jax.Gaussian] prediction from ``features``.

        Args:
            features: Model features, shaped ``(..., in_features)``.

        Returns:
            A [`Gaussian`][stagereg.jax.Gaussian] with ``loc``/``scale`` shaped
            ``(..., out_features)``.
        """
        raw_loc, raw_scale = jnp.split(self.linear(features), 2, axis=-1)
        scale = jax.nn.softplus(raw_scale) + self.eps
        return Gaussian(loc=raw_loc, scale=scale)


class GammaHead(nnx.Module):
    """An NNX head producing a shape/rate [`Gamma`][stagereg.jax.Gamma] distribution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the Gamma parameter projection.

        Args:
            in_features: Number of input feature dimensions.
            out_features: Number of predicted target dimensions.
            rngs: NNX RNG collection used to initialize parameters.
            eps: Positive finite offset added after ``softplus``.

        Raises:
            ValueError: If ``out_features`` or ``eps`` is invalid.
        """
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite.")
        self.out_features = out_features
        self.eps = eps
        self.linear = nnx.Linear(in_features, 2 * out_features, rngs=rngs)

    def __call__(self, features: jax.Array) -> Gamma:
        """Produce a Gamma prediction from model features.

        Args:
            features: Model features shaped ``(..., in_features)``.

        Returns:
            A Gamma distribution with positive concentration and rate shaped
            ``(..., out_features)``.
        """
        raw_concentration, raw_rate = jnp.split(
            self.linear(features),
            2,
            axis=-1,
        )
        concentration = jax.nn.softplus(raw_concentration) + self.eps
        rate = jax.nn.softplus(raw_rate) + self.eps
        return Gamma(concentration=concentration, rate=rate)
