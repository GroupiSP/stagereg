"""The clustered-data example's shared backbone and the networks built on it.

Every configuration uses one Tanh MLP trunk with `HIDDEN_LAYERS` hidden
layers of `HIDDEN_FEATURES` units, with an optional dropout rate. The
trunk first maps the training-input range `INPUT_RANGE` onto `[-1, 1]`
with a fixed affine map, as the xsin example does, so every network sees
inputs of order one, identically. Dropout
draws one mask per call, shared across the batch (`broadcast_dims=(0,)`),
as MC dropout requires; at rate 0 it is the identity and adds no
parameters, so networks with and without dropout share one parameter tree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from regions import CLUSTERS

from stagereg.jax import Gamma, GammaHead

HIDDEN_LAYERS = 5
"""Number of hidden layers of the trunk."""

HIDDEN_FEATURES = 10
"""Number of units of each hidden layer."""


INPUT_RANGE: tuple[float, float] = (CLUSTERS[0][0], CLUSTERS[-1][1])
"""The training inputs' range, which the trunk maps onto `[-1, 1]`."""


def normalize_inputs(inputs: jax.Array) -> jax.Array:
    """Map `INPUT_RANGE` affinely onto `[-1, 1]`.

    Args:
        inputs: Raw inputs, any shape.

    Returns:
        The normalized inputs, shaped like `inputs`; points outside the
        training range land outside `[-1, 1]`.
    """
    low, high = INPUT_RANGE
    return 2.0 * (inputs - low) / (high - low) - 1.0


class Trunk(nnx.Module):
    """The shared Tanh MLP trunk, with optional dropout after each layer."""

    def __init__(self, *, dropout_rate: float = 0.0, rngs: nnx.Rngs) -> None:
        """Build the trunk.

        Args:
            dropout_rate: Dropout rate after every hidden layer; 0 disables
                dropout.
            rngs: Randomness for the weights and the dropout masks.
        """
        self.layers = nnx.List(
            [
                nnx.Linear(
                    1 if index == 0 else HIDDEN_FEATURES, HIDDEN_FEATURES, rngs=rngs
                )
                for index in range(HIDDEN_LAYERS)
            ]
        )
        self.dropouts = nnx.List(
            [
                nnx.Dropout(dropout_rate, broadcast_dims=(0,), rngs=rngs)
                for _ in range(HIDDEN_LAYERS)
            ]
        )

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Map normalized inputs to the last hidden layer's features.

        Args:
            inputs: Raw inputs, shape `[B, 1]`; the trunk normalizes them
                with `normalize_inputs`.
            dropout_key: Key the dropout masks are drawn from; `None` draws
                them from the trunk's own stream.

        Returns:
            Hidden features, shape `[B, HIDDEN_FEATURES]`.
        """
        hidden = normalize_inputs(inputs)
        for index, (layer, dropout) in enumerate(
            zip(self.layers, self.dropouts, strict=True)
        ):
            hidden = jnp.tanh(layer(hidden))
            key = (
                None if dropout_key is None else jax.random.fold_in(dropout_key, index)
            )
            hidden = dropout(hidden, rngs=key)
        return hidden


class MeanNetwork(nnx.Module):
    """The trunk with a single linear output: the staged model's mean network."""

    def __init__(self, *, dropout_rate: float = 0.0, rngs: nnx.Rngs) -> None:
        """Build the network.

        Args:
            dropout_rate: The trunk's dropout rate.
            rngs: Randomness for the weights and the dropout masks.
        """
        self.trunk = Trunk(dropout_rate=dropout_rate, rngs=rngs)
        self.output = nnx.Linear(HIDDEN_FEATURES, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Predict the mean.

        Args:
            inputs: Inputs, shape `[B, 1]`.

        Returns:
            Predicted means, shape `[B, 1]`.
        """
        return self.output(self.trunk(inputs))


class GammaNetwork(nnx.Module):
    """The trunk with a Gamma head: the staged model's variance network."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Build the network.

        Args:
            rngs: Randomness for the weights.
        """
        self.trunk = Trunk(rngs=rngs)
        self.head = GammaHead(HIDDEN_FEATURES, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> Gamma:
        """Predict a Gamma distribution over the squared residual.

        Args:
            inputs: Inputs, shape `[B, 1]`.

        Returns:
            The Gamma distribution, whose mean is the aleatoric variance.
        """
        return self.head(self.trunk(inputs))


VARIANCE_FLOOR = 1e-6
"""Smallest variance the joint network's variance head predicts."""


class JointNetwork(nnx.Module):
    """The trunk with a mean head and a variance head: the joint model's network."""

    def __init__(self, *, dropout_rate: float = 0.0, rngs: nnx.Rngs) -> None:
        """Build the network.

        Args:
            dropout_rate: The trunk's dropout rate.
            rngs: Randomness for the weights and the dropout masks.
        """
        self.trunk = Trunk(dropout_rate=dropout_rate, rngs=rngs)
        self.mean_head = nnx.Linear(HIDDEN_FEATURES, 1, rngs=rngs)
        self.variance_head = nnx.Linear(HIDDEN_FEATURES, 1, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Predict the mean and the aleatoric variance.

        Args:
            inputs: Inputs, shape `[B, 1]`.
            dropout_key: Key the trunk's dropout masks are drawn from; `None`
                draws them from the trunk's own stream.

        Returns:
            Means and variances stacked on the last axis, shape `[B, 2]`. The
            variance is a softplus plus `VARIANCE_FLOOR`.
        """
        hidden = self.trunk(inputs, dropout_key=dropout_key)
        variance = jax.nn.softplus(self.variance_head(hidden)) + VARIANCE_FLOOR
        return jnp.concatenate([self.mean_head(hidden), variance], axis=-1)
