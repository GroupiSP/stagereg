"""The clustered-data example's shared backbone and the networks built on it.

Every configuration uses one Tanh MLP trunk with `HIDDEN_LAYERS` hidden
layers of `HIDDEN_FEATURES` units, with an optional dropout rate. Dropout
draws one mask per call, shared across the batch (`broadcast_dims=(0,)`),
as MC dropout requires; at rate 0 it is the identity and adds no
parameters, so networks with and without dropout share one parameter tree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from stagereg.jax import Gamma, GammaHead

HIDDEN_LAYERS = 5
"""Number of hidden layers of the trunk."""

HIDDEN_FEATURES = 10
"""Number of units of each hidden layer."""


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
        """Map inputs to the last hidden layer's features.

        Args:
            inputs: Inputs, shape `[B, 1]`.
            dropout_key: Key the dropout masks are drawn from; `None` draws
                them from the trunk's own stream.

        Returns:
            Hidden features, shape `[B, HIDDEN_FEATURES]`.
        """
        hidden = inputs
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
