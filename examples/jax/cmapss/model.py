"""1D-convolutional staged and joint RUL models for windowed CMAPSS data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from stagereg.jax import Gamma, GammaHead, Gaussian, GaussianHead


class Cnn1DBackbone(nnx.Module):
    """Two 1D-convolutional layers over the time axis, sensors as channels.

    Dropout follows every hidden layer, i.e. each convolutional layer, with
    one mask per call shared across the batch axis (`broadcast_dims=(0,)`),
    as [`MCDropout`][stagereg.jax.MCDropout] requires: a draw is then the
    same function at every window. Dropout adds no parameters, so the
    parameter tree does not depend on the rate. At rate 0 the dropout layers
    are the identity and hold no random state, so the backbone, its state
    and its outputs are those of a backbone without dropout.
    """

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the stacked 1D-convolutional feature extractor.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of output channels for both
                convolutional layers.
            kernel_size: Convolution kernel width along the time axis.
            dropout_rate: Dropout rate after every convolutional layer; 0
                disables dropout.
            rngs: NNX random-number generators used to initialize parameters
                and, at a positive rate, the dropout masks' streams.
        """
        self.conv1 = nnx.Conv(
            in_features=n_sensors,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=hidden_channels,
            out_features=hidden_channels,
            kernel_size=kernel_size,
            padding="SAME",
            rngs=rngs,
        )
        # Built after every parameter, and only at a positive rate, so that a
        # rate of 0 draws nothing from `rngs` and leaves the state unchanged.
        dropout_rngs = rngs if dropout_rate > 0.0 else None
        self.dropouts = nnx.List(
            [
                nnx.Dropout(dropout_rate, broadcast_dims=(0,), rngs=dropout_rngs)
                for _ in range(2)
            ]
        )

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Extract a fixed-size feature vector per windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the dropout masks are drawn from; `None` draws
                them from the backbone's own streams.

        Returns:
            Time-averaged convolutional features, shape ``(batch,
            hidden_channels)``.
        """
        hidden = inputs
        for index, (conv, dropout) in enumerate(
            zip((self.conv1, self.conv2), self.dropouts, strict=True)
        ):
            hidden = nnx.relu(conv(hidden))
            key = (
                None if dropout_key is None else jax.random.fold_in(dropout_key, index)
            )
            hidden = dropout(hidden, rngs=key)
        return jnp.mean(hidden, axis=1)


class Cnn1DMeanModel(nnx.Module):
    """Deterministic point-RUL regressor over windowed CMAPSS sensor data."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the convolutional backbone and the mean output head.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of channels produced by the
                convolutional backbone.
            kernel_size: Convolution kernel width along the time axis.
            dropout_rate: The backbone's dropout rate; 0 disables dropout.
            rngs: NNX random-number generators used to initialize parameters
                and, at a positive rate, the dropout masks' streams.
        """
        self.backbone = Cnn1DBackbone(
            n_sensors,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.output = nnx.Linear(hidden_channels, 1, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> jax.Array:
        """Predict a scalar RUL for each windowed example in the batch.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the backbone's dropout masks are drawn from;
                `None` draws them from the backbone's own streams.

        Returns:
            Predicted RUL, shape ``(batch, 1)``.
        """
        return self.output(self.backbone(inputs, dropout_key=dropout_key))


class Cnn1DGammaModel(nnx.Module):
    """Gamma residual regressor over windowed CMAPSS sensor data.

    Independently initialized from ``Cnn1DMeanModel``, sharing only the
    same backbone architecture family, and trained on Stage-1's squared
    residuals through [`GammaVarianceStage`][stagereg.jax.GammaVarianceStage].
    """

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the convolutional backbone and the Gamma output head.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of channels produced by the
                convolutional backbone.
            kernel_size: Convolution kernel width along the time axis.
            rngs: NNX random-number generators used to initialize parameters.
        """
        self.backbone = Cnn1DBackbone(
            n_sensors,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            rngs=rngs,
        )
        self.head = GammaHead(hidden_channels, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> Gamma:
        """Predict a Gamma distribution over the squared RUL residual.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            A [`Gamma`][stagereg.jax.Gamma] with positive concentration and
            rate, each shaped ``(batch, 1)``.
        """
        return self.head(self.backbone(inputs))


class CompositeGaussianModel(nnx.Module):
    """Composite predictive model combining the Stage-1/Stage-2 outputs.

    Combines the frozen Stage-1 point prediction with the Stage-2 Gamma
    mean (an estimate of the squared-residual, i.e. aleatoric variance)
    into a single [`Gaussian`][stagereg.jax.Gaussian] predictive distribution.
    """

    def __init__(
        self, mean_model: Cnn1DMeanModel, variance_model: Cnn1DGammaModel
    ) -> None:
        """Store the trained Stage-1 mean and Stage-2 Gamma variance models.

        Args:
            mean_model: Trained, frozen Stage-1 point-RUL regressor.
            variance_model: Trained Stage-2 Gamma residual regressor.
        """
        self.mean_model = mean_model
        self.variance_model = variance_model

    def __call__(self, inputs: jax.Array) -> Gaussian:
        """Predict a Gaussian RUL distribution for each windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            A [`Gaussian`][stagereg.jax.Gaussian] with ``loc`` from Stage 1 and a
            strictly positive ``scale`` derived from the Stage-2 Gamma mean,
            each shaped ``(batch, 1)``.
        """
        loc = self.mean_model(inputs)
        variance = self.variance_model(inputs).mean()
        return Gaussian(loc=loc, scale=jnp.sqrt(variance))


class Cnn1DGaussianModel(nnx.Module):
    """Joint model: one 1D-CNN backbone predicting a Gaussian RUL distribution.

    The textbook mean-variance baseline the staged model is compared against:
    a single `Cnn1DBackbone`, with the same architecture and hyperparameters
    as each staged model's backbone, followed by a
    [`GaussianHead`][stagereg.jax.GaussianHead] that predicts mean and scale
    together. Having one backbone rather than two, it has roughly half the
    staged model's parameters.
    """

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the convolutional backbone and the Gaussian output head.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of channels produced by the
                convolutional backbone.
            kernel_size: Convolution kernel width along the time axis.
            dropout_rate: The backbone's dropout rate; 0 disables dropout.
            rngs: NNX random-number generators used to initialize parameters
                and, at a positive rate, the dropout masks' streams.
        """
        self.backbone = Cnn1DBackbone(
            n_sensors,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.head = GaussianHead(hidden_channels, 1, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, *, dropout_key: jax.Array | None = None
    ) -> Gaussian:
        """Predict a Gaussian RUL distribution for each windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.
            dropout_key: Key the backbone's dropout masks are drawn from;
                `None` draws them from the backbone's own streams.

        Returns:
            A [`Gaussian`][stagereg.jax.Gaussian] with ``loc`` and a strictly
            positive ``scale``, each shaped ``(batch, 1)``.
        """
        return self.head(self.backbone(inputs, dropout_key=dropout_key))
