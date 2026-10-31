"""1D-convolutional staged and joint RUL models for windowed CMAPSS data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from stagereg.jax import Gamma, GammaHead, Gaussian, GaussianHead


class Cnn1DBackbone(nnx.Module):
    """Two 1D-convolutional layers over the time axis, sensors as channels."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the stacked 1D-convolutional feature extractor.

        Args:
            n_sensors: Number of sensor channels in the windowed input.
            hidden_channels: Number of output channels for both
                convolutional layers.
            kernel_size: Convolution kernel width along the time axis.
            rngs: NNX random-number generators used to initialize parameters.
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

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Extract a fixed-size feature vector per windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            Time-averaged convolutional features, shape ``(batch,
            hidden_channels)``.
        """
        hidden = nnx.relu(self.conv1(inputs))
        hidden = nnx.relu(self.conv2(hidden))
        return jnp.mean(hidden, axis=1)


class Cnn1DMeanModel(nnx.Module):
    """Deterministic point-RUL regressor over windowed CMAPSS sensor data."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the convolutional backbone and the mean output head.

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
        self.output = nnx.Linear(hidden_channels, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Predict a scalar RUL for each windowed example in the batch.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            Predicted RUL, shape ``(batch, 1)``.
        """
        return self.output(self.backbone(inputs))


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
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the convolutional backbone and the Gaussian output head.

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
        self.head = GaussianHead(hidden_channels, 1, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> Gaussian:
        """Predict a Gaussian RUL distribution for each windowed example.

        Args:
            inputs: Windowed sensor readings, shape ``(batch, window_length,
                n_sensors)``.

        Returns:
            A [`Gaussian`][stagereg.jax.Gaussian] with ``loc`` and a strictly
            positive ``scale``, each shaped ``(batch, 1)``.
        """
        return self.head(self.backbone(inputs))
