"""The clustered-data example's configuration, data and loader.

The training data reproduce the "Simulate data" section of the UvA DL2
Bayesian neural network tutorial 1: two clusters of 500 inputs on
`[-0.2, 0.2]` and `[0.6, 1.0]`, with the sum-of-sines target

    y = x + 0.3 sin(2 pi (x + eps)) + 0.3 sin(4 pi (x + eps)) + eps,

where `eps ~ N(0, noise_std^2)` is drawn from NumPy's legacy generator
seeded with 42. The evaluation grid has 3,000 points over `[-0.5, 1.5]`,
with fresh noisy targets drawn the same way from a different seed.

The data never change across run seeds: `ClusteredDataConfig.seed` only
varies the models' initialization, the batch order and the draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import jax
import jax.numpy as jnp
import numpy as np
from regions import CLUSTERS

from stagereg.core.protocols import LoaderFactory
from stagereg.core.types import Batch

CLUSTER_SIZE = 500
"""Number of training inputs in each cluster."""

TRAIN_SEED = 42
"""Seed of the training noise, as in the UvA tutorial."""

GRID_SEED = 7
"""Seed of the evaluation grid's noise; differs from `TRAIN_SEED`."""

GRID_BOUNDS = (-0.5, 1.5)
"""The evaluation grid's input range."""

GRID_SIZE = 3_000
"""Number of points on the evaluation grid."""


@dataclass(frozen=True)
class ClusteredDataConfig:
    """Every setting of the clustered-data comparison.

    Attributes:
        noise_std: Standard deviation of the target noise, for training and
            grid targets alike.
        batch_size: Number of training points per batch.
        learning_rate: Adam learning rate of the mean and variance stages.
        mean_epochs: Maximum number of mean-stage epochs.
        mean_patience: Epochs without a lower training loss before the
            mean stage stops early and restores its best epoch.
        variance_epochs: Maximum number of variance-stage epochs.
        variance_patience: Epochs without a lower training loss before the
            variance stage stops early and restores its best epoch.
        posterior_epochs: Fixed number of posterior-stage epochs.
        num_draws: Number of posterior draws every configuration is scored
            on.
        prior_precision: Precision of the isotropic Gaussian prior over the
            network's weights.
        bbb_learning_rate: Adam learning rate of Bayes by Backprop.
        bbb_initial_std: Initial standard deviation of Bayes by Backprop's
            mean-field Gaussian.
        seed: Seed of the models' initialization, batch order and draws.
    """

    noise_std: float = 0.02
    batch_size: int = 100
    learning_rate: float = 1e-3
    mean_epochs: int = 3_000
    mean_patience: int = 100
    variance_epochs: int = 3_000
    variance_patience: int = 100
    posterior_epochs: int = 2_000
    num_draws: int = 100
    prior_precision: float = 1.0
    bbb_learning_rate: float = 1e-2
    bbb_initial_std: float = 1e-2
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If a count is not positive, a patience is negative,
                or a scale or rate is not finite and positive (the noise
                level may be zero).
        """
        counts = {
            "batch_size": self.batch_size,
            "mean_epochs": self.mean_epochs,
            "variance_epochs": self.variance_epochs,
            "posterior_epochs": self.posterior_epochs,
            "num_draws": self.num_draws,
        }
        for name, value in counts.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        for name in ("mean_patience", "variance_patience"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative.")
        rates = {
            "learning_rate": self.learning_rate,
            "prior_precision": self.prior_precision,
            "bbb_learning_rate": self.bbb_learning_rate,
            "bbb_initial_std": self.bbb_initial_std,
        }
        for name, value in rates.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")

    @property
    def train_size(self) -> int:
        """Number of training points over both clusters."""
        return len(CLUSTERS) * CLUSTER_SIZE

    @property
    def steps_per_epoch(self) -> int:
        """Number of update steps (batches) per epoch."""
        return ceil(self.train_size / self.batch_size)


@dataclass(frozen=True)
class ClusteredData:
    """The training set and the evaluation grid.

    Attributes:
        train_inputs: Training inputs, shape `[1000, 1]`.
        train_targets: Noisy training targets, shape `[1000, 1]`.
        grid_inputs: Evaluation-grid inputs, shape `[3000, 1]`.
        grid_truth: The noise-free curve on the grid, shape `[3000]`.
        grid_targets: Fresh noisy targets on the grid, shape `[3000]`.
    """

    train_inputs: jax.Array
    train_targets: jax.Array
    grid_inputs: jax.Array
    grid_truth: jax.Array
    grid_targets: jax.Array


def true_function(inputs: np.ndarray) -> np.ndarray:
    """Return the noise-free sum-of-sines curve at the inputs.

    Args:
        inputs: Inputs of any shape.

    Returns:
        `x + 0.3 sin(2 pi x) + 0.3 sin(4 pi x)`, shaped like the inputs.
    """
    return _noisy_function(inputs, np.zeros_like(inputs))


def _noisy_function(inputs: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Return the UvA tutorial's target, with the noise inside the sines too."""
    shifted = inputs + noise
    return (
        inputs
        + 0.3 * np.sin(2 * np.pi * shifted)
        + 0.3 * np.sin(4 * np.pi * shifted)
        + noise
    )


def _noisy_targets(inputs: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    """Draw the tutorial's noisy targets with NumPy's legacy generator."""
    noise = noise_std * np.random.RandomState(seed).randn(inputs.size)
    return _noisy_function(inputs, noise)


def make_clustered_data(config: ClusteredDataConfig) -> ClusteredData:
    """Generate the training clusters and the evaluation grid.

    Args:
        config: The comparison's settings; only `noise_std` is read.

    Returns:
        The training set and the evaluation grid, as float32 arrays.
    """
    train_inputs = np.hstack(
        [np.linspace(low, high, CLUSTER_SIZE) for low, high in CLUSTERS]
    )
    grid_inputs = np.linspace(*GRID_BOUNDS, GRID_SIZE)
    return ClusteredData(
        train_inputs=jnp.asarray(train_inputs[:, None], dtype=jnp.float32),
        train_targets=jnp.asarray(
            _noisy_targets(train_inputs, config.noise_std, TRAIN_SEED)[:, None],
            dtype=jnp.float32,
        ),
        grid_inputs=jnp.asarray(grid_inputs[:, None], dtype=jnp.float32),
        grid_truth=jnp.asarray(true_function(grid_inputs), dtype=jnp.float32),
        grid_targets=jnp.asarray(
            _noisy_targets(grid_inputs, config.noise_std, GRID_SEED),
            dtype=jnp.float32,
        ),
    )


def make_loader(data: ClusteredData, config: ClusteredDataConfig) -> LoaderFactory:
    """Build a loader that reshuffles the training set every epoch.

    Args:
        data: The data whose training set to serve.
        config: The comparison's settings; `batch_size` and `seed` are read.

    Returns:
        A loader serving the shuffled training set in batches for any split
        it is asked for: the example has no held-out split.
    """

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        permutation = jax.random.permutation(
            jax.random.fold_in(jax.random.key(config.seed), epoch), config.train_size
        )
        inputs = data.train_inputs[permutation]
        targets = data.train_targets[permutation]
        return [
            Batch(
                inputs=inputs[start : start + config.batch_size],
                targets=targets[start : start + config.batch_size],
            )
            for start in range(0, config.train_size, config.batch_size)
        ]

    return loader
