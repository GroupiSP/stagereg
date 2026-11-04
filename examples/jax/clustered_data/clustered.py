"""The clustered-data example's configuration, data and loader.

The training data adapt the "Simulate data" section of the UvA DL2
Bayesian neural network tutorial 1: two clusters of 200 inputs on
`[-0.2, 0.2]` and `[0.6, 1.0]`, with the heteroscedastic sum-of-sines target

    y = x + 0.3 sin(2 pi (x + eps)) + 0.3 sin(4 pi (x + eps)) + 0.1 eps / |x|,

where `eps ~ N(0, noise_std^2)` is drawn from NumPy's legacy generator
seeded with 42. The additive noise grows as `x` approaches 0; no training
or grid input is exactly 0. The evaluation grid has 3,000 points over `[-0.5, 1.5]`,
with fresh noisy targets drawn the same way from a different seed.

The data never change across run seeds: `ClusteredDataConfig.seed` only
varies the models' initialization, the batch order and the draws. Each of
those purposes draws from its own named PRNG stream (`PRNG_STREAMS`), a
`fold_in` of the seed's key, so no two of them, joint or staged, share a
key.

As in VeBNN, every model trains on standardized targets: the loader serves
`(y - target_mean) / target_std`, with the training targets' mean and
population standard deviation, and each configuration maps its predictions
back to the targets' units before it is scored.
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

CLUSTER_SIZE = 200
"""Number of training inputs in each cluster."""

TRAIN_SEED = 42
"""Seed of the training noise, as in the UvA tutorial."""

GRID_SEED = 7
"""Seed of the evaluation grid's noise; differs from `TRAIN_SEED`."""

GRID_BOUNDS = (-0.5, 1.5)
"""The evaluation grid's input range."""

GRID_SIZE = 3_000
"""Number of points on the evaluation grid."""

SHUFFLE_STREAM = 0
"""PRNG stream of the loader's per-epoch shuffles, folded in with the epoch."""

SUPERVISED_STREAM = 1
"""PRNG stream of the staged mean and variance stages' initializations and state."""

STAGED_POSTERIOR_INIT_STREAM = 2
"""PRNG stream of the staged posterior network's initialization and dropout."""

STAGED_DRAW_STREAM = 3
"""PRNG stream of the staged posterior's draws on the grid."""

JOINT_INIT_STREAM = 4
"""PRNG stream of the joint network's initialization and dropout."""

JOINT_TRAIN_STREAM = 5
"""PRNG stream of the joint run's per-step update keys."""

JOINT_DRAW_STREAM = 6
"""PRNG stream of the joint posterior's draws on the grid."""

PRNG_STREAMS: dict[str, int] = {
    "shuffle": SHUFFLE_STREAM,
    "supervised": SUPERVISED_STREAM,
    "staged_posterior_init": STAGED_POSTERIOR_INIT_STREAM,
    "staged_draws": STAGED_DRAW_STREAM,
    "joint_init": JOINT_INIT_STREAM,
    "joint_train": JOINT_TRAIN_STREAM,
    "joint_draws": JOINT_DRAW_STREAM,
}
"""Every PRNG stream of a run, by purpose; each has its own id."""


@dataclass(frozen=True)
class ClusteredDataConfig:
    """Every setting of the clustered-data comparison.

    Attributes:
        noise_std: Standard deviation of the target noise, for training and
            grid targets alike.
        batch_size: Number of training points per batch.
        learning_rate: Adam learning rate of the mean and variance stages.
        mean_epochs: Number of mean-stage epochs, and of every joint run's
            epochs: the joint network trains as long as the staged model's
            mean network.
        variance_epochs: Number of variance-stage epochs.
        posterior_epochs: Number of posterior-stage epochs.
        num_draws: Number of posterior draws every configuration is scored
            on.
        prior_precision: Precision of the isotropic Gaussian prior over the
            network's weights: every method's posterior prior, joint or
            staged, and the mean and variance stages' L2 penalty. The
            default 0.25 is a prior standard deviation of 2.
        bbb_learning_rate: Adam learning rate of Bayes by Backprop.
        bbb_initial_std: Initial standard deviation of Bayes by Backprop's
            mean-field Gaussian.
        mc_dropout_rate: The backbone's dropout rate under MC dropout; the
            other methods run it without dropout.
        mc_dropout_learning_rate: Adam learning rate of MC dropout.
        psgld_step_size: Langevin step size of pSGLD.
        seed: Seed of the models' initialization, batch order and draws.
    """

    noise_std: float = 0.02
    batch_size: int = 100
    learning_rate: float = 1e-3
    mean_epochs: int = 3_000
    variance_epochs: int = 3_000
    posterior_epochs: int = 2_000
    num_draws: int = 100
    prior_precision: float = 0.25
    bbb_learning_rate: float = 1e-2
    bbb_initial_std: float = 1e-2
    mc_dropout_rate: float = 0.1
    mc_dropout_learning_rate: float = 1e-3
    psgld_step_size: float = 3e-3
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: If a count is not positive, a scale or rate is not
                finite and positive (the noise level may be zero), the
                dropout rate is not in `(0, 1)`, or the final half of the
                posterior epochs or of the joint run's epochs has fewer
                update steps than `num_draws`.
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
        if not isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative.")
        rates = {
            "learning_rate": self.learning_rate,
            "prior_precision": self.prior_precision,
            "bbb_learning_rate": self.bbb_learning_rate,
            "bbb_initial_std": self.bbb_initial_std,
            "mc_dropout_learning_rate": self.mc_dropout_learning_rate,
            "psgld_step_size": self.psgld_step_size,
        }
        for name, value in rates.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}.")
        if not 0.0 < self.mc_dropout_rate < 1.0:
            raise ValueError(
                f"mc_dropout_rate must lie in (0, 1), got {self.mc_dropout_rate}."
            )
        self.psgld_schedule(self.posterior_epochs)
        self.psgld_schedule(self.mean_epochs)

    @property
    def train_size(self) -> int:
        """Number of training points over both clusters."""
        return len(CLUSTERS) * CLUSTER_SIZE

    @property
    def steps_per_epoch(self) -> int:
        """Number of update steps (batches) per epoch."""
        return ceil(self.train_size / self.batch_size)

    def psgld_schedule(self, epochs: int) -> PsgldSchedule:
        """Place pSGLD's draws evenly over the final half of a run's epochs.

        The run's last update step retains the last draw, and `num_draws`
        draws are retained in all, `thinning` steps apart.

        Args:
            epochs: Epochs the pSGLD run trains: `posterior_epochs` for the
                staged model's posterior stage, `mean_epochs` for a joint run.

        Returns:
            The burn-in and thinning, in update steps.

        Raises:
            ValueError: If the final half of the run has fewer update steps
                than `num_draws`.
        """
        final_half_steps = (epochs - epochs // 2) * self.steps_per_epoch
        thinning = final_half_steps // self.num_draws
        if thinning < 1:
            raise ValueError(
                f"num_draws ({self.num_draws}) exceeds the update steps of the "
                f"final half of {epochs} epochs ({final_half_steps})."
            )
        return PsgldSchedule(
            burn_in=epochs * self.steps_per_epoch - self.num_draws * thinning,
            thinning=thinning,
        )


@dataclass(frozen=True)
class PsgldSchedule:
    """Where a pSGLD run retains its draws, in update steps (batches).

    Attributes:
        burn_in: Update steps before the first draw can be retained.
        thinning: Update steps between two retained draws.
    """

    burn_in: int
    thinning: int


@dataclass(frozen=True)
class ClusteredData:
    """The training set and the evaluation grid.

    Attributes:
        train_inputs: Training inputs, shape `[400, 1]`.
        train_targets: Noisy training targets, shape `[400, 1]`.
        grid_inputs: Evaluation-grid inputs, shape `[3000, 1]`.
        grid_truth: The noise-free curve on the grid, shape `[3000]`.
        grid_targets: Fresh noisy targets on the grid, shape `[3000]`.
    """

    train_inputs: jax.Array
    train_targets: jax.Array
    grid_inputs: jax.Array
    grid_truth: jax.Array
    grid_targets: jax.Array

    @property
    def target_mean(self) -> float:
        """Mean of the training targets, which the loader subtracts."""
        return float(jnp.mean(self.train_targets))

    @property
    def target_std(self) -> float:
        """Population standard deviation of the training targets.

        The loader divides the centered targets by it.
        """
        return float(jnp.std(self.train_targets))

    def standardize(self, targets: jax.Array) -> jax.Array:
        """Map targets to the standardized units the models train in.

        Args:
            targets: Targets in their own units, of any shape.

        Returns:
            `(targets - target_mean) / target_std`.
        """
        return (targets - self.target_mean) / self.target_std


def true_function(inputs: np.ndarray) -> np.ndarray:
    """Return the noise-free sum-of-sines curve at the inputs.

    Args:
        inputs: Inputs of any shape.

    Returns:
        `x + 0.3 sin(2 pi x) + 0.3 sin(4 pi x)`, shaped like the inputs.
    """
    return inputs + 0.3 * np.sin(2 * np.pi * inputs) + 0.3 * np.sin(4 * np.pi * inputs)


def _noisy_function(inputs: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Return the heteroscedastic target, with the noise inside the sines too."""
    shifted = inputs + noise
    return (
        inputs
        + 0.3 * np.sin(2 * np.pi * shifted)
        + 0.3 * np.sin(4 * np.pi * shifted)
        + 0.1 * noise / np.abs(inputs)
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


def seed_key(config: ClusteredDataConfig, stream: int) -> jax.Array:
    """Return the key of one PRNG stream, derived from `config.seed`.

    Args:
        config: The comparison's settings; `seed` is read.
        stream: The stream's id, one of `PRNG_STREAMS`' values.

    Returns:
        The seed's key with the stream folded in.
    """
    return jax.random.fold_in(jax.random.key(config.seed), stream)


def make_loader(data: ClusteredData, config: ClusteredDataConfig) -> LoaderFactory:
    """Build a loader that reshuffles the training set every epoch.

    Args:
        data: The data whose training set to serve.
        config: The comparison's settings; `batch_size` and `seed` are read.

    Returns:
        A loader serving the shuffled training set, with standardized
        targets, in batches for any split it is asked for: the example has
        no held-out split.
    """

    standardized_targets = data.standardize(data.train_targets)

    def loader(*, split: str, epoch: int) -> list[Batch]:
        del split
        permutation = jax.random.permutation(
            jax.random.fold_in(seed_key(config, SHUFFLE_STREAM), epoch),
            config.train_size,
        )
        inputs = data.train_inputs[permutation]
        targets = standardized_targets[permutation]
        return [
            Batch(
                inputs=inputs[start : start + config.batch_size],
                targets=targets[start : start + config.batch_size],
            )
            for start in range(0, config.train_size, config.batch_size)
        ]

    return loader
