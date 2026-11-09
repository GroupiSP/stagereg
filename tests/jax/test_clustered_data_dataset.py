"""Tests for the clustered-data example's data, grid and loader."""

from __future__ import annotations

import dataclasses
from types import ModuleType

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

jnp = pytest.importorskip("jax.numpy")


def _truth(x: np.ndarray) -> np.ndarray:
    """The noise-free sum-of-sines curve."""
    return x + 0.3 * np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x)


def test_training_data_carry_heteroscedastic_noise(
    clustered_dataset: ModuleType,
) -> None:
    config = clustered_dataset.ClusteredDataConfig()
    data = clustered_dataset.make_clustered_data(config)

    x = np.hstack([np.linspace(-0.2, 0.2, 200), np.linspace(0.6, 1.0, 200)])
    eps = 0.02 * np.random.RandomState(42).randn(400)
    y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps))
    y = y + 0.1 * eps / (np.abs(x) + 0.01)
    assert data.train_inputs.shape == data.train_targets.shape == (400, 1)
    np.testing.assert_allclose(np.asarray(data.train_inputs[:, 0]), x, atol=1e-6)
    np.testing.assert_allclose(np.asarray(data.train_targets[:, 0]), y, atol=1e-5)


def test_without_noise_the_targets_lie_on_the_truth(
    clustered_dataset: ModuleType,
) -> None:
    config = dataclasses.replace(clustered_dataset.ClusteredDataConfig(), noise_std=0.0)
    data = clustered_dataset.make_clustered_data(config)

    grid = np.asarray(data.grid_inputs[:, 0])
    np.testing.assert_allclose(grid, np.linspace(-0.5, 1.5, 3000), atol=1e-6)
    np.testing.assert_allclose(np.asarray(data.grid_truth), _truth(grid), atol=1e-5)
    np.testing.assert_allclose(np.asarray(data.grid_targets), _truth(grid), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(data.train_targets[:, 0]),
        _truth(np.asarray(data.train_inputs[:, 0])),
        atol=1e-5,
    )


def test_grid_targets_carry_fresh_noise(clustered_dataset: ModuleType) -> None:
    config = clustered_dataset.ClusteredDataConfig()
    data = clustered_dataset.make_clustered_data(config)
    train_noise = np.asarray(data.train_targets[:, 0]) - _truth(
        np.asarray(data.train_inputs[:, 0])
    )
    grid_noise = np.asarray(data.grid_targets) - np.asarray(data.grid_truth)

    assert data.grid_targets.shape == data.grid_truth.shape == (3000,)
    assert not np.allclose(grid_noise, 0.0)
    assert not np.allclose(grid_noise[: train_noise.size], train_noise)


@pytest.mark.parametrize(
    "overrides",
    [
        {"noise_std": -0.1},
        {"batch_size": 0},
        {"mean_epochs": 0},
        {"variance_epochs": 0},
        {"posterior_epochs": 0},
        {"num_draws": 0},
        {"prior_precision": 0.0},
        {"mc_dropout_rate": 0.0},
        {"mc_dropout_rate": 1.0},
        {"mc_dropout_learning_rate": 0.0},
        {"psgld_step_size": 0.0},
        {"psgld_chains": 0},
        # 3 chains cannot split 100 draws evenly.
        {"psgld_chains": 3},
        # The final posterior epoch's 4 steps cannot hold a lone chain's 5 draws.
        {"posterior_epochs": 1, "num_draws": 5, "psgld_chains": 1},
        # Nor can the joint run's final mean epoch.
        {"mean_epochs": 1, "num_draws": 5, "psgld_chains": 1},
    ],
)
def test_config_rejects_invalid_settings(
    clustered_dataset: ModuleType, overrides: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        clustered_dataset.ClusteredDataConfig(**overrides)


def test_loader_reshuffles_every_epoch_into_full_batches(
    clustered_dataset: ModuleType,
) -> None:
    config = clustered_dataset.ClusteredDataConfig()
    data = clustered_dataset.make_clustered_data(config)
    loader = clustered_dataset.make_loader(data, config)

    first, second = loader(split="train", epoch=0), loader(split="train", epoch=1)

    assert len(first) == 4
    assert all(batch.inputs.shape == (100, 1) for batch in first)
    assert not jnp.array_equal(first[0].inputs, second[0].inputs)
    np.testing.assert_allclose(
        np.sort(np.concatenate([np.asarray(b.inputs[:, 0]) for b in first])),
        np.sort(np.asarray(data.train_inputs[:, 0])),
    )


def test_loader_serves_standardized_targets(clustered_dataset: ModuleType) -> None:
    config = clustered_dataset.ClusteredDataConfig()
    data = clustered_dataset.make_clustered_data(config)
    loader = clustered_dataset.make_loader(data, config)

    targets = np.concatenate(
        [np.asarray(b.targets[:, 0]) for b in loader(split="train", epoch=0)]
    )

    np.testing.assert_allclose(targets.mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(targets.std(), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        np.sort(data.target_mean + data.target_std * targets),
        np.sort(np.asarray(data.train_targets[:, 0])),
        atol=1e-5,
    )


def test_psgld_schedule_spreads_the_draws_over_the_final_half_at_the_defaults(
    clustered_dataset: ModuleType,
) -> None:
    config = clustered_dataset.ClusteredDataConfig()

    staged = config.psgld_schedule(config.posterior_epochs)
    joint = config.psgld_schedule(config.mean_epochs)

    # 2,000 posterior epochs of 4 steps: the final 4,000 steps hold 100 draws.
    assert (staged.burn_in, staged.thinning) == (4_000, 40)
    # 3,000 joint epochs of 4 steps: the final 6,000 steps hold 100 draws.
    assert (joint.burn_in, joint.thinning) == (6_000, 60)
    # Each of the 5 staged chains holds its 20 draws in the same final 4,000.
    chain = config.psgld_schedule(config.posterior_epochs, 20)
    assert (chain.burn_in, chain.thinning) == (4_000, 200)


@settings(deadline=None, max_examples=50)
@given(
    epochs=st.integers(min_value=1, max_value=60),
    batch_size=st.sampled_from([50, 100, 300, 1000]),
    num_draws=st.integers(min_value=1, max_value=40),
)
def test_psgld_schedule_retains_every_draw_in_the_final_half_of_the_run(
    clustered_dataset: ModuleType,
    epochs: int,
    batch_size: int,
    num_draws: int,
) -> None:
    try:
        config = clustered_dataset.ClusteredDataConfig(
            batch_size=batch_size,
            num_draws=num_draws,
            psgld_chains=1,
            mean_epochs=epochs,
            posterior_epochs=epochs,
        )
    except ValueError:
        return  # too few steps in the final half for the draws
    schedule = config.psgld_schedule(epochs)

    # pSGLD retains the position after step t when t = burn_in + k * thinning, k >= 1.
    steps = config.steps_per_epoch
    total = epochs * steps
    retained = [
        t
        for t in range(1, total + 1)
        if t > schedule.burn_in and (t - schedule.burn_in) % schedule.thinning == 0
    ]
    assert len(retained) == num_draws
    assert retained[0] > epochs // 2 * steps
    assert retained[-1] == total


def test_every_prng_stream_has_its_own_id(clustered_dataset: ModuleType) -> None:
    streams = clustered_dataset.PRNG_STREAMS

    assert len(set(streams.values())) == len(streams)
    assert set(streams) == {
        "shuffle",
        "supervised",
        "staged_posterior_init",
        "staged_draws",
        "joint_init",
        "joint_train",
        "joint_draws",
    }
