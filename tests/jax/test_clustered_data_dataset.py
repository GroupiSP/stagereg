"""Tests for the clustered-data example's data, grid and loader."""

from __future__ import annotations

import dataclasses
from types import ModuleType

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")


def _truth(x: np.ndarray) -> np.ndarray:
    """The UvA tutorial's noise-free sum-of-sines curve."""
    return x + 0.3 * np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x)


def test_training_data_match_the_uva_tutorial(clustered_dataset: ModuleType) -> None:
    config = clustered_dataset.ClusteredDataConfig()
    data = clustered_dataset.make_clustered_data(config)

    x = np.hstack([np.linspace(-0.2, 0.2, 500), np.linspace(0.6, 1.0, 500)])
    eps = 0.02 * np.random.RandomState(42).randn(1000)
    y = x + 0.3 * np.sin(2 * np.pi * (x + eps)) + 0.3 * np.sin(4 * np.pi * (x + eps))
    y = y + eps
    assert data.train_inputs.shape == data.train_targets.shape == (1000, 1)
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
    assert not np.allclose(grid_noise[:1000], train_noise)


@pytest.mark.parametrize(
    "overrides",
    [
        {"noise_std": -0.1},
        {"batch_size": 0},
        {"posterior_epochs": 0},
        {"num_draws": 0},
        {"prior_precision": 0.0},
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

    assert len(first) == 10
    assert all(batch.inputs.shape == (100, 1) for batch in first)
    assert not jnp.array_equal(first[0].inputs, second[0].inputs)
    np.testing.assert_allclose(
        np.sort(np.concatenate([np.asarray(b.inputs[:, 0]) for b in first])),
        np.sort(np.asarray(data.train_inputs[:, 0])),
    )
