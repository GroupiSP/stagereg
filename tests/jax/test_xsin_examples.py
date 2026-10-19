from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import pytest
from flax import nnx

_XSIN_DIR = Path(__file__).parents[2] / "examples" / "jax" / "xsin"


def _load_xsin_module(name: str) -> ModuleType:
    """Load an XSin example module under its own name, as its scripts import it."""
    spec = importlib.util.spec_from_file_location(name, _XSIN_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the XSin {name} module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BENCHMARK = _load_xsin_module("benchmark")
_POSTERIOR_BENCHMARK = _load_xsin_module("posterior_benchmark")

XSinBackbone = _BENCHMARK.XSinBackbone
XSinConfig = _BENCHMARK.XSinConfig
make_xsin_data = _BENCHMARK.make_xsin_data
run_xsin_mve = _BENCHMARK.run_xsin_mve
run_xsin_two_step = _BENCHMARK.run_xsin_two_step
xsin_mean = _BENCHMARK.xsin_mean
xsin_variance = _BENCHMARK.xsin_variance


def test_xsin_data_is_deterministic_with_positive_variance() -> None:
    config = XSinConfig(train_size=32, evaluation_size=21, seed=4)

    first = make_xsin_data(config)
    second = make_xsin_data(config)

    assert jnp.array_equal(first.train_inputs, second.train_inputs)
    assert jnp.array_equal(first.train_targets, second.train_targets)
    assert bool(jnp.all(first.train_inputs > config.train_min))
    assert bool(jnp.all(first.train_inputs < config.train_max))
    assert first.evaluation_inputs[0, 0] == config.evaluation_min
    assert first.evaluation_inputs[-1, 0] == config.evaluation_max
    assert bool(jnp.any(first.evaluation_inputs < config.train_min))
    assert bool(jnp.any(first.evaluation_inputs > config.train_max))
    assert first.true_mean.shape == (21, 1)
    assert first.true_variance.shape == (21, 1)
    assert bool(jnp.all(first.true_variance > 0.0))
    assert jnp.allclose(first.true_mean, xsin_mean(first.evaluation_inputs))
    assert jnp.allclose(
        first.true_variance,
        xsin_variance(first.evaluation_inputs),
    )


def test_xsin_backbone_scales_inputs_from_the_training_domain() -> None:
    inputs = jnp.linspace(-5.0, 15.0, 21).reshape(-1, 1)
    backbone = XSinBackbone(8, train_domain=(0.0, 10.0), rngs=nnx.Rngs(0))
    shifted = XSinBackbone(8, train_domain=(100.0, 110.0), rngs=nnx.Rngs(0))

    assert jnp.allclose(backbone(inputs), shifted(inputs + 100.0), atol=1e-5)


def test_xsin_backbone_does_not_saturate_far_from_the_training_domain() -> None:
    backbone = XSinBackbone(8, train_domain=(0.0, 10.0), rngs=nnx.Rngs(0))
    far_inputs = jnp.array([[-1000.0], [1000.0]])

    assert float(jnp.max(jnp.abs(backbone(far_inputs)))) > 1.0


def test_xsin_methods_are_reproducible_and_fit_the_training_domain() -> None:
    config = XSinConfig(
        train_size=512,
        evaluation_size=101,
        batch_size=64,
        hidden_features=32,
        mve_epochs=300,
        mean_epochs=300,
        variance_epochs=300,
        learning_rate=0.01,
        seed=7,
    )
    data = make_xsin_data(config)

    mve = run_xsin_mve(data, config)
    duplicate_mve = run_xsin_mve(data, config)
    two_step = run_xsin_two_step(data, config)
    duplicate_two_step = run_xsin_two_step(data, config)

    assert two_step.mean.shape == mve.mean.shape == data.true_mean.shape
    assert two_step.variance.shape == mve.variance.shape == data.true_variance.shape
    assert jnp.array_equal(mve.mean, duplicate_mve.mean)
    assert jnp.array_equal(two_step.mean, duplicate_two_step.mean)
    assert all(
        math.isfinite(value)
        for result in (mve, two_step)
        for value in (
            result.mean_rmse,
            result.variance_rmse,
            result.interpolation_mean_rmse,
            result.interpolation_variance_rmse,
            result.extrapolation_mean_rmse,
            result.extrapolation_variance_rmse,
        )
    )
    # The noise variance grows with x, so it is smallest at the domain's left edge.
    smallest_noise_variance = float(xsin_variance(jnp.array(config.train_min)))
    for result in (mve, two_step):
        assert result.interpolation_mean_rmse < math.sqrt(smallest_noise_variance)
        assert result.interpolation_variance_rmse < smallest_noise_variance


run_xsin_posterior = _POSTERIOR_BENCHMARK.run_xsin_posterior
xsin_bayes_by_backprop = _POSTERIOR_BENCHMARK.xsin_bayes_by_backprop
xsin_psgld = _POSTERIOR_BENCHMARK.xsin_psgld
xsin_mc_dropout = _POSTERIOR_BENCHMARK.xsin_mc_dropout
xsin_mc_dropout_model = _POSTERIOR_BENCHMARK.xsin_mc_dropout_model


@pytest.mark.parametrize(
    ("make_method", "make_model"),
    [
        (xsin_bayes_by_backprop, None),
        (xsin_psgld, None),
        (xsin_mc_dropout, xsin_mc_dropout_model),
    ],
    ids=["bbb", "psgld", "mc_dropout"],
)
def test_xsin_posterior_is_reproducible_with_finite_scores(
    make_method: Callable[[Any], Any],
    make_model: Callable[[Any], nnx.Module] | None,
) -> None:
    config = XSinConfig(
        train_size=128,
        evaluation_size=41,
        batch_size=32,
        hidden_features=8,
        mean_epochs=5,
        variance_epochs=5,
        posterior_epochs=2,
        posterior_num_draws=4,
        psgld_burn_in=2,
        psgld_thinning=1,
        seed=3,
    )
    data = make_xsin_data(config)

    def run() -> Any:
        model = None if make_model is None else make_model(config)
        return run_xsin_posterior(data, config, make_method(config), model=model)

    first = run()
    second = run()

    assert first.mean.shape == data.true_mean.shape
    assert first.aleatoric_variance.shape == data.true_variance.shape
    assert first.epistemic_variance.shape == data.true_variance.shape
    assert bool(jnp.all(first.epistemic_variance >= 0.0))
    assert jnp.array_equal(first.mean, second.mean)
    assert jnp.array_equal(first.epistemic_variance, second.epistemic_variance)
    assert first.posterior_scores == second.posterior_scores
    assert all(
        math.isfinite(value)
        for scores in (first.posterior_scores, first.variance_stage_scores)
        for value in (
            scores.nll,
            scores.crps,
            scores.interpolation_nll,
            scores.interpolation_crps,
            scores.extrapolation_nll,
            scores.extrapolation_crps,
        )
    )


run_xsin_posteriors = _POSTERIOR_BENCHMARK.run_xsin_posteriors


def test_xsin_posteriors_share_supervised_stages_trained_once() -> None:
    config = XSinConfig(
        train_size=128,
        evaluation_size=41,
        batch_size=32,
        hidden_features=8,
        mean_epochs=3,
        variance_epochs=4,
        posterior_epochs=2,
        posterior_num_draws=4,
        psgld_burn_in=2,
        psgld_thinning=1,
        seed=3,
    )
    data = make_xsin_data(config)
    epoch_ends: Counter[str] = Counter()

    class _EpochEndCounter:
        def on_event(self, event: Any) -> None:
            if event.name == "epoch_end" and event.split == "train":
                epoch_ends[event.stage] += 1

    results = run_xsin_posteriors(data, config, event_sinks=(_EpochEndCounter(),))

    assert list(results) == ["Bayes by Backprop", "pSGLD", "MC-Dropout"]
    assert epoch_ends["mean"] == config.mean_epochs
    assert epoch_ends["variance"] == config.variance_epochs
    assert epoch_ends["posterior"] == len(results) * config.posterior_epochs
    variance_stages = [result.variance_stage for result in results.values()]
    for variance_stage in variance_stages[1:]:
        assert jnp.array_equal(variance_stage.mean, variance_stages[0].mean)
        assert jnp.array_equal(variance_stage.variance, variance_stages[0].variance)
    # Sharing the stages leaves each method's run as it was when it trained its own.
    alone = run_xsin_posterior(data, config, xsin_psgld(config))
    assert jnp.allclose(results["pSGLD"].mean, alone.mean)
    assert jnp.allclose(results["pSGLD"].epistemic_variance, alone.epistemic_variance)
