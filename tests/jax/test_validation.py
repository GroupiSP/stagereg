"""Tests for the JAX validation strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import posterior_runs
import pytest

from stagereg.core.types import Batch, TrainingState

_RAGGED_SIZE = 13
_BATCH_SIZE = 8


def ragged_loader(*, split: str, epoch: int) -> list[Batch]:
    """Thirteen points in batches of eight, so the last batch holds five."""
    del epoch
    inputs, targets = posterior_runs.regression_data()
    offset = 0.25 if split == "validation" else 0.0
    return [
        Batch(
            inputs=inputs[start : min(start + _BATCH_SIZE, _RAGGED_SIZE)],
            targets=targets[start : min(start + _BATCH_SIZE, _RAGGED_SIZE)] + offset,
        )
        for start in range(0, _RAGGED_SIZE, _BATCH_SIZE)
    ]


@pytest.mark.parametrize(
    "loader",
    [posterior_runs.regression_loader, ragged_loader],
    ids=["even", "ragged"],
)
def test_held_out_validated_stages_compile_nothing_after_their_first_epoch(
    make_stages: Callable[..., Any],
    compile_counter: Callable[[], Any],
    loader: Any,
) -> None:
    with compile_counter() as counter:
        stages = make_stages(counter, loader=loader)
        state = TrainingState(rng_state=jax.random.key(1))
        stages.mean.prepare(state)
        stages.mean.train(state)
        stages.variance.prepare(state)
        stages.variance.train(state)

    for stage in ("mean", "variance"):
        assert counter.during_warmup(stage) > 0, stage
        assert counter.after_warmup(stage) == 0, stage
