from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from stagereg.core.types import (
    Batch,
    CheckpointRef,
    ParameterRole,
    StageState,
    TrainingState,
)


def test_batch_is_immutable() -> None:
    batch = Batch(inputs=[1, 2], targets=[3], metadata={"split": "train"})

    with pytest.raises(FrozenInstanceError):
        batch.targets = [4]  # type: ignore[misc]

    assert batch.metadata == {"split": "train"}


def test_training_states_do_not_share_mutable_defaults() -> None:
    first = TrainingState()
    second = TrainingState()
    first.parameter_roles["mean"] = ParameterRole.MEAN
    first.checkpoint_registry["best"] = CheckpointRef("best-model")

    assert second.parameter_roles == {}
    assert second.checkpoint_registry == {}


def test_training_state_separates_lifecycle_from_active_stage() -> None:
    state = TrainingState()

    state.lifecycle_state = StageState.INITIALIZED
    state.active_stage = "mean"

    assert state.lifecycle_state is StageState.INITIALIZED
    assert state.active_stage == "mean"


def test_training_state_takes_the_active_stage_as_its_only_stage_label() -> None:
    state = TrainingState(active_stage="mean")

    assert state.active_stage == "mean"
    with pytest.raises(TypeError, match="stage"):
        TrainingState(stage="mean")  # type: ignore[call-arg]


@pytest.mark.parametrize("field", ["outer_iteration", "data_fingerprint"])
def test_training_state_has_no_unpopulated_fields(field: str) -> None:
    with pytest.raises(TypeError, match=field):
        TrainingState(**{field: None})


def test_training_state_allows_idempotent_component_registration() -> None:
    state = TrainingState()
    component = object()
    optimizer = object()

    state.register_component("mean", component)
    state.register_component("mean", component)
    state.register_optimizer("mean", optimizer)
    state.register_optimizer("mean", optimizer)

    assert state.model_components == {"mean": component}
    assert state.optimizer_states == {"mean": optimizer}


def test_training_state_rejects_conflicting_component_registration() -> None:
    state = TrainingState()
    state.register_component("mean", object())
    state.register_optimizer("mean", object())

    with pytest.raises(ValueError, match="model component 'mean'"):
        state.register_component("mean", object())
    with pytest.raises(ValueError, match="optimizer state 'mean'"):
        state.register_optimizer("mean", object())


def test_importing_the_core_imports_no_jax() -> None:
    probe = "import sys, stagereg.core; print('jax' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    )

    assert result.stdout.strip() == "False"
