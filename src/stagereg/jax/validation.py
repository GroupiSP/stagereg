"""JAX validation strategy implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax
from flax import nnx

from stagereg.core.protocols import LoaderFactory
from stagereg.core.types import TrainingState, ValidationResult
from stagereg.jax.evaluation import (
    SupervisedLoss,
    _build_evaluation_step,
    _evaluate_with_step,
)
from stagereg.jax.metrics import MetricSuite


@dataclass(frozen=True)
class HeldOutValidation:
    """Evaluate an NNX model against a held-out validation loader.

    The evaluation step is built and compiled once per strategy and reused
    every epoch, so validation compiles nothing after the first epoch.

    Attributes:
        model: The NNX module to evaluate.
        loader: Factory producing the validation batch loader for a
            given split and epoch.
        loss: The supervised loss function used for evaluation.
        metrics: Registered batch/epoch metrics for validation.
        metadata: Arbitrary caller-supplied metadata attached to this
            strategy.
    """

    model: nnx.Module
    loader: LoaderFactory
    loss: SupervisedLoss
    metrics: MetricSuite = field(default_factory=MetricSuite)
    metadata: dict[str, object] = field(default_factory=dict)
    _evaluation_step: Callable[..., Mapping[str, jax.Array]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Build the evaluation step every epoch reuses."""
        object.__setattr__(
            self, "_evaluation_step", _build_evaluation_step(self.metrics, self.loss)
        )

    def __call__(self, state: TrainingState, *, epoch: int) -> ValidationResult:
        """Evaluate the held-out split after a training epoch.

        Args:
            state: The current training state, whose ``rng_state`` is
                advanced in place as validation batches are consumed.
            epoch: The epoch just completed, passed to ``loader`` to
                select the validation batches for this epoch.

        Returns:
            A [`ValidationResult`][stagereg.core.ValidationResult] with bare
            metric names and ``passed=True``.

        Raises:
            TypeError: If ``state.rng_state`` is not a JAX random key.
        """
        if not isinstance(state.rng_state, jax.Array):
            raise TypeError("state.rng_state must be a JAX random key.")
        metrics, state.rng_state = _evaluate_with_step(
            self.model,
            self.loader(split="validation", epoch=epoch),
            key=state.rng_state,
            metrics=self.metrics,
            evaluation_step=self._evaluation_step,
            has_loss=True,
        )
        return ValidationResult(passed=True, metrics=metrics, message=None)
