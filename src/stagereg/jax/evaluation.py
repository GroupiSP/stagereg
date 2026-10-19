"""Shared NNX evaluation primitives used by training runners and validation strategies.

This module has no dependency on any specific training runner, so both
[`stagereg.jax.supervised`][stageregjaxsupervised] and
[`stagereg.jax.validation`][stageregjaxvalidation] may depend on it without
either depending on the other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
from flax import nnx

from stagereg.core.types import Batch, PyTree
from stagereg.jax.metrics import (
    BatchMetricSpec,
    MetricSuite,
    StepScalar,
    collect_step_metrics,
    initialize_batch_metric_values,
    maybe_collect_epoch_prediction_data,
    reduce_metric_suite,
)
from stagereg.jax.rng import split_key


class SupervisedLoss(Protocol):
    """A callable computing a supervised loss for an NNX model.

    Implementations compute a scalar loss from a model, a batch of inputs/targets/sample
    weights, a PRNG key (e.g. for stochastic layers such as dropout), and a flag
    indicating whether the call happens during training (as opposed to evaluation).
    """

    def __call__(
        self,
        model: nnx.Module,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
        training: bool,
        /,
    ) -> jax.Array:
        """Compute the scalar supervised loss for a batch.

        Args:
            model: The NNX module to evaluate.
            inputs: The batch inputs.
            targets: The batch targets.
            sample_weight: Per-sample weights for the batch.
            key: A JAX PRNG key, e.g. for stochastic layers.
            training: Whether the loss is being computed during training
                (as opposed to evaluation).

        Returns:
            The scalar loss value.
        """
        ...


def make_evaluation_step(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec] = (),
) -> Callable[..., Mapping[str, jax.Array]]:
    """Create a JIT-compiled NNX supervised evaluation step.

    Args:
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.
        metrics: Registered JAX-native batch metrics.

    Returns:
        A JIT-compiled function ``evaluate_step(model, inputs, targets,
        sample_weight, key)`` returning a mapping containing ``"loss"``
        plus one scalar value per registered batch metric.
    """

    @nnx.jit
    def evaluate_step(
        model: nnx.Module,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> Mapping[str, jax.Array]:
        loss_value = loss(model, inputs, targets, sample_weight, key, False)
        values: dict[str, jax.Array] = {"loss": loss_value}
        for spec in metrics:
            values[spec.name] = spec.metric(
                model,
                inputs,
                targets,
                sample_weight,
                key,
                False,
            )
        return values

    return evaluate_step


def _make_metrics_step(
    metrics: Sequence[BatchMetricSpec] = (),
) -> Callable[..., Mapping[str, jax.Array]]:
    """Create a JIT-compiled NNX evaluation step that computes no loss."""

    @nnx.jit
    def metrics_step(
        model: nnx.Module,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> Mapping[str, jax.Array]:
        return {
            spec.name: spec.metric(model, inputs, targets, sample_weight, key, False)
            for spec in metrics
        }

    return metrics_step


def evaluate_loader(
    model: nnx.Module,
    loader: Iterable[Batch],
    *,
    key: jax.Array,
    metrics: MetricSuite | None = None,
    loss: SupervisedLoss | None = None,
) -> tuple[dict[str, float], jax.Array]:
    """Evaluate a loader and return reduced metrics and the advanced random key.

    The evaluation step is built here from ``metrics`` and ``loss``, so a
    caller that only wants to score a loader against metrics need not
    invent a loss to do it.

    Args:
        model: The NNX module to evaluate.
        loader: An iterable of batches to evaluate.
        key: The JAX PRNG key to use, advanced once per batch.
        metrics: Optional registered batch/epoch metrics for evaluation.
            When omitted, no metrics are collected.
        loss: Optional supervised loss to average across batches. When
            omitted, the returned mapping has no ``"loss"`` key.

    Returns:
        A tuple ``(metrics, key)`` where ``metrics`` contains the
        registered metrics plus ``"loss"`` if ``loss`` was given, and
        ``key`` is advanced past all consumed batches.
    """
    metric_suite = metrics if metrics is not None else MetricSuite()
    return PreparedEvaluation.of(metric_suite, loss)(model, loader, key=key)


@dataclass(frozen=True)
class PreparedEvaluation:
    """[`evaluate_loader`][stagereg.jax.evaluate_loader] with its step built once.

    ``nnx.jit`` compiles each new step function afresh, so a caller
    evaluating every epoch, such as
    [`HeldOutValidation`][stagereg.jax.HeldOutValidation], prepares the
    evaluation once and keeps it. It is internal to the JAX backend, not
    exported (ADR 0002 keeps the evaluation step private).

    Attributes:
        metrics: The registered batch/epoch metrics.
        has_loss: Whether the step computes a loss.
        step: The jitted evaluation step.
    """

    metrics: MetricSuite
    has_loss: bool
    step: Callable[..., Mapping[str, jax.Array]]

    @classmethod
    def of(
        cls, metrics: MetricSuite, loss: SupervisedLoss | None
    ) -> PreparedEvaluation:
        """Build the evaluation step for ``metrics`` and ``loss``.

        Args:
            metrics: The suite whose batch metrics the step computes.
            loss: The loss the step also computes, if any.

        Returns:
            The prepared evaluation.
        """
        step = (
            _make_metrics_step(metrics.batch)
            if loss is None
            else make_evaluation_step(loss, metrics=metrics.batch)
        )
        return cls(metrics=metrics, has_loss=loss is not None, step=step)

    def __call__(
        self, model: nnx.Module, loader: Iterable[Batch], *, key: jax.Array
    ) -> tuple[dict[str, float], jax.Array]:
        """Evaluate a loader, as [`evaluate_loader`][stagereg.jax.evaluate_loader].

        Args:
            model: The NNX module to evaluate.
            loader: An iterable of batches to evaluate.
            key: The JAX PRNG key to use, advanced once per batch.

        Returns:
            The reduced metrics and the advanced key.
        """
        losses: list[StepScalar] | None = [] if self.has_loss else None
        batch_metric_values = initialize_batch_metric_values(self.metrics.batch)
        epoch_metric_parts = [] if self.metrics.epoch else None

        for batch in loader:
            key, batch_key = split_key(key)
            step_output = self.step(
                model,
                batch.inputs,
                batch.targets,
                batch.sample_weight,
                batch_key,
            )
            collect_step_metrics(
                step_output,
                metrics=self.metrics.batch,
                losses=losses,
                batch_metric_values=batch_metric_values,
                context="evaluation step",
            )
            maybe_collect_epoch_prediction_data(
                epoch_metric_parts,
                suite=self.metrics,
                model=model,
                batch=batch,
                batch_key=batch_key,
            )

        return (
            reduce_metric_suite(
                suite=self.metrics,
                losses=losses,
                batch_metric_values=batch_metric_values,
                epoch_metric_parts=epoch_metric_parts,
            ),
            key,
        )
