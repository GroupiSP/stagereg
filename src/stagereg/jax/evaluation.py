"""Shared NNX evaluation primitives used by training runners and validation strategies.

This module has no dependency on any specific training runner, so both
[`stagereg.jax.supervised`][stageregjaxsupervised] and
[`stagereg.jax.validation`][stageregjaxvalidation] may depend on it without
either depending on the other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

import jax
from flax import nnx

from stagereg.core.types import Batch, PyTree
from stagereg.jax.metrics import (
    BatchMetricSpec,
    MetricSuite,
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
    return _evaluate_with_step(
        model,
        loader,
        key=key,
        metrics=metric_suite,
        evaluation_step=_build_evaluation_step(metric_suite, loss),
        has_loss=loss is not None,
    )


def _build_evaluation_step(
    metrics: MetricSuite, loss: SupervisedLoss | None
) -> Callable[..., Mapping[str, jax.Array]]:
    """Build the jitted step `evaluate_loader` runs for ``metrics`` and ``loss``.

    Each call builds a new function, which ``nnx.jit`` compiles afresh; a
    caller evaluating repeatedly builds it once and keeps it.

    Args:
        metrics: The suite whose batch metrics the step computes.
        loss: The loss the step also computes, if any.

    Returns:
        The jitted evaluation step.
    """
    if loss is None:
        return _make_metrics_step(metrics.batch)
    return make_evaluation_step(loss, metrics=metrics.batch)


def _evaluate_with_step(
    model: nnx.Module,
    loader: Iterable[Batch],
    *,
    key: jax.Array,
    metrics: MetricSuite,
    evaluation_step: Callable[..., Mapping[str, jax.Array]],
    has_loss: bool,
) -> tuple[dict[str, float], jax.Array]:
    """Run `evaluate_loader` with an evaluation step already built.

    Args:
        model: The NNX module to evaluate.
        loader: An iterable of batches to evaluate.
        key: The JAX PRNG key to use, advanced once per batch.
        metrics: The registered batch/epoch metrics.
        evaluation_step: The step `_build_evaluation_step` built for
            ``metrics`` and the loss.
        has_loss: Whether the step computes a loss.

    Returns:
        The reduced metrics and the advanced key, as `evaluate_loader`.
    """
    losses: list[float] | None = [] if has_loss else None
    batch_metric_values = initialize_batch_metric_values(metrics.batch)
    epoch_metric_parts = [] if metrics.epoch else None

    for batch in loader:
        key, batch_key = split_key(key)
        step_output = evaluation_step(
            model,
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            batch_key,
        )
        collect_step_metrics(
            step_output,
            metrics=metrics.batch,
            losses=losses,
            batch_metric_values=batch_metric_values,
            context="evaluation step",
        )
        maybe_collect_epoch_prediction_data(
            epoch_metric_parts,
            suite=metrics,
            model=model,
            batch=batch,
            batch_key=batch_key,
        )

    return (
        reduce_metric_suite(
            suite=metrics,
            losses=losses,
            batch_metric_values=batch_metric_values,
            epoch_metric_parts=epoch_metric_parts,
        ),
        key,
    )
