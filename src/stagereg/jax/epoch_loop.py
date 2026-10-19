"""The epoch loop shared by every stage, independent of what a batch step trains.

Internal: not exported from ``stagereg.jax``. The loop owns epochs, training
events, validation, early stopping and the best checkpoint; a stage supplies
only a per-batch [`BatchStep`][stagereg.jax.epoch_loop.BatchStep] and a
[`StateSnapshotter`][stagereg.jax.epoch_loop.StateSnapshotter] for the best
checkpoint. [`run_supervised`][stagereg.jax.run_supervised] drives it with an
NNX model, optimizer and loss; the posterior stage drives it with an
inference method's ``update(batch, key)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import jax

from stagereg.core.checkpoints import Checkpoint, CheckpointStore
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.metric_registry import EpochPredictionData
from stagereg.core.naming import Split, metric_tag
from stagereg.core.protocols import LoaderFactory, ValidationStrategy
from stagereg.core.tracking import Decision, EventSink, TrainingEvent
from stagereg.core.types import Batch, StageResult, TrainingState
from stagereg.jax.metrics import (
    MetricSuite,
    StepScalar,
    collect_step_metrics,
    initialize_batch_metric_values,
    metric_key,
    reduce_metric_suite,
    step_outputs_to_host,
)
from stagereg.jax.rng import split_key
from stagereg.jax.state import freeze_training_state


class BatchStep(Protocol):
    """Advance training by one batch."""

    def __call__(self, batch: Batch, key: jax.Array, /) -> Mapping[str, Any]:
        """Update the trained state in place on one batch.

        Args:
            batch: The training batch.
            key: A fresh PRNG key for this batch.

        Returns:
            A mapping holding ``"loss"`` plus the value of every registered
            batch metric and every declared diagnostic.
        """
        ...


@dataclass(frozen=True)
class StateSnapshot:
    """What the best checkpoint holds of the trained state.

    Attributes:
        parameters: Snapshot of the trained parameters, stored as the
            checkpoint's ``parameters``.
        optimizer_state: Snapshot of the optimizer, or of any other state needed
            to resume, stored as the checkpoint's ``optimizer_state``. Defaults
            to ``None``.
    """

    parameters: Any
    optimizer_state: Any | None = None


class StateSnapshotter(Protocol):
    """Snapshot the trained state for the best checkpoint."""

    def __call__(self) -> StateSnapshot:
        """Return an independent snapshot of the trained state.

        Returns:
            The snapshot the best checkpoint stores.
        """
        ...


class EpochPredictionCollector(Protocol):
    """Materialize one batch's prediction data for epoch metrics."""

    def __call__(self, batch: Batch, key: jax.Array, /) -> EpochPredictionData:
        """Predict on ``batch`` before the step updates the trained state.

        Args:
            batch: The training batch.
            key: A metric-only PRNG key derived from the batch key.

        Returns:
            The batch's host-resident prediction data.
        """
        ...


def resolve_checkpoint_key(checkpoint_key: str | None, stage: str) -> str:
    """Return the checkpoint key a stage saves its best checkpoint under.

    Internal: not exported from ``stagereg.jax``. It is the single definition
    of the default key shared by the epoch loop,
    [`run_supervised`][stagereg.jax.run_supervised],
    [`MeanStage`][stagereg.jax.MeanStage] and
    [`GammaVarianceStage`][stagereg.jax.GammaVarianceStage].

    Args:
        checkpoint_key: An explicitly configured checkpoint key, or ``None``.
        stage: The stage name scoping the default key.

    Returns:
        ``checkpoint_key`` when given, otherwise ``f"{stage}/best"``.
    """
    return checkpoint_key if checkpoint_key is not None else f"{stage}/best"


def check_epoch_loop_arguments(
    *,
    state: TrainingState,
    epochs: int,
    stage: str,
    validation: ValidationStrategy | None,
    early_stopper: EarlyStopper | None,
    diagnostics: Sequence[str] = (),
    metrics: MetricSuite | None = None,
) -> None:
    """Refuse a loop configuration before anything is mutated.

    Args:
        state: The training state the loop will update.
        epochs: The maximum number of epochs.
        stage: The stage segment of every metric tag.
        validation: The optional validation strategy.
        early_stopper: The optional early-stopping policy.
        diagnostics: The declared diagnostic keys of the step's output.
        metrics: The registered metrics, whose training tags a diagnostic must
            not take.

    Raises:
        ValueError: If ``epochs`` is not positive, if ``stage`` is empty or
            contains ``/``, if ``early_stopper`` monitors a validation metric
            without a ``validation`` strategy, or if a diagnostic is empty,
            contains ``/``, is declared twice or would be recorded under a
            training tag the loop already records.
        TypeError: If ``state.rng_state`` is not a JAX random key.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not isinstance(state.rng_state, jax.Array):
        raise TypeError("state.rng_state must be a JAX random key.")
    if early_stopper and early_stopper.expects_validation() and validation is None:
        raise ValueError("validation metric monitoring requires a validation strategy.")
    metric_tag(stage, Split.TRAIN, "loss")
    _check_diagnostics(stage, diagnostics, metrics)


def _check_diagnostics(
    stage: str, diagnostics: Sequence[str], metrics: MetricSuite | None
) -> None:
    """Refuse a diagnostic whose training tag is invalid or already taken.

    Args:
        stage: The stage name the diagnostics' tags are recorded under.
        diagnostics: The diagnostic keys the stage records.
        metrics: The stage's metric suite, whose metric names are taken.

    Raises:
        ValueError: As
            [`check_epoch_loop_arguments`][stagereg.jax.epoch_loop.check_epoch_loop_arguments].
    """
    taken = {"loss"}
    if metrics is not None:
        taken |= {spec.name for spec in metrics.batch}
        taken |= {metric.name for metric in metrics.epoch}
    for name in diagnostics:
        tag = metric_tag(stage, Split.TRAIN, name)
        if name in taken:
            raise ValueError(
                f"diagnostic {name!r} collides with the tag {tag!r} "
                "that is already recorded."
            )
        taken.add(name)


def run_epoch_loop(
    *,
    step: BatchStep,
    snapshot_state: StateSnapshotter,
    train_loader: LoaderFactory,
    state: TrainingState,
    epochs: int,
    stage: str,
    checkpoint_key: str | None = None,
    validation: ValidationStrategy | None = None,
    early_stopper: EarlyStopper | None = None,
    event_sinks: Sequence[EventSink] = (),
    checkpoint_store: CheckpointStore | None = None,
    metrics: MetricSuite | None = None,
    epoch_predictions: EpochPredictionCollector | None = None,
    diagnostics: Sequence[str] = (),
) -> StageResult:
    """Run ``step`` over every training batch for up to ``epochs`` epochs.

    Each batch draws a fresh key from ``state.rng_state``. When ``metrics``
    registers epoch metrics, ``epoch_predictions`` predicts on each batch
    before ``step`` updates it. After each epoch the loop records and emits the
    training metrics, validates, and lets ``early_stopper`` decide; on every
    improvement it saves a best checkpoint holding ``snapshot_state()``.

    Args:
        step: The per-batch step, which updates the trained state in place.
        snapshot_state: Snapshots the trained state for the best checkpoint.
            Called only when a best checkpoint is saved.
        train_loader: Factory producing the training batches of each epoch.
        state: The training state updated in place across epochs.
        epochs: The maximum number of epochs to run.
        stage: The stage recorded on ``state`` and events, and the stage
            segment of every metric tag.
        checkpoint_key: The key the best checkpoint is saved under. Defaults
            to ``None``, which resolves to ``f"{stage}/best"``.
        validation: An optional strategy evaluating ``state`` after each epoch.
        early_stopper: An optional policy stopping training early.
        event_sinks: Sinks notified of epoch, validation, best-model and
            early-stop events.
        checkpoint_store: An optional store for the best checkpoint.
        metrics: Registered batch and epoch metrics. When omitted, only loss
            is collected.
        epoch_predictions: Predicts on each batch for epoch metrics. Required
            when ``metrics`` registers epoch metrics.
        diagnostics: Keys of ``step``'s output recorded, as their epoch mean,
            alongside the training metrics. Other undeclared keys are ignored.

    Returns:
        A [`StageResult`][stagereg.core.StageResult] with ``state``, the last
        epoch's training metrics and its training loss.

    Raises:
        ValueError: If the arguments are refused by
            [`check_epoch_loop_arguments`][stagereg.jax.epoch_loop.check_epoch_loop_arguments],
            if epoch metrics are registered without ``epoch_predictions``, if
            ``step`` does not produce a declared diagnostic or produces a
            non-finite one, or if the monitored metric is not produced.
        TypeError: If ``state.rng_state`` is not a JAX random key.
    """
    check_epoch_loop_arguments(
        state=state,
        epochs=epochs,
        stage=stage,
        validation=validation,
        early_stopper=early_stopper,
        diagnostics=diagnostics,
        metrics=metrics,
    )
    metric_suite = metrics if metrics is not None else MetricSuite()
    if metric_suite.epoch and epoch_predictions is None:
        raise ValueError("epoch metrics require an epoch prediction collector.")
    resolved_key = resolve_checkpoint_key(checkpoint_key, stage)

    state.active_stage = stage
    latest_metrics: Mapping[str, float] = {}
    for epoch in range(epochs):
        epoch_metrics = _run_training_epoch(
            step=step,
            train_loader=train_loader,
            state=state,
            epoch=epoch,
            metrics=metric_suite,
            epoch_predictions=epoch_predictions,
            diagnostics=diagnostics,
        )
        latest_metrics = epoch_metrics
        _record_metrics(state, stage, Split.TRAIN, epoch_metrics)
        _emit(
            event_sinks,
            "epoch_end",
            stage=stage,
            split=Split.TRAIN,
            epoch=epoch,
            state=state,
            metrics=epoch_metrics,
        )
        validation_metrics = _run_validation_epoch(
            validation=validation,
            state=state,
            stage=stage,
            epoch=epoch,
            event_sinks=event_sinks,
        )
        if _should_stop_early(
            early_stopper=early_stopper,
            training_metrics=epoch_metrics,
            validation_metrics=validation_metrics,
            checkpoint_store=checkpoint_store,
            checkpoint_key=resolved_key,
            snapshot_state=snapshot_state,
            state=state,
            epoch=epoch,
            stage=stage,
            event_sinks=event_sinks,
        ):
            break

    return StageResult(state=state, metrics=latest_metrics, loss=latest_metrics["loss"])


def _run_training_epoch(
    *,
    step: BatchStep,
    train_loader: LoaderFactory,
    state: TrainingState,
    epoch: int,
    metrics: MetricSuite,
    epoch_predictions: EpochPredictionCollector | None,
    diagnostics: Sequence[str],
) -> dict[str, float]:
    """Run one training epoch and reduce all registered metrics.

    Args:
        step: The per-batch step.
        train_loader: Factory producing training batches.
        state: The live training state containing the RNG key.
        epoch: Epoch index passed to ``train_loader``.
        metrics: Registered batch and epoch metrics.
        epoch_predictions: Predicts on each batch when epoch metrics are
            registered.
        diagnostics: Keys of each step's output reduced to their epoch mean.

    Returns:
        Reduced epoch metrics containing ``"loss"``, any registered metrics
        and every diagnostic.

    Raises:
        ValueError: If a step does not produce a diagnostic, or a diagnostic's
            epoch mean is not finite.
    """
    # Step outputs stay on the device until the epoch ends, so no batch waits
    # for the previous step to finish before being dispatched.
    losses: list[StepScalar] = []
    batch_metric_values = initialize_batch_metric_values(metrics.batch)
    epoch_metric_parts: list[EpochPredictionData] | None = [] if metrics.epoch else None
    diagnostic_values: dict[str, list[StepScalar]] = {name: [] for name in diagnostics}

    for batch in train_loader(split="train", epoch=epoch):
        state.rng_state, batch_key = split_key(state.rng_state)
        if epoch_metric_parts is not None and epoch_predictions is not None:
            epoch_metric_parts.append(epoch_predictions(batch, metric_key(batch_key)))
        output = step(batch, batch_key)
        collect_step_metrics(
            output,
            metrics=metrics.batch,
            losses=losses,
            batch_metric_values=batch_metric_values,
            context="train step",
        )
        for name, values in diagnostic_values.items():
            if name not in output:
                raise ValueError(f"train step did not produce diagnostic {name!r}.")
            values.append(output[name])

    losses, batch_metric_values, diagnostic_values = step_outputs_to_host(
        (losses, batch_metric_values, diagnostic_values)
    )
    reduced = reduce_metric_suite(
        suite=metrics,
        losses=losses,
        batch_metric_values=batch_metric_values,
        epoch_metric_parts=epoch_metric_parts,
    )
    for name, values in diagnostic_values.items():
        mean = math.fsum(values) / len(values) if values else math.nan
        if not math.isfinite(mean):
            raise ValueError(f"diagnostic {name!r} is not finite.")
        reduced[name] = mean
    return reduced


def _run_validation_epoch(
    *,
    validation: ValidationStrategy | None,
    state: TrainingState,
    stage: str,
    epoch: int,
    event_sinks: Sequence[EventSink],
) -> Mapping[str, float]:
    """Run validation for one epoch when configured.

    Args:
        validation: Optional validation strategy.
        state: The live training state.
        stage: Stage name used in emitted events and metric tags.
        epoch: Epoch index being validated.
        event_sinks: Event sinks notified on validation completion.

    Returns:
        Validation metrics, or an empty mapping when validation is disabled.
    """
    if validation is None:
        return {}

    validation_metrics = validation(state, epoch=epoch).metrics
    _record_metrics(state, stage, Split.VALIDATION, validation_metrics)
    _emit(
        event_sinks,
        "validation_end",
        stage=stage,
        split=Split.VALIDATION,
        epoch=epoch,
        state=state,
        metrics=validation_metrics,
    )
    return validation_metrics


def _should_stop_early(
    *,
    early_stopper: EarlyStopper | None,
    training_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    checkpoint_store: CheckpointStore | None,
    checkpoint_key: str,
    snapshot_state: StateSnapshotter,
    state: TrainingState,
    epoch: int,
    stage: str,
    event_sinks: Sequence[EventSink],
) -> bool:
    """Observe metrics with the early stopper and emit side effects.

    Args:
        early_stopper: Optional early-stopping policy.
        training_metrics: Metrics produced by the training epoch.
        validation_metrics: Metrics produced by validation.
        checkpoint_store: Optional checkpoint store for best-model snapshots.
        checkpoint_key: Checkpoint key used for best-model snapshots.
        snapshot_state: Snapshots the trained state for the best checkpoint.
        state: The live training state.
        epoch: Epoch index being observed.
        stage: Stage name used in emitted events.
        event_sinks: Event sinks notified on improvement or early stop.

    Returns:
        ``True`` when training should stop early, otherwise ``False``.

    Raises:
        ValueError: If the monitored metric is missing from the selected metric
            mapping.
    """
    if early_stopper is None:
        return False

    monitored_metrics = (
        validation_metrics if early_stopper.expects_validation() else training_metrics
    )
    metric_name = early_stopper.monitored_metric_name()
    if metric_name not in monitored_metrics:
        raise ValueError(f"monitored metric {metric_name!r} was not produced.")

    value = monitored_metrics[metric_name]
    verdict = early_stopper.observe(value, epoch=epoch)
    split = verdict.state.source
    decision = Decision(metric=metric_name, value=value)
    if verdict.improved:
        if checkpoint_store is not None:
            _save_checkpoint(
                checkpoint_store,
                checkpoint_key,
                state,
                snapshot_state(),
                epoch,
                verdict.state,
            )
        _emit(
            event_sinks,
            "best_model",
            stage=stage,
            split=split,
            epoch=epoch,
            state=state,
            decision=decision,
        )
    if verdict.should_stop:
        _emit(
            event_sinks,
            "early_stop",
            stage=stage,
            split=split,
            epoch=epoch,
            state=state,
            decision=decision,
        )
    return verdict.should_stop


def _record_metrics(
    state: TrainingState,
    stage: str,
    split: Split,
    metrics: Mapping[str, float],
) -> None:
    """Append metric values to ``state.metric_history`` under their tags.

    Args:
        state: The training state whose ``metric_history`` is updated.
        stage: Stage that produced the metrics.
        split: Split the metrics were measured on.
        metrics: Mapping of bare metric name to the value observed this
            epoch.
    """
    for name, value in metrics.items():
        state.record_metric(metric_tag(stage, split, name), value)


def _emit(
    sinks: Sequence[EventSink],
    name: str,
    *,
    stage: str,
    split: Split,
    epoch: int,
    state: TrainingState,
    metrics: Mapping[str, float] | None = None,
    decision: Decision | None = None,
) -> None:
    """Build a training event and dispatch it to every sink.

    Args:
        sinks: The event sinks to notify.
        name: The event name, e.g. ``"epoch_end"`` or ``"early_stop"``.
        stage: The stage name associated with the event.
        split: The split the event concerns.
        epoch: The epoch at which the event occurred.
        state: The training state associated with the event.
        metrics: The metrics measured at this point, keyed by bare metric
            name. Defaults to none, as for a decision event.
        decision: The measurement a decision event judged. Defaults to
            ``None``, as for an event that reports measurements.
    """
    event = TrainingEvent(
        name=name,
        stage=stage,
        split=split,
        step=epoch,
        metrics={} if metrics is None else metrics,
        state=state,
        decision=decision,
    )
    for sink in sinks:
        sink.on_event(event)


def _save_checkpoint(
    store: CheckpointStore,
    key: str,
    state: TrainingState,
    trained: StateSnapshot,
    epoch: int,
    early_stopping_state: object,
) -> None:
    """Persist a best-model checkpoint.

    Args:
        store: The checkpoint store to write to.
        key: The key under which the checkpoint is saved.
        state: The current training state, frozen into an independent
            snapshot before being embedded in the checkpoint.
        trained: The snapshot of the trained state.
        epoch: The epoch at which the improvement was observed.
        early_stopping_state: The early-stopping state at the time of
            the improvement.
    """
    store.save(
        key,
        Checkpoint(
            state=freeze_training_state(state),
            epoch=epoch,
            parameters=trained.parameters,
            optimizer_state=trained.optimizer_state,
            early_stopping_state=early_stopping_state,
        ),
    )
