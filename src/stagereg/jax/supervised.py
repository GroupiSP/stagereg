"""A single-model NNX supervised training runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
from flax import nnx

from stagereg.core.checkpoints import CheckpointStore
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.protocols import LoaderFactory, ValidationStrategy
from stagereg.core.tracking import EventSink
from stagereg.core.types import Batch, PyTree, StageResult, TrainingState
from stagereg.jax.epoch_loop import (
    StateSnapshot,
    check_epoch_loop_arguments,
    run_epoch_loop,
)
from stagereg.jax.evaluation import SupervisedLoss
from stagereg.jax.metrics import (
    BatchMetricSpec,
    MetricSuite,
    make_epoch_prediction_resolver,
)
from stagereg.jax.state import snapshot


def make_train_step(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec] = (),
) -> Callable[..., Mapping[str, jax.Array]]:
    """Create a JIT-compiled NNX/Optax supervised training step.

    Args:
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.

    Returns:
        A JIT-compiled function ``train_step(model, optimizer, inputs,
        targets, sample_weight, key)`` that performs one gradient update
        in place and returns a mapping containing ``"loss"`` plus
        registered batch metric values. Loss is evaluated on the
        pre-update training-state model, while batch metrics are computed
        on the same pre-update parameters with ``training=False`` to
        avoid a second state-mutating training-mode forward pass.
    """

    return nnx.jit(_train_step_body(loss, metrics=metrics))


def _train_step_body(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec],
) -> Callable[..., Mapping[str, jax.Array]]:
    """Return the untransformed training step behind ``make_train_step``.

    Args:
        loss: The supervised loss to minimise.
        metrics: Batch metrics evaluated on the pre-update parameters.

    Returns:
        A function ``train_step(model, optimizer, inputs, targets,
        sample_weight, key)`` that updates ``model`` through ``optimizer``
        and returns the loss and batch metric values.
    """

    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        inputs: PyTree,
        targets: jax.Array,
        sample_weight: jax.Array | None,
        key: jax.Array,
    ) -> Mapping[str, jax.Array]:
        def loss_fn(current_model: nnx.Module) -> jax.Array:
            return loss(
                current_model,
                inputs,
                targets,
                sample_weight,
                key,
                True,
            )

        loss_value, gradients = nnx.value_and_grad(loss_fn)(model)
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
        optimizer.update(model, gradients)
        return values

    return train_step


def _bind_train_step(
    loss: SupervisedLoss,
    *,
    metrics: Sequence[BatchMetricSpec],
    model: nnx.Module,
    optimizer: nnx.Optimizer,
) -> Callable[..., Mapping[str, jax.Array]]:
    """Compile a training step with ``model`` and ``optimizer`` bound once.

    ``make_train_step`` walks the module graph of ``model`` and ``optimizer``
    on every call, which costs more than the compiled step itself for the
    small models of the examples. Binding them flattens the graph once; the
    bound step still reads and writes the same Variables, so in-place
    updates made between calls, such as a restored checkpoint, are seen.
    The module structure and static attributes, such as a training flag,
    are fixed when binding, and changes to them inside the step are not
    written back to ``model``.

    Args:
        loss: The supervised loss to minimise.
        metrics: Batch metrics evaluated on the pre-update parameters.
        model: The live model, updated in place by every call.
        optimizer: The live optimizer, updated in place by every call.

    Returns:
        A function ``bound_step(inputs, targets, sample_weight, key)``.
    """
    return nnx.jit_partial(
        _train_step_body(loss, metrics=metrics),
        model,
        optimizer,
        graph=True,
        graph_updates=False,
    )


def run_supervised(
    *,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_loader: LoaderFactory,
    loss: SupervisedLoss,
    state: TrainingState,
    epochs: int,
    validation: ValidationStrategy | None = None,
    early_stopper: EarlyStopper | None = None,
    event_sinks: Sequence[EventSink] = (),
    checkpoint_store: CheckpointStore | None = None,
    checkpoint_key: str | None = None,
    stage: str = "supervised",
    model_name: str = "model",
    optimizer_name: str = "optimizer",
    metrics: MetricSuite | None = None,
) -> StageResult:
    """Train a single NNX model for a fixed or early-stopped number of epochs.

    The runner does not restore the best checkpoint: when training ends,
    ``model`` holds the parameters of the last epoch run. To continue from the
    best model, load the checkpoint from ``checkpoint_store`` and pass it to
    [`restore_checkpoint`][stagereg.jax.restore_checkpoint].

    The training step is compiled once with ``model`` and ``optimizer`` bound
    to it. It updates their variables in place and sees changes made to them
    between steps, but their module structure and plain Python attributes
    are fixed for the run: an attribute set inside the step, e.g. by the
    loss, is not written back to ``model``, as it would be by
    [`make_train_step`][stagereg.jax.make_train_step].

    Epoch metrics run the model's inference-mode forward pass as one
    compiled call, so with ``metrics`` holding any epoch metric the model's
    output must be a JAX pytree, as every built-in distribution is.

    Args:
        model: The NNX module to train, mutated in place.
        optimizer: The NNX optimizer used to update ``model``, mutated
            in place.
        train_loader: Factory producing the training batch loader for a
            given split and epoch.
        loss: A callable computing the supervised loss given a model,
            inputs, targets, sample weights, a PRNG key, and a
            ``training`` flag.
        state: The training state to update in place across epochs.
        epochs: The maximum number of epochs to run.
        validation: An optional strategy used to evaluate ``state``
            after each epoch. Required if ``early_stopper`` monitors a
            validation metric.
        early_stopper: An optional policy that stops training early
            based on a monitored training or validation metric.
        event_sinks: Sinks notified of epoch, validation, best-model,
            and early-stop events.
        checkpoint_store: An optional store used to persist the best
            checkpoint when ``early_stopper`` reports an improvement.
        checkpoint_key: The checkpoint key under which the best checkpoint
            is saved. Defaults to ``None``, which resolves to the stage-scoped
            key ``f"{stage}/best"`` so that stages sharing one
            ``checkpoint_store`` never overwrite each other; the
            [`MeanStage`][stagereg.jax.MeanStage] and
            [`GammaVarianceStage`][stagereg.jax.GammaVarianceStage] defaults
            resolve the same way.
        stage: The stage name recorded on ``state`` and emitted events,
            and the stage segment of every metric tag recorded in
            ``state.metric_history``. Defaults to ``"supervised"``.
        model_name: Name under which ``model`` is registered in ``state``.
            Defaults to ``"model"``.
        optimizer_name: Name under which ``optimizer`` is registered in
            ``state``. Defaults to ``"optimizer"``.
        metrics: Optional registered batch/epoch metrics for training. When
            omitted, only loss is collected.

    Returns:
        A [`StageResult`][stagereg.core.StageResult] with the final
        ``state``, the last recorded training metrics under their bare
        names, and the final training loss. ``state.metric_history``
        records every metric under its metric tag ``stage/split/metric``.

    Raises:
        ValueError: If ``epochs`` is not positive, if ``stage`` is empty
            or contains ``/``, if ``early_stopper`` monitors a validation
            metric without a ``validation`` strategy, or if the monitored
            metric is not produced by training or validation.
        TypeError: If ``state.rng_state`` is not a JAX random key.
    """
    check_epoch_loop_arguments(
        state=state,
        epochs=epochs,
        stage=stage,
        validation=validation,
        early_stopper=early_stopper,
    )
    state.register_component(model_name, model)
    state.register_optimizer(optimizer_name, optimizer)
    metric_suite = metrics if metrics is not None else MetricSuite()
    train_step = _bind_train_step(
        loss, metrics=metric_suite.batch, model=model, optimizer=optimizer
    )

    def step(batch: Batch, key: jax.Array, /) -> Mapping[str, jax.Array]:
        return train_step(
            batch.inputs,
            batch.targets,
            batch.sample_weight,
            key,
        )

    def snapshot_state() -> StateSnapshot:
        return StateSnapshot(
            parameters=snapshot(model), optimizer_state=snapshot(optimizer)
        )

    epoch_predictions = (
        make_epoch_prediction_resolver(suite=metric_suite, model=model)
        if metric_suite.epoch
        else None
    )

    return run_epoch_loop(
        step=step,
        snapshot_state=snapshot_state,
        train_loader=train_loader,
        state=state,
        epochs=epochs,
        stage=stage,
        checkpoint_key=checkpoint_key,
        validation=validation,
        early_stopper=early_stopper,
        event_sinks=event_sinks,
        checkpoint_store=checkpoint_store,
        metrics=metric_suite,
        epoch_predictions=epoch_predictions,
    )
