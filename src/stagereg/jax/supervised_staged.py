"""Explicit supervised stages for mean and Gamma variance training."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from stagereg.core.checkpoints import Checkpoint, CheckpointStore
from stagereg.core.early_stopping import EarlyStopper
from stagereg.core.losses import (
    NegativeLogLikelihoodLoss,
    SquaredErrorLoss,
    add_epsilon,
)
from stagereg.core.protocols import LoaderFactory, ValidationStrategy
from stagereg.core.tracking import EventSink
from stagereg.core.types import (
    Batch,
    CheckpointRef,
    ParameterRole,
    StageResult,
    StageState,
    TrainingState,
    ValidationResult,
)
from stagereg.jax.distributions import ResidualScale
from stagereg.jax.evaluation import SupervisedLoss
from stagereg.jax.losses import make_supervised_loss
from stagereg.jax.metrics import MetricSuite
from stagereg.jax.epoch_loop import resolve_checkpoint_key
from stagereg.jax.stage_checkpoints import (
    finalized_checkpoint,
    latest_training_result,
    load_best_checkpoint,
    require_component_names,
    require_finalized,
    select_checkpoint,
)
from stagereg.jax.state import restore_checkpoint, snapshot
from stagereg.jax.supervised import run_supervised


def materialize_residual_loader(
    mean_model: nnx.Module,
    source_loader: LoaderFactory,
    *,
    splits: Sequence[str] = ("train", "validation"),
    source_epoch: int = 0,
) -> LoaderFactory:
    """Materialize detached squared residual targets for fixed data splits.

    The source loader is consumed exactly once per configured split. The
    returned loader replays the cached batches for every later epoch, making
    the Step 1 mean predictions and source data a fixed Step 2 training
    snapshot.

    Args:
        mean_model: Trained deterministic mean model.
        source_loader: Loader providing inputs and original regression targets.
        splits: Split names to materialize. Defaults to training and validation.
        source_epoch: Source-loader epoch used for the one-time snapshot.

    Returns:
        A loader factory serving cached batches whose targets are detached
        squared residuals.

    Raises:
        ValueError: If splits are empty, duplicated, unknown at replay time, or
            produce no batches; if a source batch lacks targets; or if mean
            predictions and targets have different shapes.
    """
    split_names = tuple(splits)
    if not split_names:
        raise ValueError("splits must contain at least one split name.")
    if any(not name for name in split_names):
        raise ValueError("split names must not be empty.")
    if len(set(split_names)) != len(split_names):
        raise ValueError("split names must be unique.")

    prediction_model = nnx.clone(mean_model)
    prediction_model.eval()
    cached: dict[str, tuple[Batch, ...]] = {}

    for split in split_names:
        residual_batches = tuple(
            _materialize_residual_batch(prediction_model, batch)
            for batch in source_loader(split=split, epoch=source_epoch)
        )
        if not residual_batches:
            raise ValueError(f"source split {split!r} must provide at least one batch.")
        cached[split] = residual_batches

    def residual_loader(*, split: str, epoch: int) -> tuple[Batch, ...]:
        del epoch
        if split not in cached:
            raise ValueError(f"residual split {split!r} was not materialized.")
        return cached[split]

    return residual_loader


def _materialize_residual_batch(mean_model: nnx.Module, batch: Batch) -> Batch:
    """Return one batch with detached squared residual targets."""
    if batch.targets is None:
        raise ValueError("source batches must provide targets.")
    predictions = mean_model(batch.inputs)
    if predictions.shape != batch.targets.shape:
        raise ValueError("mean predictions and targets must have matching shapes.")
    residuals = jax.lax.stop_gradient(jnp.square(batch.targets - predictions))
    return Batch(
        inputs=batch.inputs,
        targets=residuals,
        sample_weight=batch.sample_weight,
        metadata=batch.metadata,
    )


def _mean_squared_residual(residual_loader: LoaderFactory) -> float:
    """Return the mean of the materialized training squared residuals.

    The mean is unweighted and taken over every target element, so a
    multi-output model gets one scalar scale.

    Args:
        residual_loader: The materialized residual loader.

    Returns:
        The mean squared residual of the ``"train"`` split.

    Raises:
        ValueError: If the training split was not materialized, or if the mean
            is not positive and finite.
    """
    targets = [
        batch.targets
        for batch in residual_loader(split="train", epoch=0)
        if batch.targets is not None
    ]
    total = sum(float(jnp.sum(target)) for target in targets)
    count = sum(target.size for target in targets)
    mean = total / count if count else math.nan
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(
            f"training squared residuals must have a positive finite mean, got {mean}."
        )
    return mean


def _holds_residual_scale(model: nnx.Module) -> bool:
    """Return whether ``model`` holds a [`ResidualScale`][stagereg.jax.ResidualScale].

    Args:
        model: The variance model.

    Returns:
        ``True`` if ``model`` holds at least one ``ResidualScale``.
    """
    return bool(jax.tree.leaves(nnx.state(model, ResidualScale)))


def _set_residual_scale(model: nnx.Module, scale: float) -> None:
    """Set every [`ResidualScale`][stagereg.jax.ResidualScale] in ``model``.

    Args:
        model: The variance model; one without a ``ResidualScale`` is left
            unchanged.
        scale: The value every ``ResidualScale`` is set to.
    """
    scales = nnx.state(model, ResidualScale)
    nnx.update(model, jax.tree.map(lambda value: jnp.full_like(value, scale), scales))


@dataclass(frozen=True)
class SupervisedStageOptions:
    """Shared epoch-runner options for concrete supervised stages.

    Attributes:
        epochs: Maximum number of training epochs.
        validation: Optional validation strategy.
        early_stopper: Optional early-stopping policy.
        event_sinks: Event consumers notified by the runner.
        checkpoint_store: Optional store for the stage's best and finalized
            checkpoints.
        checkpoint_key: The checkpoint key the stage saves its best and
            finalized checkpoints under. Defaults to ``None``, which resolves to the stage-scoped
            key ``f"{stage}/best"`` (``mean/best``, ``variance/best``), the
            same default as [`run_supervised`][stagereg.jax.run_supervised],
            so stages sharing one ``checkpoint_store`` never overwrite each
            other's checkpoints.
        metrics: Batch and epoch metric registrations.
    """

    epochs: int
    validation: ValidationStrategy | None = None
    early_stopper: EarlyStopper | None = None
    event_sinks: Sequence[EventSink] = ()
    checkpoint_store: CheckpointStore | None = None
    checkpoint_key: str | None = None
    metrics: MetricSuite = field(default_factory=MetricSuite)


@dataclass
class MeanStage:
    """Concrete Step 1 stage training a deterministic mean model with MSE.

    Attributes:
        model: NNX model producing deterministic mean predictions.
        optimizer: NNX optimizer bound to ``model``.
        train_loader: Factory producing original regression batches.
        options: Shared supervised-runner options.
        model_name: State registry name for the mean model.
        optimizer_name: State registry name for the mean optimizer.
        loss: Scalar supervised loss used to train the mean model.
    """

    model: nnx.Module
    optimizer: nnx.Optimizer
    train_loader: LoaderFactory
    options: SupervisedStageOptions
    model_name: str = "mean_model"
    optimizer_name: str = "mean_optimizer"
    loss: SupervisedLoss = field(
        default_factory=lambda: make_supervised_loss(SquaredErrorLoss())
    )
    name: str = field(default="mean", init=False)
    requires: frozenset[str] = field(default_factory=frozenset, init=False)
    produces: frozenset[str] = field(
        default_factory=lambda: frozenset({"mean"}),
        init=False,
    )

    def prepare(self, state: TrainingState) -> None:
        """Register Step 1 ownership and initialize the staged lifecycle.

        Args:
            state: Shared staged training state.

        Raises:
            ValueError: If the lifecycle or component ownership is invalid.
        """
        if state.lifecycle_state not in (StageState.NEW, StageState.INITIALIZED):
            raise ValueError("mean stage requires NEW or INITIALIZED lifecycle state.")
        _validate_named_registration(
            state.model_components,
            self.model_name,
            self.model,
            kind="model component",
        )
        _validate_named_registration(
            state.optimizer_states,
            self.optimizer_name,
            self.optimizer,
            kind="optimizer state",
        )
        _validate_parameter_role(state, self.model_name, ParameterRole.MEAN)
        state.register_component(self.model_name, self.model)
        state.register_optimizer(self.optimizer_name, self.optimizer)
        state.parameter_roles[self.model_name] = ParameterRole.MEAN
        state.lifecycle_state = StageState.INITIALIZED
        state.active_stage = self.name

    def train(self, state: TrainingState) -> StageResult:
        """Train the mean model and transition the workflow to ``MEAN_READY``.

        With a checkpoint store configured, the stage then saves a finalized
        ``MEAN_READY`` checkpoint under its checkpoint key, with metadata
        ``{"stage": "mean", "stage_complete": True}``. With an early stopper,
        it first restores its best checkpoint, so the live mean model holds the
        best epoch's weights and the finalized checkpoint is the best epoch's;
        without one, the finalized checkpoint is the last epoch's.

        Args:
            state: Prepared staged training state.

        Returns:
            The supervised runner result, or the restored epoch's training
            metrics when the best checkpoint was restored.

        Raises:
            ValueError: If the stage was not prepared or training is non-finite.
        """
        if state.lifecycle_state is not StageState.INITIALIZED:
            raise ValueError("mean stage must be prepared before training.")
        return _train_stage(
            self,
            state,
            train_loader=self.train_loader,
            validation=self.options.validation,
            ready=StageState.MEAN_READY,
        )

    def restore(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore the mean stage's finalized checkpoint into ``state``.

        The live model and optimizer take the checkpoint's weights in place and
        are registered under ``model_name`` and ``optimizer_name``. The restore
        is clean-slate, like
        [`restore_checkpoint`][stagereg.jax.restore_checkpoint]: other registered
        components and optimizers are dropped. Afterwards the state passes
        [`validate`][stagereg.jax.MeanStage.validate], so the variance stage can
        be prepared or restored next.

        Args:
            state: Training state to restore into, typically a fresh one.
            checkpoint: The mean stage's finalized checkpoint.

        Raises:
            ValueError: If ``checkpoint`` is not finalized by this stage, that
                is its lifecycle state is not ``MEAN_READY`` or its metadata
                lacks ``"stage": "mean"`` and ``"stage_complete": True``, or
                if it was saved with another ``model_name``; this is checked
                before ``state`` or the live objects change. Also
                raised, as by
                [`restore_checkpoint`][stagereg.jax.restore_checkpoint], if the
                live model or optimizer is incompatible with the checkpoint,
                which may leave them and ``state`` partially restored.
            TypeError: As by
                [`restore_checkpoint`][stagereg.jax.restore_checkpoint], if the
                checkpoint does not hold NNX snapshots or a JAX random key.
        """
        require_finalized(checkpoint, stage=self.name, ready=StageState.MEAN_READY)
        require_component_names(
            checkpoint, model_name=self.model_name, role=ParameterRole.MEAN
        )
        self._restore_live(state, checkpoint)

    def _restore_live(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore a mean checkpoint into the live model and mark Step 1 ready."""
        restore_checkpoint(
            checkpoint,
            state=state,
            model=self.model,
            optimizer=self.optimizer,
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
        )
        state.lifecycle_state = StageState.MEAN_READY
        state.active_stage = self.name

    def validate(self, state: TrainingState) -> ValidationResult:
        """Validate mean-stage lifecycle and ownership invariants.

        Args:
            state: Shared staged training state.

        Returns:
            A validation result describing whether Step 1 is ready.
        """
        passed = (
            state.lifecycle_state is StageState.MEAN_READY
            and state.model_components.get(self.model_name) is self.model
            and state.optimizer_states.get(self.optimizer_name) is self.optimizer
            and state.parameter_roles.get(self.model_name) is ParameterRole.MEAN
        )
        return ValidationResult(
            passed=passed,
            message=None if passed else "mean stage invariants are not satisfied.",
        )

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef:
        """Return a reference to the mean stage's best checkpoint.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the checkpoint key the stage saves under: the
            configured ``checkpoint_key``, or ``f"{stage}/best"`` when none
            was configured.

        Raises:
            ValueError: If no checkpoint exists under that key.
        """
        del state
        return _select_checkpoint(self)


@dataclass
class GammaVarianceStage:
    """Concrete Step 2 stage fitting Gamma-distributed squared residuals.

    Attributes:
        model: NNX model producing Gamma residual predictions.
        optimizer: NNX optimizer bound to ``model``.
        source_loader: Factory producing original regression batches.
        options: Shared supervised-runner options.
        mean_model_name: Registry name of the prepared mean model.
        mean_optimizer_name: Registry name of the mean optimizer, kept
            registered when the stage restores its best checkpoint.
        model_name: State registry name for the variance model.
        optimizer_name: State registry name for the variance optimizer.
        splits: Source split names materialized as residual batches.
        source_epoch: Source-loader epoch used for residual materialization.
        validation_factory: Optional factory adapting the residual loader into
            a validation strategy.
        loss: Scalar supervised loss used to train the variance model.
    """

    model: nnx.Module
    optimizer: nnx.Optimizer
    source_loader: LoaderFactory
    options: SupervisedStageOptions
    mean_model_name: str = "mean_model"
    mean_optimizer_name: str = "mean_optimizer"
    model_name: str = "variance_model"
    optimizer_name: str = "variance_optimizer"
    splits: Sequence[str] = ("train", "validation")
    source_epoch: int = 0
    validation_factory: Callable[[LoaderFactory], ValidationStrategy] | None = None
    loss: SupervisedLoss = field(
        default_factory=lambda: make_supervised_loss(
            NegativeLogLikelihoodLoss(target_transform=add_epsilon())
        )
    )
    name: str = field(default="variance", init=False)
    requires: frozenset[str] = field(
        default_factory=lambda: frozenset({"mean"}),
        init=False,
    )
    produces: frozenset[str] = field(
        default_factory=lambda: frozenset({"variance"}),
        init=False,
    )
    _residual_loader: LoaderFactory | None = field(default=None, init=False, repr=False)
    _validation: ValidationStrategy | None = field(default=None, init=False, repr=False)

    def prepare(self, state: TrainingState) -> None:
        """Validate Step 1 output and materialize fixed residual targets.

        If the variance model holds a
        [`ResidualScale`][stagereg.jax.ResidualScale], also sets every one to
        the mean squared residual of the ``"train"`` split, so a
        [`GammaHead`][stagereg.jax.GammaHead] trains at unit scale. A model
        without one, such as ``GammaHead(..., residual_scaling=False)``, trains
        on the raw residuals.

        Args:
            state: Shared state whose mean component is ready.

        Raises:
            TypeError: If the registered mean component is not an NNX module.
            ValueError: If lifecycle, ownership, or registrations are invalid,
                or, when the variance model holds a ``ResidualScale``, if
                ``"train"`` is not among the materialized splits or the
                training squared residuals' mean is not positive and finite.
        """
        if state.lifecycle_state is not StageState.MEAN_READY:
            raise ValueError("variance stage requires MEAN_READY lifecycle state.")
        if self.mean_model_name not in state.model_components:
            raise ValueError(
                f"mean model component {self.mean_model_name!r} is not registered."
            )
        mean_model = state.model_components[self.mean_model_name]
        if not isinstance(mean_model, nnx.Module):
            raise TypeError("registered mean model must be an NNX module.")
        if state.parameter_roles.get(self.mean_model_name) is not ParameterRole.MEAN:
            raise ValueError("registered mean model must have the MEAN parameter role.")
        _validate_named_registration(
            state.model_components,
            self.model_name,
            self.model,
            kind="model component",
        )
        _validate_named_registration(
            state.optimizer_states,
            self.optimizer_name,
            self.optimizer,
            kind="optimizer state",
        )
        _validate_parameter_role(state, self.model_name, ParameterRole.VARIANCE)

        residual_loader = materialize_residual_loader(
            mean_model,
            self.source_loader,
            splits=self.splits,
            source_epoch=self.source_epoch,
        )
        residual_scale = (
            _mean_squared_residual(residual_loader)
            if _holds_residual_scale(self.model)
            else None
        )
        validation = (
            self.validation_factory(residual_loader)
            if self.validation_factory is not None
            else self.options.validation
        )

        if residual_scale is not None:
            _set_residual_scale(self.model, residual_scale)
        state.register_component(self.model_name, self.model)
        state.register_optimizer(self.optimizer_name, self.optimizer)
        state.parameter_roles[self.model_name] = ParameterRole.VARIANCE
        state.frozen_components = state.frozen_components | {self.mean_model_name}
        state.active_stage = self.name
        self._residual_loader = residual_loader
        self._validation = validation

    def train(self, state: TrainingState) -> StageResult:
        """Train the variance model and transition to ``VARIANCE_READY``.

        With a checkpoint store configured, the stage then saves a finalized
        ``VARIANCE_READY`` checkpoint under its checkpoint key, with metadata
        ``{"stage": "variance", "stage_complete": True}``. With an early
        stopper, it first restores its best checkpoint, so the live variance
        model holds the best epoch's weights, while the mean model and, if
        registered, the mean optimizer stay registered; the finalized
        checkpoint is then the best epoch's. Without an early stopper, it is
        the last epoch's.

        Args:
            state: Prepared state retaining a frozen mean component.

        Returns:
            The supervised runner result, or the restored epoch's training
            metrics when the best checkpoint was restored.

        Raises:
            ValueError: If preparation is incomplete or training is non-finite.
        """
        if state.lifecycle_state is not StageState.MEAN_READY:
            raise ValueError("variance stage requires MEAN_READY lifecycle state.")
        if self._residual_loader is None:
            raise ValueError("variance stage must be prepared before training.")
        return _train_stage(
            self,
            state,
            train_loader=self._residual_loader,
            validation=self._validation,
            ready=StageState.VARIANCE_READY,
        )

    def restore(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore the variance stage's finalized checkpoint into ``state``.

        Call it after the mean stage's
        [`restore`][stagereg.jax.MeanStage.restore]. The live variance model and
        optimizer take the checkpoint's weights in place and are registered
        under ``model_name`` and ``optimizer_name``. The mean model stays
        registered under ``mean_model_name``, and the mean optimizer under
        ``mean_optimizer_name`` if it was registered before. Afterwards the
        state passes [`validate`][stagereg.jax.GammaVarianceStage.validate].

        Args:
            state: Training state the mean stage has been restored into.
            checkpoint: The variance stage's finalized checkpoint.

        Raises:
            ValueError: If ``checkpoint`` is not finalized by this stage, that
                is its lifecycle state is not ``VARIANCE_READY`` or its
                metadata lacks ``"stage": "variance"`` and ``"stage_complete":
                True``, if it was saved with another ``model_name`` or
                ``mean_model_name``, or if no mean model is registered under
                ``mean_model_name``; all are checked before ``state`` or the
                live objects change. Also raised, as by
                [`restore_checkpoint`][stagereg.jax.restore_checkpoint], if the
                live model or optimizer is incompatible with the checkpoint,
                which may leave them and ``state`` partially restored.
            TypeError: As by
                [`restore_checkpoint`][stagereg.jax.restore_checkpoint], if the
                checkpoint does not hold NNX snapshots or a JAX random key.
        """
        require_finalized(checkpoint, stage=self.name, ready=StageState.VARIANCE_READY)
        require_component_names(
            checkpoint,
            model_name=self.model_name,
            role=ParameterRole.VARIANCE,
            frozen=self.mean_model_name,
        )
        if self.mean_model_name not in state.model_components:
            raise ValueError(
                f"mean model component {self.mean_model_name!r} is not registered."
            )
        self._restore_live(state, checkpoint)

    def _restore_live(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore a variance checkpoint, keeping the mean registrations live.

        The variance checkpoint snapshots only the variance model and optimizer,
        and ``restore_checkpoint`` restores clean-slate, so the mean model and,
        if registered, the mean optimizer are carried across the restore.
        """
        mean_model = state.model_components[self.mean_model_name]
        mean_optimizer = state.optimizer_states.get(self.mean_optimizer_name)
        restore_checkpoint(
            checkpoint,
            state=state,
            model=self.model,
            optimizer=self.optimizer,
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
        )
        state.register_component(self.mean_model_name, mean_model)
        if mean_optimizer is not None:
            state.register_optimizer(self.mean_optimizer_name, mean_optimizer)
        state.lifecycle_state = StageState.VARIANCE_READY
        state.active_stage = self.name

    def validate(self, state: TrainingState) -> ValidationResult:
        """Validate variance-stage lifecycle, ownership, and freezing.

        Args:
            state: Shared staged training state.

        Returns:
            A validation result describing whether Step 2 is ready.
        """
        passed = (
            state.lifecycle_state is StageState.VARIANCE_READY
            and state.model_components.get(self.model_name) is self.model
            and state.optimizer_states.get(self.optimizer_name) is self.optimizer
            and state.parameter_roles.get(self.model_name) is ParameterRole.VARIANCE
            and self.mean_model_name in state.frozen_components
        )
        return ValidationResult(
            passed=passed,
            message=None if passed else "variance stage invariants are not satisfied.",
        )

    def select_checkpoint(self, state: TrainingState) -> CheckpointRef:
        """Return a reference to the variance stage's best checkpoint.

        Args:
            state: Shared staged training state.

        Returns:
            Reference to the checkpoint key the stage saves under: the
            configured ``checkpoint_key``, or ``f"{stage}/best"`` when none
            was configured.

        Raises:
            ValueError: If no checkpoint exists under that key.
        """
        del state
        return _select_checkpoint(self)


class _CheckpointingStage(Protocol):
    """A concrete supervised stage, as the shared training helpers see it.

    Attributes:
        name: Stage name, scoping metric tags and the default checkpoint key.
        model: The stage's live model.
        optimizer: The stage's live optimizer, bound to ``model``.
        options: The stage's supervised-runner options.
        model_name: State registry name for ``model``.
        optimizer_name: State registry name for ``optimizer``.
        loss: Scalar supervised loss used to train ``model``.
    """

    name: str
    model: nnx.Module
    optimizer: nnx.Optimizer
    options: SupervisedStageOptions
    model_name: str
    optimizer_name: str
    loss: SupervisedLoss

    def _restore_live(self, state: TrainingState, checkpoint: Checkpoint) -> None:
        """Restore a checkpoint into ``state`` and mark the stage ready."""
        ...


def _checkpoint_key(stage: _CheckpointingStage) -> str:
    """Return the checkpoint key ``stage`` saves its best checkpoint under."""
    return resolve_checkpoint_key(stage.options.checkpoint_key, stage.name)


def _train_stage(
    stage: _CheckpointingStage,
    state: TrainingState,
    *,
    train_loader: LoaderFactory,
    validation: ValidationStrategy | None,
    ready: StageState,
) -> StageResult:
    """Run a stage's supervised training, mark it ready and finalize it.

    Args:
        stage: The prepared stage to train.
        state: State the stage has been prepared on.
        train_loader: Factory producing the stage's training batches.
        validation: The stage's validation strategy, if any.
        ready: The lifecycle state the stage reaches once trained.

    Returns:
        The supervised runner result, or the restored epoch's training metrics
        when the best checkpoint was restored.

    Raises:
        ValueError: If training produced a non-finite final loss.
    """
    options = stage.options
    result = run_supervised(
        model=stage.model,
        optimizer=stage.optimizer,
        train_loader=train_loader,
        loss=stage.loss,
        state=state,
        epochs=options.epochs,
        validation=validation,
        early_stopper=options.early_stopper,
        event_sinks=options.event_sinks,
        checkpoint_store=options.checkpoint_store,
        checkpoint_key=_checkpoint_key(stage),
        stage=stage.name,
        model_name=stage.model_name,
        optimizer_name=stage.optimizer_name,
        metrics=options.metrics,
    )
    if result.loss is None or not math.isfinite(result.loss):
        raise ValueError(f"{stage.name} stage produced a non-finite final loss.")
    state.lifecycle_state = ready
    return _finalize_checkpoint(stage, state, result)


def _select_checkpoint(stage: _CheckpointingStage) -> CheckpointRef:
    """Return a reference to the checkpoint ``stage`` saves its best under.

    Args:
        stage: The stage whose checkpoint to reference.

    Returns:
        Reference to the stage's checkpoint key, with the stage-name metadata.

    Raises:
        ValueError: If no checkpoint exists under that key.
    """
    return select_checkpoint(
        stage.options.checkpoint_store, key=_checkpoint_key(stage), stage=stage.name
    )


def _finalize_checkpoint(
    stage: _CheckpointingStage,
    state: TrainingState,
    result: StageResult,
) -> StageResult:
    """Save a trained stage's finalized checkpoint under its checkpoint key.

    With an early stopper, the stage's best checkpoint is restored and saved
    again as finalized; without one, the last epoch's state is (ADR 0011).
    Without a checkpoint store, or with an early stopper but no saved best
    checkpoint, nothing is saved.

    Args:
        stage: The stage that has just trained.
        state: State the stage has just trained and marked ready.
        result: The supervised runner result of the stage.

    Returns:
        A result reporting the restored epoch's training metrics when the best
        checkpoint was restored, otherwise ``result``.
    """
    options = stage.options
    store = options.checkpoint_store
    if store is None:
        return result
    if options.early_stopper is None:
        store.save(
            _checkpoint_key(stage),
            finalized_checkpoint(
                state,
                stage=stage.name,
                epoch=options.epochs - 1,
                early_stopping_state=None,
                parameters=snapshot(stage.model),
                optimizer_state=snapshot(stage.optimizer),
            ),
        )
        return result
    return _restore_and_finalize_best_checkpoint(stage, state, result, store=store)


def _restore_and_finalize_best_checkpoint(
    stage: _CheckpointingStage,
    state: TrainingState,
    result: StageResult,
    *,
    store: CheckpointStore,
) -> StageResult:
    """Restore a stage's best checkpoint and save it again as finalized.

    The checkpoint is not finalized yet, so it is restored through the stage's
    unchecked restore path rather than its public ``restore``. Without a saved
    best checkpoint, ``result`` is returned unchanged.

    Args:
        stage: The stage that has just trained, with an early stopper.
        state: State the stage has just trained and marked ready.
        result: The supervised runner result of the stage.
        store: The stage's checkpoint store.

    Returns:
        ``result`` if nothing was restored, otherwise a result reporting the
        restored epoch's training metrics.
    """
    key = _checkpoint_key(stage)
    checkpoint = load_best_checkpoint(
        early_stopper=stage.options.early_stopper, checkpoint_store=store, key=key
    )
    if checkpoint is None:
        return result

    stage._restore_live(state, checkpoint)
    store.save(
        key,
        finalized_checkpoint(
            state,
            stage=stage.name,
            epoch=checkpoint.epoch,
            early_stopping_state=checkpoint.early_stopping_state,
            parameters=snapshot(stage.model),
            optimizer_state=snapshot(stage.optimizer),
            metadata=checkpoint.metadata,
        ),
    )
    return latest_training_result(state, stage.name)


def _validate_named_registration(
    registry: dict[str, Any],
    name: str,
    value: Any,
    *,
    kind: str,
) -> None:
    """Reject a conflicting named object without mutating the registry."""
    if name in registry and registry[name] is not value:
        raise ValueError(f"{kind} {name!r} is already registered.")


def _validate_parameter_role(
    state: TrainingState,
    component_name: str,
    role: ParameterRole,
) -> None:
    """Reject conflicting component ownership without mutating state."""
    registered = state.parameter_roles.get(component_name)
    if registered is not None and registered is not role:
        raise ValueError(
            f"model component {component_name!r} already has role {registered.value!r}."
        )
