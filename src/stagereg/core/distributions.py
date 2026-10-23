"""Protocols for distribution-valued model predictions and objectives."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from stagereg.core.types import Array, Batch


class PredictiveDistribution(Protocol):
    """A probability distribution produced by a regression model.

    This protocol is the backend-neutral model/loss boundary. Epoch metrics use backend
    adapters to materialize host arrays rather than retaining distribution objects,
    which intentionally have no concatenation or indexing contract.
    """

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    def log_prob(self, targets: Array) -> Array: ...

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array: ...

    def mean(self) -> Array: ...

    def variance(self) -> Array: ...


@runtime_checkable
class ClosedFormCRPS(Protocol):
    """A predictive distribution whose CRPS has a closed form.

    This protocol is optional: a
    [`PredictiveDistribution`][stagereg.core.PredictiveDistribution] need not
    implement it. A distribution that does can be scored by the closed-form
    CRPS epoch metric, which never falls back to a sample estimate (ADR 0015). ``isinstance`` checks
    only that a ``crps`` method exists.
    """

    def crps(self, targets: Array) -> Array:
        """Return the expected CRPS of each scoring unit.

        Args:
            targets: One target per scoring unit, shaped ``[*batch]``, or
                ``R`` reference samples per scoring unit, shaped
                ``[R, *batch]``.

        Returns:
            The CRPS of each scoring unit, averaged over its reference
            samples, shaped ``[*batch]``.
        """
        ...


class DistributionHead(Protocol):
    """Produces a predictive distribution from model features."""

    def __call__(self, features: Array) -> PredictiveDistribution: ...


class Likelihood(Protocol):
    """Evaluates a target under a predictive distribution."""

    def log_prob(self, prediction: PredictiveDistribution, targets: Array) -> Array: ...


class DistributionLoss(Protocol):
    """Computes an unreduced loss from a predictive distribution."""

    def per_example(
        self, prediction: PredictiveDistribution, batch: Batch
    ) -> Array: ...


class PredictionLoss(Protocol):
    """Computes an unreduced loss from deterministic array predictions."""

    def per_example(self, prediction: Array, batch: Batch) -> Array: ...


# Backward-compatible alias for the original distribution-loss protocol.
Loss = DistributionLoss
