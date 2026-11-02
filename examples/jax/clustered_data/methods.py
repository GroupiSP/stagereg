"""The clustered-data example's inference methods, shared by both trainings.

Joint and staged runs build each method from the same factory, so they share its
hyperparameters. The methods differ from one another only in the inference method and
the backbone's dropout rate: MC dropout runs it at `mc_dropout_rate`, the others at 0.
"""

from __future__ import annotations

from collections.abc import Callable

import optax
from clustered import ClusteredDataConfig

from stagereg.jax import (
    BayesByBackprop,
    InferenceMethod,
    MCDropout,
    PreconditionedSGLD,
)

MethodFactory = Callable[[ClusteredDataConfig], InferenceMethod]
"""Builds a fresh inference method from the comparison's settings."""

BBB = "BBB"
"""Name of Bayes by Backprop in the results and the table."""

MCD = "MCD"
"""Name of MC dropout in the results and the table."""

PSGLD = "pSGLD"
"""Name of preconditioned SGLD in the results and the table."""


def bayes_by_backprop(config: ClusteredDataConfig) -> InferenceMethod:
    """Build Bayes by Backprop with the comparison's hyperparameters.

    Args:
        config: The comparison's settings; `bbb_learning_rate` and
            `bbb_initial_std` are read.

    Returns:
        A fresh `BayesByBackprop`.
    """
    return BayesByBackprop(
        optimizer=optax.adam(config.bbb_learning_rate),
        initial_std=config.bbb_initial_std,
    )


def mc_dropout(config: ClusteredDataConfig) -> InferenceMethod:
    """Build MC dropout with the comparison's hyperparameters.

    It infers over a backbone with dropout; see `dropout_rate`.

    Args:
        config: The comparison's settings; `mc_dropout_learning_rate` is
            read.

    Returns:
        A fresh `MCDropout`.
    """
    return MCDropout(optimizer=optax.adam(config.mc_dropout_learning_rate))


def psgld(config: ClusteredDataConfig, *, preceding_epochs: int = 0) -> InferenceMethod:
    """Build pSGLD retaining `num_draws` draws over the final half of the posterior.

    Args:
        config: The comparison's settings; `psgld_step_size` is read, and
            the draw schedule comes from `config.psgld_schedule`.
        preceding_epochs: Epochs the same run trains before its posterior
            epochs, burned in as well; 0 for the staged model's posterior
            stage.

    Returns:
        A fresh `PreconditionedSGLD`.
    """
    schedule = config.psgld_schedule(preceding_epochs=preceding_epochs)
    return PreconditionedSGLD(
        step_size=config.psgld_step_size,
        burn_in=schedule.burn_in,
        thinning=schedule.thinning,
    )


def dropout_rate(method: str, config: ClusteredDataConfig) -> float:
    """Return the backbone's dropout rate under an inference method.

    Args:
        method: The inference method's name, a key of `METHODS`.
        config: The comparison's settings; `mc_dropout_rate` is read.

    Returns:
        `config.mc_dropout_rate` for MC dropout, 0 for the other methods.
    """
    return config.mc_dropout_rate if method == MCD else 0.0


METHODS: dict[str, MethodFactory] = {
    BBB: bayes_by_backprop,
    MCD: mc_dropout,
    PSGLD: psgld,
}
"""Every inference method the comparison runs, by name, in table order."""
