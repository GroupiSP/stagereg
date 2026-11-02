"""The clustered-data example's inference methods, shared by both trainings.

Joint and staged runs build each method from the same factory, so they share its
hyperparameters.
"""

from __future__ import annotations

from collections.abc import Callable

import optax
from clustered import ClusteredDataConfig

from stagereg.jax import BayesByBackprop, InferenceMethod

MethodFactory = Callable[[ClusteredDataConfig], InferenceMethod]
"""Builds a fresh inference method from the comparison's settings."""

BBB = "BBB"
"""Name of Bayes by Backprop in the results and the table."""


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


METHODS: dict[str, MethodFactory] = {BBB: bayes_by_backprop}
"""Every inference method the comparison runs, by name, in table order."""
