"""Numerical, NumPy-only metrics for probabilistic regression."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_nonempty_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a finite, nonempty floating-point vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _as_nonempty_matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Return a finite, two-dimensional floating-point matrix without empty axes.

    Args:
        values: The values to convert.
        name: The argument name used in error messages.

    Returns:
        ``values`` as a floating-point matrix.

    Raises:
        ValueError: If ``values`` is not two-dimensional, is empty, or holds a
            non-finite value.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if matrix.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _evaluation_grid(values: ArrayLike) -> NDArray[np.float64]:
    """Validate and return a numerical integration grid."""
    grid = _as_nonempty_vector(values, name="x_grid")
    if grid.size < 2:
        raise ValueError("x_grid must contain at least two values.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")
    return grid


def _matching_vectors(
    *named_values: tuple[str, ArrayLike],
) -> tuple[NDArray[np.float64], ...]:
    vectors = tuple(
        _as_nonempty_vector(values, name=name) for name, values in named_values
    )
    if len({vector.shape for vector in vectors}) != 1:
        names = ", ".join(name for name, _ in named_values)
        raise ValueError(f"{names} must have matching shapes.")
    return vectors


def cdf(x: float, samples: ArrayLike) -> float:
    """Return the empirical cumulative distribution at ``x``."""
    sample_vector = np.asarray(samples, dtype=float)
    if sample_vector.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if sample_vector.size == 0:
        return 0.0
    if not np.isfinite(x) or not np.all(np.isfinite(sample_vector)):
        raise ValueError("x and samples must contain only finite values.")
    return float(np.mean(sample_vector <= x))


def crps(samples_true: ArrayLike, samples_pred: ArrayLike, x_grid: ArrayLike) -> float:
    """Approximate expected CRPS under a scalar empirical reference distribution.

    Args:
        samples_true: Samples from the reference scalar distribution.
        samples_pred: Samples from the predictive scalar distribution.
        x_grid: Finite, strictly increasing integration coordinates.

    Returns:
        The predictive CRPS averaged over the empirical reference distribution.

    Raises:
        ValueError: If samples or the grid are empty, non-finite, or malformed.
    """
    grid = _evaluation_grid(x_grid)
    true_samples = _as_nonempty_vector(samples_true, name="samples_true")
    predicted_samples = _as_nonempty_vector(samples_pred, name="samples_pred")
    true_cdf = np.mean(true_samples[None, :] <= grid[:, None], axis=1)
    predicted_cdf = np.mean(predicted_samples[None, :] <= grid[:, None], axis=1)
    expected_squared_distance = (
        true_cdf * (1.0 - predicted_cdf) ** 2 + (1.0 - true_cdf) * predicted_cdf**2
    )
    return float(np.trapezoid(expected_squared_distance, grid))


def point_crps(y_true: float, samples_pred: ArrayLike, x_grid: ArrayLike) -> float:
    """Approximate CRPS between a scalar target and empirical predictions.

    Args:
        y_true: Scalar reference target.
        samples_pred: Samples from the predictive scalar distribution.
        x_grid: Finite, strictly increasing integration coordinates.

    Returns:
        The grid-integrated squared distance between the point and empirical CDFs.

    Raises:
        ValueError: If inputs are empty, non-finite, or malformed.
    """
    if not np.isfinite(y_true):
        raise ValueError("y_true must be finite.")
    grid = _evaluation_grid(x_grid)
    predicted_samples = _as_nonempty_vector(samples_pred, name="samples_pred")
    true_cdf = np.where(grid < y_true, 0.0, 1.0)
    predicted_cdf = np.mean(predicted_samples[None, :] <= grid[:, None], axis=1)
    return float(np.trapezoid((true_cdf - predicted_cdf) ** 2, grid))


def sample_crps(y_true: float, samples_pred: ArrayLike) -> float:
    """Return the exact CRPS between a scalar target and empirical predictions.

    It uses the energy form ``E|X - y| - E|X - X'| / 2`` of the CRPS, with ``X``
    and ``X'`` drawn from the empirical distribution of ``samples_pred``, so
    no evaluation grid is needed. It equals the integral of the squared
    distance between the empirical and the target's CDFs over the whole line.

    Args:
        y_true: Scalar reference target.
        samples_pred: Samples from the predictive scalar distribution.

    Returns:
        The CRPS of the empirical predictive distribution at ``y_true``.

    Raises:
        ValueError: If the target is not finite, or the samples are empty,
            non-finite, or not one-dimensional.
    """
    if not np.isfinite(y_true):
        raise ValueError("y_true must be finite.")
    samples = np.sort(_as_nonempty_vector(samples_pred, name="samples_pred"))
    count = samples.size
    absolute_error = np.mean(np.abs(samples - y_true))
    ranks = 2.0 * np.arange(count) - count + 1.0
    mean_pairwise_distance = 2.0 * np.dot(ranks, samples) / count**2
    return float(absolute_error - mean_pairwise_distance / 2.0)


def _mean_pairwise_distance(sorted_samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``E|X - X'|`` for each row of row-sorted samples."""
    count = sorted_samples.shape[-1]
    ranks = 2.0 * np.arange(count) - count + 1.0
    return 2.0 * (sorted_samples @ ranks) / count**2


def expected_sample_crps(
    predictive_samples: ArrayLike, reference_samples: ArrayLike
) -> NDArray[np.float64]:
    """Return the exact expected CRPS of each unit's empirical predictive.

    For each scoring unit (row) it evaluates ``E|X - Y| - E|X - X'| / 2``,
    with ``X`` and ``X'`` drawn from the unit's predictive samples and ``Y``
    from its reference samples. This equals the mean of
    [`sample_crps`][stagereg.core.sample_crps] over the unit's reference
    samples. No evaluation grid is needed, and all units are scored at once
    by sorting rather than looping over reference samples.

    Args:
        predictive_samples: Predictive samples of shape ``(units, S)``.
        reference_samples: Reference samples of shape ``(units, R)``, such as
            draws from a simulator.

    Returns:
        The expected CRPS of each unit, of shape ``(units,)``.

    Raises:
        ValueError: If either input is empty, non-finite or not
            two-dimensional, or if their numbers of units differ.
    """
    predictive = _as_nonempty_matrix(predictive_samples, name="predictive_samples")
    reference = _as_nonempty_matrix(reference_samples, name="reference_samples")
    if predictive.shape[0] != reference.shape[0]:
        raise ValueError(
            "predictive_samples and reference_samples must have the same number "
            "of units."
        )
    predictive_count = predictive.shape[1]
    reference_count = reference.shape[1]

    # Sort predictive and reference samples jointly per row. For each reference
    # sample, the predictive samples before it in that order are not larger
    # than it, which gives sum_i |x_i - y| from a running count and sum.
    pooled = np.concatenate([predictive, reference], axis=1)
    order = np.argsort(pooled, axis=1, kind="stable")
    pooled = np.take_along_axis(pooled, order, axis=1)
    is_predictive = order < predictive_count
    predictive_below = np.cumsum(is_predictive, axis=1)
    sum_below = np.cumsum(np.where(is_predictive, pooled, 0.0), axis=1)
    total = predictive.sum(axis=1, keepdims=True)
    distance_to_each = (
        pooled * (2.0 * predictive_below - predictive_count) + total - 2.0 * sum_below
    )
    absolute_error = np.where(is_predictive, 0.0, distance_to_each).sum(axis=1) / (
        predictive_count * reference_count
    )
    spread = _mean_pairwise_distance(np.sort(predictive, axis=1))
    return absolute_error - spread / 2.0


_erf = np.vectorize(math.erf, otypes=[np.float64])


def gaussian_mixture_crps_kernel(
    xp: ModuleType,
    erf: Callable[[Any], Any],
    draw_means: Any,
    variance: Any,
    references: Any,
    weights: Any,
) -> Any:
    """Return the closed-form expected CRPS of a weighted Gaussian mixture.

    The one implementation of the Grimit et al. (2006) formula, written against
    an array module so that the core stays free of any backend (ADR 0012): the
    core passes NumPy, and the JAX backend passes ``jax.numpy``, which keeps the
    result traceable under ``jax.jit``. It does no validation; callers check
    their inputs. With ``A(mu, v) = E|Z|`` for ``Z ~ N(mu, v)``, the CRPS of a
    mixture with component weights ``w_m`` at a target ``y`` is::

        sum_m w_m A(mu_m - y, s**2) - (1/2) sum_{m,m'} w_m w_m' A(mu_m - mu_m', 2 s**2)

    Args:
        xp: The array module, ``numpy`` or ``jax.numpy``.
        erf: The error function matching ``xp``.
        draw_means: Component means, shaped ``[M, *batch]``; all must be finite.
        variance: The positive variance every component shares, shaped
            ``[*batch]``.
        references: Reference samples, shaped ``[R, *batch]``.
        weights: Component weights summing to one, shaped ``[M]``.

    Returns:
        The CRPS averaged over each unit's reference samples, shaped
        ``[*batch]``.
    """

    def expected_absolute(location: Any, location_variance: Any) -> Any:
        scale = xp.sqrt(location_variance)
        z = location / scale
        density = xp.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        return 2.0 * scale * density + location * erf(z / math.sqrt(2.0))

    weights = xp.reshape(weights, (-1,) + (1,) * (draw_means.ndim - 1))
    absolute_error = xp.sum(
        weights
        * expected_absolute(draw_means[:, None] - references[None, :], variance).mean(
            axis=1
        ),
        axis=0,
    )
    pairwise = expected_absolute(
        draw_means[:, None] - draw_means[None, :], 2.0 * variance
    )
    spread = xp.sum(weights[:, None] * weights[None, :] * pairwise, axis=(0, 1))
    return absolute_error - spread / 2.0


def expected_gaussian_mixture_crps(
    draw_means: ArrayLike, variance: ArrayLike, reference_samples: ArrayLike
) -> NDArray[np.float64]:
    """Return the closed-form expected CRPS of each unit's Gaussian mixture.

    Each scoring unit (row) predicts an equally weighted mixture of Gaussians
    ``N(mu_m, s**2)``, one per draw mean, all sharing the unit's variance
    ``s**2``. With ``A(mu, v) = E|Z|`` for ``Z ~ N(mu, v)``, its CRPS at a
    target ``y`` is (Grimit et al. 2006)::

        (1/M) sum_m A(mu_m - y, s**2) - (1/(2 M**2)) sum_{m,m'} A(mu_m - mu_m', 2 s**2)

    and this function returns its mean over the unit's reference samples. A
    point target is a single reference sample (``R = 1``). With one draw it is
    the closed-form CRPS of a Gaussian. Unlike
    [`expected_sample_crps`][stagereg.core.expected_sample_crps] it scores the
    predictive's own formula rather than samples drawn from it, so it carries no
    Monte Carlo noise. The cost is ``O(M R + M**2)`` per unit.

    Args:
        draw_means: Mean of each mixture component, of shape ``(units, M)``.
        variance: The variance every component of a unit shares, of shape
            ``(units,)``.
        reference_samples: Reference samples of shape ``(units, R)``, such as
            the observed target (``R = 1``) or draws from a simulator.

    Returns:
        The expected CRPS of each unit, of shape ``(units,)``.

    Raises:
        ValueError: If an input is empty or non-finite, if ``draw_means`` or
            ``reference_samples`` is not two-dimensional, if ``variance`` is
            not one-dimensional or not positive, or if the numbers of units
            differ.
    """
    means = _as_nonempty_matrix(draw_means, name="draw_means")
    reference = _as_nonempty_matrix(reference_samples, name="reference_samples")
    shared_variance = _as_nonempty_vector(variance, name="variance")
    if not means.shape[0] == reference.shape[0] == shared_variance.shape[0]:
        raise ValueError(
            "draw_means, variance and reference_samples must have the same number "
            "of units."
        )
    if np.any(shared_variance <= 0.0):
        raise ValueError("variance must be positive.")
    draws = means.shape[1]
    return gaussian_mixture_crps_kernel(
        np, _erf, means.T, shared_variance, reference.T, np.full(draws, 1.0 / draws)
    )


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the root mean squared error."""
    target, prediction = _matching_vectors(("y_true", y_true), ("y_pred", y_pred))
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Return the fraction of targets within inclusive prediction intervals."""
    target, lower_bound, upper_bound = _matching_vectors(
        ("y_true", y_true), ("lower", lower), ("upper", upper)
    )
    if np.any(lower_bound > upper_bound):
        raise ValueError("lower must not exceed upper.")
    return float(np.mean((lower_bound <= target) & (target <= upper_bound)))


def wsu(lower: ArrayLike, upper: ArrayLike, x: ArrayLike) -> float:
    """Return the normalized weighted spread of uncertainty intervals.

    The calculation preserves QMoDeM's established WSU definition while using a generic,
    strictly increasing coordinate rather than battery-specific time.
    """
    lower_bound, upper_bound, coordinate = _matching_vectors(
        ("lower", lower), ("upper", upper), ("x", x)
    )
    if coordinate.size < 3:
        raise ValueError("x, lower, and upper must contain at least three values.")
    if np.any(lower_bound > upper_bound):
        raise ValueError("lower must not exceed upper.")
    if np.any(np.diff(coordinate) <= 0):
        raise ValueError("x must be strictly increasing.")

    interval_widths = (upper_bound[2:] + upper_bound[1:-1]) / 2 - (
        lower_bound[2:] + lower_bound[1:-1]
    ) / 2
    period = coordinate[-1] - coordinate[0]
    return float(np.dot(interval_widths, coordinate[1:-1] - coordinate[0]) / period**2)
