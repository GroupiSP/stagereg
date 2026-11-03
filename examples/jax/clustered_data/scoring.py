"""The clustered-data example's shared scoring of per-draw predictions.

Every configuration, joint or staged, is reduced to the same
`PredictiveDraws`: one predictive mean and one aleatoric variance per
posterior draw and grid point. Its predictive distribution is the
equal-weight mixture of those Gaussians, and one scoring path serves every
configuration.

A target `y` lies inside the mixture's central `p`-interval iff its
probability integral transform `F(y)` satisfies `|F(y) - 1/2| <= p/2`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from regions import IN_DISTRIBUTION, REGIONS, RegionMasks


@dataclass(frozen=True)
class PredictiveDraws:
    """A posterior predictive reduced to per-draw Gaussians on the grid.

    Attributes:
        means: Each draw's predictive mean, shape `[S, N]`.
        variances: Each draw's aleatoric variance, shape `[S, N]`. A staged
            model broadcasts its single aleatoric variance across draws.
    """

    means: jax.Array
    variances: jax.Array

    def in_target_units(self, mean: float, std: float) -> PredictiveDraws:
        """Map draws predicted on standardized targets back to the targets' units.

        Args:
            mean: The targets' mean, which standardization subtracted.
            std: The targets' standard deviation, which standardization
                divided by.

        Returns:
            The draws with means `mean + std * means` and variances
            `std**2 * variances`.
        """
        return PredictiveDraws(
            means=mean + std * self.means, variances=std**2 * self.variances
        )


def mixture_cdf(draws: PredictiveDraws, targets: jax.Array) -> jax.Array:
    """Evaluate the equal-weight Gaussian mixture's CDF at the targets.

    Args:
        draws: The per-draw predictive means and variances, shape `[S, N]`.
        targets: One target per grid point, shape `[N]`.

    Returns:
        `F(y_n) = mean_s Phi((y_n - mu_sn) / sigma_sn)`, shape `[N]`.
    """
    scale = jnp.sqrt(draws.variances)
    return jnp.mean(norm.cdf(targets, loc=draws.means, scale=scale), axis=0)


def central_coverage(
    draws: PredictiveDraws,
    targets: jax.Array,
    level: float,
    *,
    where: jax.Array | None = None,
) -> float:
    """Return the fraction of targets inside the central `level`-interval.

    Args:
        draws: The per-draw predictive means and variances, shape `[S, N]`.
        targets: One target per grid point, shape `[N]`.
        level: The interval's nominal level, in `(0, 1)`.
        where: Optional boolean mask of shape `[N]` restricting the points
            counted, e.g. one region's.

    Returns:
        The observed coverage, in `[0, 1]`.
    """
    inside = jnp.abs(mixture_cdf(draws, targets) - 0.5) <= level / 2
    return float(jnp.mean(inside, where=where))


COVERAGE_LEVEL = 0.95
"""The nominal level of the central interval every region's coverage uses."""


@dataclass(frozen=True)
class RegionScores:
    """One region's uncertainty and coverage.

    Attributes:
        aleatoric_std: `sqrt` of the region average of the draw-averaged
            aleatoric variance, in target units.
        epistemic_std: `sqrt` of the region average of the variance over
            draws of the predictive mean, in target units.
        coverage: Fraction of the region's targets inside the mixture's
            central `COVERAGE_LEVEL`-interval.
    """

    aleatoric_std: float
    epistemic_std: float
    coverage: float


@dataclass(frozen=True)
class Scores:
    """One configuration's scores on the evaluation grid.

    Attributes:
        rmse: RMSE of the predictive mean against the targets, over the
            in-distribution region only.
        regions: Each region's uncertainty and coverage, by region name.
    """

    rmse: float
    regions: dict[str, RegionScores]


def score(draws: PredictiveDraws, targets: jax.Array, masks: RegionMasks) -> Scores:
    """Score per-draw predictions on the grid, region by region.

    Args:
        draws: The per-draw predictive means and variances, shape `[S, N]`.
        targets: One noisy target per grid point, shape `[N]`.
        masks: The grid's region masks, as `region_masks` returns them.

    Returns:
        The in-distribution RMSE and every region's `RegionScores`.
    """
    aleatoric = jnp.mean(draws.variances, axis=0)
    epistemic = jnp.var(draws.means, axis=0)
    squared_error = (jnp.mean(draws.means, axis=0) - targets) ** 2

    def region_scores(mask: jax.Array) -> RegionScores:
        return RegionScores(
            aleatoric_std=float(jnp.sqrt(jnp.mean(aleatoric, where=mask))),
            epistemic_std=float(jnp.sqrt(jnp.mean(epistemic, where=mask))),
            coverage=central_coverage(draws, targets, COVERAGE_LEVEL, where=mask),
        )

    return Scores(
        rmse=float(jnp.sqrt(jnp.mean(squared_error, where=masks[IN_DISTRIBUTION]))),
        regions={name: region_scores(masks[name]) for name in REGIONS},
    )


CALIBRATION_LEVELS: tuple[float, ...] = tuple(step / 20 for step in range(1, 20))
"""The nominal levels a calibration curve is drawn at: 0.05, 0.10, ..., 0.95."""


def calibration_curve(
    draws: PredictiveDraws,
    targets: jax.Array,
    levels: Sequence[float] = CALIBRATION_LEVELS,
    *,
    where: jax.Array | None = None,
) -> tuple[float, ...]:
    """Return the observed coverage of the central interval at each level.

    Each point is `central_coverage` at that level, so the calibration
    curve and the table's coverage share one scoring path.

    Args:
        draws: The per-draw predictive means and variances, shape `[S, N]`.
        targets: One target per grid point, shape `[N]`.
        levels: The nominal levels, each in `(0, 1)`.
        where: Optional boolean mask of shape `[N]` restricting the points
            counted, e.g. the extrapolation region's.

    Returns:
        One observed coverage in `[0, 1]` per level, in the levels' order.
    """
    return tuple(
        central_coverage(draws, targets, level, where=where) for level in levels
    )
