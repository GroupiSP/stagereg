"""The clustered-data example's shared scoring of posterior predictives.

Every configuration, joint or staged, is reduced to the library's
[`PosteriorPredictive`][stagereg.jax.PosteriorPredictive] on the evaluation
grid: the equal-weight mixture of one Gaussian per posterior draw. A joint
draw carries its own aleatoric variance; a staged chain's draws share its
variance stage's. One scoring path serves every configuration.

A target `y` lies inside the mixture's central `p`-interval iff its
probability integral transform `F(y)` satisfies `|F(y) - 1/2| <= p/2`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from regions import IN_DISTRIBUTION, REGIONS, RegionMasks

from stagereg.jax import PosteriorPredictive


def in_target_units(
    predictive: PosteriorPredictive, mean: float, std: float
) -> PosteriorPredictive:
    """Map a predictive on standardized targets back to the targets' units.

    Args:
        predictive: The predictive on standardized targets.
        mean: The targets' mean, which standardization subtracted.
        std: The targets' standard deviation, which standardization
            divided by.

    Returns:
        The predictive with draws `mean + std * draws` and aleatoric
        variance `std**2 * aleatoric_variance`, shared or per draw as given.
    """
    return PosteriorPredictive(
        draws=mean + std * predictive.draws,
        aleatoric_variance=std**2 * predictive.aleatoric_variance,
        draw_mask=predictive.draw_mask,
    )


def pool_chains(chains: Sequence[PosteriorPredictive]) -> PosteriorPredictive:
    """Pool several chains' predictives into one equal-weight predictive.

    Args:
        chains: The chains' predictives, all on the same grid.

    Returns:
        Every chain's draws, concatenated in order along the draw axis, each
        with its own chain's aleatoric variance.
    """
    return PosteriorPredictive(
        draws=jnp.concatenate([chain.draws for chain in chains]),
        aleatoric_variance=jnp.concatenate(
            [
                jnp.broadcast_to(chain.aleatoric_variance, chain.draws.shape)
                for chain in chains
            ]
        ),
    )


def central_coverage(
    predictive: PosteriorPredictive,
    targets: jax.Array,
    level: float,
    *,
    where: jax.Array | None = None,
) -> float:
    """Return the fraction of targets inside the central `level`-interval.

    Args:
        predictive: The predictive on the grid, batch shape `[N]`.
        targets: One target per grid point, shape `[N]`.
        level: The interval's nominal level, in `(0, 1)`.
        where: Optional boolean mask of shape `[N]` restricting the points
            counted, e.g. one region's.

    Returns:
        The observed coverage, in `[0, 1]`.
    """
    inside = jnp.abs(predictive.cdf(targets) - 0.5) <= level / 2
    return float(jnp.mean(inside, where=where))


COVERAGE_LEVEL = 0.95
"""The nominal level of the central interval every region's coverage uses."""


@dataclass(frozen=True)
class RegionScores:
    """One region's uncertainty and coverage.

    Attributes:
        aleatoric_std: `sqrt` of the region average of the moment-matched
            aleatoric variance, the draws' average, in target units.
        epistemic_std: `sqrt` of the region average of the moment-matched
            epistemic variance, the variance of the draws' means, in target
            units.
        coverage: Fraction of the region's targets inside the mixture's
            central `COVERAGE_LEVEL`-interval.
    """

    aleatoric_std: float
    epistemic_std: float
    coverage: float

    @property
    def epistemic_share(self) -> float:
        """`σ_epi / σ_total`, with `σ_total = hypot(σ_epi, σ_ale)`; 0 if both are 0."""
        total = math.hypot(self.epistemic_std, self.aleatoric_std)
        return self.epistemic_std / total if total > 0.0 else 0.0


@dataclass(frozen=True)
class Scores:
    """One configuration's scores on the evaluation grid.

    Attributes:
        r2: Coefficient of determination of the predictive mean against
            the targets, over the in-distribution region only.
        regions: Each region's uncertainty and coverage, by region name.
    """

    r2: float
    regions: dict[str, RegionScores]


def score(
    predictive: PosteriorPredictive, targets: jax.Array, masks: RegionMasks
) -> Scores:
    """Score a predictive on the grid, region by region.

    Args:
        predictive: The predictive on the grid, batch shape `[N]`.
        targets: One noisy target per grid point, shape `[N]`.
        masks: The grid's region masks, as `region_masks` returns them.

    Returns:
        The in-distribution R² and every region's `RegionScores`.
    """
    summary = predictive.moment_matched()
    in_distribution = masks[IN_DISTRIBUTION]
    residual = jnp.sum((summary.loc - targets) ** 2, where=in_distribution)
    spread = jnp.sum(
        (targets - jnp.mean(targets, where=in_distribution)) ** 2, where=in_distribution
    )

    def region_scores(mask: jax.Array) -> RegionScores:
        return RegionScores(
            aleatoric_std=float(
                jnp.sqrt(jnp.mean(summary.aleatoric_variance, where=mask))
            ),
            epistemic_std=float(
                jnp.sqrt(jnp.mean(summary.epistemic_variance, where=mask))
            ),
            coverage=central_coverage(predictive, targets, COVERAGE_LEVEL, where=mask),
        )

    return Scores(
        r2=float(1.0 - residual / spread),
        regions={name: region_scores(masks[name]) for name in REGIONS},
    )


CALIBRATION_LEVELS: tuple[float, ...] = tuple(step / 20 for step in range(1, 20))
"""The nominal levels a calibration curve is drawn at: 0.05, 0.10, ..., 0.95."""


def calibration_curve(
    predictive: PosteriorPredictive,
    targets: jax.Array,
    levels: Sequence[float] = CALIBRATION_LEVELS,
    *,
    where: jax.Array | None = None,
) -> tuple[float, ...]:
    """Return the observed coverage of the central interval at each level.

    Each point is `central_coverage` at that level, so the calibration
    curve and the table's coverage share one scoring path.

    Args:
        predictive: The predictive on the grid, batch shape `[N]`.
        targets: One target per grid point, shape `[N]`.
        levels: The nominal levels, each in `(0, 1)`.
        where: Optional boolean mask of shape `[N]` restricting the points
            counted, e.g. the extrapolation region's.

    Returns:
        One observed coverage in `[0, 1]` per level, in the levels' order.
    """
    return tuple(
        central_coverage(predictive, targets, level, where=where) for level in levels
    )
