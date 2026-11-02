"""The clustered-data example's regions of the evaluation grid.

The glossary's three regions partition the grid over `[-0.5, 1.5]`:

* the **in-distribution region**, `[-0.2, 0.2]` and `[0.6, 1.0]`, which the
  two training clusters cover;
* the **gap region**, `(0.2, 0.6)`, between the clusters;
* the **extrapolation region**, `[-0.5, -0.2)` and `(1.0, 1.5]`, beyond them.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

IN_DISTRIBUTION = "in_distribution"
"""Name of the in-distribution region."""

GAP = "gap"
"""Name of the gap region."""

EXTRAPOLATION = "extrapolation"
"""Name of the extrapolation region."""

REGIONS: tuple[str, ...] = (IN_DISTRIBUTION, GAP, EXTRAPOLATION)
"""The region names, in the order the metrics table reports them."""

CLUSTERS: tuple[tuple[float, float], ...] = ((-0.2, 0.2), (0.6, 1.0))
"""The closed input intervals the two training clusters cover."""

RegionMasks = Mapping[str, jax.Array]
"""Boolean masks over the grid's points, shape `[N]`, keyed by region name."""


def region_masks(inputs: jax.Array) -> RegionMasks:
    """Split grid inputs into the in-distribution, gap and extrapolation regions.

    Args:
        inputs: Grid inputs, shape `[N, 1]` or `[N]`.

    Returns:
        One boolean mask of shape `[N]` per region in `REGIONS`; every point
        is in exactly one region.
    """
    points = jnp.reshape(inputs, (-1,))
    (first_low, first_high), (second_low, second_high) = CLUSTERS
    in_distribution = ((points >= first_low) & (points <= first_high)) | (
        (points >= second_low) & (points <= second_high)
    )
    gap = (points > first_high) & (points < second_low)
    return {
        IN_DISTRIBUTION: in_distribution,
        GAP: gap,
        EXTRAPOLATION: ~(in_distribution | gap),
    }
