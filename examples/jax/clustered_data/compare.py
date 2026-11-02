"""Compare how confident the staged model is in and out of distribution.

Generates the UvA tutorial's two clusters of training data and a dense
evaluation grid over `[-0.5, 1.5]`, then, for each seed, trains the staged
model on a Tanh MLP backbone of 5 hidden layers of 10 units: a mean stage
and a Gamma variance stage, shared by three runs of the library's posterior
stage, with Bayes by Backprop, MC dropout and pSGLD. The three differ only
in the inference method and the backbone's dropout rate, which is non-zero
for MC dropout alone.

Each configuration is reduced to per-draw predictive means and variances
on the grid and scored on the in-distribution region, the gap region
between the clusters and the extrapolation region beyond them. The table
reports the in-distribution RMSE of the predictive mean, then each region's
aleatoric sigma, epistemic sigma and 95% coverage of the predictive
mixture's central interval.

Run it with:

    uv run --extra jax python examples/jax/clustered_data/compare.py

Every setting is a flag; `--help` lists them with their defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow sibling-module imports both when run as a script and when loaded
# from an arbitrary working directory, e.g. via `importlib` in tests.
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from clustered import ClusteredData, ClusteredDataConfig, make_clustered_data
from methods import METHODS, dropout_rate
from regions import EXTRAPOLATION, GAP, IN_DISTRIBUTION, REGIONS, region_masks
from scoring import PredictiveDraws, Scores, score
from staged import posterior_network, run_staged, train_supervised_stages

STAGED = "staged"
"""Name of the staged training in the results and the table."""

REGION_LABELS: dict[str, str] = {
    IN_DISTRIBUTION: "ID",
    GAP: "gap",
    EXTRAPOLATION: "extrap",
}
"""Each region's short label in the table's column headers."""


@dataclass(frozen=True)
class ConfigurationResult:
    """One configuration's draws on the grid, and their scores.

    Attributes:
        training: How the posterior was inferred, e.g. `"staged"`.
        method: The inference method's name, e.g. `"BBB"`.
        draws: The per-draw predictive means and variances on the grid.
        scores: The draws' scores against the grid's noisy targets.
    """

    training: str
    method: str
    draws: PredictiveDraws
    scores: Scores

    @property
    def name(self) -> str:
        """The configuration's row label, e.g. `"staged BBB"`."""
        return f"{self.training} {self.method}"


@dataclass(frozen=True)
class SeedComparison:
    """Every configuration trained from one seed.

    Attributes:
        seed: The seed the configurations were trained from.
        point_epochs: Epochs the staged model's mean and variance stages ran
            together.
        configurations: Each configuration's result, in table order.
    """

    seed: int
    point_epochs: int
    configurations: tuple[ConfigurationResult, ...]


RunReporter = Callable[[int, ConfigurationResult], None]
"""Called as `report(seed, result)` once each configuration of a seed is scored."""


def _ignore_run(seed: int, result: ConfigurationResult) -> None:
    """Report nothing; the default `RunReporter`."""
    del seed, result


def compare_seed(
    data: ClusteredData,
    config: ClusteredDataConfig,
    *,
    on_run: RunReporter = _ignore_run,
) -> SeedComparison:
    """Train and score every configuration from `config.seed`, silently.

    Args:
        data: The training set and the evaluation grid.
        config: The comparison's settings.
        on_run: Called once per configuration as soon as it is scored.

    Returns:
        The seed's configurations and their scores.
    """
    masks = region_masks(data.grid_inputs)
    supervised = train_supervised_stages(data, config)
    configurations = []
    for method, make_method in METHODS.items():
        draws = run_staged(
            data,
            config,
            supervised,
            make_method(config),
            network=posterior_network(config, dropout_rate(method, config)),
        )
        result = ConfigurationResult(
            training=STAGED,
            method=method,
            draws=draws,
            scores=score(draws, data.grid_targets, masks),
        )
        on_run(config.seed, result)
        configurations.append(result)
    return SeedComparison(
        seed=config.seed,
        point_epochs=supervised.point_epochs,
        configurations=tuple(configurations),
    )


def compare(
    data: ClusteredData,
    config: ClusteredDataConfig,
    *,
    seeds: int,
    on_run: RunReporter = _ignore_run,
) -> list[SeedComparison]:
    """Repeat the comparison over consecutive seeds on the same data.

    Args:
        data: The training set and the evaluation grid, fixed across seeds.
        config: The comparison's settings; seed `i` runs with
            `config.seed + i`.
        seeds: Number of seeds.
        on_run: Called once per seed and configuration as soon as it is
            scored.

    Returns:
        One `SeedComparison` per seed, in seed order.
    """
    return [
        compare_seed(
            data, dataclasses.replace(config, seed=config.seed + index), on_run=on_run
        )
        for index in range(seeds)
    ]


def _metric_columns() -> list[tuple[str, Callable[[Scores], float]]]:
    """The table's columns: in-distribution RMSE, then three per region."""
    columns: list[tuple[str, Callable[[Scores], float]]] = [
        (f"{REGION_LABELS[IN_DISTRIBUTION]} RMSE", lambda scores: scores.rmse)
    ]
    for region in REGIONS:
        label = REGION_LABELS[region]
        columns += [
            (
                f"{label} σ_ale",
                lambda scores, r=region: scores.regions[r].aleatoric_std,
            ),
            (
                f"{label} σ_epi",
                lambda scores, r=region: scores.regions[r].epistemic_std,
            ),
            (f"{label} cov95", lambda scores, r=region: scores.regions[r].coverage),
        ]
    return columns


def _cell(values: Sequence[float]) -> str:
    """Format one cell: the plain value for one seed, else mean ± std."""
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        return f"{float(array[0]):.4f}"
    return f"{float(array.mean()):.4f} ± {float(array.std(ddof=1)):.4f}"


def format_table(results: Sequence[SeedComparison]) -> str:
    """Format the comparison as one row per configuration.

    Args:
        results: The per-seed results, all with the same configurations.

    Returns:
        The table: a header, then each configuration's row of in-distribution
        RMSE and every region's aleatoric sigma, epistemic sigma and 95%
        coverage, aggregated over seeds.
    """
    columns = _metric_columns()
    names = [result.name for result in results[0].configurations]
    rows = [
        (
            name,
            [
                _cell([metric(seed.configurations[index].scores) for seed in results])
                for _, metric in columns
            ],
        )
        for index, name in enumerate(names)
    ]
    label_width = max(len(name) for name in names)
    widths = [
        max(len(header), *(len(cells[column]) for _, cells in rows))
        for column, (header, _) in enumerate(columns)
    ]

    def line(label: str, cells: Sequence[str]) -> str:
        padded = (f"{cell:>{width}}" for cell, width in zip(cells, widths, strict=True))
        return "  ".join([f"{label:<{label_width}}", *padded])

    return "\n".join(
        [line("", [header for header, _ in columns])]
        + [line(name, cells) for name, cells in rows]
    )


def print_run(seed: int, result: ConfigurationResult) -> None:
    """Print one scored configuration as a progress line.

    Args:
        seed: The seed the configuration was trained from.
        result: The scored configuration.
    """
    regions = ", ".join(
        f"{REGION_LABELS[name]} cov95={region.coverage:.4f}"
        for name, region in result.scores.regions.items()
    )
    print(f"[{result.name} seed={seed}] ID RMSE={result.scores.rmse:.4f}, {regions}")


_FLAG_HELP: dict[str, str] = {
    "noise_std": "Standard deviation of the target noise",
    "batch_size": "Training points per batch",
    "learning_rate": "Adam learning rate of the mean and variance stages",
    "mean_epochs": "Maximum mean-stage epochs",
    "mean_patience": "Mean-stage early-stopping patience, in epochs",
    "variance_epochs": "Maximum variance-stage epochs",
    "variance_patience": "Variance-stage early-stopping patience, in epochs",
    "posterior_epochs": "Fixed posterior-stage epochs",
    "num_draws": "Posterior draws to score",
    "prior_precision": "Precision of the isotropic Gaussian prior",
    "bbb_learning_rate": "Adam learning rate of Bayes by Backprop",
    "bbb_initial_std": "Initial standard deviation of Bayes by Backprop",
    "mc_dropout_rate": "Dropout rate of the backbone under MC dropout",
    "mc_dropout_learning_rate": "Adam learning rate of MC dropout",
    "psgld_step_size": "Langevin step size of pSGLD",
    "seed": "Seed of the initialization, batch order and draws",
}
"""Each `ClusteredDataConfig` field's command-line help."""


def _parser() -> argparse.ArgumentParser:
    """Build the command line: one flag per `ClusteredDataConfig` field."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for field in dataclasses.fields(ClusteredDataConfig):
        default = field.default
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=type(default),
            default=default,
            help=f"{_FLAG_HELP[field.name]} (default: {default}).",
        )
    return parser


def main() -> None:
    """Run the comparison from the command line and print its table."""
    parser = _parser()
    arguments = vars(parser.parse_args())
    try:
        config = ClusteredDataConfig(**arguments)
    except ValueError as error:
        parser.error(str(error))
    data = make_clustered_data(config)
    results = compare(data, config, seeds=1, on_run=print_run)
    print(f"\nClustered data, seed {config.seed}:")
    print(format_table(results))


if __name__ == "__main__":
    main()
