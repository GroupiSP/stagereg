"""Run mean, variance and posterior stages on the XSin-inspired benchmark.

The VeBNN-style workflow: the mean and Gamma variance stages of
``two_steps.py``, trained once, then a posterior stage that replaces the
trained mean with a posterior over mean functions, inferred in turn by Bayes by
Backprop, pSGLD and MC-Dropout, the last on the mean network with batch-shared
dropout. All three posterior stages start from the same finalized stages. Each run
prints the NLL and closed-form CRPS of the variance stage's Gaussian and of the posterior
predictive on noisy evaluation targets, and plots the posterior predictive band
against the variance stage's band.

Run with:

    uv run --extra jax --extra plot python examples/jax/xsin/posterior.py
"""

from benchmark import StagePrintingEventSink, XSinConfig, make_xsin_data
from posterior_benchmark import (
    plot_xsin_posterior,
    print_xsin_scores,
    run_xsin_posteriors,
)


def main() -> None:
    """Train the supervised stages once, then a posterior per inference method."""
    config = XSinConfig()
    data = make_xsin_data(config)
    results = run_xsin_posteriors(data, config, event_sinks=(StagePrintingEventSink(),))
    for label, result in results.items():
        print_xsin_scores(label, result)
        plot_xsin_posterior(label, data, result, config)


if __name__ == "__main__":
    main()
