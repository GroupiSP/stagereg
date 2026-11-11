# RUL curves refuse padded windows, while training accepts them

_Status: the CMAPSS example has moved out of this repository, to an application-layer repo (#186). This record is kept as history._

`build_windows` left-pads a unit shorter than the window length by repeating its first observed cycle, so that every unit contributes at least one training window. `build_unit_rul_curve`, which prepares a single unit's per-cycle RUL curve, deliberately does the opposite: it raises `ValueError` for such a unit rather than padding it. The two padding policies live in the same module and disagree on purpose.

The reason is what a padded window means in each setting. A padded window's sensor rows are fabricated — repeated readings that were never observed — while its RUL target is always computed from the unit's real `time_cycles`. As a training sample that is a tolerable approximation, and dropping the unit entirely would lose real data. As a plotted point it is a claim about the model's behaviour at a cycle where the model was never given genuine history, drawn indistinguishably from the points either side of it. The figure exists to let a reader judge the model's accuracy and its stated uncertainty, so a point with no support in the data is worse than a missing one.

## Considered options

- **Pad in both places.** Rejected: it manufactures predictions and reads as a real curve. Issue #21 asked explicitly that "no left-padding is introduced to manufacture predictions".
- **Refuse in both places.** Rejected: it would silently discard short units from training, changing what the models are fitted on to serve a plotting concern.
- **Return an empty curve instead of raising.** Rejected: a caller that selected a unit and got nothing back would draw an empty column, or narrow the figure without saying why. The exception is loud, and the plotting entry point's caller — `run.py`, which chooses the units — is the right place to hear it.

## Consequences

`plot_validation_rul_curves` propagates the `ValueError` rather than skipping the offending column, so a caller selecting a unit shorter than the window length fails loudly. FD001's train-split lifetimes are far above the example's 30-cycle window, so this cannot arise from the real dataset; it is reachable only from synthetic or subsetted frames.
