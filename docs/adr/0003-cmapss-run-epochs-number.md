# Number of epochs for the CMAPSS run

_Status: the CMAPSS example has moved out of this repository, to an application-layer repo (#186). This record is kept as history._

The CMAPSS example used to run for 50 epochs both for the mean and the variance stage. By inspecting the logs, I noticed that after epoch ~20, the training loss still reduced, while the validation loss stagnated or even increased. I interpreted this as a symptom of overfitting.

Consequently, I decided to reduce the number of epochs for both stages to 20, which seems to strike a better balance between training and validation performance.

## Considered options

- Keep the original 50 epochs for both stages. This was excluded since an example showing overfitting is not a good example to leave in a probabilistic regression tutorial.
