# Distributions are pytrees; `Batch` is not

The JAX backend's distributions, `Gaussian`, `Gamma`, `MomentMatchedPredictive` and `PosteriorPredictive`, are registered as JAX pytrees, so they pass into and out of jitted functions as whole values. Compiled training, validation and prediction code (#120) returns distributions from jitted functions, and users composing their own JAX code should not have to unpack a distribution's fields by hand to cross a `jax.jit` boundary. Every field of these distributions is a child array; none is static.

`Batch` is deliberately not registered. It lives in `stagereg.core`, which imports no backend, so registering it would either pull JAX into the core or need the JAX backend to register a core type as a side effect of being imported. Its `metadata` is also an arbitrary caller mapping, which may hold strings, file paths or other non-array values that cannot be pytree leaves and would force every value in it to be hashable static data. Jitted code receives a batch's `inputs`, `targets` and `sample_weight` directly instead.

## The rule for future distributions

Every field of a distribution that is a pytree must be an array, or be declared explicitly static (auxiliary data that is hashable and compared for equality when JAX caches a compiled function). A Python scalar, string or object stored as a plain field would become a leaf: it breaks under `jax.jit`, or silently triggers a recompile per value. Adding such a field means deciding which of the two it is, not leaving it to whatever the registration does.

Unflattening rebuilds a distribution without running `__init__` or `__post_init__`, because JAX unflattens with placeholder leaves (tracers, `None`, `vmap` axis specs) that validation would reject. Validation in `__post_init__`, such as `PosteriorPredictive`'s non-empty draw axis, therefore guards construction by callers only.

## Considered Options

- **Register `Batch` too, with `metadata` as static auxiliary data.** Rejected: it gives the core a JAX dependency (or a registration side effect from the backend), and requires the metadata to be hashable.
- **Keep distributions as plain dataclasses and unpack fields at every jit boundary.** Rejected: every compiled path and every user would repeat the unpacking, and the field lists would drift from the classes.
