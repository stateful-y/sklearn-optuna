# How to Configure Samplers

This guide shows you how to choose and configure Optuna samplers for your
hyperparameter search. Use this when you want to control the optimization
algorithm or need reproducible results.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search

## Choose a Sampler

Wrap any Optuna sampler with the `Sampler` class to make it compatible with
scikit-learn's `get_params()` / `set_params()` / `clone()` API:

```python
from sklearn_optuna import OptunaSearchCV, Sampler
import optuna

search = OptunaSearchCV(
    estimator, param_distributions,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
)
```

### TPE (default)

Tree-structured Parzen Estimator is the default when no sampler is specified.
It works well for most search spaces:

```python
sampler = Sampler(sampler=optuna.samplers.TPESampler, seed=42)
```

### CMA-ES

CMA-ES works well for low-dimensional continuous search spaces. It is not
suitable for categorical parameters:

```python
sampler = Sampler(sampler=optuna.samplers.CmaEsSampler, seed=0)
```

### Random

Use `RandomSampler` as a baseline comparison or when you want uniform coverage
of the search space:

```python
sampler = Sampler(sampler=optuna.samplers.RandomSampler, seed=0)
```

## Set a Seed for Reproducibility

Pass `seed=` to the `Sampler` constructor. Results are deterministic when
`n_jobs=1`:

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    n_jobs=1,
)
```

If you use `n_jobs > 1`, trial ordering is non-deterministic and results may
vary between runs even with the same seed.

## Use a Sampler in Nested Cross-Validation

Because the `Sampler` wrapper supports `get_params()` and `set_params()`, it
survives `clone()` and can be used as a tunable hyperparameter:

```python
from sklearn.model_selection import cross_val_score

search = OptunaSearchCV(
    estimator, param_distributions,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
)

# The sampler is cloned correctly for each outer fold
scores = cross_val_score(search, X, y, cv=5)
```

## See Also

- [API Reference](../reference/api.md) - full wrapper API
- [Concepts and Architecture](../explanation/concepts.md) - how wrappers work
- [Optuna Samplers documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) - all available samplers
