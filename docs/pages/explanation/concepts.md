# Concepts and Architecture

Sklearn-optuna bridges two ecosystems: scikit-learn's estimator API and
Optuna's hyperparameter optimization framework. This page explains how the
integration works and what trade-offs were made.

## The Two Ecosystems

Scikit-learn organizes machine learning around **estimators** - objects with a
`fit()` / `predict()` interface, parameter introspection via `get_params()` /
`set_params()`, and deep cloning via `clone()`. Hyperparameter search is built
on `BaseSearchCV`, which adds cross-validation, result aggregation, and refitting
to any estimator.

Optuna organizes optimization around **studies** and **trials**. A study manages
the optimization loop; each trial suggests parameters from a sampler, evaluates
an objective function, and stores results. Optuna provides sample-efficient
algorithms (TPE, CMA-ES) and persistence via pluggable storage backends.

The challenge is that these two systems have incompatible assumptions about how
objects are constructed, cloned, and parameterized.

## How OptunaSearchCV Works

`OptunaSearchCV` extends scikit-learn's `BaseSearchCV`. When you call `fit()`:

1. **Study creation** - An Optuna study is created (or resumed if you pass an
   existing study). The sampler, storage, and study name are configured from
   the wrapper objects you provided at construction time.

2. **Trial loop** - The study's `optimize()` method calls an internal
   `_Objective` instance for each trial. The objective asks Optuna to suggest
   parameter values from the distributions you provided, clones the estimator
   with those values, and runs `cross_validate()`.

3. **Score storage** - Per-fold scores are stored as trial user attributes
   (`mean_test_score`, `split0_test_score`, etc.), making them available
   through both Optuna's trial API and scikit-learn's `cv_results_` dict.

4. **Result assembly** - After optimization completes, `_build_cv_results()`
   transforms the list of `FrozenTrial` objects into scikit-learn's standard
   `cv_results_` format. `best_params_`, `best_score_`, and `best_index_` are
   set from the best trial.

5. **Refit** - If `refit=True` (the default), the best estimator is trained on
   the full dataset and stored as `best_estimator_`.

Because `OptunaSearchCV` inherits from `BaseSearchCV`, everything that works
with scikit-learn's search API works here too: pipelines, `clone()`, metadata
routing, and serialization.

## Wrapper Classes

Optuna's samplers, storage backends, and callbacks are plain Python objects.
They do not implement `get_params()` or `set_params()`, which means scikit-learn
cannot clone them, serialize them, or route parameters through them.

The `Sampler`, `Storage`, and `Callback` wrappers solve this by storing the
**class** and its **constructor arguments** separately:

```python
from sklearn_optuna import Sampler
import optuna

# This stores: class=TPESampler, kwargs={seed: 42}
sampler = Sampler(sampler=optuna.samplers.TPESampler, seed=42)

# scikit-learn can now introspect and clone it
sampler.get_params()
# {'sampler': <class 'optuna.samplers.TPESampler'>, 'seed': 42}
```

When `OptunaSearchCV` needs the actual Optuna object, it calls
`wrapper.instantiate()`, which constructs a fresh instance from the stored
class and arguments.

This design means wrappers survive `clone()`, work in nested cross-validation,
and can even be treated as tunable hyperparameters themselves.

## Parameter Distributions

Search spaces are defined with Optuna distribution objects rather than
scikit-learn-style lists or ranges:

- `FloatDistribution(low, high, log=False, step=None)` - continuous float range
- `IntDistribution(low, high, log=False, step=1)` - integer range
- `CategoricalDistribution(choices)` - discrete set of values

Using `log=True` for parameters like regularization strength (`C`, `alpha`) or
learning rates is important because these parameters typically span several
orders of magnitude. Log-scaling ensures the sampler explores each order of
magnitude equally.

The distribution keys must match the estimator's constructor parameter names.
For pipeline sub-estimators, use `__` (double underscore) syntax:
`"clf__C"` addresses the `C` parameter of the `clf` step.

## Parallelism

`OptunaSearchCV` supports parallel trial execution via the `n_jobs` parameter.
Trials run concurrently using threading, which works well with Optuna's
thread-safe study implementation.

Within each trial, `cross_validate` runs with `n_jobs=1` to avoid nested
parallelism. This is a deliberate trade-off: nested parallel workers (trials x
CV folds) compete for CPU resources and often slow things down due to overhead.

For CPU-bound workloads, this means the effective parallelism is at the trial
level, not the fold level.

## Limitations

- **No pruning** - Optuna's pruning API (early stopping of unpromising trials)
  is not wired into `OptunaSearchCV`. All trials run full cross-validation.

- **Single-objective only** - `OptunaSearchCV` creates a single-objective study
  (`direction="maximize"`). Multi-objective Pareto optimization requires using
  Optuna directly.

- **Threading-based parallelism** - Parallel trials use threading, not
  multiprocessing. For most scikit-learn estimators that release the GIL during
  fitting, this works well. For pure-Python estimators, `n_jobs=1` may be
  faster.

## FAQ

### Can I use OptunaSearchCV in a Pipeline?

Yes. `OptunaSearchCV` is a valid scikit-learn estimator. You can use it as a
step in a `Pipeline` or wrap a `Pipeline` with it. See
[How to Use in Pipelines](../how-to/use-in-pipelines.md).

### How do I access the Optuna study after fitting?

The study is available as `search.study_` and the trial list as
`search.trials_`. You can pass `search.study_` to Optuna's visualization
functions directly. See the
Visualization example notebook ([View](/examples/visualization/) · [Open in marimo](/examples/visualization/edit/)) for plotting
optimization history and parameter importance.

### Can I resume a previous search?

Pass the study from a previous run to `fit()`:

```python
search.fit(X, y, study=search.study_)
```

This appends new trials to the existing study. See
[How to Persist and Resume Studies](../how-to/persist-studies.md).

## Further Reading

- [Getting Started](../tutorials/getting-started.md) - hands-on tutorial
- [Configuration Reference](../reference/configuration.md) - all OptunaSearchCV parameters
- [API Reference](../reference/api.md) - full API documentation
