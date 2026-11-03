---
description: How the Scikit-Learn and Optuna ecosystems fit together, and the trade-offs the integration makes.
---

# Concepts and Architecture

Sklearn-Optuna bridges two ecosystems: Scikit-Learn's estimator API and
Optuna's hyperparameter optimization framework. This page explains how the
integration works and what trade-offs were made.

## The Two Ecosystems

Scikit-Learn organizes machine learning around **estimators**: objects with a
`fit()` / `predict()` interface, parameter introspection via `get_params()` /
`set_params()`, and deep cloning via `clone()`. Hyperparameter search is built
on `BaseSearchCV`, which adds cross-validation, result aggregation, and refitting
to any estimator.

Optuna organizes optimization around **studies** and **trials**. A study manages
the optimization loop; each trial suggests parameters from a sampler, evaluates
an objective function, and stores results. Optuna provides sample-efficient
algorithms (TPE, CMA-ES) and persistence via pluggable storage backends.

The challenge is that these two systems have incompatible assumptions about how
objects are constructed, cloned, and parameterized. Scikit-Learn requires every
object to expose its constructor arguments through `get_params()` so that
`clone()` can reconstruct it, and `set_params()` can modify it. Optuna objects
like `TPESampler` or `RDBStorage` do not implement this protocol: they hold
internal state (RNG history, database connections) that is not exposed as
constructor kwargs, cannot be serialized, and cannot be reconstructed from
parameters alone. Without a bridge layer, Scikit-Learn cannot clone, route
parameters through, or nest Optuna objects in pipelines.

## How OptunaSearchCV Works

`OptunaSearchCV` extends Scikit-Learn's `BaseSearchCV`. When you call `fit()`:

1. **Study creation**: An Optuna study is created (or resumed if you pass an
   existing study). The sampler, storage, and study name are configured from
   the wrapper objects you provided at construction time.

2. **Trial loop**: The study's `optimize()` method calls an internal
   `_Objective` instance for each trial. The objective asks Optuna to suggest
   parameter values from the distributions you provided, clones the estimator
   with those values, and fits and scores each cross-validation fold in turn,
   using the same per-fold routine Scikit-Learn's own searches use.

3. **Score storage**: As each fold finishes, its scores and timings are stored
   as trial user attributes (`split0_test_score`, `split0_fit_time`, etc.) and
   its test score is reported to the trial as an intermediate value. The means
   (`mean_test_score`, `mean_fit_time`, etc.) are stored once every fold has
   run. Everything is available through Optuna's trial API; the per-fold
   scores, the means, and the timing means and standard deviations also
   appear in Scikit-Learn's `cv_results_` dict, while the per-fold timings
   stay on the trial.

4. **Result assembly**: After optimization completes, `_build_cv_results()`
   transforms the list of `FrozenTrial` objects into Scikit-Learn's standard
   `cv_results_` format. `best_params_`, `best_score_`, and `best_index_` are
   set from the best trial.

5. **Refit**: If `refit=True` (the default), the best estimator is trained on
   the full dataset and stored as `best_estimator_`.

Because `OptunaSearchCV` inherits from `BaseSearchCV`, everything that works
with Scikit-Learn's search API works here too: pipelines, `clone()`, metadata
routing, and serialization.

## Wrapper Classes

Optuna's samplers, storage backends, pruners, and callbacks are plain Python
objects. They do not implement `get_params()` or `set_params()`, which means
Scikit-Learn cannot clone them, serialize them, or route parameters through
them.

The `Sampler`, `Storage`, `Pruner`, and `Callback` wrappers solve this using
[Sklearn-Wrap](https://github.com/stateful-y/sklearn-wrap)'s `BaseClassWrapper`,
which stores the **class** and its **constructor arguments** separately:

```python
from sklearn_optuna import Sampler
import optuna

# This stores: class=TPESampler, kwargs={seed: 42}
sampler = Sampler(sampler=optuna.samplers.TPESampler, seed=42)

# Scikit-Learn can now introspect and clone it
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
Scikit-Learn-style lists or ranges:

- `FloatDistribution(low, high, log=False, step=None)`: continuous float range
- `IntDistribution(low, high, log=False, step=1)`: integer range
- `CategoricalDistribution(choices)`: discrete set of values

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

Within each trial, the folds run one after another. This is a deliberate
trade-off: nested parallel workers (trials x CV folds) compete for CPU
resources and often slow things down due to overhead, and running folds in
sequence is what lets a trial be stopped between folds.

For CPU-bound workloads, this means the effective parallelism is at the trial
level, not the fold level.

### Multi-node optimization

Because Optuna studies can be shared through a database backend, you can
distribute trials across multiple machines. Each node runs its own
`OptunaSearchCV.fit()` pointing to the same `Storage` and `study_name`, and
Optuna coordinates trial suggestions automatically. See
[How to Persist and Resume Studies](../how-to/persist-studies.md) for storage
setup and Optuna's
[distributed optimization guide](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-node-optimization)
for the full pattern.

## Pruning

Every trial reports each fold's test score to Optuna as an intermediate value
at step `i` for fold `i`. That record exists whether or not you ask for
pruning: it is what `optuna.visualization.plot_intermediate_values` draws, and
it is what a pruner reads.

When you pass a `Pruner`, the objective asks it after every fold whether the
trial should stop. The default wrapped pruner is Optuna's `WilcoxonPruner`,
which runs a paired statistical test between this trial's fold scores and the
best trial's scores on the same folds. Pairing fold `i` with fold `i` is only
meaningful when fold `i` is the same data in every trial, which is why `fit()`
splits the data once and scores every trial on those folds, whatever the
splitter. The test needs several pairs before it
can be confident: at its default threshold it cannot stop a trial before the
fourth fold, so a three-fold search never prunes.

A stopped trial ends in Optuna's `PRUNED` state. It keeps its parameters and
the per-fold attributes of the folds that ran, but has no mean score, and it
is left out of `cv_results_` so that every row of that table remains a full
evaluation. `trials_` and `study_` still hold it, and `n_pruned_trials_`
counts them. A fold whose fit failed under `error_score=np.nan` is not
reported at all: a failed fold is not a measurement, and reporting NaN would
make the pruners disagree about what it means.

See [How to Prune Unpromising Trials](../how-to/prune-trials.md) for the
procedure.

## Limitations

- **Pruning between folds only**: a trial can be stopped after a fold, not in
  the middle of a fit. Stopping a boosting model after some of its rounds, or
  an iterative model after some epochs, is not supported.

- **Single-objective only**: `OptunaSearchCV` creates a single-objective study
  (`direction="maximize"`). Multi-objective Pareto optimization requires using
  Optuna directly.

- **Threading-based parallelism**: Parallel trials on a single machine use
  threading, not multiprocessing. For true multi-process or multi-node parallelism,
  use a shared database storage so each process or machine runs its own
  `OptunaSearchCV.fit()` against the same study (see
  [Multi-node optimization](#multi-node-optimization) above).

## FAQ

### Can I use OptunaSearchCV in a Pipeline?

Yes. `OptunaSearchCV` is a valid Scikit-Learn estimator. You can use it as a
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

- [Getting Started](../tutorials/getting-started.md): hands-on tutorial
- [Configuration Reference](../reference/configuration.md): all OptunaSearchCV parameters
- [API Reference](../reference/api.md): full API documentation
