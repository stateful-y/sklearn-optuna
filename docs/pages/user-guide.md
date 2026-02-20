# User Guide

This guide provides comprehensive documentation for Sklearn-Optuna.

## Overview

Sklearn-Optuna bridges two ecosystems: Scikit-Learn's estimator API and Optuna's hyperparameter optimization framework. The main entry point, `OptunaSearchCV`, extends `BaseSearchCV` so that every Scikit-Learn pattern such as `fit()`/`score()`, `Pipeline`, `clone()`, `get_params()`/`set_params()` as well as metadata routing works without modification. Under the hood it creates an Optuna study, runs trials that call `cross_validate`, and exposes results through the standard Scikit-Learn `cv_results_` dict and Optuna `study_` object.

## Prerequisites

Before diving into Sklearn-Optuna, it's helpful to understand:

### Scikit-learn search API

Sklearn-Optuna extends Scikit-Learn's `BaseSearchCV` class, which provides the foundation for hyperparameter search estimators like `GridSearchCV` and `RandomizedSearchCV`. This means `OptunaSearchCV` inherits the complete search API: `fit(X, y)` runs the search, `score(X, y)` evaluates the best model, `best_params_` and `best_estimator_` expose results, and `cv_results_` provides detailed per-trial metrics. The search object itself is an estimator that works in pipelines, supports `clone()`, and integrates with Scikit-Learn's metadata routing system.

Learn more: [Scikit-Learn Model Selection documentation](https://scikit-learn.org/stable/model_selection.html)

### Optuna distributions

Search spaces are defined with Optuna distribution objects (`FloatDistribution`, `IntDistribution`, `CategoricalDistribution`). Each distribution describes bounds, log-scaling, and step size for a single hyperparameter.

Learn more: [Optuna Distributions API](https://optuna.readthedocs.io/en/stable/reference/distributions.html)

## Why Sklearn-Optuna?

### Compared to GridSearchCV / RandomizedSearchCV

Scikit-learn's built-in searches are either exhaustive (grid) or uniformly random. `OptunaSearchCV` replaces them with sample-efficient algorithms — primarily TPE — that focus trials on promising regions of the search space. The API is identical: swap `GridSearchCV` for `OptunaSearchCV`, replace `param_grid` with `param_distributions` using Optuna distribution objects, and everything else (`cv`, `scoring`, `refit`, `n_jobs`, `cv_results_`) stays the same.

### Compared to using Optuna directly

Raw Optuna gives you full control (multi-objective, pruning, dashboards) but requires you to write the objective function, manage cross-validation, and reconstruct results manually. Sklearn-Optuna handles all of that: it builds the objective, runs `cross_validate`, stores per-fold scores in trial user attributes, and assembles `cv_results_` automatically.

### Compared to optuna-integration's OptunaSearchCV

The existing `optuna_integration.OptunaSearchCV` provides similar functionality but with a different design philosophy. Sklearn-Optuna directly extends `BaseSearchCV`, inheriting Scikit-Learn's complete search interface including metadata routing, `clone()` support, and pipeline compatibility without additional glue code. Additionally, Sklearn-Optuna wraps Optuna components (samplers, storages, callbacks) in Scikit-Learn-compatible classes that expose `get_params()` and `set_params()`, which means they survive cloning and serialization and can be used as tunable hyperparameters in nested cross-validation scenarios. This tighter integration makes Sklearn-Optuna feel like a native Scikit-Learn component rather than a bridge between two libraries.

## Core Concepts

### OptunaSearchCV lifecycle

1. **Construction** — You pass an estimator, `param_distributions`, and optional `Sampler`/`Storage`/`Callback` wrappers.
2. **`fit(X, y)`** — Creates (or reuses) an Optuna study. Each trial suggests parameters from the distributions, clones the estimator, and runs `cross_validate`. Scores are stored as trial user attributes.
3. **Results** — After optimization, `cv_results_` is built from trial data, `best_params_` / `best_score_` / `best_index_` are set, and (if `refit=True`) `best_estimator_` is trained on the full dataset.

```python
from sklearn_optuna import OptunaSearchCV

search = OptunaSearchCV(estimator, param_distributions, n_trials=50)
search.fit(X, y)
search.best_params_      # best hyperparameters
search.cv_results_       # full results dict
search.best_estimator_   # refitted model
```

**Example:** See the Quickstart notebook ([View](/examples/quickstart/) | [Editable](/examples/quickstart/edit/)) for a complete walkthrough.

### Wrapper classes: Sampler, Storage, Callback

Optuna objects (samplers, storages, callbacks) are not Scikit-Learn-compatible by default — they lack `get_params()` / `set_params()`. Sklearn-Optuna provides thin wrappers (`Sampler`, `Storage`, `Callback`) that store the class and its constructor arguments separately. When `OptunaSearchCV` needs the actual object it calls `wrapper.instantiate()`.

This design means wrappers survive `clone()` and can be tuned as hyperparameters in nested searches.

```python
from sklearn_optuna import Sampler, Storage, Callback
import optuna

sampler = Sampler(sampler=optuna.samplers.TPESampler, seed=42)
storage = Storage(storage=optuna.storages.RDBStorage, url="sqlite:///study.db")
callback = Callback(callback=optuna.study.MaxTrialsCallback, n_trials=100)
```

### Parameter distributions

Pass a `dict[str, BaseDistribution]` as `param_distributions`. The keys must match the estimator's constructor parameter names (or use `__` syntax for pipeline sub-estimators).

```python
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

param_distributions = {
    "C": FloatDistribution(1e-3, 100.0, log=True),
    "max_iter": IntDistribution(50, 500, step=50),
    "solver": CategoricalDistribution(["lbfgs", "saga"]),
}
```

## Key Features

### Sampler selection

Choose any Optuna sampler through the `Sampler` wrapper. TPE is the default; CMA-ES works well for low-dimensional continuous spaces; `RandomSampler` gives a baseline.

```python
from sklearn_optuna import Sampler
import optuna

search = OptunaSearchCV(
    estimator, param_distributions,
    sampler=Sampler(sampler=optuna.samplers.CmaEsSampler, seed=0),
)
```

### Callbacks

Pass a dictionary of `Callback` wrappers to control optimization. Each callback is invoked at the end of every trial.

```python
from sklearn_optuna import Callback
from optuna.study import MaxTrialsCallback

search = OptunaSearchCV(
    estimator, param_distributions,
    callbacks={"max_trials": Callback(callback=MaxTrialsCallback, n_trials=50)},
)
```

**Example:** See the Callbacks notebook ([View](/examples/callbacks/) | [Editable](/examples/callbacks/edit/)) for early stopping patterns.

### Study persistence and reuse

Persist trials to a database with `Storage`, or resume a previous run by passing an existing study to `fit()`.

```python
# Persist to SQLite
search = OptunaSearchCV(
    estimator, param_distributions,
    storage=Storage(storage=optuna.storages.RDBStorage, url="sqlite:///study.db"),
)
search.fit(X, y)

# Resume later
search.fit(X, y, study=search.study_)
```

**Example:** See the Study Management notebook ([View](/examples/study_management/) | [Editable](/examples/study_management/edit/)) for reproducibility patterns.

### Multi-metric scoring

Pass multiple scorers via `scoring` and set `refit` to the metric name used for selecting the best model. All metrics appear in `cv_results_`.

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    scoring=["accuracy", "f1"],
    refit="accuracy",
)
```

**Example:** See the Metadata Routing notebook ([View](/examples/metadata_routing/) | [Editable](/examples/metadata_routing/edit/)) for advanced scoring patterns with `sample_weight`.

### Training scores and error handling

Set `return_train_score=True` to include training fold scores in `cv_results_`. Use `error_score` to control what happens when a parameter combination causes fitting to fail.

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    return_train_score=True,
    error_score="raise",  # or np.nan to skip failures
)
```

### Pipeline integration

`OptunaSearchCV` works inside and around Scikit-Learn pipelines. Use `__` syntax to address sub-estimator parameters.

```python
from Scikit-Learn.pipeline import Pipeline
from Scikit-Learn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
param_distributions = {
    "clf__C": FloatDistribution(1e-2, 10.0, log=True),
}
search = OptunaSearchCV(pipe, param_distributions, n_trials=20)
```

**Example:** See the Nested Pipeline notebook ([View](/examples/nested_pipeline/) | [Editable](/examples/nested_pipeline/edit/)) for advanced nested search patterns.

## Configuration

### Basic parameters

| Parameter     | Default | Description |
|:-------------|:--------|:-----------|
| `n_trials`   | `10`    | Number of Optuna trials to run |
| `timeout`    | `None`  | Maximum seconds for the study |
| `cv`         | `None`  | Cross-validation strategy (default 5-fold) |
| `n_jobs`     | `None`  | Parallel jobs (`-1` = all CPUs) |
| `refit`      | `True`  | Refit best estimator on full data |
| `verbose`    | `0`     | Verbosity level |

### Advanced parameters

| Parameter              | Default | Description |
|:----------------------|:--------|:-----------|
| `sampler`             | `None`  | `Sampler` wrapper (defaults to TPE internally) |
| `storage`             | `None`  | `Storage` wrapper for trial persistence |
| `callbacks`           | `None`  | `dict[str, Callback]` invoked per trial |
| `scoring`             | `None`  | Scorer string, callable, list, or dict |
| `return_train_score`  | `False` | Include training fold scores |
| `error_score`         | `np.nan`| Value on fit failure, or `"raise"` |

## Best Practices

### 1. Start small, then scale

Run a handful of trials first to verify the search space makes sense, then increase `n_trials`. Use `timeout` as a safety net for long-running experiments.

### 2. Use log-scaling for magnitude parameters

Regularization constants (`C`, `alpha`) and learning rates typically span several orders of magnitude. Use `FloatDistribution(..., log=True)` so trials are sampled uniformly in log-space.

### 3. Seed the sampler for reproducibility

Pass `seed=` to the `Sampler` wrapper so that results are deterministic when `n_jobs=1`.

```python
sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42)
```

## Limitations and Considerations

1. **No pruning yet**: Optuna's pruning API (early stopping of unpromising trials) is not wired into `OptunaSearchCV`. All trials run full cross-validation.

2. **Single-objective only**: `OptunaSearchCV` creates a single-objective (`direction="maximize"`) study. Multi-objective Pareto optimization requires using Optuna directly.

3. **Parallelism**: `n_jobs` controls Optuna's threading-based parallel trial execution. Within each trial, `cross_validate` runs with `n_jobs=1` to avoid nested parallelism issues. For CPU-bound workloads, consider using `n_jobs=1` and parallelizing at the trial level with multiprocessing storage backends.

## FAQ

### Can I use OptunaSearchCV in a Pipeline?

Yes. `OptunaSearchCV` is a valid Scikit-Learn estimator. You can use it as a step in a `Pipeline` or wrap a `Pipeline` with it.

### How do I access the Optuna study after fitting?

The study is available as `search.study_` and the trial list as `search.trials_`. You can pass `search.study_` to Optuna's visualization functions directly.

**Example:** See the Visualization notebook ([View](/examples/visualization/) | [Editable](/examples/visualization/edit/)) for plotting optimization history and parameter importance.

### Can I resume a previous search?

Pass the study from a previous run to `fit()`:

```python
search.fit(X, y, study=search.study_)
```

This appends new trials to the existing study.

## Next Steps

- Follow the [Getting Started](getting-started.md) guide for installation and first run
- Explore the [Examples](examples.md) for interactive notebooks
- Check the [API Reference](api-reference.md) for full parameter documentation
- Join the community on [GitHub Discussions](https://github.com/stateful-y/Scikit-Learn-optuna/discussions)
