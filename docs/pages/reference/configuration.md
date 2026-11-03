# Configuration Reference

All parameters accepted by `OptunaSearchCV`.

## Basic Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `estimator` | estimator | *required* | The Scikit-Learn estimator to tune |
| `param_distributions` | `dict[str, BaseDistribution]` | *required* | Mapping of parameter names to Optuna distributions |
| `n_trials` | `int` | `10` | Number of Optuna trials to run |
| `timeout` | `float \| None` | `None` | Maximum seconds for the study. `None` means no limit |
| `cv` | `int \| CV splitter \| None` | `None` | Cross-validation strategy. `None` defaults to 5-fold |
| `n_jobs` | `int \| None` | `None` | Parallel trial jobs. `-1` uses all CPUs. `None` means 1 |
| `refit` | `bool \| str` | `True` | Refit the best estimator on the full dataset. Pass a metric name when using multi-metric scoring |
| `verbose` | `int` | `0` | Verbosity level for Optuna logging |

## Advanced Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `sampler` | `Sampler \| None` | `None` | Sampler wrapper. `None` uses Optuna's default (TPE) |
| `storage` | `Storage \| None` | `None` | Storage wrapper for trial persistence |
| `callbacks` | `dict[str, Callback] \| None` | `None` | Dictionary of callback wrappers invoked after each trial |
| `pruner` | `Pruner \| None` | `None` | Pruner wrapper consulted after each fold. `None` never stops a trial. Fold scores are reported to the study either way |
| `scoring` | `str \| callable \| list \| dict \| None` | `None` | Scoring metric(s). `None` uses the estimator's default scorer |
| `return_train_score` | `bool` | `False` | Include training fold scores in `cv_results_` |
| `error_score` | `float \| "raise"` | `np.nan` | Value assigned on fit failure. `"raise"` stops the search |
| `study_name` | `str \| None` | `None` | Name for the Optuna study |

## Attributes (after `fit()`)

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `best_params_` | `dict` | Parameter setting that gave the best score |
| `best_score_` | `float` | Mean cross-validated score of the best trial |
| `best_index_` | `int` | Index of the best trial in `cv_results_` |
| `best_estimator_` | estimator | Estimator refitted on full data (when `refit=True`) |
| `cv_results_` | `dict` | Dictionary with per-trial results in Scikit-Learn format: parameters, per-fold and mean scores, ranks, and `mean_fit_time`, `std_fit_time`, `mean_score_time`, `std_score_time`. One row per completed trial; pruned trials are left out |
| `study_` | `optuna.Study` | The Optuna study object |
| `trials_` | `list[FrozenTrial]` | List of all trials, including pruned ones |
| `n_splits_` | `int` | Number of cross-validation folds the resolved `cv` produced |
| `n_pruned_trials_` | `int` | Number of trials stopped by the pruner. `0` when `pruner` is `None` |

## See Also

- [API Reference](api.md): full API documentation with docstrings
- [Concepts and Architecture](../explanation/concepts.md): how these parameters interact
- [Getting Started](../tutorials/getting-started.md): basic usage tutorial
