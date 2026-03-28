# How to Persist and Resume Studies

This guide shows you how to save optimization results to a database and resume
a search from where it left off. Use this when experiments are long-running or
you want to build on previous results.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search

## Save Trials to a Database

Wrap an Optuna storage backend with the `Storage` class and pass it to
`OptunaSearchCV`:

```python
import optuna
from sklearn_optuna import OptunaSearchCV, Storage

search = OptunaSearchCV(
    estimator, param_distributions,
    storage=Storage(storage=optuna.storages.RDBStorage, url="sqlite:///study.db"),
)
search.fit(X, y)
```

Trials are persisted to `study.db` as they complete. If the process is
interrupted, completed trials are not lost.

## Resume a Previous Search

After fitting, the study is available as `search.study_`. Pass it back to
`fit()` to continue from where the previous run stopped:

```python
# First run: 50 trials
search.fit(X, y)

# Resume: 50 more trials appended to the same study
search.fit(X, y, study=search.study_)
```

If you are using a `Storage` backend, you can also resume across sessions by
loading the study from the database.

## Use a Study Name

Pass `study_name` to organize multiple studies in the same database:

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    storage=Storage(storage=optuna.storages.RDBStorage, url="sqlite:///experiments.db"),
    study_name="logistic_regression_v2",
)
```

## See Also

- [API Reference](../reference/api.md) - full wrapper API
- [Study Management example notebook](/examples/study_management/) - interactive persistence patterns
- [Concepts and Architecture](../explanation/concepts.md) - how wrappers work
