# How to Use Callbacks

This guide shows you how to attach callbacks to control optimization behavior.
Use this when you need to stop the search early, log progress, or implement
custom stopping criteria.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search

## Add a Callback

Wrap any Optuna callback with the `Callback` class and pass a dictionary of
callbacks to `OptunaSearchCV`:

```python
from sklearn_optuna import OptunaSearchCV, Callback
from optuna.study import MaxTrialsCallback

search = OptunaSearchCV(
    estimator, param_distributions,
    callbacks={"max_trials": Callback(callback=MaxTrialsCallback, n_trials=50)},
)
```

Each callback in the dictionary is invoked at the end of every trial. The
dictionary key is an arbitrary name used for parameter routing.

## Stop After a Fixed Number of Trials

Use `MaxTrialsCallback` to cap the total number of trials. This is useful when
you also set `timeout` and want whichever limit is reached first:

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    n_trials=200,      # upper bound
    timeout=600,       # 10 minutes max
    callbacks={"stop": Callback(callback=MaxTrialsCallback, n_trials=100)},
)
```

## Use Multiple Callbacks

Pass multiple entries in the callbacks dictionary:

```python
from optuna.study import MaxTrialsCallback

search = OptunaSearchCV(
    estimator, param_distributions,
    callbacks={
        "max_trials": Callback(callback=MaxTrialsCallback, n_trials=100),
        "custom": Callback(callback=MyCustomCallback, threshold=0.95),
    },
)
```

## Write a Custom Callback

Any callable that accepts `study` and `trial` arguments works as a callback
class:

```python
class EarlyStoppingCallback:
    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, study, trial):
        if trial.number >= self.patience:
            best = study.best_trial.number
            if trial.number - best >= self.patience:
                study.stop()

# Wrap it
search = OptunaSearchCV(
    estimator, param_distributions,
    callbacks={"early_stop": Callback(callback=EarlyStoppingCallback, patience=10)},
)
```

## See Also

- [API Reference](../reference/api.md) - full wrapper API
- [Callbacks example notebook](/examples/callbacks/) - interactive early stopping patterns
- [Concepts and Architecture](../explanation/concepts.md) - how wrappers work
