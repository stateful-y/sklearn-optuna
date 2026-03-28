# How to Handle Errors

This guide shows you how to control what happens when a hyperparameter
combination causes fitting to fail, and how to include training scores in
results. Use this when your search space includes configurations that may
not converge.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search

## Control Behavior on Fit Failures

By default, `OptunaSearchCV` records `NaN` for failing trials and continues
the search. Change this with `error_score`:

### Skip failures silently (default)

```python
import numpy as np
from sklearn_optuna import OptunaSearchCV

search = OptunaSearchCV(
    estimator, param_distributions,
    error_score=np.nan,  # default
)
```

Failed trials appear in `cv_results_` with `NaN` scores but do not stop the
search.

### Raise on failure

If you want the search to stop immediately when a trial fails:

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    error_score="raise",
)
```

This is useful during development to catch unexpected parameter combinations.

## Include Training Scores

Set `return_train_score=True` to add training fold scores alongside test scores
in `cv_results_`:

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    return_train_score=True,
)
search.fit(X, y)
```

After fitting, `cv_results_` includes `mean_train_score`, `std_train_score`,
and per-split `split0_train_score`, `split1_train_score`, etc.

Comparing training and test scores helps diagnose overfitting: a large gap
between the two suggests the model memorizes the training data.

## Filter Failed Trials from Results

After fitting, you can inspect which trials failed:

```python
import pandas as pd

results = pd.DataFrame(search.cv_results_)
failed = results[results["mean_test_score"].isna()]
print(f"{len(failed)} trials failed out of {len(results)}")
```

## See Also

- [API Reference](../reference/api.md) - `error_score` and `return_train_score` parameters
- [Configuration Reference](../reference/configuration.md) - all parameter options
