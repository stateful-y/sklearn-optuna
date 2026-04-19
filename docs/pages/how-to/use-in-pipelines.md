# How to Use in Pipelines

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/pipelines/) · [Open in marimo](/examples/pipelines/edit/)

This guide shows you how to use `OptunaSearchCV` with Scikit-Learn pipelines.
Use this when you need to tune hyperparameters of estimators inside a pipeline,
or nest `OptunaSearchCV` as a pipeline step.

## Prerequisites

- Sklearn-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Scikit-Learn's `Pipeline`

## Tune Pipeline Sub-Estimator Parameters

Use `__` (double underscore) syntax to address parameters of estimators nested
inside a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from optuna.distributions import FloatDistribution

from sklearn_optuna import OptunaSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200)),
])

param_distributions = {
    "clf__C": FloatDistribution(1e-2, 10.0, log=True),
}

search = OptunaSearchCV(pipe, param_distributions, n_trials=20)
search.fit(X, y)
```

The search tunes `C` on the `LogisticRegression` step while applying
`StandardScaler` in each cross-validation fold.

## Nest OptunaSearchCV Inside a Pipeline

`OptunaSearchCV` is a valid Scikit-Learn estimator, so you can use it as a
pipeline step:

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

search = OptunaSearchCV(
    LogisticRegression(max_iter=200),
    {"C": FloatDistribution(1e-2, 10.0, log=True)},
    n_trials=10,
)

pipe = Pipeline([
    ("pca", PCA(n_components=5)),
    ("search", search),
])
pipe.fit(X, y)
```

## Tune Parameters Across Multiple Pipeline Steps

Address parameters from different steps in the same search:

```python
from optuna.distributions import IntDistribution

pipe = Pipeline([
    ("pca", PCA()),
    ("clf", LogisticRegression(max_iter=200)),
])

param_distributions = {
    "pca__n_components": IntDistribution(2, 10),
    "clf__C": FloatDistribution(1e-2, 10.0, log=True),
}

search = OptunaSearchCV(pipe, param_distributions, n_trials=30)
search.fit(X, y)
```

## Manage Computational Cost

Nested searches multiply the total number of evaluations. An outer search with
10 trials wrapping an inner search with 5 trials produces 50 total evaluations
(plus cross-validation folds). Keep `n_trials` low on the inner search and
increase only after initial exploration.

## See Also

- [API Reference](../reference/api.md): full parameter documentation
- [Concepts and Architecture](../explanation/concepts.md): how OptunaSearchCV integrates with Scikit-Learn
