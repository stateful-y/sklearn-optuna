# How to Use in Pipelines

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/nested_pipeline/) · [Open in marimo](/examples/nested_pipeline/edit/)

This guide shows you how to use `OptunaSearchCV` with scikit-learn pipelines.
Use this when you need to tune hyperparameters of estimators inside a pipeline,
or nest `OptunaSearchCV` as a pipeline step.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with scikit-learn's `Pipeline`

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

`OptunaSearchCV` is a valid scikit-learn estimator, so you can use it as a
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

## See Also

- [API Reference](../reference/api.md) - full parameter documentation
- [Concepts and Architecture](../explanation/concepts.md) - how OptunaSearchCV integrates with scikit-learn
