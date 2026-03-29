# How to Route Metadata

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/metadata_routing/) · [Open in marimo](/examples/metadata_routing/edit/)

This guide shows you how to pass sample metadata (weights, groups, or custom
properties) through `OptunaSearchCV` to the underlying estimator. Use this when
your scoring or fitting logic depends on per-sample information like class
weights or group labels.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search
- scikit-learn's metadata routing enabled

## Enable Metadata Routing

Activate scikit-learn's metadata routing before creating your search:

```python
import sklearn
sklearn.set_config(enable_metadata_routing=True)
```

This is required for any metadata to flow through cross-validation and into
estimators.

## Route Sample Weights to fit and score

Configure your estimator to request `sample_weight` on both `fit` and `score`,
then pass the weights to `OptunaSearchCV.fit()`:

```python
from sklearn.svm import SVC
from sklearn_optuna import OptunaSearchCV
from optuna.distributions import FloatDistribution

estimator = SVC().set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

search = OptunaSearchCV(
    estimator,
    {"C": FloatDistribution(0.1, 10.0, log=True)},
    n_trials=10,
)
search.fit(X, y, sample_weight=weights)
```

The weights are forwarded to every cross-validation fold's `fit()` and
`score()` calls.

## Use with Multiple Metrics

Combine metadata routing with multi-metric scoring. Set `scoring` to a
dictionary of scorers that accept `sample_weight`:

```python
from sklearn.metrics import make_scorer, accuracy_score

weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
unweighted_acc = make_scorer(accuracy_score)

search = OptunaSearchCV(
    estimator, param_distributions,
    scoring={"weighted": weighted_acc, "unweighted": unweighted_acc},
    refit="weighted",
)
search.fit(X, y, sample_weight=weights)
```

Both metrics appear in `cv_results_` and the search optimizes for the `refit`
metric.

## Route Metadata in Pipelines

When routing metadata through a pipeline, configure both the pipeline step and
the search to forward metadata correctly:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC().set_fit_request(sample_weight=True)),
])

search = OptunaSearchCV(pipe, param_distributions)
search.fit(X, y, clf__sample_weight=weights)
```

Use the `step__param` syntax to target metadata at a specific pipeline step.

## See Also

- [Score Multiple Metrics](score-multiple-metrics.md) - use multiple scorers with metadata routing
- [Use in Pipelines](use-in-pipelines.md) - general pipeline integration
- [API Reference](../reference/api.md) - full parameter documentation
- [Concepts and Architecture](../explanation/concepts.md) - how OptunaSearchCV integrates with scikit-learn
