# How to Score Multiple Metrics

This guide shows you how to evaluate hyperparameter configurations against
multiple scoring metrics simultaneously. Use this when you need to track
accuracy alongside F1 score, precision, or any other metric.

## Prerequisites

- sklearn-optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search

## Pass Multiple Scorers

Provide a list of scorer names to `scoring` and set `refit` to the metric used
for selecting the best model:

```python
from sklearn_optuna import OptunaSearchCV

search = OptunaSearchCV(
    estimator, param_distributions,
    scoring=["accuracy", "f1"],
    refit="accuracy",
)
search.fit(X, y)
```

All metrics appear in `cv_results_` as `mean_test_accuracy`, `mean_test_f1`,
and their per-split variants.

## Use a Dictionary of Scorers

For custom scorers, pass a dictionary:

```python
from sklearn.metrics import make_scorer, precision_score, recall_score

search = OptunaSearchCV(
    estimator, param_distributions,
    scoring={
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
    },
    refit="precision",
)
```

## Access Per-Metric Results

After fitting, each metric has its own columns in `cv_results_`:

```python
import pandas as pd

results = pd.DataFrame(search.cv_results_)
print(results[["mean_test_precision", "mean_test_recall"]].head())
```

## Combine with Metadata Routing

If your scorer requires additional metadata (such as `sample_weight`), use
scikit-learn's metadata routing:

```python
from sklearn.metrics import make_scorer, f1_score

weighted_f1 = make_scorer(f1_score, average="weighted")
weighted_f1.set_score_request(sample_weight=True)

search = OptunaSearchCV(
    estimator, param_distributions,
    scoring=weighted_f1,
)
search.fit(X, y, sample_weight=weights)
```

## See Also

- [API Reference](../reference/api.md) - full parameter documentation
- Metadata Routing example notebook: [View](/examples/metadata_routing/) · [Open in marimo](/examples/metadata_routing/edit/) - advanced scoring with sample weights
- [Configuration Reference](../reference/configuration.md) - all `scoring` and `refit` options
