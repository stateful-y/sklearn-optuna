# Getting Started

!!! tip "Interactive notebook"
    Follow along in the Quickstart notebook for a hands-on version of this tutorial.
    [View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)

In this tutorial, we will run an Optuna-powered hyperparameter search using
scikit-learn's familiar `fit()` / `best_params_` API. Along the way, we will
install the package, define an Optuna search space, create an `OptunaSearchCV`
instance, and inspect the results.

## Prerequisites

- Python 3.11+ installed
- A terminal or command prompt

## Installation

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install sklearn_optuna
    ```

=== "uv"

    ```bash
    uv add sklearn_optuna
    ```

Verify the installation:

```python
import sklearn_optuna

print(sklearn_optuna.__version__)
```

The output should look something like:

```text
0.1.0a3
```

## Your First Hyperparameter Search

Now let's set up a classification problem and find the best regularization
strength for a logistic regression model.

### Prepare the data

We will use scikit-learn's `make_classification` to generate a small synthetic
dataset:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200, n_features=6, n_informative=3,
    n_redundant=0, random_state=42,
)
```

### Define the search space

We tell Optuna which hyperparameter to tune and over what range. Here we search
for the regularization parameter `C` on a log scale between 0.01 and 10:

```python
from optuna.distributions import FloatDistribution

param_distributions = {
    "C": FloatDistribution(1e-2, 10.0, log=True),
}
```

Notice that we use Optuna's `FloatDistribution` instead of a plain list or
range. The `log=True` argument means trials are sampled uniformly in log-space,
which is important for parameters that span several orders of magnitude.

### Create and run the search

Now we bring it all together with `OptunaSearchCV`. We pass a `Sampler` wrapper
so that results are reproducible:

```python
import optuna
from sklearn.linear_model import LogisticRegression

from sklearn_optuna import OptunaSearchCV, Sampler

search = OptunaSearchCV(
    LogisticRegression(max_iter=200),
    param_distributions,
    n_trials=10,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=0),
    cv=3,
)
search.fit(X, y)
```

After a moment, you should see Optuna's trial log. The search runs 10 trials,
each evaluating a different value of `C` using 3-fold cross-validation.

### Inspect the results

Let's check what the search found:

```python
print(f"Best params: {search.best_params_}")
print(f"Best score:  {search.best_score_:.3f}")
```

You should see output similar to:

```text
Best params: {'C': 1.234}
Best score:  0.870
```

The best model is already refitted on the full dataset and ready to use:

```python
predictions = search.best_estimator_.predict(X[:5])
print(predictions)
```

Notice that `best_estimator_` is a regular scikit-learn estimator. Everything
you normally do with a fitted model (predict, score, serialize) works here too.

## What We Built

We ran an Optuna-powered hyperparameter search through scikit-learn's standard
API. Along the way, we:

- Installed sklearn-optuna and verified it
- Defined an Optuna search space with `FloatDistribution`
- Created an `OptunaSearchCV` instance with a reproducible sampler
- Inspected `best_params_`, `best_score_`, and `best_estimator_`

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can:

- Run code directly in your browser via WebAssembly
- Experiment with different parameters
- See visual outputs in real-time

Or run locally:

=== "just"

    ```bash
    just example quickstart
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/quickstart.py
    ```

## Next Steps

- [How to Configure Samplers](../how-to/configure-samplers.md) - choose the right optimization algorithm
- [How to Use Callbacks](../how-to/use-callbacks.md) - control when optimization stops
- [How to Use in Pipelines](../how-to/use-in-pipelines.md) - tune hyperparameters inside scikit-learn pipelines
- [Concepts and Architecture](../explanation/concepts.md) - understand how OptunaSearchCV works under the hood
- [Examples](examples.md) - interactive notebooks with more advanced use cases
