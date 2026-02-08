# Getting Started

This guide will help you install and start using Sklearn-Optuna in minutes.

## Installation

### Step 1: Install the package

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install sklearn_optuna
    ```

=== "uv"

    ```bash
    uv add sklearn_optuna
    ```

=== "conda"

    ```bash
    conda install -c conda-forge sklearn_optuna
    ```

=== "mamba"

    ```bash
    mamba install -c conda-forge sklearn_optuna
    ```

> **Note**: For conda/mamba, ensure the package is published to conda-forge first.

### Step 2: Verify installation

```python
import sklearn_optuna
print(sklearn_optuna.__version__)
```

## Basic Usage

### Step 1: Import and define a search space

```python
import optuna
from optuna.distributions import FloatDistribution
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn_optuna import OptunaSearchCV, Sampler

X, y = make_classification(n_samples=200, n_features=6, random_state=42)

param_distributions = {
    "C": FloatDistribution(1e-2, 10.0, log=True),
}
```

### Step 2: Create an OptunaSearchCV instance

```python
search = OptunaSearchCV(
    LogisticRegression(max_iter=200),
    param_distributions,
    n_trials=20,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    cv=3,
)
```

### Step 3: Fit and inspect results

```python
search.fit(X, y)

print(search.best_params_)   # e.g. {'C': 1.23}
print(search.best_score_)    # e.g. 0.87
search.best_estimator_.predict(X[:5])
```

## Complete Example

Here's a complete working example (mirrors `examples/quickstart.py`):

```python
import optuna
from optuna.distributions import FloatDistribution
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn_optuna import OptunaSearchCV, Sampler

# Prepare data
X, y = make_classification(
    n_samples=200, n_features=6, n_informative=3,
    n_redundant=0, random_state=42,
)

# Define search space
param_distributions = {
    "C": FloatDistribution(1e-2, 10.0, log=True),
}

# Run search
search = OptunaSearchCV(
    LogisticRegression(max_iter=200),
    param_distributions,
    n_trials=10,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=0),
    cv=3,
)
search.fit(X, y)

print(f"Best params: {search.best_params_}")
print(f"Best score:  {search.best_score_:.3f}")
```

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

Now that you have Sklearn-Optuna installed and running:

- **Learn the concepts**: Read the [User Guide](user-guide.md) to understand core concepts and capabilities
- **Explore examples**: Check out the [Examples](examples.md) for real-world use cases
- **Dive into the API**: Browse the [API Reference](api-reference.md) for detailed documentation
- **Get help**: Visit [GitHub Discussions](https://github.com/stateful-y/sklearn-optuna/discussions) or [open an issue](https://github.com/stateful-y/sklearn-optuna/issues)
