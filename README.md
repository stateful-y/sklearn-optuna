<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stateful-y/sklearn-optuna/main/docs/assets/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stateful-y/sklearn-optuna/main/docs/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/stateful-y/sklearn-optuna/main/docs/assets/logo_light.png" alt="Sklearn-Optuna">
  </picture>
</p>


[![Python Version](https://img.shields.io/pypi/pyversions/sklearn_optuna)](https://pypi.org/project/sklearn_optuna/)
[![License](https://img.shields.io/github/license/stateful-y/sklearn-optuna)](https://github.com/stateful-y/sklearn-optuna/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/sklearn_optuna)](https://pypi.org/project/sklearn_optuna/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sklearn_optuna)](https://anaconda.org/conda-forge/sklearn_optuna)
[![codecov](https://codecov.io/gh/stateful-y/sklearn-optuna/branch/main/graph/badge.svg)](https://codecov.io/gh/stateful-y/sklearn-optuna)

## What is Sklearn-Optuna?

Sklearn-Optuna provides `OptunaSearchCV`, a drop-in replacement for Scikit-Learn's
`GridSearchCV` and `RandomizedSearchCV` that uses [Optuna](https://optuna.org/) for hyperparameter optimization. It extends `BaseSearchCV` so that `fit()`, `score()`, `best_params_`, `cv_results_`, and all other Scikit-Learn search attributes work exactly as expected.

With Sklearn-Optuna, you gain immediate access to sophisticated Bayesian optimization while maintaining full compatibility with Scikit-Learn's Pipeline, cross-validation, and serialization. Perfect for data scientists who want smarter hyperparameter search without learning a new API.

> **Note**: This project is inspired by [Optuna-Integration's OptunaSearchCV](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.OptunaSearchCV.html).

## What are the features of Sklearn-Optuna?

- **Drop-in Scikit-Learn search**: Extends `BaseSearchCV` and works with `Pipeline`, `clone()`, `get_params()`/`set_params()`, and any estimator
- **Smart optimization**: TPE, CMA-ES, and other samplers explore the search space more efficiently than grid or random search
- **Flexible distributions**: Optuna distributions (`FloatDistribution`, `IntDistribution`, `CategoricalDistribution`) for log-scaled, bounded, and categorical spaces
- **Study persistence**: Pass a `Storage` backend to persist trials to a database, or resume optimization by passing an existing study to `fit()`
- **Callbacks**: Attach Optuna callbacks (e.g., `MaxTrialsCallback`) to control optimization flow
- **Nested pipeline search**: Use `OptunaSearchCV` inside another `OptunaSearchCV` for hierarchical hyperparameter optimization

## How to install Sklearn-Optuna?

Install the Sklearn-Optuna package using `pip`:

```bash
pip install sklearn_optuna
```

or using `uv`:

```bash
uv pip install sklearn_optuna
```

or using `conda`:

```bash
conda install -c conda-forge sklearn_optuna
```

or using `mamba`:

```bash
mamba install -c conda-forge sklearn_optuna
```

or alternatively, add `sklearn_optuna` to your `requirements.txt` or `pyproject.toml` file.

## How to get started with Sklearn-Optuna?

### 1. Define a search space with Optuna distributions

```python
from optuna.distributions import FloatDistribution
from sklearn.linear_model import LogisticRegression

param_distributions = {
    "C": FloatDistribution(1e-2, 10.0, log=True),
}
```

### 2. Create and fit an OptunaSearchCV

```python
from sklearn_optuna import OptunaSearchCV, Sampler
import optuna

search = OptunaSearchCV(
    LogisticRegression(max_iter=200),
    param_distributions,
    n_trials=20,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    cv=5,
)
search.fit(X_train, y_train)
```

### 3. Inspect results

```python
print(search.best_params_)   # {'C': 1.23}
print(search.best_score_)    # 0.95
search.best_estimator_.predict(X_test)
```

## How do I use Sklearn-Optuna?

Full documentation is available at [https://sklearn-optuna.readthedocs.io/](https://sklearn-optuna.readthedocs.io/).

Interactive examples are available in the `examples/` directory:

- **Online**: [https://sklearn-optuna.readthedocs.io/en/latest/pages/examples/](https://sklearn-optuna.readthedocs.io/en/latest/pages/examples/)
- **Locally**: Run `just example quickstart` or `uv run marimo edit examples/quickstart.py` to open an interactive notebook

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features**: [GitHub Issues](https://github.com/stateful-y/sklearn-optuna/issues)
- **Join the discussion**: [GitHub Discussions](https://github.com/stateful-y/sklearn-optuna/discussions)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/stateful-y/sklearn-optuna/blob/main/CONTRIBUTING.md)

If you are interested in becoming a maintainer or taking a more active role, please reach out to Guillaume Tauzin on [stateful-y.io contact page](https://stateful-y.io/contact).

## Where can I learn more?

- **Full documentation**: [https://sklearn-optuna.readthedocs.io/](https://sklearn-optuna.readthedocs.io/)
- **Interactive Examples**: [https://sklearn-optuna.readthedocs.io/en/latest/pages/examples/](https://sklearn-optuna.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/sklearn-optuna/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/sklearn-optuna/blob/main/LICENSE).

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
