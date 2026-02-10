![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Sklearn-Optuna's documentation

`OptunaSearchCV` is a drop-in replacement for Scikit-Learn's `GridSearchCV` and `RandomizedSearchCV` powered by [Optuna](https://optuna.org/). It extends `BaseSearchCV`, so `fit()`, `score()`, `best_params_`, `cv_results_`, pipelines, and `clone()` all work out of the box. Optuna samplers (TPE, CMA-ES, …) explore search spaces more efficiently than grid or random search, while Optuna distributions give you log-scaled, bounded, and categorical parameter spaces.

!!! note "Inspiration"
    This project is inspired by [optuna-integration's OptunaSearchCV](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.OptunaSearchCV.html).

<div class="grid cards" markdown>

-  **Get Started in 5 Minutes**

    ---

    Install Sklearn-Optuna and run your first hyperparameter search

    Install → Define distributions → Fit → Done

    [Getting Started](pages/getting-started.md)

- **Learn the Concepts**

    ---

    Understand OptunaSearchCV, samplers, distributions, and callbacks

    [User Guide](pages/user-guide.md)

- **See It In Action**

    ---

    Explore 7 interactive notebooks from quickstart to pipelines

    [Examples](pages/examples.md)

- **API Reference**

    ---

    Complete API documentation for OptunaSearchCV and wrapper classes

    [API Reference](pages/api-reference.md)


</div>

## Table of Contents

### [Getting started](pages/getting-started.md)

Step-by-step guide to installing and setting up Sklearn-Optuna in your project.

- [1. Install the package](pages/getting-started.md#step-1-install-the-package)
- [2. Verify installation](pages/getting-started.md#step-2-verify-installation)
- [3. Basic usage](pages/getting-started.md#basic-usage)


### [Examples](pages/examples.md)

Interactive notebooks demonstrating real-world use cases.

- [What can Sklearn-Optuna do?](pages/examples.md#what-can-sklearn-optuna-do)
- [Running examples locally](pages/examples.md#running-examples-locally)


### [User guide](pages/user-guide.md)

In-depth documentation on the design, architecture, and core concepts.

- [Core Concepts](pages/user-guide.md#core-concepts)
- [Configuration](pages/user-guide.md#configuration)
- [Best Practices](pages/user-guide.md#best-practices)

### [Reference](pages/api-reference.md)

Detailed reference for the Sklearn-Optuna API, including classes, functions, and configuration options.

## License

Sklearn-Optuna is open source and licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0). You are free to use, modify, and distribute this software under the terms of this license.
