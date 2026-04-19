![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Sklearn-Optuna's documentation

`OptunaSearchCV` is a drop-in replacement for Scikit-Learn's `GridSearchCV` and `RandomizedSearchCV` powered by [Optuna](https://optuna.org/). It extends `BaseSearchCV`, so `fit()`, `score()`, `best_params_`, `cv_results_`, pipelines, and `clone()` all work out of the box. Optuna samplers (TPE, CMA-ES, ...) explore search spaces more efficiently than grid or random search, while Optuna distributions give you log-scaled, bounded, and categorical parameter spaces.

!!! note "Inspiration"
    This project is inspired by [optuna-integration's OptunaSearchCV](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.OptunaSearchCV.html).

<div class="grid cards" markdown>

-  **Get Started in 5 Minutes**

    ---

    Install Sklearn-Optuna and run your first hyperparameter search.

    [Getting Started](pages/tutorials/getting-started.md)

- **How-to Guides**

    ---

    Task-oriented guides for samplers, callbacks, persistence, pipelines, and more.

    [How-to Guides](pages/how-to/configure-samplers.md)

- **See It In Action**

    ---

    Explore interactive notebooks from quickstart to pipelines.

    [Examples](pages/tutorials/examples.md)

- **API Reference**

    ---

    Complete API documentation for OptunaSearchCV and wrapper classes.

    [API Reference](pages/reference/api.md)


</div>

## Table of Contents

### [Getting Started](pages/tutorials/getting-started.md)

Step-by-step tutorial to installing and running your first hyperparameter search.

- [Installation](pages/tutorials/getting-started.md#installation)
- [Your First Search](pages/tutorials/getting-started.md#your-first-hyperparameter-search)

### [How-to Guides](pages/how-to/configure-samplers.md)

Task-oriented guides for common workflows.

- [Configure Samplers](pages/how-to/configure-samplers.md)
- [Use Callbacks](pages/how-to/use-callbacks.md)
- [Persist and Resume Studies](pages/how-to/persist-studies.md)
- [Score Multiple Metrics](pages/how-to/score-multiple-metrics.md)
- [Use in Pipelines](pages/how-to/use-in-pipelines.md)
- [Handle Errors](pages/how-to/handle-errors.md)
- [Troubleshooting](pages/how-to/troubleshooting.md)

### [Examples](pages/tutorials/examples.md)

Interactive notebooks demonstrating real-world use cases.

### [Reference](pages/reference/api.md)

Detailed reference for the Sklearn-Optuna API, including classes, functions, and configuration options.

- [API Reference](pages/reference/api.md)
- [Configuration](pages/reference/configuration.md)

### [Concepts and Architecture](pages/explanation/concepts.md)

Understanding how OptunaSearchCV works and what trade-offs were made.

## License

Sklearn-Optuna is open source and licensed under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0).
