# How-to Guides

Task-oriented guides for common workflows. Each guide assumes you have already completed the [Getting Started](../tutorials/getting-started.md) tutorial.

## Optimization

- [Configure Samplers](configure-samplers.md): choose and configure Optuna samplers, set seeds for reproducibility
- [Use Callbacks](use-callbacks.md): stop optimization early with trial limits, timeouts, or custom conditions
- [Score Multiple Metrics](score-multiple-metrics.md): evaluate configurations against multiple scoring metrics simultaneously
- [Visualize Results](visualize-results.md): plot optimization history and parameter importance with Optuna's visualization tools

## Integration

- [Use in Pipelines](use-in-pipelines.md): wrap or embed `OptunaSearchCV` in Scikit-Learn pipelines
- [Route Metadata](route-metadata.md): pass sample weights and other metadata through to estimators and scorers
- [Persist and Resume Studies](persist-studies.md): save optimization state to a database and resume from prior trials

## Debugging

- [Handle Errors](handle-errors.md): control what happens when a hyperparameter combination causes fitting to fail
- [Troubleshooting](troubleshooting.md): solutions to common problems with installation, search, pipelines, and storage

## Project

- [Contributing](contribute.md): set up a development environment, run tests, and submit changes
