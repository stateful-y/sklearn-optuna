# Troubleshooting

Solutions to common problems when using Sklearn-Optuna.

## Installation Issues

**Problem: Package not found**
: Verify the package name: `pip install sklearn-optuna` or `uv add sklearn-optuna`.

**Problem: Import error after installation**
: Make sure you installed in the active environment: `python -c "import sklearn_optuna"`.

**Problem: Optuna version conflict**
: Sklearn-Optuna requires Optuna 3.5-3.x. If you have Optuna 4.x installed, downgrade with `pip install "optuna>=3.5,<4"`.

## Search Issues

**Problem: Results are not reproducible**
: Wrap the sampler with `Sampler` and pass `seed=`. Use `n_jobs=1` because parallel trial ordering is non-deterministic. See [How to Configure Samplers](configure-samplers.md#set-a-seed-for-reproducibility).

**Problem: All trials return NaN**
: The estimator may be failing silently. Set `error_score="raise"` to surface the underlying error. See [How to Handle Errors](handle-errors.md#raise-on-failure).

**Problem: Search is slow**
: Reduce `n_trials` or set a `timeout` to cap execution time. Check that `cv` is not set to a large number of folds. Consider using `n_jobs=-1` for parallel trial execution.

**Problem: CMA-ES sampler raises an error**
: CMA-ES does not support `CategoricalDistribution` parameters. Use `TPESampler` for mixed search spaces.

## Pipeline Issues

**Problem: Parameter not found in pipeline**
: Use double-underscore syntax to address nested parameters: `"clf__C"` for the `C` parameter of the `clf` step. See [How to Use in Pipelines](use-in-pipelines.md).

**Problem: OptunaSearchCV does not work as a pipeline step**
: `OptunaSearchCV` is a valid Scikit-Learn estimator and works as a pipeline step. Make sure you pass it as a step tuple: `("search", search)`.

## Storage Issues

**Problem: Study not resuming from database**
: Make sure you use the same `study_name` and `Storage` configuration. See [How to Persist and Resume Studies](persist-studies.md).

## Getting Help

- [Open an issue on GitHub](https://github.com/stateful-y/sklearn-optuna/issues/new)
- [Start a discussion](https://github.com/stateful-y/sklearn-optuna/discussions)
