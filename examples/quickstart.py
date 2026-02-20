"""OptunaSearchCV Quickstart.

Run a short Optuna-powered hyperparameter search and interpret the best
parameters and score.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install(["scikit-learn", "optuna", "sklearn-optuna"])
    return


@app.cell(hide_code=True)
def _():
    import optuna
    from optuna.distributions import FloatDistribution
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from sklearn_optuna import OptunaSearchCV, Sampler

    return (
        FloatDistribution,
        LogisticRegression,
        OptunaSearchCV,
        Sampler,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OptunaSearchCV Quickstart

    ## What You'll Learn

    - How to set up and run `OptunaSearchCV` as a drop-in replacement for sklearn search estimators
    - How to define search spaces using Optuna distributions
    - How to read best parameters and score from the completed search

    ## Prerequisites

    Basic familiarity with scikit-learn's fit/predict API and hyperparameter tuning concepts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Prepare Data and Estimator

    Create a classification dataset and initialize a scikit-learn estimator. The estimator
    can be any sklearn-compatible model — here we use LogisticRegression as a simple example.
    OptunaSearchCV will tune its hyperparameters automatically.
    """)
    return


@app.cell
def _(make_classification):
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    return X, y


@app.cell
def _(LogisticRegression):
    estimator = LogisticRegression(max_iter=200)
    return (estimator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Define a Search Space

    Define the hyperparameter search space using Optuna distributions. Here we tune the
    regularization parameter `C` with `FloatDistribution` using log-scale, which is ideal
    for parameters that span several orders of magnitude. Optuna distributions replace
    sklearn's `param_grid` or `param_distributions`.
    """)
    return


@app.cell
def _(FloatDistribution):
    param_distributions = {
        "C": FloatDistribution(1e-2, 10.0, log=True),
    }
    return (param_distributions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Run OptunaSearchCV
    """)
    return


@app.cell
def _(OptunaSearchCV, Sampler, X, estimator, optuna, param_distributions, y):
    search = OptunaSearchCV(
        estimator,
        param_distributions,
        n_trials=10,
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=0),
        cv=3,
        n_jobs=1,
    )
    search.fit(X, y)
    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    mo.md(f"""
    **Best params:** {search.best_params_}
    **Best score:** {search.best_score_:.3f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Drop-in replacement** -- `OptunaSearchCV` works like other sklearn search estimators
    - **Flexible search spaces** -- Optuna distributions define continuous, discrete, and categorical parameters
    - **Standard results** -- `best_params_` and `best_score_` summarize the best trial

    ## Next Steps

    - **Study management**: See study_management.py for resuming and reusing optimization runs
    - **Callbacks**: See callbacks.py to stop runs early with custom stopping criteria
    """)
    return


if __name__ == "__main__":
    app.run()
