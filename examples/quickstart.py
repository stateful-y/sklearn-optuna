"""OptunaSearchCV Quickstart.

Run a short Optuna-powered hyperparameter search and interpret the best
parameters and score.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optuna",
#     "scikit-learn",
#     "sklearn-optuna",
# ]
# ///
import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

__gallery__ = {
    "title": "OptunaSearchCV Quickstart",
    "description": "Run a fast hyperparameter search and read the best parameters and score.",
    "category": "tutorial",
    "companion": "pages/tutorials/getting-started.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


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

    In this notebook, we will run a hyperparameter search using
    `OptunaSearchCV` and inspect the best parameters and score.
    Along the way, we will define a search space with Optuna
    distributions and see how the results API works.

    **Prerequisites:** Basic familiarity with scikit-learn's fit/predict API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Prepare Data and Estimator

    We start by creating a classification dataset and initializing a
    LogisticRegression estimator. `OptunaSearchCV` will tune its
    hyperparameters automatically.
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

    Now we define the hyperparameter search space. We use
    `FloatDistribution` with log-scale for the regularization
    parameter `C`.
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

    Notice that `best_params_` and `best_score_` work exactly like
    scikit-learn's `GridSearchCV` -- the same attributes, the same format.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What We Built

    We ran a hyperparameter search with `OptunaSearchCV` and found
    the best regularization parameter for a LogisticRegression.
    Along the way, we:

    - Defined a search space with `FloatDistribution`
    - Ran the search with `OptunaSearchCV.fit()`
    - Inspected `best_params_` and `best_score_`

    **Next steps:**

    - How to resume optimization from prior trials:
      [View](/examples/study_management/) · [Open in marimo](/examples/study_management/edit/)
    - How to stop optimization early with callbacks:
      [View](/examples/callbacks/) · [Open in marimo](/examples/callbacks/edit/)
    """)
    return


if __name__ == "__main__":
    app.run()
