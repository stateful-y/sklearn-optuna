"""How to Handle Failing Trials.

Control what happens when a hyperparameter combination causes fitting to fail.
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
    "title": "How to Handle Failing Trials",
    "description": "Control what happens when a hyperparameter combination causes fitting to fail.",
    "category": "how-to",
    "companion": "pages/how-to/handle-errors.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import math

    from optuna.distributions import FloatDistribution, IntDistribution
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from sklearn_optuna import OptunaSearchCV

    return (
        FloatDistribution,
        IntDistribution,
        LogisticRegression,
        OptunaSearchCV,
        PCA,
        Pipeline,
        make_classification,
        math,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Handle Failing Trials

    This notebook shows how to control what happens when some
    hyperparameter combinations fail during fitting.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Default Behavior: NaN for Failures

    By default, `error_score=np.nan` records `NaN` for failing
    trials and continues the search. Here, PCA `n_components`
    values above 5 exceed the feature count and fail.
    """)
    return


@app.cell
def _(
    FloatDistribution,
    IntDistribution,
    LogisticRegression,
    OptunaSearchCV,
    PCA,
    Pipeline,
    make_classification,
):
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=0
    )

    pipe = Pipeline([
        ("pca", PCA()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    # n_components > 5 will fail (dataset has 5 features)
    search = OptunaSearchCV(
        pipe,
        param_distributions={
            "pca__n_components": IntDistribution(2, 8),
            "clf__C": FloatDistribution(0.1, 10.0, log=True),
        },
        n_trials=15,
        cv=3,
    )
    search.fit(X, y)
    return X, search, y


@app.cell(hide_code=True)
def _(math, mo, search):
    _scores = search.cv_results_["mean_test_score"]
    _n_failed = sum(1 for s in _scores if math.isnan(s))
    _n_total = len(_scores)
    _best = search.best_score_
    mo.md(f"""
    **Total trials:** {_n_total}
    **Failed trials (NaN):** {_n_failed}
    **Successful trials:** {_n_total - _n_failed}
    **Best score:** `{_best:.3f}`

    The search completed despite failures. Failed trials appear as
    `NaN` in `cv_results_`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Raise on Failure

    Set `error_score="raise"` to stop the search immediately when
    a trial fails. This is useful during development to catch
    unexpected parameter combinations.

    ```python
    search = OptunaSearchCV(
        pipe, param_distributions,
        error_score="raise",  # stops on first failure
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Filter Failed Trials

    Inspect which trials failed and which parameter combinations
    caused the failures.
    """)
    return


@app.cell
def _(math, mo, search):
    _scores = search.cv_results_["mean_test_score"]
    _params = search.cv_results_["params"]
    _failed = [
        {"params": p, "mean_test_score": s}
        for p, s in zip(_params, _scores)
        if math.isnan(s)
    ]
    mo.ui.table(_failed)
    return


if __name__ == "__main__":
    app.run()
