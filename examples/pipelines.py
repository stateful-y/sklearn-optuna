"""How to Tune Pipeline Parameters.

Tune hyperparameters across multiple pipeline steps with OptunaSearchCV.
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
    "title": "How to Tune Pipeline Parameters",
    "description": "Tune hyperparameters across multiple pipeline steps with OptunaSearchCV.",
    "category": "how-to",
    "companion": "pages/how-to/use-in-pipelines.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from optuna.distributions import FloatDistribution, IntDistribution
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn_optuna import OptunaSearchCV

    return (
        FloatDistribution,
        IntDistribution,
        LogisticRegression,
        OptunaSearchCV,
        PCA,
        Pipeline,
        StandardScaler,
        make_classification,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Tune Pipeline Parameters

    This notebook shows how to tune hyperparameters of estimators
    inside a scikit-learn `Pipeline` using `OptunaSearchCV`.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/))
    and sklearn pipelines.
    """)
    return


@app.cell
def _(make_classification):
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Tune Sub-Estimator Parameters

    Use `__` (double underscore) syntax to address parameters of
    estimators nested inside a pipeline.
    """)
    return


@app.cell
def _(
    FloatDistribution,
    LogisticRegression,
    OptunaSearchCV,
    Pipeline,
    StandardScaler,
    X,
    y,
):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    search = OptunaSearchCV(
        pipe,
        param_distributions={
            "clf__C": FloatDistribution(1e-2, 10.0, log=True),
        },
        n_trials=10,
        cv=3,
    )
    search.fit(X, y)
    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    mo.md(f"""
    **Best C:** `{search.best_params_['clf__C']:.4f}`
    **Best Score:** `{search.best_score_:.3f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Tune Across Multiple Steps

    Address parameters from different pipeline steps in the same
    search. Here, tune PCA's `n_components` alongside the
    classifier's `C`.
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
    X,
    y,
):
    multi_pipe = Pipeline([
        ("pca", PCA()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    multi_search = OptunaSearchCV(
        multi_pipe,
        param_distributions={
            "pca__n_components": IntDistribution(2, 8),
            "clf__C": FloatDistribution(1e-2, 10.0, log=True),
        },
        n_trials=15,
        cv=3,
    )
    multi_search.fit(X, y)
    return (multi_search,)


@app.cell(hide_code=True)
def _(mo, multi_search):
    mo.md(f"""
    **Best PCA components:** `{multi_search.best_params_['pca__n_components']}`
    **Best C:** `{multi_search.best_params_['clf__C']:.4f}`
    **Best Score:** `{multi_search.best_score_:.3f}`
    """)
    return


if __name__ == "__main__":
    app.run()
