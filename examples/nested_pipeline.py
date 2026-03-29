"""How to Nest OptunaSearchCV in a Pipeline.

Tune preprocessing and sampler parameters with nested optimization.
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
    "title": "How to Nest OptunaSearchCV in a Pipeline",
    "description": "Tune preprocessing choices and sampler parameters simultaneously with nested optimization.",
    "category": "how-to",
    "companion": "pages/how-to/use-in-pipelines.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from sklearn_optuna import OptunaSearchCV, Sampler

    return (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
        LogisticRegression,
        MinMaxScaler,
        OptunaSearchCV,
        Pipeline,
        Sampler,
        StandardScaler,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Nest OptunaSearchCV in a Pipeline

    This notebook shows how to place `OptunaSearchCV` inside a
    pipeline and tune both preprocessing and inner-search parameters
    from an outer search.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/))
    and sklearn pipelines.
    """)
    return


@app.cell
def _(make_classification):
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=42,
    )
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Create the Inner Search

    Build an `OptunaSearchCV` that optimizes model hyperparameters.
    Set `n_trials=3` to keep the nested search fast.
    """)
    return


@app.cell
def _(FloatDistribution, LogisticRegression, OptunaSearchCV, Sampler, optuna):
    inner_search = OptunaSearchCV(
        LogisticRegression(max_iter=200),
        param_distributions={
            "C": FloatDistribution(0.01, 10.0, log=True),
        },
        n_trials=3,  # Keep low for nested search
        cv=2,
        refit=True,  # Inner search needs to be refitted
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    )
    return (inner_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Build the Pipeline

    Place the inner search as the final pipeline step.
    """)
    return


@app.cell
def _(Pipeline, StandardScaler, inner_search):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", inner_search),
    ])
    return (pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Run the Outer Search

    Use double-underscore syntax to tune nested parameters like
    `classifier__sampler__n_startup_trials`.
    """)
    return


@app.cell
def _(
    CategoricalDistribution,
    IntDistribution,
    MinMaxScaler,
    OptunaSearchCV,
    StandardScaler,
    X,
    pipeline,
    y,
):
    outer_search = OptunaSearchCV(
        pipeline,
        param_distributions={
            "scaler": CategoricalDistribution([
                StandardScaler(),
                MinMaxScaler(),
            ]),
            "classifier__sampler__n_startup_trials": IntDistribution(1, 2),
        },
        n_trials=5,  # Keep low for demonstration
        cv=2,
    )
    outer_search.fit(X, y)
    return (outer_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Inspect Results
    """)
    return


@app.cell(hide_code=True)
def _(mo, outer_search):
    mo.md(f"""
    **Best Parameters:**
    - Scaler: `{type(outer_search.best_params_['scaler']).__name__}`
    - Inner sampler n_startup_trials: `{outer_search.best_params_['classifier__sampler__n_startup_trials']}`

    **Best Cross-Validated Score:** `{outer_search.best_score_:.3f}`

    **Total Outer Trials:** `{len(outer_search.study_.trials)}`
    **Inner Trials per Outer:** `{outer_search.best_estimator_.named_steps['classifier'].n_trials}`
    """)
    return


if __name__ == "__main__":
    app.run()
