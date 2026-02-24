"""Nested OptunaSearchCV in Pipelines.

Demonstrate an advanced pattern where OptunaSearchCV is used as a final
estimator in a pipeline that is itself tuned by another OptunaSearchCV.
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
    # Nested OptunaSearchCV in Pipelines

    ## What You'll Learn

    - How to nest `OptunaSearchCV` as a final estimator inside a pipeline tuned by another `OptunaSearchCV`
    - How to use double-underscore syntax to tune inner search parameters (e.g., sampler settings)
    - The computational trade-offs of nested optimization

    ## Prerequisites

    Familiarity with the OptunaSearchCV quickstart (see quickstart.py) and sklearn pipelines.

    **Performance Note:** Nested searches multiply computational costs
    (`outer_trials x inner_trials` evaluations). Use this pattern sparingly
    and with minimal trial counts for exploratory work.
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
    ## 1. Create Inner Search

    The inner `OptunaSearchCV` optimizes model hyperparameters and will be used as the final
    estimator in a pipeline. Setting `refit=False` avoids unnecessary refitting after each
    inner optimization completes because the outer search doesn't need the fitted inner estimator,
    only its cross-validation performance. This saves computation in nested search scenarios.
    We keep `n_trials` low (3) because nested searches multiply computational costs:
    `outer_trials × inner_trials` total evaluations.
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
    ## 2. Build Pipeline with Inner Search

    Create a pipeline where the inner search is the final estimator. The outer
    search can then optimize both preprocessing steps and inner search parameters.
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
    ## 3. Run Outer Search with Sampler Tuning

    The outer search optimizes:
    - Preprocessing choice (`StandardScaler` vs `MinMaxScaler`)
    - Inner search's sampler parameter (`n_startup_trials`)

    Use double-underscore syntax to access nested parameters:
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

    **Total Trials:** `{len(outer_search.study_.trials)}`

    The outer search evaluated {len(outer_search.study_.trials)} combinations,
    with each running {outer_search.best_estimator_.named_steps['classifier'].n_trials}
    inner trials.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Nesting** -- `OptunaSearchCV` can be nested inside pipelines tuned by another `OptunaSearchCV`
    - **Refit control** -- Use `refit=False` on inner searches to save computation time
    - **Nested parameters** -- Access nested parameters with double-underscore syntax (`classifier__sampler__param`)
    - **Sampler tuning** -- Sampler parameters like `n_startup_trials` can be tuned as hyperparameters
    - **Cost awareness** -- Keep trial counts minimal due to multiplicative computational costs

    ## Next Steps

    - **Metadata routing**: See metadata_routing.py to route sample weights through `OptunaSearchCV`
    - **Sampler exploration**: Try tuning other sampler parameters or different sampler types
    """)
    return


if __name__ == "__main__":
    app.run()
