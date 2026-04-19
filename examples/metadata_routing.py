"""How to Route Sample Weights Through OptunaSearchCV.

Pass sample_weight through OptunaSearchCV to model fitting and scoring.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "optuna",
#     "scikit-learn",
#     "sklearn-optuna",
# ]
# ///
import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

__gallery__ = {
    "title": "How to Route Sample Weights Through OptunaSearchCV",
    "description": "Pass sample_weight through OptunaSearchCV to both fitting and scoring.",
    "category": "how-to",
    "companion": "pages/how-to/route-metadata.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import sklearn
    from optuna.distributions import FloatDistribution
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, make_scorer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    from sklearn_optuna import OptunaSearchCV

    return (
        FloatDistribution,
        LogisticRegression,
        OptunaSearchCV,
        Pipeline,
        StandardScaler,
        accuracy_score,
        compute_sample_weight,
        make_classification,
        make_scorer,
        sklearn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Route Sample Weights Through OptunaSearchCV

    This notebook shows how to enable scikit-learn's metadata routing
    and pass `sample_weight` through `OptunaSearchCV` to fitting,
    scoring, and pipelines.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/))
    and scikit-learn metadata routing (requires sklearn >= 1.4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Create an Imbalanced Dataset

    Generate a dataset with 90/10 class imbalance and compute
    balanced sample weights.
    """)
    return


@app.cell
def _(compute_sample_weight, make_classification):
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced: 90% class 0, 10% class 1
        flip_y=0.01,
        random_state=42,
    )

    # Compute balanced sample weights
    sample_weight = compute_sample_weight("balanced", y)
    return X, sample_weight, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Enable Routing and Run a Search

    Enable metadata routing, configure the estimator to request
    `sample_weight`, and pass it to `fit()`.
    """)
    return


@app.cell
def _(
    FloatDistribution,
    LogisticRegression,
    OptunaSearchCV,
    X,
    sample_weight,
    sklearn,
    y,
):
    with sklearn.config_context(enable_metadata_routing=True):
        # Configure estimator to request sample_weight
        lr = LogisticRegression(max_iter=300, random_state=42)
        lr.set_fit_request(sample_weight=True)
        lr.set_score_request(sample_weight=True)

        # Create search with parameter distributions
        search = OptunaSearchCV(
            lr,
            param_distributions={
                "C": FloatDistribution(0.01, 10.0, log=True),
            },
            n_trials=10,
            cv=3,
        )

        # Fit with sample_weight - it will be routed to fit() and score()
        search.fit(X, y, sample_weight=sample_weight)

    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    mo.md(f"""
    **Best Parameters:** `C = {search.best_params_['C']:.4f}`
    **Best Weighted Score:** `{search.best_score_:.3f}`
    **Trials run:** `{len(search.cv_results_['params'])}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Multi-Metric Scoring with Mixed Routing

    Create scorers with different routing preferences: one weighted,
    one unweighted.
    """)
    return


@app.cell
def _(
    FloatDistribution,
    LogisticRegression,
    OptunaSearchCV,
    X,
    accuracy_score,
    make_scorer,
    sample_weight,
    sklearn,
    y,
):
    with sklearn.config_context(enable_metadata_routing=True):
        # Configure estimator
        lr_multi = LogisticRegression(max_iter=300, random_state=42)
        lr_multi.set_fit_request(sample_weight=True)
        lr_multi.set_score_request(sample_weight=True)

        # Create scorers with different routing
        weighted_scorer = make_scorer(accuracy_score)
        weighted_scorer.set_score_request(sample_weight=True)

        unweighted_scorer = make_scorer(accuracy_score)
        unweighted_scorer.set_score_request(sample_weight=False)

        scoring = {
            "weighted_accuracy": weighted_scorer,
            "unweighted_accuracy": unweighted_scorer,
        }

        search_multi = OptunaSearchCV(
            lr_multi,
            param_distributions={
                "C": FloatDistribution(0.01, 10.0, log=True),
            },
            n_trials=10,
            cv=3,
            scoring=scoring,
            refit="weighted_accuracy",  # Optimize for weighted metric
        )

        search_multi.fit(X, y, sample_weight=sample_weight)

    return (search_multi,)


@app.cell(hide_code=True)
def _(mo, search_multi):
    mo.md(f"""
    **Best Parameters:** `C = {search_multi.best_params_['C']:.4f}`
    **Weighted Accuracy:** `{search_multi.cv_results_['mean_test_weighted_accuracy'][search_multi.best_index_]:.3f}`
    **Unweighted Accuracy:** `{search_multi.cv_results_['mean_test_unweighted_accuracy'][search_multi.best_index_]:.3f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Route Metadata in a Pipeline

    Configure each pipeline step independently: the scaler ignores
    weights while the classifier uses them.
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
    sample_weight,
    sklearn,
    y,
):
    with sklearn.config_context(enable_metadata_routing=True):
        # Configure pipeline components
        scaler = StandardScaler()
        scaler.set_fit_request(sample_weight=False)  # Scaler ignores weights

        lr_pipe = LogisticRegression(max_iter=300, random_state=42)
        lr_pipe.set_fit_request(sample_weight=True)  # Classifier uses weights
        lr_pipe.set_score_request(sample_weight=True)

        pipe = Pipeline([
            ("scaler", scaler),
            ("classifier", lr_pipe),
        ])

        search_pipe = OptunaSearchCV(
            pipe,
            param_distributions={
                "classifier__C": FloatDistribution(0.01, 10.0, log=True),
            },
            n_trials=10,
            cv=3,
        )

        search_pipe.fit(X, y, sample_weight=sample_weight)

    return (search_pipe,)


@app.cell(hide_code=True)
def _(mo, search_pipe):
    mo.md(f"""
    **Best Parameters:** `C = {search_pipe.best_params_['classifier__C']:.4f}`
    **Best Score:** `{search_pipe.best_score_:.3f}`

    The sample weights were correctly routed only to the classifier, not the scaler.
    """)
    return


if __name__ == "__main__":
    app.run()
