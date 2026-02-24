"""Metadata Routing with sample_weight.

Demonstrate how to use scikit-learn's metadata routing to pass sample_weight
through OptunaSearchCV to both model fitting and scoring.
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
    # Metadata Routing with sample_weight

    ## What You'll Learn

    - How to enable and configure scikit-learn's metadata routing for `sample_weight`
    - How to route metadata through `OptunaSearchCV` to both model fitting and scoring
    - How to set up multi-metric scoring with different routing preferences
    - How to handle metadata routing in pipelines with mixed requirements

    ## Prerequisites

    Familiarity with the OptunaSearchCV quickstart (see quickstart.py) and scikit-learn metadata routing (requires sklearn >= 1.4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Create Imbalanced Dataset

    Generate a classification dataset with significant class imbalance (90% class 0, 10% class 1).
    Class imbalance is common in real-world problems and can lead to poor model performance if not
    addressed. We compute balanced sample weights using sklearn's `compute_sample_weight`, which
    assigns higher weights to minority class samples. These weights will be routed through
    `OptunaSearchCV` to both the estimator's `fit()` method and the scorer.
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
    ## 2. Enable Metadata Routing and Configure Estimator

    Metadata routing must be explicitly enabled via sklearn's config context.
    Then configure the estimator to request `sample_weight` for both fitting
    and scoring.
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

    The model was trained and evaluated using balanced sample weights,
    giving equal importance to both classes despite the 9:1 imbalance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Multi-Metric Scoring with Different Routing

    Use multiple scorers with different metadata routing preferences. One scorer
    uses sample weights (weighted accuracy) while another ignores them
    (unweighted accuracy).
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
    **Weighted Accuracy:** `{search_multi.best_score_:.3f}`

    The search optimized for weighted accuracy while also tracking unweighted
    accuracy. Check `cv_results_` for both metrics across all trials.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Metadata Routing in Pipelines

    When using pipelines, different steps can have different metadata requirements.
    Here, the scaler doesn't need sample weights, but the classifier does.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Enable routing** -- Use `sklearn.config_context(enable_metadata_routing=True)` to activate metadata routing
    - **Request configuration** -- Use `.set_fit_request(sample_weight=True)` and `.set_score_request(sample_weight=True)` on estimators
    - **Pass as kwargs** -- Pass metadata as keyword arguments to `fit()`: `search.fit(X, y, sample_weight=weights)`
    - **Mixed routing** -- Different scorers can have different routing preferences in multi-metric scenarios
    - **Pipeline support** -- Each pipeline step can independently configure its metadata requirements
    - **Imbalanced data** -- Sample weights are essential for handling imbalanced datasets properly

    ## Next Steps

    - **Group cross-validation**: Explore routing `groups` for GroupKFold cross-validation
    - **Custom metadata**: Try custom metadata for domain-specific information
    - **Callbacks**: Combine metadata routing with callbacks for advanced workflows (see callbacks.py)
    """)
    return


if __name__ == "__main__":
    app.run()
