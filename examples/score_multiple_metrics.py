"""How to Score Multiple Metrics.

Evaluate hyperparameter configurations against multiple scoring metrics at once.
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
    "title": "How to Score Multiple Metrics",
    "description": "Evaluate hyperparameter configurations against multiple scoring metrics at once.",
    "category": "how-to",
    "companion": "pages/how-to/score-multiple-metrics.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from optuna.distributions import FloatDistribution
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from sklearn_optuna import OptunaSearchCV

    return (
        FloatDistribution,
        LogisticRegression,
        OptunaSearchCV,
        make_classification,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Score Multiple Metrics

    This notebook shows how to evaluate hyperparameter configurations
    against multiple scoring metrics and access per-metric results.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)).
    """)
    return


@app.cell
def _(make_classification):
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=0
    )
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Pass Multiple Scorer Names

    Provide a list of scorer names to `scoring` and set `refit` to
    the metric used for selecting the best model.
    """)
    return


@app.cell
def _(FloatDistribution, LogisticRegression, OptunaSearchCV, X, y):
    search = OptunaSearchCV(
        LogisticRegression(max_iter=200),
        {"C": FloatDistribution(1e-2, 10.0, log=True)},
        scoring=["accuracy", "f1"],
        refit="accuracy",
        n_trials=10,
        cv=3,
    )
    search.fit(X, y)
    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    _cols = [c for c in search.cv_results_ if c.startswith("mean_test_")]
    mo.md(f"""
    **Available metric columns:** `{_cols}`
    **Best accuracy:** `{search.best_score_:.3f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Access Per-Metric Results

    Each metric has its own columns in `cv_results_`. Use pandas to
    compare metrics side by side.
    """)
    return


@app.cell
def _(mo, search):
    _results = search.cv_results_
    _rows = [
        {
            "params": _results["params"][i],
            "mean_test_accuracy": round(_results["mean_test_accuracy"][i], 4),
            "mean_test_f1": round(_results["mean_test_f1"][i], 4),
        }
        for i in range(len(_results["params"]))
    ]
    _rows.sort(key=lambda r: r["mean_test_accuracy"], reverse=True)
    mo.ui.table(_rows[:5])
    return


if __name__ == "__main__":
    app.run()
