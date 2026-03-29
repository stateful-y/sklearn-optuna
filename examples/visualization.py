"""How to Visualize Optimization History.

Plot optimization progress and parameter relationships from a completed search.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optuna",
#     "plotly",
#     "scikit-learn",
#     "sklearn-optuna",
# ]
# ///
import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

__gallery__ = {
    "title": "How to Visualize Optimization History",
    "description": "Plot optimization progress and parameter relationships from a completed search.",
    "category": "how-to",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import optuna
    from optuna.distributions import FloatDistribution, IntDistribution
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    from sklearn_optuna import OptunaSearchCV

    return (
        IntDistribution,
        OptunaSearchCV,
        RandomForestClassifier,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Visualize Optimization History

    This notebook shows how to retrieve the Optuna `study_` from a
    completed `OptunaSearchCV` and plot optimization history and
    parameter contours.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)). Plotly is
    used for interactive plots.
    """)
    return


@app.cell
def _(make_classification):
    # Setup data
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Run a Search

    Run a search with enough trials to produce meaningful plots.
    """)
    return


@app.cell
def _(IntDistribution, OptunaSearchCV, RandomForestClassifier, X, y):
    search = OptunaSearchCV(
        RandomForestClassifier(n_jobs=1),
        {
            "n_estimators": IntDistribution(10, 50, step=10),
            "max_depth": IntDistribution(2, 10, log=True),
        },
        n_trials=15,
    )
    search.fit(X, y)
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Plot the Results

    Pass `search.study_` to Optuna's visualization functions.
    """)
    return


@app.cell
def _(optuna, search):
    fig_history = optuna.visualization.plot_optimization_history(search.study_)
    fig_history
    return


@app.cell
def _(optuna, search):
    fig_contour = optuna.visualization.plot_contour(search.study_)
    fig_contour
    return


if __name__ == "__main__":
    app.run()
