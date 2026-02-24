"""Visualization.

Visualize optimization history using Optuna's plotting functions.
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
    # Visualization

    ## What You'll Learn

    - How to access the Optuna `study_` attribute from a completed `OptunaSearchCV`
    - How to generate optimization history and contour plots using Optuna's visualization module
    - How to interpret search progress and parameter relationships

    ## Prerequisites

    Familiarity with the OptunaSearchCV quickstart (see quickstart.py). Plotly is used for interactive plots.
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
    ## 1. Run Search

    Run a hyperparameter search with multiple trials to generate enough data for meaningful
    visualizations. The more trials you run, the clearer the patterns in the optimization history
    and parameter relationships will be. Here we use a RandomForestClassifier with two hyperparameters
    to demonstrate different visualization types.
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
    ## 2. Visualize Results

    Use Optuna's visualization module to create interactive Plotly figures. The optimization
    history plot shows how objective values improved over trials, revealing convergence patterns
    and potential plateaus. The contour plot displays the relationship between pairs of hyperparameters,
    showing which regions of the search space yield better performance.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Study access** -- Use `search.study_` to get the Optuna Study object after fitting
    - **Built-in plots** -- Pass `search.study_` to standard Optuna visualization functions for interactive Plotly figures

    ## Next Steps

    - **Callbacks**: See callbacks.py to stop trials early based on custom criteria
    - **Nested pipelines**: See nested_pipeline.py for advanced optimization patterns
    """)
    return


if __name__ == "__main__":
    app.run()
