"""Study Management.

Learn how to reuse studies and manage optimization history.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip
        await micropip.install(["scikit-learn", "optuna", "sklearn-optuna"])
    return


@app.cell(hide_code=True)
def _():
    import optuna
    from optuna.distributions import FloatDistribution
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from sklearn_optuna import OptunaSearchCV

    return (
        FloatDistribution,
        LogisticRegression,
        OptunaSearchCV,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Study Management

    ## What You'll Learn

    - How to create an Optuna study manually and pass it to `OptunaSearchCV`
    - How to resume optimization from prior trials without starting from scratch
    - How to maintain reproducible experiment histories across runs

    ## Prerequisites

    Familiarity with the OptunaSearchCV quickstart (see quickstart.py).
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
    ## 1. Create a Study First

    Create an Optuna study explicitly before running `OptunaSearchCV`. By creating the study
    upfront, you can control its name, direction, and sampler. This study object can then be
    passed to multiple search operations, accumulating trials across runs.
    """)
    return


@app.cell
def _(optuna):
    # Create a persistent study
    study = optuna.create_study(direction="maximize", study_name="my_study")
    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Pass Study to fit()

    Pass the existing study to `OptunaSearchCV.fit()` using the `study=` parameter. This
    appends new trials to the study instead of creating a new one. The search will suggest
    parameters based on all historical trials, making the optimization more efficient by
    learning from previous runs.
    """)
    return


@app.cell
def _(FloatDistribution, LogisticRegression, OptunaSearchCV, X, study, y):
    search = OptunaSearchCV(
        LogisticRegression(),
        {"C": FloatDistribution(0.1, 10.0)},
        n_trials=5,
    )

    # Pass the existing study to fit()
    search.fit(X, y, study=study)
    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    mo.md(f"""
    Study name: {search.study_.study_name}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Study reuse** -- Create an Optuna Study manually and pass it to `fit(study=...)` to accumulate trials
    - **Resume optimization** -- Pausing and resuming searches or sharing studies across runs avoids redundant computation

    ## Next Steps

    - **Visualizations**: See visualization.py to plot optimization history and parameter importance
    - **Callbacks**: See callbacks.py to add custom stopping criteria to your searches
    """)
    return


if __name__ == "__main__":
    app.run()
