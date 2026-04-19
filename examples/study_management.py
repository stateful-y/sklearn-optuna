"""How to Resume Optimization from Prior Trials.

Reuse an Optuna study to continue optimization runs.
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
    "title": "How to Resume Optimization from Prior Trials",
    "description": "Reuse an Optuna study to continue optimization runs and keep experiments reproducible.",
    "category": "how-to",
    "companion": "pages/how-to/persist-studies.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


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
    # How to Resume Optimization from Prior Trials

    This notebook shows how to create an Optuna study manually
    and pass it to `OptunaSearchCV` to accumulate trials across
    runs.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)).
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
    ## 1. Create a Study

    Create an Optuna study explicitly to control its name, direction,
    and sampler.
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
    ## 2. Pass the Study to fit()

    Pass the existing study to `OptunaSearchCV.fit()` via the
    `study=` parameter. New trials are appended to the study
    instead of creating a new one.
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
    **Study name:** {search.study_.study_name}

    **Total trials:** {len(search.study_.trials)}

    **Best trial number:** {search.study_.best_trial.number}

    **Best score:** {search.study_.best_value:.3f}
    """)
    return


if __name__ == "__main__":
    app.run()
