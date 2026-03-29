"""How to Stop Optimization Early with Callbacks.

Stop unneeded work early by adding Optuna callbacks to OptunaSearchCV.
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
    "title": "How to Stop Optimization Early with Callbacks",
    "description": "Stop unneeded work early by adding Optuna callbacks to your search.",
    "category": "how-to",
    "companion": "pages/how-to/use-callbacks.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.study import MaxTrialsCallback
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC

    from sklearn_optuna import Callback, OptunaSearchCV

    return (
        Callback,
        FloatDistribution,
        MaxTrialsCallback,
        OptunaSearchCV,
        SVC,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Stop Optimization Early with Callbacks

    This notebook shows how to stop optimization early by adding
    Optuna callbacks to `OptunaSearchCV`.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)).
    """)
    return


@app.cell
def _(make_classification):
    # Setup data
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=2, random_state=0
    )
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Wrap the Callback

    Wrap Optuna's `MaxTrialsCallback` with the `Callback` wrapper
    for sklearn compatibility.
    """)
    return


@app.cell
def _(Callback, MaxTrialsCallback, optuna):
    # Wrap the callback using the Callback wrapper class
    max_trials_cb = Callback(MaxTrialsCallback, n_trials=5, states=(optuna.trial.TrialState.COMPLETE,))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Pass to OptunaSearchCV

    Pass a dictionary of callbacks to `OptunaSearchCV` via the
    `callbacks=` parameter. The callback stops the search after
    5 completed trials, even though `n_trials=20`.
    """)
    return


@app.cell
def _(
    Callback,
    FloatDistribution,
    MaxTrialsCallback,
    OptunaSearchCV,
    SVC,
    X,
    optuna,
    y,
):
    search = OptunaSearchCV(
        SVC(),
        {
            "C": FloatDistribution(0.1, 10.0, log=True),
        },
        # Pass a dictionary of name -> Callback wrapper
        callbacks={
            "max_trials": Callback(
                MaxTrialsCallback,
                n_trials=5,
                states=(optuna.trial.TrialState.COMPLETE,)
            )
        },
        n_trials=20,  # Will be stopped early by callback
    )
    search.fit(X, y)
    return (search,)


@app.cell
def _(search):
    # Check how many trials actually ran
    n_trials_run = len(search.study_.trials)
    return (n_trials_run,)


@app.cell(hide_code=True)
def _(mo, n_trials_run):
    mo.md(f"Requested trials: 20\nActual trials run: {n_trials_run}")
    return


if __name__ == "__main__":
    app.run()
