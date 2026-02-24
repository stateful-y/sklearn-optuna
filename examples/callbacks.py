"""Callbacks.

Learn how to use Optuna callbacks in OptunaSearchCV.
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
    from optuna.distributions import FloatDistribution
    from optuna.study import MaxTrialsCallback
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC

    from sklearn_optuna import Callback, OptunaSearchCV, Sampler

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
    # Callbacks

    ## What You'll Learn

    - How to wrap Optuna callbacks with sklearn-optuna's `Callback` wrapper
    - How to pass a dictionary of callbacks to `OptunaSearchCV`
    - How callbacks can override default stopping behavior (e.g., early stopping)

    ## Prerequisites

    Familiarity with the OptunaSearchCV quickstart (see quickstart.py).
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
    ## 1. Using MaxTrialsCallback

    Wrap Optuna's `MaxTrialsCallback` using sklearn-optuna's `Callback` wrapper. This makes
    the callback compatible with sklearn's `get_params()` and `set_params()` interface, allowing
    it to survive cloning. The callback stops optimization after a specified number of completed
    trials, even if `n_trials` is set higher.
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

    Pass a dictionary of callbacks to `OptunaSearchCV` using the `callbacks=` parameter. Each
    callback is instantiated and invoked at the end of every trial. Here, the callback will
    stop the search after 5 completed trials, even though `n_trials=20` is specified. This
    demonstrates how callbacks can override default stopping behavior.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    - **Callback wrapper** -- Wrap Optuna callbacks with `Callback` for sklearn compatibility
    - **Dictionary interface** -- Pass a dictionary of callbacks to `OptunaSearchCV`
    - **Custom stopping** -- Useful for stopping criteria beyond simple trial counts or timeout

    ## Next Steps

    - **Nested pipelines**: See nested_pipeline.py for advanced optimization patterns with pipelines
    - **Metadata routing**: See metadata_routing.py to route sample weights through the search
    """)
    return


if __name__ == "__main__":
    app.run()
