"""How to Prune Unpromising Trials.

Stop a trial after a few cross-validation folds when it is reliably worse than the best one.
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
    "title": "How to Prune Unpromising Trials",
    "description": "Stop a trial after a few folds when it is reliably worse than the best one, and read what it left behind.",
    "category": "how-to",
    "companion": "pages/how-to/prune-trials.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import warnings

    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.exceptions import ExperimentalWarning
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC

    from sklearn_optuna import OptunaSearchCV, Pruner, Sampler

    # Optuna marks WilcoxonPruner experimental; the warning is informational.
    warnings.simplefilter("ignore", ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        FloatDistribution,
        OptunaSearchCV,
        Pruner,
        SVC,
        Sampler,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Prune Unpromising Trials

    This notebook shows how to stop a trial after a few cross-validation
    folds when it is reliably worse than the best trial so far, instead of
    running every fold.

    **Prerequisites:** Familiarity with the
    OptunaSearchCV quickstart
    ([View](/examples/quickstart/) · [Open in marimo](/examples/quickstart/edit/)).
    """)
    return


@app.cell
def _(make_classification):
    # Setup data
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=4, random_state=0
    )
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Enable Pruning

    Pass a `Pruner` wrapper. With no arguments it wraps Optuna's
    `WilcoxonPruner`, which compares the current trial's fold scores with
    the best trial's scores on the same folds. It needs at least four
    folds to act, so the search uses `cv=5`.
    """)
    return


@app.cell
def _(FloatDistribution, OptunaSearchCV, Pruner, SVC, Sampler, X, optuna, y):
    search = OptunaSearchCV(
        SVC(),
        {
            "C": FloatDistribution(1e-3, 1e3, log=True),
            "gamma": FloatDistribution(1e-4, 1e1, log=True),
        },
        pruner=Pruner(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=0),
        cv=5,
        n_trials=40,
    )
    search.fit(X, y)
    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    mo.md(f"""
    Trials run: {len(search.trials_)}
    Pruned trials: {search.n_pruned_trials_}
    Rows in `cv_results_`: {len(search.cv_results_["params"])}
    Best score: {search.best_score_:.3f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Read a Pruned Trial

    Pruned trials are left out of `cv_results_` but stay in `trials_`.
    Each one keeps the fold scores it reported and the per-fold attributes
    of the folds that ran, and has no mean score.
    """)
    return


@app.cell
def _(optuna, search):
    pruned = [t for t in search.trials_ if t.state == optuna.trial.TrialState.PRUNED]
    first_pruned = pruned[0] if pruned else None
    return first_pruned, pruned


@app.cell(hide_code=True)
def _(first_pruned, mo, pruned):
    if first_pruned is None:
        _text = "No trial was pruned in this run."
    else:
        _folds = sorted(first_pruned.intermediate_values)
        _attrs = {k: v for k, v in first_pruned.user_attrs.items() if k.startswith("split")}
        _text = f"""
    {len(pruned)} pruned trials. The first one (trial {first_pruned.number}):

    - parameters: `{first_pruned.params}`
    - folds reported: `{_folds}`
    - fold scores: `{first_pruned.intermediate_values}`
    - has a mean score: `{"mean_test_score" in first_pruned.user_attrs}`
    - per-fold attributes: `{sorted(_attrs)}`
    """
    mo.md(_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Plot Fold Scores Per Trial

    Fold scores are reported to the study whether or not a pruner is set,
    so Optuna's intermediate-value plot works on any search. Each line is
    one trial across folds; pruned trials end early.
    """)
    return


@app.cell
def _(optuna, search):
    fig_intermediate = optuna.visualization.plot_intermediate_values(search.study_)
    fig_intermediate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Choose a Different Pruner

    Any Optuna pruner class can be wrapped. `MedianPruner` stops a trial
    whose fold score is below the median of earlier trials at the same
    fold, and acts from the first fold after its warm-up. A warm-up at or
    above the fold count would disable it, and `fit()` warns in that case.
    """)
    return


@app.cell
def _(FloatDistribution, OptunaSearchCV, Pruner, SVC, Sampler, X, optuna, y):
    median_search = OptunaSearchCV(
        SVC(),
        {
            "C": FloatDistribution(1e-3, 1e3, log=True),
            "gamma": FloatDistribution(1e-4, 1e1, log=True),
        },
        pruner=Pruner(pruner=optuna.pruners.MedianPruner, n_startup_trials=5, n_warmup_steps=1),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=0),
        cv=5,
        n_trials=40,
    )
    median_search.fit(X, y)
    return (median_search,)


@app.cell(hide_code=True)
def _(median_search, mo):
    mo.md(f"""
    With `MedianPruner`: {median_search.n_pruned_trials_} of
    {len(median_search.trials_)} trials pruned, best score
    {median_search.best_score_:.3f}.
    """)
    return


if __name__ == "__main__":
    app.run()
