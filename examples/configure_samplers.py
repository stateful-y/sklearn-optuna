"""How to Choose and Configure a Sampler.

Control the optimization algorithm and get reproducible results with Optuna samplers.
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
    "title": "How to Choose and Configure a Sampler",
    "description": "Control the optimization algorithm and get reproducible results with Optuna samplers.",
    "category": "how-to",
    "companion": "pages/how-to/configure-samplers.md",
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
    from sklearn.svm import SVC

    from sklearn_optuna import OptunaSearchCV, Sampler

    return (
        FloatDistribution,
        OptunaSearchCV,
        SVC,
        Sampler,
        make_classification,
        optuna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # How to Choose and Configure a Sampler

    This notebook shows how to choose an Optuna sampler and configure
    it for reproducible hyperparameter searches.

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
    ## 1. Configure TPESampler with a Seed

    Wrap `TPESampler` with the `Sampler` wrapper and pass `seed=`
    for deterministic optimization. Results are reproducible when
    `n_jobs=1`.
    """)
    return


@app.cell
def _(FloatDistribution, OptunaSearchCV, SVC, Sampler, X, optuna, y):
    tpe_search = OptunaSearchCV(
        SVC(),
        {"C": FloatDistribution(0.1, 10.0, log=True)},
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        cv=3,
        n_jobs=1,
    )
    tpe_search.fit(X, y)
    return (tpe_search,)


@app.cell(hide_code=True)
def _(mo, tpe_search):
    mo.md(f"""
    **Sampler:** TPESampler
    **Best C:** `{tpe_search.best_params_['C']:.4f}`
    **Best score:** `{tpe_search.best_score_:.3f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Switch to RandomSampler

    Swap the sampler class to `RandomSampler` for uniform coverage
    of the search space.
    """)
    return


@app.cell
def _(FloatDistribution, OptunaSearchCV, SVC, Sampler, X, optuna, y):
    random_search = OptunaSearchCV(
        SVC(),
        {"C": FloatDistribution(0.1, 10.0, log=True)},
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=42),
        n_trials=10,
        cv=3,
        n_jobs=1,
    )
    random_search.fit(X, y)
    return (random_search,)


@app.cell(hide_code=True)
def _(mo, random_search):
    mo.md(f"""
    **Sampler:** RandomSampler
    **Best C:** `{random_search.best_params_['C']:.4f}`
    **Best score:** `{random_search.best_score_:.3f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Verify Reproducibility

    Run the same search again with the same seed. The results
    should match exactly.
    """)
    return


@app.cell
def _(FloatDistribution, OptunaSearchCV, SVC, Sampler, X, optuna, y):
    repeat_search = OptunaSearchCV(
        SVC(),
        {"C": FloatDistribution(0.1, 10.0, log=True)},
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        cv=3,
        n_jobs=1,
    )
    repeat_search.fit(X, y)
    return (repeat_search,)


@app.cell(hide_code=True)
def _(mo, repeat_search, tpe_search):
    _match = tpe_search.best_params_["C"] == repeat_search.best_params_["C"]
    mo.md(f"""
    **Original best C:** `{tpe_search.best_params_['C']:.4f}`
    **Repeated best C:** `{repeat_search.best_params_['C']:.4f}`
    **Results match:** `{_match}`
    """)
    return


if __name__ == "__main__":
    app.run()
