# How to Prune Unpromising Trials

This guide shows you how to stop a trial after a few cross-validation folds
when it is reliably worse than the best trial so far, instead of running every
fold. Use this when fits are expensive and your search uses enough folds for
an early verdict to mean something.

## Prerequisites

- Sklearn-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search
- At least four cross-validation folds (see step 2)

<!-- COMPANION_NOTEBOOKS -->

## Enable Pruning

Pass a `Pruner` wrapper. With no arguments it wraps Optuna's `WilcoxonPruner`,
which compares the current trial's fold scores with the best trial's scores on
the same folds:

```python
from sklearn_optuna import OptunaSearchCV, Pruner

search = OptunaSearchCV(
    estimator, param_distributions,
    pruner=Pruner(),
    cv=5,
    n_trials=50,
)
search.fit(X, y)
```

Optuna marks `WilcoxonPruner` as experimental and emits an
`ExperimentalWarning` when it is created. The warning is informational.

## Give the Pruner Enough Folds

The default pruner runs a paired statistical test and cannot be confident
before it has seen four folds. With `cv=3` it never stops a trial, and
`fit()` warns:

```text
UserWarning: The pruner cannot stop any trial: WilcoxonPruner with
p_threshold=0.1 and n_startup_steps=2 needs at least 4 folds, but cv yields 3.
```

If you see this warning, raise `cv` to at least 5, or switch to a pruner with
a lower warm-up (next section). Raising `p_threshold` makes the test act
earlier but also makes it wrong more often.

## Choose a Different Pruner

Any Optuna pruner class can be wrapped. `MedianPruner` stops a trial whose
score at fold `i` is below the median of earlier trials at fold `i`, and acts
from the first fold after its warm-up:

```python
import optuna

search = OptunaSearchCV(
    estimator, param_distributions,
    pruner=Pruner(pruner=optuna.pruners.MedianPruner, n_startup_trials=5, n_warmup_steps=1),
    cv=5,
)
```

`ThresholdPruner` stops a trial as soon as a fold score falls below a floor
you choose, with no comparison to other trials:

```python
search = OptunaSearchCV(
    estimator, param_distributions,
    pruner=Pruner(pruner=optuna.pruners.ThresholdPruner, lower=0.6),
    cv=5,
)
```

The median and percentile pruners skip the first `n_warmup_steps` folds; a
warm-up at or above the fold count disables them, and `fit()` warns in that
case too. Refer to the
[Optuna pruners reference](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
for the full list and their parameters.

## Read the Pruned Trials

Pruned trials do not appear in `cv_results_`, so every row of that table is a
full evaluation. Count them and read them from the study:

```python
import optuna

print(search.n_pruned_trials_)

pruned = [t for t in search.trials_ if t.state == optuna.trial.TrialState.PRUNED]
for trial in pruned:
    print(trial.params, trial.intermediate_values)
```

`trial.intermediate_values` maps fold index to that fold's test score for
the metric the search optimizes. A pruned trial also keeps
`split<i>_test_score` and `split<i>_fit_time` in `trial.user_attrs` for the
folds that ran, and has no `mean_test_score`.

If a fold's fit failed under `error_score=np.nan`, that fold is not reported
at all, so its index is missing from `intermediate_values` while
`split<i>_test_score` is `NaN`.

## Plot Fold Scores Per Trial

Fold scores are reported to the study whether or not a pruner is set, so
Optuna's intermediate-value plot works on any search:

```python
import optuna

fig = optuna.visualization.plot_intermediate_values(search.study_)
fig.show()
```

Each line is one trial across folds; pruned trials end early.

## Resume a Study With a Pruner

Load the study and pass it to `fit()` with a pruner:

```python
import optuna
from sklearn.model_selection import KFold

existing_study = optuna.load_study(study_name="svc_search", storage="sqlite:///study.db")

search = OptunaSearchCV(
    estimator, param_distributions,
    pruner=Pruner(),
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
)
search.fit(X, y, study=existing_study)
```

Every `fit()` splits the data once and scores all its trials on those folds.
A resumed `fit()` splits again, so its trials are only comparable with the
earlier ones when the splitter returns the same folds every time: a
non-shuffling splitter, or a shuffling one with an integer `random_state`, as
above.

If the study was created before pruning was enabled, its completed trials
have no intermediate values. The default pruner compares against the best
completed trial's fold scores, so it stays silent until a trial run with the
current release completes.

See [How to Persist and Resume Studies](persist-studies.md) for the storage
setup. With a database storage, reporting adds one row per fold per trial.

## Let the Sampler Learn From Pruned Trials

`TPESampler` ignores pruned trials by default and can keep sampling the same
poor region. Tell it to use their last reported value:

```python
from sklearn_optuna import Sampler

search = OptunaSearchCV(
    estimator, param_distributions,
    pruner=Pruner(),
    sampler=Sampler(sampler=optuna.samplers.TPESampler, consider_pruned_trials=True, seed=0),
    cv=5,
)
```

## Troubleshooting

**Problem: `n_pruned_trials_` is 0 and nothing was pruned**
: Check for the "cannot stop any trial" warning and raise `cv` or lower the
  warm-up. With few trials, the default pruner also has nothing to compare
  against until the first trial completes. Finally, a search whose candidates
  are all close to the best one has nothing to prune.

**Problem: `TypeError: pruner must be a Pruner instance`**
: You passed a raw Optuna pruner. Wrap the class, not an instance:
  `Pruner(pruner=optuna.pruners.MedianPruner, n_warmup_steps=1)`.

**Problem: a parameter combination I expected to see is missing from `cv_results_`**
: It was pruned. Look it up in `search.trials_` or `search.study_.trials`.

## See Also

- [Configuration Reference](../reference/configuration.md): the `pruner`
  parameter and the `n_pruned_trials_` attribute
- [Concepts: Pruning](../explanation/concepts.md#pruning): why fold scores are
  paired across trials and what a pruned trial leaves behind
- [How to Use Callbacks](use-callbacks.md): stopping the whole search, rather
  than one trial, early
