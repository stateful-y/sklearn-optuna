"""Objective function for Optuna hyperparameter optimization."""

import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _fit_and_score,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)


class _Objective:
    """Objective function for Optuna trials in OptunaSearchCV.

    This class encapsulates the logic for evaluating hyperparameter
    configurations during Optuna optimization. It handles parameter
    suggestion, fold-by-fold cross-validation, score extraction, reporting
    of fold scores to the trial, pruning, and error handling.

    Parameters
    ----------
    estimator : estimator object
        Base estimator to evaluate.

    param_distributions : dict[str, optuna.distributions.BaseDistribution]
        Dictionary mapping parameter names to Optuna distributions.

    X : array-like
        Training data.

    y : array-like
        Target values.

    cv : cross-validation generator
        Resolved cross-validation splitting strategy (the output of
        ``check_cv``).

    scorers : callable or _MultimetricScorer
        Scoring function(s), as returned by ``BaseSearchCV._get_scorers``.

    routed_params : Bunch
        Metadata routed for ``fit``: ``routed_params.estimator.fit``,
        ``routed_params.scorer.score`` and ``routed_params.splitter.split``.

    verbose : int, default=0
        Verbosity level.

    return_train_score : bool, default=False
        Whether to include training scores.

    error_score : numeric or 'raise', default=np.nan
        Value to assign on error, or 'raise' to propagate exceptions.

    multimetric : bool, default=False
        Whether multiple metrics are being optimized.

    refit : bool or str, default=True
        Primary metric name for multi-metric optimization.

    prune : bool, default=False
        Whether to ask the study's pruner after each fold whether the trial
        should stop. Fold scores are reported to the trial regardless.

    Notes
    -----
    The trial evaluation flow is:

    1. Suggest parameters from distributions using ``trial._suggest()``.
    2. Store parameters as trial user attributes (``param_{name}``).
    3. Fit and score each fold in turn. After each fold, store that fold's
       scores and timings as user attributes, report the optimized metric's
       score to the trial when it is finite, and, when pruning is enabled,
       raise ``optuna.TrialPruned`` if the pruner says so.
    4. Store aggregated timings, then apply scikit-learn's fit-failure
       policy (warn, or raise when every fold failed).
    5. Store mean scores and return the mean test score (or the primary
       metric in multi-metric mode).

    A pruned trial therefore keeps its parameters and the per-fold
    attributes of the folds that ran, and has no mean score.

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any],
        X: Any,
        y: Any,
        cv: Any,
        scorers: Any,
        routed_params: Any,
        *,
        verbose: int = 0,
        return_train_score: bool = False,
        error_score: float | str = np.nan,
        multimetric: bool = False,
        refit: bool | str = True,
        prune: bool = False,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.X = X
        self.y = y
        self.cv = cv
        self.scorers = scorers
        self.routed_params = routed_params
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.error_score = error_score
        self.multimetric = multimetric
        self.refit = refit
        self.prune = prune

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Evaluate a single trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object for suggesting parameters.

        Returns
        -------
        float
            Optimization objective value (score to maximize).

        Raises
        ------
        optuna.TrialPruned
            If the pruner stopped the trial, or user code raised it.
        """
        # Suggest parameters
        study_params = self._suggest_parameters(trial)

        # Store parameters as user attributes
        self._store_parameters(trial, study_params)

        try:
            # Run cross-validation fold by fold
            scores = self._run_cross_validation(trial, study_params)

            # Aggregate and store scores
            if self.multimetric:
                return self._handle_multimetric_scores(trial, scores)
            else:
                return self._handle_single_metric_scores(trial, scores)

        except optuna.TrialPruned:
            # A prune is a verdict, not a failure: let Optuna record it.
            raise
        except Exception as e:
            return self._handle_error(trial, e)

    def _suggest_parameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest parameters from distributions."""
        study_params = {}
        for param_name, distribution in self.param_distributions.items():
            study_params[param_name] = trial._suggest(param_name, distribution)
        return study_params

    def _store_parameters(self, trial: optuna.trial.Trial, params: dict[str, Any]) -> None:
        """Store parameters as trial user attributes."""
        for param_name, param_value in params.items():
            trial.set_user_attr(f"param_{param_name}", param_value)

    def _run_cross_validation(self, trial: optuna.trial.Trial, params: dict[str, Any]) -> dict[str, Any]:
        """Fit and score every fold, reporting each fold's score to the trial.

        Returns the same dictionary ``sklearn.model_selection.cross_validate``
        would return (``test_<metric>``, ``train_<metric>``, ``fit_time``,
        ``score_time``), so the aggregation step is unchanged. Unlike
        ``cross_validate`` the folds run here, one at a time, which is what
        makes per-fold reporting and pruning possible.
        """
        base_estimator = clone(self.estimator)
        splits = list(self.cv.split(self.X, self.y, **self.routed_params.splitter.split))

        if self.verbose > 0:
            # Same shape as BaseSearchCV's per-batch message; the per-fold
            # messages come from _fit_and_score at verbose > 1.
            print(f"[Trial {trial.number}] Fitting {len(splits)} folds for parameters {params}")  # noqa: T201

        results: list[dict[str, Any]] = []
        for fold_idx, (train, test) in enumerate(splits):
            result = _fit_and_score(
                clone(base_estimator),
                self.X,
                self.y,
                scorer=self.scorers,
                train=train,
                test=test,
                verbose=self.verbose,
                parameters=params,
                fit_params=self.routed_params.estimator.fit,
                score_params=self.routed_params.scorer.score,
                return_train_score=self.return_train_score,
                return_times=True,
                error_score=self.error_score,
            )
            results.append(result)

            self._store_fold(trial, fold_idx, result)
            self._report_fold(trial, fold_idx, result["test_scores"])

        self._store_timing(trial, results)

        # Same fit-failure policy as cross_validate: expand error scores for
        # callable scorers, warn about failed fits, and raise when every fold
        # failed. Timings are already stored, so that raise still leaves a
        # trial with timing columns.
        _insert_error_scores(results, self.error_score)
        _warn_or_raise_about_fit_failures(results, self.error_score)
        aggregated = _aggregate_score_dicts(results)

        ret: dict[str, Any] = {"fit_time": aggregated["fit_time"], "score_time": aggregated["score_time"]}
        test_scores_dict = _normalize_score_results(aggregated["test_scores"])
        for name in test_scores_dict:
            ret[f"test_{name}"] = test_scores_dict[name]
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(aggregated["train_scores"])
            for name in train_scores_dict:
                ret[f"train_{name}"] = train_scores_dict[name]
        return ret

    def _store_fold(self, trial: optuna.trial.Trial, fold_idx: int, result: dict[str, Any]) -> None:
        """Store one fold's scores and timings as trial user attributes."""
        trial.set_user_attr(f"split{fold_idx}_fit_time", float(result["fit_time"]))
        trial.set_user_attr(f"split{fold_idx}_score_time", float(result["score_time"]))

        self._store_fold_scores(trial, fold_idx, "test", result["test_scores"])
        if self.return_train_score and "train_scores" in result:
            self._store_fold_scores(trial, fold_idx, "train", result["train_scores"])

    @staticmethod
    def _store_fold_scores(trial: optuna.trial.Trial, fold_idx: int, kind: str, scores: Any) -> None:
        """Store ``split{i}_{kind}_{metric}`` attributes for one fold."""
        if isinstance(scores, dict):
            for metric_name, value in scores.items():
                trial.set_user_attr(f"split{fold_idx}_{kind}_{metric_name}", _as_float(value))
        else:
            trial.set_user_attr(f"split{fold_idx}_{kind}_score", _as_float(scores))

    def _optimized_fold_score(self, test_scores: Any) -> float:
        """Return the fold score of the metric the search optimizes."""
        if isinstance(test_scores, dict):
            if self.refit and isinstance(self.refit, str):
                return _as_float(test_scores.get(self.refit, float("nan")))
            # First scorer, which is the metric _handle_multimetric_scores optimizes.
            return _as_float(next(iter(test_scores.values()), float("nan")))
        return _as_float(test_scores)

    def _report_fold(self, trial: optuna.trial.Trial, fold_idx: int, test_scores: Any) -> None:
        """Report a finite fold score to the trial and prune when asked to.

        A non-finite score (a failed fold under ``error_score=np.nan``) is
        not a measurement, so it is not reported: the pruners disagree on
        what NaN means and none of them should act on it.
        """
        value = self._optimized_fold_score(test_scores)
        if not np.isfinite(value):
            return

        trial.report(value, step=fold_idx)
        if self.prune and trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned after fold {fold_idx}.")

    @staticmethod
    def _store_timing(trial: optuna.trial.Trial, results: list[dict[str, Any]]) -> None:
        """Store aggregated fit and score times as trial user attributes."""
        fit_times = [float(r["fit_time"]) for r in results]
        score_times = [float(r["score_time"]) for r in results]
        trial.set_user_attr("mean_fit_time", float(np.mean(fit_times)))
        trial.set_user_attr("std_fit_time", float(np.std(fit_times)))
        trial.set_user_attr("mean_score_time", float(np.mean(score_times)))
        trial.set_user_attr("std_score_time", float(np.std(score_times)))

    def _handle_multimetric_scores(self, trial: optuna.trial.Trial, scores: dict) -> float:
        """Store multi-metric mean scores and return primary metric."""
        for key, val_array in scores.items():
            if key.startswith("test_"):
                metric_name = key[5:]  # remove 'test_'
                trial.set_user_attr(f"mean_test_{metric_name}", _nanmean(val_array))
            elif key.startswith("train_") and self.return_train_score:
                metric_name = key[6:]  # remove 'train_'
                trial.set_user_attr(f"mean_train_{metric_name}", _nanmean(val_array))

        # Determine primary metric to optimize
        if self.refit and isinstance(self.refit, str):
            metric_to_optimize = trial.user_attrs[f"mean_test_{self.refit}"]
        else:
            # Use first scorer from scores result if available
            test_keys = [k for k in scores if k.startswith("test_")]
            if not test_keys:
                raise ValueError("No test scores found in cross-validation result.")
            first_scorer = test_keys[0][5:]

            metric_to_optimize = trial.user_attrs[f"mean_test_{first_scorer}"]

        if np.isnan(metric_to_optimize):
            return float("-inf")
        return metric_to_optimize

    def _handle_single_metric_scores(self, trial: optuna.trial.Trial, scores: dict) -> float:
        """Store single metric mean scores and return value."""
        mean_test_score = _nanmean(scores["test_score"])
        trial.set_user_attr("mean_test_score", mean_test_score)

        if self.return_train_score:
            trial.set_user_attr("mean_train_score", _nanmean(scores["train_score"]))

        if np.isnan(mean_test_score):
            return float("-inf")
        return mean_test_score

    def _handle_error(self, trial: optuna.trial.Trial, exception: Exception) -> int | float:
        """Handle exceptions during trial evaluation."""
        trial.set_user_attr("exception", str(exception))
        trial.set_user_attr("exception_type", type(exception).__name__)

        if isinstance(self.error_score, str) and self.error_score == "raise":
            raise exception

        # At this point error_score must be numeric (not "raise")
        error_value = self.error_score
        assert isinstance(error_value, int | float)  # noqa: S101 -- type-narrowing guard; the "raise" case is handled above

        if np.isnan(error_value):
            return float("-inf")
        return error_value


def _as_float(value: Any) -> float:
    """Coerce a score to float, mapping anything non-numeric to NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(values: Any) -> float:
    """Mean ignoring NaN, without the empty-slice warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Mean of empty slice
        return float(np.nanmean(np.asarray(values, dtype=float)))
