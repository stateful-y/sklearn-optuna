"""Objective function for Optuna hyperparameter optimization."""

import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_validate


class _Objective:
    """Objective function for Optuna trials in OptunaSearchCV.

    This class encapsulates the logic for evaluating hyperparameter
    configurations during Optuna optimization. It handles parameter
    suggestion, cross-validation, score extraction, and error handling.

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
        Cross-validation splitting strategy.

    scorers : dict or callable
        Scoring functions.

    fit_params : dict
        Additional parameters passed to estimator.fit().

    groups : array-like, optional
        Group labels for cross-validation splits.

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
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any],
        X: Any,
        y: Any,
        cv: Any,
        scorers: Any,
        fit_params: dict[str, Any],
        *,
        groups: Any = None,
        verbose: int = 0,
        return_train_score: bool = False,
        error_score: float | str = np.nan,
        multimetric: bool = False,
        refit: bool | str = True,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.X = X
        self.y = y
        self.cv = cv
        self.scorers = scorers
        self.fit_params = fit_params
        self.groups = groups
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.error_score = error_score
        self.multimetric = multimetric
        self.refit = refit

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
        """
        # Suggest parameters
        study_params = self._suggest_parameters(trial)

        # Store parameters as user attributes
        self._store_parameters(trial, study_params)

        try:
            # Run cross-validation
            scores = self._run_cross_validation(study_params)

            # Extract and store scores
            if self.multimetric:
                return self._handle_multimetric_scores(trial, scores)
            else:
                return self._handle_single_metric_scores(trial, scores)

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

    def _run_cross_validation(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run cross-validation with given parameters."""
        cloned_estimator = clone(self.estimator)
        cloned_estimator.set_params(**params)

        return cross_validate(
            cloned_estimator,
            self.X,
            self.y,
            groups=self.groups,
            cv=self.cv,
            scoring=self.scorers,
            params=self.fit_params,
            n_jobs=1,  # Optuna handles parallelization
            verbose=self.verbose,
            return_train_score=self.return_train_score,
            error_score=self.error_score,
        )

    def _handle_multimetric_scores(self, trial: optuna.trial.Trial, scores: dict) -> float:
        """Store multi-metric scores and return primary metric."""
        # Store all test metrics
        for key, val_array in scores.items():
            if key.startswith("test_"):
                metric_name = key[5:]  # remove 'test_'
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # Mean of empty slice
                    mean_val = np.nanmean(val_array)
                trial.set_user_attr(f"mean_test_{metric_name}", mean_val)

                for i, val in enumerate(val_array):
                    trial.set_user_attr(f"split{i}_test_{metric_name}", val)

            elif key.startswith("train_") and self.return_train_score:
                metric_name = key[6:]  # remove 'train_'
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # Mean of empty slice
                    mean_val = np.nanmean(val_array)
                trial.set_user_attr(f"mean_train_{metric_name}", mean_val)

                for i, val in enumerate(val_array):
                    trial.set_user_attr(f"split{i}_train_{metric_name}", val)

        # Determine primary metric to optimize
        if self.refit and isinstance(self.refit, str):
            metric_to_optimize = trial.user_attrs[f"mean_test_{self.refit}"]
        else:
            # Use first scorer from scores result if available
            test_keys = [k for k in scores if k.startswith("test_")]
            if not test_keys:
                raise ValueError("No test scores found in cross_validate result.")
            first_scorer = test_keys[0][5:]

            metric_to_optimize = trial.user_attrs[f"mean_test_{first_scorer}"]

        if np.isnan(metric_to_optimize):
            return float("-inf")
        return metric_to_optimize

    def _handle_single_metric_scores(self, trial: optuna.trial.Trial, scores: dict) -> float:
        """Store single metric scores and return value."""
        test_scores = scores["test_score"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Mean of empty slice
            mean_test_score = np.nanmean(test_scores)
        trial.set_user_attr("mean_test_score", mean_test_score)

        for i, val in enumerate(test_scores):
            trial.set_user_attr(f"split{i}_test_score", val)

        if self.return_train_score:
            train_scores = scores["train_score"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Mean of empty slice
                mean_train_score = np.nanmean(train_scores)
            trial.set_user_attr("mean_train_score", mean_train_score)
            for i, val in enumerate(train_scores):
                trial.set_user_attr(f"split{i}_train_score", val)

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
        assert isinstance(error_value, int | float)

        if np.isnan(error_value):
            return float("-inf")
        return error_value
