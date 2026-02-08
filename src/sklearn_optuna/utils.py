"""Utility functions for sklearn-optuna."""

from typing import Any

import numpy as np
import optuna
from scipy.stats import rankdata


def _build_cv_results(
    trials: list[optuna.trial.FrozenTrial],
    multimetric: bool = False,
    return_train_score: bool = False,
) -> dict[str, Any]:
    """Build cv_results_ dict from Optuna trials.

    Converts Optuna trial data into scikit-learn's cv_results_ format.

    Parameters
    ----------
    trials : list of FrozenTrial
        List of all trials from an Optuna study.

    multimetric : bool, default=False
        Whether multiple metrics were used during optimization.

    return_train_score : bool, default=False
        Whether training scores should be included.

    Returns
    -------
    dict
        Dictionary compatible with sklearn's cv_results_ format.

    Notes
    -----
    This function filters for completed trials only and expects
    trial user attributes to be formatted as:
    - 'param_{name}' for parameter values
    - 'mean_test_{metric}' and 'split{i}_test_{metric}' for test scores
    - 'mean_train_{metric}' and 'split{i}_train_{metric}' for train scores
    """
    # Filter completed trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_trials = len(completed_trials)

    if n_trials == 0:
        return {"params": []}

    # Infer metric names
    metric_names = _infer_metric_names(completed_trials[0], multimetric)

    # Initialize result arrays
    results = _initialize_result_arrays(completed_trials, n_trials, metric_names, return_train_score)

    # Fill results from trials
    _fill_results_from_trials(results, completed_trials, metric_names, return_train_score)

    # Compute rankings
    _compute_rankings(results)

    return results


def _infer_metric_names(trial: optuna.trial.FrozenTrial, multimetric: bool) -> list[str]:
    """Infer metric names from trial user attributes."""
    if multimetric:
        metric_names = []
        for key in trial.user_attrs:
            if key.startswith("mean_test_"):
                metric_name = key[10:]  # len("mean_test_")
                metric_names.append(metric_name)
        metric_names.sort()
        return metric_names
    else:
        return ["score"]


def _initialize_result_arrays(
    completed_trials: list[optuna.trial.FrozenTrial],
    n_trials: int,
    metric_names: list[str],
    return_train_score: bool,
) -> dict[str, Any]:
    """Initialize empty result arrays."""
    results = {}

    # Metric arrays
    for name in metric_names:
        suffix = "_score" if name == "score" else f"_{name}"
        results[f"mean_test{suffix}"] = np.full(n_trials, np.nan)
        results[f"std_test{suffix}"] = np.full(n_trials, np.nan)
        if return_train_score:
            results[f"mean_train{suffix}"] = np.full(n_trials, np.nan)
            results[f"std_train{suffix}"] = np.full(n_trials, np.nan)

    # Split arrays
    split_keys = set()
    for trial in completed_trials:
        for key in trial.user_attrs:
            if key.startswith("split"):
                split_keys.add(key)

    for key in split_keys:
        results[key] = np.full(n_trials, np.nan)

    # Parameter arrays
    param_names = set()
    for trial in completed_trials:
        for key in trial.user_attrs:
            if key.startswith("param_"):
                param_names.add(key)

    results["params"] = [None] * n_trials
    for name in param_names:
        results[name] = np.ma.masked_all(n_trials, dtype=object)

    return results


def _fill_results_from_trials(
    results: dict[str, Any],
    completed_trials: list[optuna.trial.FrozenTrial],
    metric_names: list[str],
    return_train_score: bool,
) -> None:
    """Fill result arrays from trial data."""
    for trial_idx, trial in enumerate(completed_trials):
        # Extract parameters
        params = {}
        for key in trial.user_attrs:
            if key.startswith("param_"):
                p_name = key[6:]
                params[p_name] = trial.user_attrs[key]
                results[key][trial_idx] = trial.user_attrs[key]
                results[key].mask[trial_idx] = False
        results["params"][trial_idx] = params

        # Copy metrics and splits
        for key, value in results.items():
            if key in trial.user_attrs:
                value[trial_idx] = trial.user_attrs[key]

        # Compute standard deviations
        for name in metric_names:
            suffix = "_score" if name == "score" else f"_{name}"

            # Test std
            split_keys = [k for k in trial.user_attrs if k.endswith(f"_test{suffix}") and k.startswith("split")]
            if split_keys:
                vals = [trial.user_attrs[k] for k in split_keys]
                results[f"std_test{suffix}"][trial_idx] = np.std(vals)

            # Train std
            if return_train_score:
                split_keys = [k for k in trial.user_attrs if k.endswith(f"_train{suffix}") and k.startswith("split")]
                if split_keys:
                    vals = [trial.user_attrs[k] for k in split_keys]
                    results[f"std_train{suffix}"][trial_idx] = np.std(vals)


def _compute_rankings(results: dict[str, Any]) -> None:
    """Compute rankings for test scores (rank 1 = best)."""
    mean_test_keys = [k for k in results if k.startswith("mean_test")]

    for key in mean_test_keys:
        scores = results[key]
        # Use rankdata with method='min' to handle ties (same as sklearn)
        # Negate scores because rankdata ranks in ascending order
        ranks = rankdata(-scores, method="min")

        suffix = key[9:]  # remove "mean_test" -> leaves "_score" or "_metricname"
        rank_key = f"rank_test{suffix}"

        results[rank_key] = ranks.astype(int)
