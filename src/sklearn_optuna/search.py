"""Optuna-powered hyperparameter search for scikit-learn."""

from numbers import Integral, Real
from typing import cast

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from sklearn_optuna.optuna import Sampler, Storage


class OptunaSearchCV(BaseSearchCV):
    """Hyperparameter search using Optuna optimization.

    OptunaSearchCV implements a "fit" and a "score" method and provides
    hyperparameter optimization using Optuna's trial-based optimization
    framework. It automatically manages the Optuna study, suggests parameters
    using the specified sampler, and optionally supports early stopping via pruning.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings defined using Optuna
    distributions.

    Parameters
    ----------
    estimator : estimator object
        An object of that type is instantiated for each search point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict[str, optuna.distributions.BaseDistribution]
        Dictionary with parameter names (str) as keys and Optuna distribution
        objects as values. Distributions define the search space for each
        hyperparameter.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

    n_trials : int, default=10
        Number of trials for hyperparameter search. Each trial evaluates one
        set of hyperparameters.

    timeout : float, default=None
        Stop study after the given number of seconds. If this argument is set
        to None, the study is executed without time limitation.

    study : optuna.study.Study, default=None
        A study corresponds to an optimization task. If None, a new study is
        created.

    sampler : Sampler, default=None
        A sampler wrapped in the Sampler class. If None, TPESampler is used.

    storage : Storage, default=None
        A storage wrapped in the Storage class. If None, in-memory storage
        is used.

    enable_pruning : bool, default=False
        If True, enable early stopping of unpromising trials. Requires the
        estimator to have a ``partial_fit`` method.

        .. note::
           Pruning is not yet implemented in this version.

    max_iter : int, default=10
        Maximum number of iterations for pruning. Only used when
        ``enable_pruning=True``.

        .. note::
           Pruning is not yet implemented in this version.

    subsample : float or int, default=1.0
        Proportion of the training set to use for hyperparameter search.
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset. If int, represents the absolute number of samples.
        The full training set is used for the final refit.

        .. note::
           Subsampling is not yet implemented in this version.

    callbacks : list of callable, default=None
        List of callback functions that are invoked at the end of each trial.
        Each function must accept two parameters with the following types in
        this order: :class:`~optuna.study.Study` and
        :class:`~optuna.trial.FrozenTrial`.

    catch : tuple of Exception, default=()
        A tuple of exception classes to catch during trial execution. Caught
        exceptions are stored in the trial's ``user_attrs`` and do not stop
        the study.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator which gave
        highest score (or smallest loss if specified) on the left out data.
        Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the
        best candidate parameter setting.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    study_ : optuna.study.Study
        The Optuna study object containing all trials and optimization history.

    trials_ : list of optuna.trial.FrozenTrial
        The list of all trials executed during the search.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn_optuna import OptunaSearchCV, Sampler
    >>> from optuna.distributions import FloatDistribution
    >>> import optuna
    >>> X, y = load_iris(return_X_y=True)
    >>> param_distributions = {
    ...     "C": FloatDistribution(0.01, 10.0, log=True),
    ...     "gamma": FloatDistribution(0.001, 1.0, log=True),
    ... }
    >>> search = OptunaSearchCV(
    ...     SVC(),
    ...     param_distributions,
    ...     n_trials=20,
    ...     sampler=Sampler(optuna.samplers.TPESampler, seed=42),
    ... )
    >>> search.fit(X, y)
    OptunaSearchCV(...)
    >>> search.best_params_
    {...}
    """

    _parameter_constraints = {
        **BaseSearchCV._parameter_constraints,
        "n_trials": [Interval(Integral, 1, None, closed="left"), None],
        "timeout": [Interval(Real, 0, None, closed="neither"), None],
        "study": [optuna.study.Study, None],
        "sampler": [Sampler, None],
        "storage": [Storage, None],
        "enable_pruning": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "subsample": [
            Interval(Real, 0.0, 1.0, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "callbacks": [list, None],
        "catch": [tuple],
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        cv=None,
        scoring=None,
        n_jobs=None,
        refit=True,
        verbose=0,
        error_score=np.nan,
        return_train_score=False,
        n_trials=10,
        timeout=None,
        study=None,
        sampler=None,
        storage=None,
        enable_pruning=False,
        max_iter=10,
        subsample=1.0,
        callbacks=None,
        catch=(),
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = study
        self.sampler = sampler
        self.storage = storage
        self.enable_pruning = enable_pruning
        self.max_iter = max_iter
        self.subsample = subsample
        self.callbacks = callbacks
        self.catch = catch

    def _run_search(self, evaluate_candidates):
        """Search all candidates by running Optuna optimization.

        Parameters
        ----------
        evaluate_candidates : callable
            A function that evaluates a list of parameter dictionaries and
            returns a dict of arrays containing the scores.
        """
        # Validate param_distributions
        for param_name, distribution in self.param_distributions.items():
            if not isinstance(distribution, BaseDistribution):
                raise ValueError(
                    f"Parameter '{param_name}' has an invalid distribution. "
                    f"Expected optuna.distributions.BaseDistribution, got {type(distribution)}."
                )

        # Instantiate sampler and storage from wrappers if provided
        sampler_instance = None
        if self.sampler is not None:
            sampler_instance = self.sampler.instantiate().instance_

        storage_instance = None
        if self.storage is not None:
            storage_instance = self.storage.instantiate().instance_

        # Create or use existing study
        if self.study is None:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler_instance,
                storage=storage_instance,
            )
        else:
            study = self.study
            if sampler_instance is not None:
                study.sampler = sampler_instance

        # Define objective function
        def objective(trial):
            # Suggest parameters using trial
            params = {}
            for param_name, distribution in self.param_distributions.items():
                params[param_name] = trial._suggest(param_name, distribution)

            # Store parameters in trial user attributes
            for param_name, param_value in params.items():
                trial.set_user_attr(f"param_{param_name}", param_value)

            try:
                # TODO: It seems `results` is global and gets extended on each call?
                # TODO: How to handle this? I could take -1 index instead of 0, but is that reliable?
                # Use BaseSearchCV's evaluate_candidates to evaluate the params
                results = evaluate_candidates([params])

                # Extract scores from results
                # For multi-metric scoring, keys are like "mean_test_{metric_name}"
                # For single metric, key is "mean_test_score"
                if "mean_test_score" in results:
                    # Single metric
                    test_score = results["mean_test_score"][0]
                    train_score = results.get("mean_train_score", [None])[0]
                    trial.set_user_attr("mean_test_score", test_score)
                    if train_score is not None:
                        trial.set_user_attr("mean_train_score", train_score)
                else:
                    # Multi-metric - find primary metric to optimize
                    # Use first metric key as default or the refit metric
                    test_score_keys = [k for k in results if k.startswith("mean_test_")]
                    if test_score_keys:
                        primary_key = test_score_keys[0]
                        test_score = results[primary_key][0]
                        # Store all metrics in user attrs
                        for key in test_score_keys:
                            metric_name = key.replace("mean_test_", "")
                            trial.set_user_attr(f"mean_test_{metric_name}", results[key][0])
                    else:
                        # Fallback if no test scores found
                        test_score = 0.0

                # Extract and store per-split scores
                # Determine number of splits from results keys
                split_keys = [key for key in results if key.startswith("split") and key.endswith("_test_score")]

                for split_key in split_keys:
                    split_test_score = results[split_key][0]
                    trial.set_user_attr(split_key, split_test_score)

                if self.return_train_score:
                    train_split_keys = [
                        key for key in results if key.startswith("split") and key.endswith("_train_score")
                    ]
                    for split_key in train_split_keys:
                        split_train_score = results[split_key][0]
                        trial.set_user_attr(split_key, split_train_score)

                return test_score

            except self.catch as e:
                # Store exception info and return worst possible score
                trial.set_user_attr("exception", str(e))
                trial.set_user_attr("exception_type", type(e).__name__)
                if isinstance(self.error_score, str) and self.error_score == "raise":
                    raise
                return self.error_score

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=self.callbacks,
            catch=self.catch if self.catch else (),
            n_jobs=self.n_jobs if self.n_jobs is not None else 1,
        )

        # Store study and trials
        self.study_ = study
        self.trials_ = study.trials

        # Build cv_results_ from trials
        self._build_cv_results()

    def _build_cv_results(self):
        """Build cv_results_ dict from Optuna trials."""
        n_trials = len(self.trials_)

        # Initialize result arrays
        results: dict[str, list | np.ndarray] = {
            "mean_test_score": np.zeros(n_trials),
            "std_test_score": np.zeros(n_trials),
            "rank_test_score": np.zeros(n_trials, dtype=int),
            "params": [],
        }

        # Add per-split test scores
        if n_trials > 0:
            # Determine number of splits from first trial
            split_keys = [
                key for key in self.trials_[0].user_attrs if key.startswith("split") and key.endswith("_test_score")
            ]
            n_splits = len(split_keys)

            for split_idx in range(n_splits):
                results[f"split{split_idx}_test_score"] = np.zeros(n_trials)

        # Add train scores if requested
        if self.return_train_score:
            results["mean_train_score"] = np.zeros(n_trials)
            results["std_train_score"] = np.zeros(n_trials)

            if n_trials > 0 and "split0_train_score" in self.trials_[0].user_attrs:
                for split_idx in range(n_splits):
                    results[f"split{split_idx}_train_score"] = np.zeros(n_trials)

        # Fill in results from trials
        params_list = cast(list, results["params"])
        for trial_idx, trial in enumerate(self.trials_):
            # Extract parameters
            params = {}
            for key, value in trial.user_attrs.items():
                if key.startswith("param_"):
                    param_name = key[6:]  # Remove "param_" prefix
                    params[param_name] = value
            params_list.append(params)

            # Extract test scores
            mean_test_score = trial.user_attrs.get("mean_test_score", np.nan)
            results["mean_test_score"][trial_idx] = mean_test_score

            # Extract per-split test scores and compute std
            split_scores = []
            for split_idx in range(n_splits):
                score = trial.user_attrs.get(f"split{split_idx}_test_score", np.nan)
                results[f"split{split_idx}_test_score"][trial_idx] = score
                split_scores.append(score)

            if split_scores:
                results["std_test_score"][trial_idx] = np.std(split_scores)

            # Extract train scores if available
            if self.return_train_score:
                mean_train_score = trial.user_attrs.get("mean_train_score", np.nan)
                results["mean_train_score"][trial_idx] = mean_train_score

                # Extract per-split train scores and compute std
                split_train_scores = []
                for split_idx in range(n_splits):
                    score = trial.user_attrs.get(f"split{split_idx}_train_score", np.nan)
                    if f"split{split_idx}_train_score" in results:
                        results[f"split{split_idx}_train_score"][trial_idx] = score
                    split_train_scores.append(score)

                if split_train_scores:
                    results["std_train_score"][trial_idx] = np.std(split_train_scores)

        # Compute rankings (lower rank = better score)
        mean_test_scores = np.asarray(results["mean_test_score"])
        results["rank_test_score"] = np.argsort(-mean_test_scores) + 1

        self.cv_results_ = results

    @property
    def classes_(self):
        """Class labels (only available for classification).

        Returns
        -------
        ndarray of shape (n_classes,)
            The class labels.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.classes_

    @property
    def n_features_in_(self):
        """Number of features seen during fit.

        Returns
        -------
        int
            Number of features.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.n_features_in_
