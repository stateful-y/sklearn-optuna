"""Optuna-powered hyperparameter search for scikit-learn."""

from numbers import Integral, Real

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from sklearn.base import clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import indexable
from sklearn.utils._param_validation import Interval

from sklearn_optuna.objective import _Objective
from sklearn_optuna.optuna import Callback, Sampler, Storage
from sklearn_optuna.utils import _build_cv_results


class OptunaSearchCV(BaseSearchCV):
    """Hyperparameter search using Optuna optimization.

    OptunaSearchCV implements a "fit" and a "score" method and provides
    hyperparameter optimization using Optuna's trial-based optimization
    framework. It automatically manages the Optuna study and suggests parameters
    using the specified sampler.

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

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see [scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter));
        - a callable (see [implementing scorers](https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object)) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.

        See [multimetric grid search](https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search) for an example.

    sampler : Sampler, default=None
        A wrapped Optuna sampler. If None, TPESampler is used.

    storage : Storage, default=None
        A wrapped Optuna storage. If None, in-memory storage is used.

    callbacks : dict of str to Callback, default=None
        Dictionary mapping callback names to Callback instances. Each callback
        is invoked at the end of each trial with the study and trial objects.

    n_trials : int, default=10
        Number of trials for hyperparameter search. Each trial evaluates one
        set of hyperparameters.

    timeout : float, default=None
        Stop study after the given number of seconds. If this argument is set
        to None, the study is executed without time limitation.

    n_jobs : int, default=None
        Number of parallel trials to run. This parameter is passed directly to
        Optuna's ``study.optimize(n_jobs=...)`` which uses threading for
        parallelization. ``None`` or ``1`` runs trials sequentially. ``-1`` uses
        all available CPU cores. Note that each trial runs cross-validation with
        ``n_jobs=1`` internally to avoid nested parallelism.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - [CV splitter](https://scikit-learn.org/stable/glossary.html#term-CV-splitter),
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer to the [cross-validation user guide](https://scikit-learn.org/stable/modules/cross_validation.html) for the various
        cross-validation strategies that can be used here.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

        - `>1` : the computation time for each fold and parameter candidate is
          displayed;
        - `>2` : the score is also displayed;
        - `>3` : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

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

    classes_ : ndarray of shape (n_classes,)
        Class labels. Only available when `refit=True` and the underlying
        estimator is a classifier.

    study_ : optuna.study.Study
        The Optuna study object containing all trials and optimization history.

    trials_ : list of optuna.trial.FrozenTrial
        The list of all trials executed during the search.

    n_features_in_ : int
        Number of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit).

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during [fit](https://scikit-learn.org/stable/glossary.html#term-fit). Defined only when `X`
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
    ...     sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
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
        "sampler": [Sampler, None],
        "storage": [Storage, None],
        "callbacks": [dict, None],
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        scoring=None,
        sampler=None,
        storage=None,
        callbacks=None,
        n_trials=10,
        timeout=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        error_score=np.nan,
        return_train_score=False,
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
        self.sampler = sampler
        self.storage = storage
        self.timeout = timeout
        self.callbacks = callbacks

    def fit(self, X, y=None, *, study=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        study : optuna.study.Study, default=None
            An existing Optuna study to continue optimization from. If None, a new
            study will be created.

        **params : dict of str -> object
            Parameters passed to the `fit` method of the estimator, the scorer,
            and the CV splitter.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        X, y = indexable(X, y)
        scorers, refit_metric = super()._get_scorers()
        self.scorer_ = scorers
        self.multimetric_ = not (
            callable(self.scoring) or self.scoring is None or isinstance(self.scoring, str | bytes)
        )

        # Validate param_distributions
        for param_name, distribution in self.param_distributions.items():
            if not isinstance(distribution, BaseDistribution):
                raise ValueError(
                    f"Parameter '{param_name}' has an invalid distribution. "
                    f"Expected optuna.distributions.BaseDistribution, got {type(distribution)}."
                )

        # Handle metadata routing
        routed_params = self._get_routed_params_for_fit(params)

        fit_params = params
        groups = params.get("groups")
        if "estimator" in routed_params and "fit" in routed_params["estimator"]:
            fit_params = routed_params["estimator"]["fit"]

        # Store the study if provided, otherwise clear previous study
        if study is not None:
            self.study_ = study
        else:
            # Reset study for fresh fit
            self.study_ = None

        # Instantiate sampler and storage from wrappers if provided
        sampler_instance = None
        if self.sampler is not None:
            sampler_instance = self.sampler.instantiate().instance_

        storage_instance = None
        if self.storage is not None:
            storage_instance = self.storage.instantiate().instance_

        # Validate and prepare callbacks
        callback_list = None
        if self.callbacks is not None:
            if not isinstance(self.callbacks, dict):
                raise TypeError(f"callbacks must be a dict of str to Callback, got {type(self.callbacks)}")
            callback_list = []
            for name, callback in self.callbacks.items():
                if not isinstance(callback, Callback):
                    raise TypeError(f"Callback '{name}' must be a Callback instance, got {type(callback)}")
                callback.instantiate()
                callback_list.append(callback)

        # Create or use existing study
        if not hasattr(self, "study_") or self.study_ is None:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler_instance,
                storage=storage_instance,
            )
        else:
            study = self.study_
            if sampler_instance is not None:
                study.sampler = sampler_instance

        # Create objective function
        objective = _Objective(
            estimator=estimator,
            param_distributions=self.param_distributions,
            X=X,
            y=y,
            cv=self.cv,
            scorers=scorers,
            fit_params=fit_params,
            groups=groups,
            verbose=self.verbose,
            return_train_score=self.return_train_score,
            error_score=self.error_score,
            multimetric=self.multimetric_,
            refit=self.refit,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callback_list,
            n_jobs=self.n_jobs if self.n_jobs is not None else 1,
        )

        # Store study and trials
        self.study_ = study
        self.trials_ = study.trials

        # Build cv_results_ attribute from trials
        self.cv_results_ = _build_cv_results(self.trials_, self.multimetric_, self.return_train_score)

        # If no completed trials, we can't find best params
        if not self.cv_results_["params"] or len(self.cv_results_["params"]) == 0:
            # This can happen if all trials failed
            # We should probably follow BaseSearchCV behavior which leaves best_index_ unset?
            # Or raise ValueError if refit=True
            if self.refit:
                raise ValueError("No trials were completed. 'refit' cannot be true.")
            return self

        # Find best parameters and refit
        if self.refit or not self.multimetric_:
            if callable(self.refit):
                self.best_estimator_ = self.refit(self.cv_results_)
            else:
                if self.multimetric_:
                    if isinstance(self.refit, str):
                        self.best_index_ = self.cv_results_[f"rank_test_{self.refit}"].argmin()
                        self.best_score_ = self.cv_results_[f"mean_test_{self.refit}"][self.best_index_]
                    else:  # pragma: no cover
                        # Unreachable due to BaseSearchCV validation
                        pass
                else:
                    self.best_index_ = self.cv_results_["rank_test_score"].argmin()
                    self.best_score_ = self.cv_results_["mean_test_score"][self.best_index_]

                self.best_params_ = self.cv_results_["params"][self.best_index_]

                # Standard refit
                if self.refit:
                    self.best_estimator_ = clone(estimator).set_params(**self.best_params_)
                    # Should we pop groups from fit_params?
                    # BaseSearchCV does: self.best_estimator_.fit(X, y, **fit_params)
                    # Metadata Routing usually handles this.
                    # We pass fit_params as is.
                    self.best_estimator_.fit(X, y, **fit_params)

        return self
