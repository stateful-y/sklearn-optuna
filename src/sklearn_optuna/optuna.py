"""Optuna wrapper classes for hyperparameter optimization."""

import optuna
from sklearn_wrap import BaseClassWrapper


class Sampler(BaseClassWrapper):
    """Wrapper for Optuna samplers used in hyperparameter optimization.

    Parameters
    ----------
    sampler : type, default=optuna.samplers.TPESampler
        Optuna sampler class to instantiate.

    **params : dict
        Parameters to pass to the sampler constructor.

    """

    _estimator_name = "sampler"
    _estimator_base_class = optuna.samplers.BaseSampler
    _estimator_default_class = optuna.samplers.TPESampler


class Storage(BaseClassWrapper):
    """Wrapper for Optuna storage backends for optimization history.

    Parameters
    ----------
    storage : type, default=optuna.storages.RDBStorage
        Optuna storage class to instantiate.

    **params : dict
        Parameters to pass to the storage constructor.

    """

    _estimator_name = "storage"
    _estimator_base_class = optuna.storages.BaseStorage
    _estimator_default_class = optuna.storages.RDBStorage


class Callback(BaseClassWrapper):
    """Wrapper for Optuna callback classes invoked during optimization.

    Parameters
    ----------
    callback : type
        Optuna callback class to instantiate. The class must implement
        ``__call__(study, trial)`` to be invoked at the end of each trial.

    **params : dict
        Parameters to pass to the callback constructor.

    Examples
    --------
    >>> from optuna.study import MaxTrialsCallback
    >>> callback = Callback(callback=MaxTrialsCallback, n_trials=100)

    """

    _estimator_name = "callback"
    _estimator_base_class = object

    def __init__(self, callback: type, **params: dict[str, object]) -> None:
        if not isinstance(callback, type):
            raise TypeError(f"callback must be a class, got {type(callback)}")
        BaseClassWrapper.__init__(self, callback=callback, **params)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Invoke the callback by instantiating it and calling it.

        Parameters
        ----------
        study : optuna.study.Study
            The study object.

        trial : optuna.trial.FrozenTrial
            The completed trial.

        """
        return self.instance_(study, trial)
