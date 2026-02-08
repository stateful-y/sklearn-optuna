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

    def __init__(self, sampler: type = optuna.samplers.TPESampler, **params: dict[str, object]) -> None:
        BaseClassWrapper.__init__(self, estimator_class=sampler, **params)


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

    def __init__(self, storage: type = optuna.storages.RDBStorage, **params: dict[str, object]) -> None:
        BaseClassWrapper.__init__(self, estimator_class=storage, **params)
