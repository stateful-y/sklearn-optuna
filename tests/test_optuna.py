"""Tests for Optuna wrapper classes."""

import optuna
import pytest
from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    GridSampler,
    NSGAIISampler,
    RandomSampler,
    TPESampler,
)
from optuna.storages import (
    BaseStorage,
    InMemoryStorage,
    RDBStorage,
)
from optuna.study import MaxTrialsCallback
from sklearn.base import BaseEstimator
from sklearn_wrap import BaseClassWrapper
from sklearn_wrap.base import REQUIRED_PARAM_VALUE

from sklearn_optuna.optuna import Callback, Sampler, Storage


# Tests for Sampler class
class TestSampler:
    """Tests for the Sampler wrapper class."""

    def test_sampler_inherits_base_wrapper(self):
        """Test that Sampler inherits from BaseClassWrapper."""
        sampler = Sampler()
        assert isinstance(sampler, BaseClassWrapper)
        assert isinstance(sampler, BaseEstimator)

    def test_sampler_default_initialization(self):
        """Test Sampler with default parameters (TPESampler)."""
        sampler = Sampler()
        assert sampler.estimator_class == TPESampler
        assert sampler.estimator_name == "sampler"
        assert sampler.estimator_base_class == BaseSampler

    def test_sampler_custom_class(self):
        """Test Sampler with custom sampler class."""
        sampler = Sampler(sampler=RandomSampler)
        assert sampler.estimator_class == RandomSampler

    def test_sampler_with_seed_parameter(self):
        """Test Sampler with seed parameter."""
        sampler = Sampler(sampler=TPESampler, seed=42)
        assert sampler.params["seed"] == 42

    def test_sampler_with_multiple_params(self):
        """Test Sampler with multiple parameters."""
        sampler = Sampler(
            sampler=TPESampler,
            seed=123,
            n_startup_trials=15,
            n_ei_candidates=30,
        )
        assert sampler.params["seed"] == 123
        assert sampler.params["n_startup_trials"] == 15
        assert sampler.params["n_ei_candidates"] == 30

    def test_sampler_instantiate_tpe(self):
        """Test instantiating a TPESampler."""
        sampler = Sampler(sampler=TPESampler, seed=42)
        sampler.instantiate()

        assert hasattr(sampler, "instance_")
        assert isinstance(sampler.instance_, TPESampler)
        assert isinstance(sampler.instance_, BaseSampler)

    def test_sampler_instantiate_random(self):
        """Test instantiating a RandomSampler."""
        sampler = Sampler(sampler=RandomSampler, seed=99)
        sampler.instantiate()

        assert isinstance(sampler.instance_, RandomSampler)

    def test_sampler_instantiate_grid(self):
        """Test instantiating a GridSampler with required parameters."""
        search_space = {
            "x": [0, 1, 2],
            "y": [0.0, 0.5, 1.0],
        }
        sampler = Sampler(sampler=GridSampler, search_space=search_space)
        sampler.instantiate()

        assert isinstance(sampler.instance_, GridSampler)

    def test_sampler_instantiate_cmaes(self):
        """Test instantiating a CmaEsSampler."""
        sampler = Sampler(sampler=CmaEsSampler, seed=42)
        sampler.instantiate()

        assert isinstance(sampler.instance_, CmaEsSampler)

    def test_sampler_instantiate_nsgaii(self):
        """Test instantiating a NSGAIISampler."""
        sampler = Sampler(sampler=NSGAIISampler, seed=42)
        sampler.instantiate()

        assert isinstance(sampler.instance_, NSGAIISampler)

    def test_sampler_get_params(self):
        """Test get_params returns correct parameters."""
        sampler = Sampler(sampler=TPESampler, seed=42, n_startup_trials=10)
        params = sampler.get_params()

        assert "sampler" in params
        assert params["sampler"] == TPESampler
        assert params["seed"] == 42
        assert params["n_startup_trials"] == 10

    def test_sampler_get_params_default(self):
        """Test get_params with default sampler."""
        sampler = Sampler()
        params = sampler.get_params()

        assert params["sampler"] == TPESampler
        # Check that default parameters are included
        assert "seed" in params

    def test_sampler_set_params(self):
        """Test set_params updates parameters."""
        sampler = Sampler(sampler=TPESampler, seed=42)
        sampler.set_params(seed=100, n_startup_trials=20)

        assert sampler.params["seed"] == 100
        assert sampler.params["n_startup_trials"] == 20

    def test_sampler_set_params_change_class(self):
        """Test that set_params cannot change sampler class (sklearn_wrap restriction)."""
        sampler = Sampler(sampler=TPESampler)

        # sklearn_wrap prevents changing estimator class via set_params
        with pytest.raises(ValueError, match="Cannot change estimator class"):
            sampler.set_params(sampler=RandomSampler)

    def test_sampler_set_params_returns_self(self):
        """Test that set_params returns self for chaining."""
        sampler = Sampler()
        result = sampler.set_params(seed=42)

        assert result is sampler

    def test_sampler_estimator_name(self):
        """Test that _estimator_name is correctly set."""
        assert Sampler._estimator_name == "sampler"
        sampler = Sampler()
        assert sampler.estimator_name == "sampler"

    def test_sampler_estimator_base_class(self):
        """Test that _estimator_base_class is correctly set."""
        assert Sampler._estimator_base_class == BaseSampler
        sampler = Sampler()
        assert sampler.estimator_base_class == BaseSampler

    def test_sampler_invalid_sampler_class(self):
        """Test that invalid sampler class raises error on initialization."""

        class NotASampler:
            def __init__(self):
                pass

        with pytest.raises(ValueError, match="Invalid sampler class.*Valid estimator class should be derived from"):
            Sampler(sampler=NotASampler)

    def test_sampler_invalid_parameter(self):
        """Test that invalid parameter raises ValueError."""
        with pytest.raises(ValueError, match="not a valid parameter"):
            Sampler(sampler=TPESampler, invalid_param=123)

    def test_sampler_grid_missing_required_param(self):
        """Test GridSampler without required search_space parameter."""
        sampler = Sampler(sampler=GridSampler)

        # Should have REQUIRED_PARAM_VALUE for search_space
        assert sampler.params["search_space"] == REQUIRED_PARAM_VALUE

        with pytest.raises(ValueError, match="requires parameter.*search_space"):
            sampler.instantiate()

    def test_sampler_with_all_default_params(self):
        """Test that sampler includes all default parameters from TPESampler."""
        sampler = Sampler(sampler=TPESampler)

        # TPESampler has many default parameters
        assert "consider_prior" in sampler.params
        assert "prior_weight" in sampler.params
        assert "consider_magic_clip" in sampler.params
        assert "consider_endpoints" in sampler.params

    def test_sampler_sklearn_compatibility(self):
        """Test that Sampler works with sklearn's parameter interface."""
        sampler = Sampler(sampler=TPESampler, seed=42)

        # Get params
        params = sampler.get_params()
        assert "seed" in params
        assert params["seed"] == 42

        # Modify parameters (but not estimator class)
        sampler.set_params(seed=999)
        assert sampler.params["seed"] == 999

    def test_sampler_multiple_instantiation(self):
        """Test that sampler can be instantiated multiple times."""
        sampler = Sampler(sampler=RandomSampler, seed=42)

        sampler.instantiate()
        instance1 = sampler.instance_

        sampler.instantiate()
        instance2 = sampler.instance_

        # Both should be RandomSampler instances
        assert isinstance(instance1, RandomSampler)
        assert isinstance(instance2, RandomSampler)


# Tests for Storage class
class TestStorage:
    """Tests for the Storage wrapper class."""

    def test_storage_inherits_base_wrapper(self):
        """Test that Storage inherits from BaseClassWrapper."""
        storage = Storage(storage=InMemoryStorage)
        assert isinstance(storage, BaseClassWrapper)
        assert isinstance(storage, BaseEstimator)

    def test_storage_default_initialization(self):
        """Test Storage with default parameters (RDBStorage)."""
        storage = Storage()
        assert storage.estimator_class == RDBStorage
        assert storage.estimator_name == "storage"
        assert storage.estimator_base_class == BaseStorage

    def test_storage_custom_class(self):
        """Test Storage with custom storage class."""
        storage = Storage(storage=InMemoryStorage)
        assert storage.estimator_class == InMemoryStorage

    def test_storage_instantiate_inmemory(self):
        """Test instantiating an InMemoryStorage."""
        storage = Storage(storage=InMemoryStorage)
        storage.instantiate()

        assert hasattr(storage, "instance_")
        assert isinstance(storage.instance_, InMemoryStorage)
        assert isinstance(storage.instance_, BaseStorage)

    def test_storage_instantiate_rdb(self):
        """Test instantiating an RDBStorage with required url."""
        storage = Storage(storage=RDBStorage, url="sqlite:///:memory:")
        storage.instantiate()

        assert isinstance(storage.instance_, RDBStorage)

    def test_storage_rdb_missing_url(self):
        """Test RDBStorage without required url parameter."""
        storage = Storage(storage=RDBStorage)

        # Should have REQUIRED_PARAM_VALUE for url
        assert storage.params["url"] == REQUIRED_PARAM_VALUE

        with pytest.raises(ValueError, match="requires parameter.*url"):
            storage.instantiate()

    def test_storage_get_params(self):
        """Test get_params returns correct parameters."""
        storage = Storage(storage=InMemoryStorage)
        params = storage.get_params()

        assert "storage" in params
        assert params["storage"] == InMemoryStorage

    def test_storage_get_params_with_url(self):
        """Test get_params with RDBStorage and url."""
        storage = Storage(storage=RDBStorage, url="sqlite:///:memory:")
        params = storage.get_params()

        assert params["storage"] == RDBStorage
        assert params["url"] == "sqlite:///:memory:"

    def test_storage_set_params(self):
        """Test set_params updates parameters."""
        storage = Storage(storage=RDBStorage, url="sqlite:///old.db")
        storage.set_params(url="sqlite:///new.db")

        assert storage.params["url"] == "sqlite:///new.db"

    def test_storage_set_params_change_class(self):
        """Test that set_params cannot change storage class."""
        storage = Storage(storage=RDBStorage, url="sqlite:///:memory:")

        with pytest.raises(ValueError, match="Cannot change estimator class"):
            storage.set_params(storage=InMemoryStorage)

    def test_storage_set_params_returns_self(self):
        """Test that set_params returns self for chaining."""
        storage = Storage(storage=InMemoryStorage)
        result = storage.set_params()

        assert result is storage

    def test_storage_estimator_name(self):
        """Test that _estimator_name is correctly set."""
        assert Storage._estimator_name == "storage"
        storage = Storage(storage=InMemoryStorage)
        assert storage.estimator_name == "storage"

    def test_storage_estimator_base_class(self):
        """Test that _estimator_base_class is correctly set."""
        assert Storage._estimator_base_class == BaseStorage
        storage = Storage(storage=InMemoryStorage)
        assert storage.estimator_base_class == BaseStorage

    def test_storage_invalid_storage_class(self):
        """Test that invalid storage class raises error on initialization."""

        class NotAStorage:
            def __init__(self):
                pass

        with pytest.raises(ValueError, match="Invalid storage class.*Valid estimator class should be derived from"):
            Storage(storage=NotAStorage)

    def test_storage_invalid_parameter(self):
        """Test that invalid parameter raises ValueError."""
        with pytest.raises(ValueError, match="not a valid parameter"):
            Storage(storage=InMemoryStorage, invalid_param=123)

    def test_storage_sklearn_compatibility(self):
        """Test that Storage works with sklearn's parameter interface."""
        storage = Storage(storage=RDBStorage, url="sqlite:///:memory:")

        # Get params
        params = storage.get_params()
        assert "url" in params
        assert params["url"] == "sqlite:///:memory:"

        # Modify parameters (but not estimator class)
        storage.set_params(url="sqlite:///test.db")
        assert storage.params["url"] == "sqlite:///test.db"

    def test_storage_rdb_with_engine_kwargs(self):
        """Test RDBStorage with engine_kwargs parameter."""
        storage = Storage(
            storage=RDBStorage,
            url="sqlite:///:memory:",
            engine_kwargs={"echo": False},
        )
        assert storage.params["url"] == "sqlite:///:memory:"
        assert storage.params["engine_kwargs"] == {"echo": False}

    def test_storage_multiple_instantiation(self):
        """Test that storage can be instantiated multiple times."""
        storage = Storage(storage=InMemoryStorage)

        storage.instantiate()
        instance1 = storage.instance_

        storage.instantiate()
        instance2 = storage.instance_

        # Both should be InMemoryStorage instances
        assert isinstance(instance1, InMemoryStorage)
        assert isinstance(instance2, InMemoryStorage)


# Integration tests
class TestSamplerStorageIntegration:
    """Integration tests for Sampler and Storage with Optuna."""

    def test_sampler_in_optuna_study(self):
        """Test that instantiated Sampler works in an Optuna study."""
        sampler_wrapper = Sampler(sampler=RandomSampler, seed=42)
        sampler_wrapper.instantiate()

        study = optuna.create_study(sampler=sampler_wrapper.instance_)

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        study.optimize(objective, n_trials=5)

        assert len(study.trials) == 5
        assert isinstance(study.sampler, RandomSampler)

    def test_storage_in_optuna_study(self):
        """Test that instantiated Storage works in an Optuna study."""
        storage_wrapper = Storage(storage=InMemoryStorage)
        storage_wrapper.instantiate()

        study = optuna.create_study(storage=storage_wrapper.instance_)

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2

        study.optimize(objective, n_trials=5)

        assert len(study.trials) == 5
        assert isinstance(study._storage, InMemoryStorage)

    def test_sampler_and_storage_together(self):
        """Test using both Sampler and Storage in a study."""
        sampler_wrapper = Sampler(sampler=TPESampler, seed=123)
        sampler_wrapper.instantiate()

        storage_wrapper = Storage(storage=InMemoryStorage)
        storage_wrapper.instantiate()

        study = optuna.create_study(
            sampler=sampler_wrapper.instance_,
            storage=storage_wrapper.instance_,
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_float("y", 0, 1)
            return x**2 + y**2

        study.optimize(objective, n_trials=10)

        assert len(study.trials) == 10
        assert isinstance(study.sampler, TPESampler)

    def test_change_sampler_params_between_studies(self):
        """Test changing sampler parameters between studies."""
        sampler_wrapper = Sampler(sampler=RandomSampler, seed=1)
        sampler_wrapper.instantiate()

        study1 = optuna.create_study(sampler=sampler_wrapper.instance_)

        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        study1.optimize(objective, n_trials=3)

        # Change parameters and create new sampler
        sampler_wrapper.set_params(seed=2)
        sampler_wrapper.instantiate()

        study2 = optuna.create_study(sampler=sampler_wrapper.instance_)
        study2.optimize(objective, n_trials=3)

        assert len(study1.trials) == 3
        assert len(study2.trials) == 3


# Edge cases and additional tests
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_sampler_with_none_seed(self):
        """Test Sampler with explicitly None seed."""
        sampler = Sampler(sampler=TPESampler, seed=None)
        assert sampler.params["seed"] is None
        sampler.instantiate()
        assert isinstance(sampler.instance_, TPESampler)

    def test_storage_rdb_with_complex_url(self):
        """Test RDBStorage with complex database URL."""
        url = "postgresql://user:password@localhost:5432/optuna"
        storage = Storage(storage=RDBStorage, url=url)
        assert storage.params["url"] == url

    def test_sampler_params_from_get_params(self):
        """Test using params from get_params to create new sampler."""
        sampler1 = Sampler(sampler=TPESampler, seed=42, n_startup_trials=15)
        params = sampler1.get_params()

        # Create new sampler with same params
        sampler2 = Sampler(**params)

        assert sampler2.estimator_class == sampler1.estimator_class
        assert sampler2.params["seed"] == sampler1.params["seed"]
        assert sampler2.params["n_startup_trials"] == sampler1.params["n_startup_trials"]

    def test_storage_params_from_get_params(self):
        """Test using params from get_params to create new storage."""
        storage1 = Storage(storage=InMemoryStorage)
        params = storage1.get_params()

        # Create new storage with same params
        storage2 = Storage(**params)

        assert storage2.estimator_class == storage1.estimator_class

    def test_sampler_repr(self):
        """Test that Sampler has a reasonable string representation."""
        sampler = Sampler(sampler=TPESampler, seed=42)
        repr_str = repr(sampler)

        assert isinstance(repr_str, str)
        assert "Sampler" in repr_str

    def test_storage_repr(self):
        """Test that Storage has a reasonable string representation."""
        storage = Storage(storage=InMemoryStorage)
        repr_str = repr(storage)

        assert isinstance(repr_str, str)
        assert "Storage" in repr_str


# Tests for Callback class
class TestCallback:
    """Tests for the Callback wrapper class."""

    def test_callback_inherits_base_wrapper(self):
        """Test that Callback inherits from BaseClassWrapper."""
        callback = Callback(MaxTrialsCallback, n_trials=10)
        assert isinstance(callback, BaseClassWrapper)
        assert isinstance(callback, BaseEstimator)

    def test_callback_initialization_with_class(self):
        """Test Callback initialization with callback class."""
        callback = Callback(MaxTrialsCallback, n_trials=20)
        assert callback.estimator_class == MaxTrialsCallback
        assert callback.estimator_name == "callback"
        assert callback.params["n_trials"] == 20

    def test_callback_initialization_non_class_raises_error(self):
        """Test that passing non-class to Callback raises TypeError."""
        callback_instance = MaxTrialsCallback(n_trials=10, states=None)
        with pytest.raises(TypeError, match="callback must be a class"):
            Callback(callback_instance)

    def test_callback_initialization_with_instance_raises_error(self):
        """Test that passing an instance instead of class raises TypeError."""
        with pytest.raises(TypeError, match="callback must be a class"):
            Callback("not a class")

    def test_callback_get_params(self):
        """Test get_params returns correct parameters."""
        callback = Callback(MaxTrialsCallback, n_trials=15)
        params = callback.get_params()
        assert params["callback"] == MaxTrialsCallback
        assert params["n_trials"] == 15

    def test_callback_set_params(self):
        """Test set_params modifies parameters."""
        callback = Callback(MaxTrialsCallback, n_trials=10)
        result = callback.set_params(n_trials=25)
        assert result is callback
        assert callback.params["n_trials"] == 25

    def test_callback_instantiate(self):
        """Test instantiate creates callback instance."""
        callback = Callback(MaxTrialsCallback, n_trials=10)
        callback.instantiate()
        assert hasattr(callback, "instance_")
        assert isinstance(callback.instance_, MaxTrialsCallback)

    def test_callback_call_invokes_instance(self, optuna_study):
        """Test __call__ method invokes the callback instance."""
        callback = Callback(MaxTrialsCallback, n_trials=5)
        callback.instantiate()

        # Create a simple trial
        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        # Run one trial
        optuna_study.optimize(objective, n_trials=1)
        trial = optuna_study.trials[0]

        # Call the callback
        callback(optuna_study, trial)
        # Should not raise any errors

    def test_callback_custom_class(self):
        """Test Callback with custom callback class."""

        class CustomCallback:
            def __init__(self, threshold=0.5):
                self.threshold = threshold

            def __call__(self, study, trial):
                pass

        callback = Callback(CustomCallback, threshold=0.8)
        assert callback.estimator_class == CustomCallback
        assert callback.params["threshold"] == 0.8

    def test_callback_integration_with_optuna(self, optuna_study):
        """Test that callback works in Optuna optimization."""
        callback = Callback(MaxTrialsCallback, n_trials=3)
        callback.instantiate()

        def objective(trial):
            return trial.suggest_float("x", 0, 1)

        # Should stop after 3 trials due to callback
        optuna_study.optimize(objective, n_trials=10, callbacks=[callback.instance_])
        assert len(optuna_study.trials) == 3

    def test_callback_repr(self):
        """Test that Callback has a reasonable string representation."""
        callback = Callback(MaxTrialsCallback, n_trials=10)
        repr_str = repr(callback)
        assert isinstance(repr_str, str)
        assert "Callback" in repr_str

    def test_callback_multiple_instantiate(self):
        """Test that calling instantiate multiple times works."""
        callback = Callback(MaxTrialsCallback, n_trials=5)
        callback.instantiate()
        first_instance = callback.instance_
        callback.instantiate()
        second_instance = callback.instance_
        # Should create new instances each time
        assert first_instance is not second_instance
        assert isinstance(first_instance, type(second_instance))
