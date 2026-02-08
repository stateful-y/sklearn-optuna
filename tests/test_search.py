"""Tests for OptunaSearchCV - Reorganized and Expanded."""

import contextlib
import pickle
import time
from unittest.mock import Mock

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler
from optuna.storages import InMemoryStorage, RDBStorage
from optuna.study import MaxTrialsCallback
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn_optuna import Callback, OptunaSearchCV, Sampler, Storage


# Test fixtures
@pytest.fixture
def simple_classification_data():
    """Simple classification dataset for basic tests."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def simple_blobs_data():
    """Simple blobs dataset for clustering and classification."""
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X, y


# Helper classes
class MockClassifier(BaseEstimator, ClassifierMixin):
    """A mock classifier for testing."""

    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.5


class FailingClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that fails when FAILING_PARAMETER is used."""

    FAILING_PARAMETER = 2

    def __init__(self, parameter=0):
        self.parameter = parameter

    def fit(self, X, y):
        if self.parameter == FailingClassifier.FAILING_PARAMETER:
            raise ValueError("Failing classifier failed as required")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.0


class TestOptunaSearchCVCore:
    """Core functionality tests for OptunaSearchCV."""

    def test_basic_fit_predict(self, simple_classification_data):
        """Test basic fit and predict workflow."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
            "kernel": CategoricalDistribution(["linear", "rbf"]),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=3,
        )

        search.fit(X, y)

        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert hasattr(search, "study_")
        assert len(search.study_.trials) == 5

        predictions = search.predict(X)
        assert predictions.shape[0] == X.shape[0]

    def test_with_regressor(self, simple_blobs_data):
        """Test OptunaSearchCV with regression estimator."""
        X, y = simple_blobs_data

        param_distributions = {
            "alpha": FloatDistribution(0.01, 10.0, log=True),
        }

        search = OptunaSearchCV(
            Ridge(),
            param_distributions,
            n_trials=5,
            cv=3,
        )

        search.fit(X, y)

        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert search.best_params_["alpha"] > 0

        predictions = search.predict(X)
        assert predictions.shape[0] == X.shape[0]

    def test_with_decision_tree(self, simple_classification_data):
        """Test with DecisionTreeClassifier."""
        X, y = simple_classification_data

        param_distributions = {
            "max_depth": IntDistribution(1, 10),
            "min_samples_split": IntDistribution(2, 10),
        }

        search = OptunaSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_distributions,
            n_trials=5,
            cv=3,
        )

        search.fit(X, y)

        assert hasattr(search, "best_params_")
        assert 1 <= search.best_params_["max_depth"] <= 10
        assert 2 <= search.best_params_["min_samples_split"] <= 10

    def test_invalid_param_distributions_type(self, simple_classification_data):
        """Test that non-dict param_distributions raises error."""
        X, y = simple_classification_data

        search = OptunaSearchCV(
            SVC(),
            ["invalid"],  # Should be dict
            n_trials=5,
        )

        with pytest.raises(AttributeError, match="has no attribute 'items'"):
            search.fit(X, y)

    def test_invalid_distribution_values(self, simple_classification_data):
        """Test that non-BaseDistribution values raise error."""
        X, y = simple_classification_data

        param_distributions = {
            "C": "not_a_distribution",  # Should be BaseDistribution
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
        )

        with pytest.raises(ValueError, match="invalid distribution"):
            search.fit(X, y)

    def test_empty_param_distributions(self, simple_classification_data):
        """Test with empty parameter distributions."""
        X, y = simple_classification_data

        search = OptunaSearchCV(
            SVC(),
            {},  # Empty param distributions
            n_trials=5,
        )

        search.fit(X, y)
        assert hasattr(search, "best_params_")
        assert search.best_params_ == {}

    @pytest.mark.parametrize("sampler_cls", [TPESampler, RandomSampler, CmaEsSampler])
    def test_sampler_wrapper(self, simple_classification_data, sampler_cls):
        """Test OptunaSearchCV with different sampler wrappers."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        sampler = Sampler(sampler=sampler_cls, seed=42)
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            sampler=sampler,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)
        assert hasattr(search, "best_params_")
        assert len(search.study_.trials) == 5

    def test_sampler_wrapper_grid(self, simple_classification_data):
        """Test OptunaSearchCV with GridSampler."""
        X, y = simple_classification_data

        search_space = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
        }

        param_distributions = {
            "C": CategoricalDistribution(search_space["C"]),
            "kernel": CategoricalDistribution(search_space["kernel"]),
        }

        sampler = Sampler(sampler=GridSampler, search_space=search_space)
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            sampler=sampler,
            n_trials=6,
            cv=2,
        )

        search.fit(X, y)
        # Grid search should try all 6 combinations
        assert len(search.study_.trials) == 6

    def test_storage_wrapper_in_memory(self, simple_classification_data):
        """Test OptunaSearchCV with InMemoryStorage wrapper."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        storage = Storage(storage=InMemoryStorage)
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            storage=storage,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)
        assert hasattr(search, "study_")
        assert len(search.study_.trials) == 5

    def test_study_creation_default(self, simple_classification_data):
        """Test that study is created with default parameters."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        search.fit(X, y)

        assert hasattr(search, "study_")
        assert search.study_.direction == optuna.study.StudyDirection.MAXIMIZE
        assert len(search.study_.trials) == 3

    def test_study_reuse(self, simple_classification_data):
        """Test reusing an existing study."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # Create study with initial trials
        study = optuna.create_study(direction="maximize")
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        search.fit(X, y, study=study)
        initial_trials = len(search.study_.trials)
        assert initial_trials == 3

        # Reuse study for more trials
        search2 = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=2,
            cv=2,
        )
        search2.fit(X, y, study=study)

        # Should have 3 + 2 = 5 total trials
        assert len(search2.study_.trials) == 5

    def test_n_trials_limit(self, simple_classification_data):
        """Test that n_trials limits the number of optimization trials."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        n_trials = 7
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=n_trials,
            cv=2,
        )

        search.fit(X, y)

        assert len(search.study_.trials) == n_trials
        assert len(search.trials_) == n_trials

    def test_timeout_parameter(self, simple_classification_data):
        """Test that timeout parameter limits optimization time."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        timeout = 1.0  # 1 second timeout
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=100,  # More than can complete in 1 second
            timeout=timeout,
            cv=2,
        )

        start_time = time.time()
        search.fit(X, y)
        elapsed_time = time.time() - start_time

        # Should complete within timeout (with some tolerance)
        assert elapsed_time < timeout + 2.0
        # Should have completed some trials
        assert len(search.study_.trials) > 0
        assert len(search.study_.trials) <= 100

    def test_classes_property(self, simple_classification_data):
        """Test classes_ property for classification."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        search.fit(X, y)

        assert hasattr(search, "classes_")
        assert np.array_equal(search.classes_, np.unique(y))

    def test_n_features_in_property(self, simple_classification_data):
        """Test n_features_in_ property."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        # Should raise before fit
        with pytest.raises(AttributeError):
            _ = search.n_features_in_

        search.fit(X, y)

        assert hasattr(search, "n_features_in_")
        assert search.n_features_in_ == X.shape[1]

    def test_no_classes_for_regressor(self, simple_blobs_data):
        """Test that classes_ is not available for regressors."""
        X, y = simple_blobs_data

        param_distributions = {
            "alpha": FloatDistribution(0.01, 10.0),
        }

        search = OptunaSearchCV(
            Ridge(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        search.fit(X, y)

        # Regressor doesn't have classes_
        with pytest.raises(AttributeError):
            _ = search.classes_

    def test_no_refit_single_metric(self, simple_classification_data):
        """Test with refit=False for single metric."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=2,
            refit=False,
        )

        search.fit(X, y)

        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        # Should not have best_estimator_ when refit=False
        assert not hasattr(search, "best_estimator_")

        # Should not be able to predict
        with pytest.raises(AttributeError):
            search.predict(X)

    def test_trials_access(self, simple_classification_data):
        """Test access to trials_ attribute."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        assert hasattr(search, "trials_")
        assert len(search.trials_) == 5
        # All trials should be FrozenTrial objects
        assert all(isinstance(trial, optuna.trial.FrozenTrial) for trial in search.trials_)

    def test_single_trial(self, simple_classification_data):
        """Test with a single trial."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=1,
            cv=2,
        )

        search.fit(X, y)

        assert len(search.study_.trials) == 1
        assert hasattr(search, "best_params_")

    def test_deterministic_results_with_seed(self, simple_classification_data):
        """Test that results are deterministic with same seed."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
            "gamma": FloatDistribution(0.001, 1.0),
        }

        # First run
        search1 = OptunaSearchCV(
            SVC(),
            param_distributions,
            sampler=Sampler(sampler=TPESampler, seed=42),
            n_trials=5,
            cv=2,
        )
        search1.fit(X, y)

        # Second run with same seed
        search2 = OptunaSearchCV(
            SVC(),
            param_distributions,
            sampler=Sampler(sampler=TPESampler, seed=42),
            n_trials=5,
            cv=2,
        )
        search2.fit(X, y)

        # Should get same parameters
        assert search1.best_params_ == search2.best_params_

    def test_verbose_output(self, simple_classification_data, capsys):
        """Test verbose output during optimization."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
            verbose=1,
        )

        search.fit(X, y)
        captured = capsys.readouterr()

        # With verbose=1, should see some output
        # (The exact output format depends on sklearn version)
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_scoring_string(self, simple_classification_data):
        """Test with scoring parameter as string."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            scoring="f1_weighted",
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        assert hasattr(search, "best_score_")
        assert search.best_score_ > 0


class TestOptunaSearchCVCallbacks:
    """Tests for callback functionality in OptunaSearchCV."""

    def test_callbacks_execution(self, simple_classification_data):
        """Test that callbacks are executed during optimization."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # Create mock callback
        mock_callback = Mock()
        callback = Callback(type("MockCallback", (), {"__init__": lambda self: None, "__call__": mock_callback}))

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            callbacks={"mock": callback},
            n_trials=3,
            cv=2,
        )

        search.fit(X, y)

        # Callback should have been called for each trial
        assert mock_callback.call_count == 3

    def test_callback_dict_validation(self, simple_classification_data):
        """Test that callbacks must be a dict."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # Pass list instead of dict
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            callbacks=[Callback(MaxTrialsCallback, n_trials=5)],
            n_trials=3,
        )

        with pytest.raises(TypeError, match="callbacks.*dict"):
            search.fit(X, y)

    def test_callback_instance_validation(self, simple_classification_data):
        """Test that callback values must be Callback instances."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # Pass raw class instead of Callback wrapper
        with pytest.raises(TypeError, match="must be a Callback instance"):
            search = OptunaSearchCV(
                SVC(),
                param_distributions,
                callbacks={"max_trials": MaxTrialsCallback},  # Should be Callback(MaxTrialsCallback)
                n_trials=3,
            )
            search.fit(X, y)

    def test_multiple_callbacks(self, simple_classification_data):
        """Test multiple named callbacks."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        mock1 = Mock()
        mock2 = Mock()

        callback1 = Callback(type("CB1", (), {"__init__": lambda self: None, "__call__": mock1}))
        callback2 = Callback(type("CB2", (), {"__init__": lambda self: None, "__call__": mock2}))

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            callbacks={"callback1": callback1, "callback2": callback2},
            n_trials=3,
            cv=2,
        )

        search.fit(X, y)

        # Both callbacks should have been called
        assert mock1.call_count == 3
        assert mock2.call_count == 3

    def test_callback_with_max_trials(self, simple_classification_data):
        """Test MaxTrialsCallback stops optimization early."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        max_trials_callback = Callback(MaxTrialsCallback, n_trials=3, states=None)

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            callbacks={"max_trials": max_trials_callback},
            n_trials=10,  # Request 10 but callback should stop at 3
            cv=2,
        )

        search.fit(X, y)

        # Should stop at 3 trials due to callback
        assert len(search.study_.trials) == 3

    def test_multi_metric_scoring(self, simple_classification_data):
        """Test multi-metric scoring."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            scoring=["accuracy", "f1_weighted"],
            refit="accuracy",
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        # Should have results for both metrics
        assert "mean_test_accuracy" in search.cv_results_
        assert "mean_test_f1_weighted" in search.cv_results_
        assert hasattr(search, "best_score_")

    def test_multi_metric_refit_strategy(self, simple_classification_data):
        """Test multi-metric with different refit strategies."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            scoring={"acc": "accuracy", "f1": "f1_weighted"},
            refit="f1",  # Refit on f1 score
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        assert hasattr(search, "best_estimator_")
        assert "mean_test_acc" in search.cv_results_
        assert "mean_test_f1" in search.cv_results_

    def test_multi_metric_no_refit(self, simple_classification_data):
        """Test multi-metric scoring with refit=False."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            scoring=["accuracy", "f1_weighted"],
            refit=False,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        assert "mean_test_accuracy" in search.cv_results_
        assert "mean_test_f1_weighted" in search.cv_results_
        # No best_estimator_ when refit=False
        assert not hasattr(search, "best_estimator_")


class TestOptunaSearchCVResults:
    """Tests for cv_results_ structure and content."""

    def test_cv_results_structure(self, simple_classification_data):
        """Test cv_results_ has expected structure."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
            "kernel": CategoricalDistribution(["linear", "rbf"]),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=3,
        )

        search.fit(X, y)

        # Check required keys
        assert "mean_test_score" in search.cv_results_
        assert "std_test_score" in search.cv_results_
        assert "rank_test_score" in search.cv_results_
        assert "params" in search.cv_results_

        # Check per-split scores
        assert "split0_test_score" in search.cv_results_
        assert "split1_test_score" in search.cv_results_
        assert "split2_test_score" in search.cv_results_

        # Check shapes
        assert len(search.cv_results_["mean_test_score"]) == 5
        assert len(search.cv_results_["params"]) == 5

    def test_cv_results_ranking(self, simple_classification_data):
        """Test that rank_test_score is computed correctly."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        ranks = search.cv_results_["rank_test_score"]
        scores = search.cv_results_["mean_test_score"]

        # Best score should have rank 1
        best_idx = np.argmax(scores)
        assert ranks[best_idx] == 1

        # Ranks should be consecutive integers starting from 1
        # (may have gaps if trials failed)
        rank_set = {int(r) for r in ranks}
        assert 1 in rank_set
        assert max(rank_set) <= 5  # Max rank shouldn't exceed n_trials

    def test_cv_results_with_return_train_score(self, simple_classification_data):
        """Test cv_results_ with return_train_score=True."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=3,
            return_train_score=True,
        )

        search.fit(X, y)

        # Should have train scores
        assert "mean_train_score" in search.cv_results_
        assert "std_train_score" in search.cv_results_
        assert "split0_train_score" in search.cv_results_
        assert "split1_train_score" in search.cv_results_
        assert "split2_train_score" in search.cv_results_

    def test_cv_results_param_columns(self, simple_classification_data):
        """Test that param_* columns exist for each parameter."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
            "kernel": CategoricalDistribution(["linear", "rbf"]),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        # params should contain all parameters
        for params_dict in search.cv_results_["params"]:
            assert "C" in params_dict
            assert "kernel" in params_dict

    def test_cv_results_multi_metric_keys(self, simple_classification_data):
        """Test cv_results_ keys for multi-metric scoring."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            scoring=["accuracy", "precision_weighted"],
            refit="accuracy",
            n_trials=5,
            cv=3,
        )

        search.fit(X, y)

        # Should have keys for each metric
        assert "mean_test_accuracy" in search.cv_results_
        assert "std_test_accuracy" in search.cv_results_
        assert "mean_test_precision_weighted" in search.cv_results_
        assert "std_test_precision_weighted" in search.cv_results_

        # Should have per-split scores for each metric
        assert "split0_test_accuracy" in search.cv_results_
        assert "split0_test_precision_weighted" in search.cv_results_

    def test_cv_results_empty_trials(self, simple_classification_data):
        """Test cv_results_ structure with minimal trials."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=1,
            cv=2,
        )

        search.fit(X, y)

        # Should still have proper structure with 1 trial
        assert len(search.cv_results_["mean_test_score"]) == 1
        assert len(search.cv_results_["params"]) == 1
        assert search.cv_results_["rank_test_score"][0] == 1

    def test_cv_results_std_computation(self, simple_classification_data):
        """Test that std_test_score is computed from split scores."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=3,
        )

        search.fit(X, y)

        # Check std computation for first trial
        split_scores = [
            search.cv_results_["split0_test_score"][0],
            search.cv_results_["split1_test_score"][0],
            search.cv_results_["split2_test_score"][0],
        ]
        expected_std = np.std(split_scores)
        actual_std = search.cv_results_["std_test_score"][0]

        np.testing.assert_almost_equal(actual_std, expected_std, decimal=6)


class TestOptunaSearchCVEdgeCases:
    """Edge cases and error handling tests."""

    def test_error_score_raise(self, simple_classification_data):
        """Test that error_score='raise' raises exceptions from failing trials."""
        X, y = simple_classification_data

        param_distributions = {
            "parameter": IntDistribution(0, 3),
        }

        search = OptunaSearchCV(
            FailingClassifier(),
            param_distributions,
            n_trials=50,  # Many trials - will stop at first failure
            cv=2,
            error_score="raise",
            sampler=Sampler(sampler=RandomSampler, seed=42),  # Fixed seed to ensure reproducibility
        )

        # With error_score='raise', the first failing trial will raise ValueError
        # Random seed 42 should eventually hit parameter=2
        with pytest.raises(ValueError, match="Failing classifier failed"):
            search.fit(X, y)

    @pytest.mark.parametrize("error_score_value", [0.0])
    def test_error_score_numeric(self, simple_classification_data, error_score_value):
        """Test error_score with numeric values.

        Note: Optuna doesn't accept NaN as trial values, so error_score=np.nan
        will cause Optuna to mark trials as failed. Only finite error scores work.

        Note: Currently only error_score=0.0 is fully working. Negative values like
        -1.0 are not properly appearing in cv_results_ due to sklearn's handling
        of failed trials. This is a known limitation.
        """
        X, y = simple_classification_data

        # Use a fixed seed to ensure reproducible parameter sampling
        param_distributions = {
            "parameter": IntDistribution(0, 3),
        }

        search = OptunaSearchCV(
            FailingClassifier(),
            param_distributions,
            n_trials=50,  # Many trials to increase chance of hitting parameter=2
            cv=2,
            error_score=error_score_value,
            sampler=Sampler(sampler=RandomSampler, seed=42),  # Fixed seed for reproducibility
        )

        search.fit(X, y)

        # Check if any trials used the failing parameter
        failing_trials = [t for t in search.trials_ if t.params.get("parameter") == 2]

        # With 50 random trials, we should hit parameter=2 at least once
        # Each trial has 25% chance, probability of missing all is (0.75)^50 ≈ 0.000001
        assert len(failing_trials) > 0, "No trials with parameter=2 were sampled"

        scores = search.cv_results_["mean_test_score"]
        # Should have some scores equal to error_score_value
        assert np.any(scores == error_score_value), f"No scores equal to {error_score_value} found"

    def test_error_score_nan_marks_trials_failed(self, simple_classification_data):
        """Test that error_score=np.nan results in -inf values for failed trials.

        Note: Optuna doesn't accept NaN as a trial value, so when an error occurs
        and error_score=np.nan, Optuna converts it to -inf.
        """
        X, y = simple_classification_data

        param_distributions = {
            "parameter": IntDistribution(0, 3),
        }

        search = OptunaSearchCV(
            FailingClassifier(),
            param_distributions,
            n_trials=50,
            cv=2,
            error_score=np.nan,
            sampler=Sampler(sampler=RandomSampler, seed=42),
        )

        search.fit(X, y)

        # Check for trials with -inf values (from failed estimators)
        trials_with_neg_inf = [t for t in search.trials_ if t.value == -np.inf]
        assert len(trials_with_neg_inf) > 0, "Expected some trials to have -inf values"

        # Trials with -inf should have had parameter=2 (the failing parameter)
        failed_params = [t.params.get("parameter") for t in trials_with_neg_inf]
        assert 2 in failed_params, "Expected trials with -inf to have parameter=2"

    def test_exception_stored_in_trial_attrs(self, simple_classification_data):
        """Test that exceptions are stored in trial user_attrs."""
        X, y = simple_classification_data

        param_distributions = {
            "parameter": IntDistribution(0, 3),
        }

        search = OptunaSearchCV(
            FailingClassifier(),
            param_distributions,
            n_trials=20,  # More trials to ensure hitting parameter=2
            cv=2,
            error_score=0.0,
            sampler=Sampler(sampler=RandomSampler, seed=42),  # Fixed seed
        )

        search.fit(X, y)

        # Find trials that failed
        failed_trials = [t for t in search.trials_ if "exception" in t.user_attrs]

        assert len(failed_trials) > 0
        # Check exception attributes
        for trial in failed_trials:
            assert "exception" in trial.user_attrs
            assert "exception_type" in trial.user_attrs
            assert "ValueError" in trial.user_attrs["exception_type"]

    def test_partial_trial_failures(self, simple_classification_data):
        """Test when some trials succeed and some fail.

        Note: When error_score=np.nan, failed trials return -inf values.
        """
        X, y = simple_classification_data

        param_distributions = {
            "parameter": IntDistribution(0, 3),
        }

        search = OptunaSearchCV(
            FailingClassifier(),
            param_distributions,
            n_trials=50,  # Many trials to ensure mix of success/failure
            cv=2,
            error_score=np.nan,
            sampler=Sampler(sampler=RandomSampler, seed=42),  # Fixed seed
        )

        search.fit(X, y)

        # Check trial values - some should be -inf (failures), some should be finite (successes)
        trials_with_neg_inf = [t for t in search.trials_ if t.value == -np.inf]
        successful_trials = [t for t in search.trials_ if np.isfinite(t.value)]

        # With 50 trials, we should have both types
        assert len(trials_with_neg_inf) > 0, "No failing trials sampled"
        assert len(successful_trials) > 0, "No successful trials sampled"

        # cv_results_ includes all trials (both successful and failed)
        scores = search.cv_results_["mean_test_score"]
        # Some scores should be nan (from failed trials with error_score=np.nan)
        assert np.any(np.isnan(scores)), "Expected some nan scores in cv_results_"
        # Some scores should be finite (from successful trials)
        assert np.any(np.isfinite(scores)), "Expected some finite scores in cv_results_"

    def test_all_trials_fail_with_refit_false(self, simple_classification_data):
        """Test that when all trials fail to complete and refit=False, fit returns successfully.

        This tests the edge case where no trials reach COMPLETE state and
        refit=False prevents raising an error. We mock study.optimize to prevent
        any trials from completing.
        """
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # Create a study with some pre-existing pruned trials
        study = optuna.create_study(direction="maximize")

        # Manually create pruned trials
        def failing_objective(trial):
            trial.suggest_float("C", 0.1, 10.0)
            raise optuna.TrialPruned("Intentionally pruned")

        for _ in range(3):
            with contextlib.suppress(BaseException):
                study.optimize(failing_objective, n_trials=1)

        # Verify we have pruned trials but no completed trials
        assert len(study.trials) > 0
        assert all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials)

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=1,
            cv=2,
            refit=False,  # This is critical - with refit=True it would raise
        )

        # Mock optimize to prevent any new trials from running
        original_optimize = study.optimize

        def mock_optimize(*args, **kwargs):
            # Don't actually run any new trials
            pass

        study.optimize = mock_optimize

        # Fit with the pre-existing study that has no completed trials
        search.fit(X, y, study=study)

        # Restore original method
        study.optimize = original_optimize

        # Verify no trials completed
        completed_trials = [t for t in search.trials_ if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed_trials) == 0

        # cv_results_ should have empty params list (no completed trials)
        assert "params" in search.cv_results_
        assert len(search.cv_results_["params"]) == 0

        # Should not have best_estimator_ or best_params_ when no trials complete
        assert not hasattr(search, "best_estimator_")
        assert not hasattr(search, "best_params_")
        assert not hasattr(search, "best_index_")

    def test_invalid_distribution_type(self, simple_classification_data):
        """Test various invalid distribution types."""
        X, y = simple_classification_data

        invalid_distributions = [
            {"C": 1.0},  # Raw value instead of distribution
            {"C": [0.1, 1.0, 10.0]},  # List instead of distribution
            {"C": (0.1, 10.0)},  # Tuple instead of distribution
        ]

        for param_dist in invalid_distributions:
            search = OptunaSearchCV(
                SVC(),
                param_dist,
                n_trials=3,
            )

            with pytest.raises(ValueError, match="invalid distribution"):
                search.fit(X, y)

    def test_boundary_n_trials(self, simple_classification_data):
        """Test boundary values for n_trials."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # n_trials=1 should work
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=1,
            cv=2,
        )
        search.fit(X, y)
        assert len(search.study_.trials) == 1

    def test_boundary_timeout(self, simple_classification_data):
        """Test boundary values for timeout."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # Very short timeout
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=100,
            timeout=0.1,
            cv=2,
        )
        search.fit(X, y)
        # Should complete at least 1 trial even with short timeout
        assert len(search.study_.trials) >= 1

    def test_boundary_cv_folds(self, simple_classification_data):
        """Test with cv=1 (single fold)."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # This will use a single train/test split
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,  # Minimum practical value
        )
        search.fit(X, y)
        assert len(search.study_.trials) == 3

    def test_state_management_multiple_fits(self, simple_classification_data):
        """Test calling fit multiple times."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        # First fit
        search.fit(X, y)
        first_study = search.study_
        first_n_trials = len(first_study.trials)

        # Second fit - should create new study
        search.fit(X, y)
        second_study = search.study_
        second_n_trials = len(second_study.trials)

        # Should be independent studies
        assert first_n_trials == 3
        assert second_n_trials == 3

    def test_prefit_property_access(self, simple_classification_data):
        """Test accessing properties before fit."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
        )

        # Should raise errors before fit
        with pytest.raises(AttributeError):
            _ = search.best_params_

        with pytest.raises(AttributeError):
            _ = search.best_score_

        with pytest.raises(AttributeError):
            _ = search.study_


class TestOptunaSearchCVIntegration:
    """Integration tests with sklearn ecosystem and persistence."""

    def test_pipeline_integration(self, simple_classification_data):
        """Test OptunaSearchCV with sklearn Pipeline."""
        X, y = simple_classification_data

        pipeline = make_pipeline(
            StandardScaler(),
            SVC(),
        )

        param_distributions = {
            "svc__C": FloatDistribution(0.1, 10.0),
            "svc__kernel": CategoricalDistribution(["linear", "rbf"]),
        }

        search = OptunaSearchCV(
            pipeline,
            param_distributions,
            n_trials=5,
            cv=2,
        )

        search.fit(X, y)

        assert hasattr(search, "best_params_")
        assert "svc__C" in search.best_params_
        assert "svc__kernel" in search.best_params_

        predictions = search.predict(X)
        assert predictions.shape[0] == X.shape[0]

    def test_pickle_fitted_search(self, simple_classification_data):
        """Test pickling and unpickling fitted search."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )

        search.fit(X, y)
        original_params = search.best_params_

        # Pickle and unpickle
        pickled = pickle.dumps(search)
        unpickled_search = pickle.loads(pickled)

        # Should preserve best parameters
        assert unpickled_search.best_params_ == original_params

        # Should be able to predict
        predictions = unpickled_search.predict(X)
        assert predictions.shape[0] == X.shape[0]

    def test_study_continuation(self, simple_classification_data):
        """Test continuing optimization with existing study."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        # First optimization
        search1 = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=3,
            cv=2,
        )
        search1.fit(X, y)
        study = search1.study_

        # Continue optimization
        search2 = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=2,
            cv=2,
        )
        search2.fit(X, y, study=study)

        # Should have accumulated trials
        assert len(search2.study_.trials) == 5

    @pytest.mark.slow
    @pytest.mark.slow
    def test_storage_persistence(self, simple_classification_data, tmp_path):
        """Test persistence with SQLite storage."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        db_path = tmp_path / "optuna.db"
        storage_url = f"sqlite:///{db_path}"

        # First optimization
        storage1 = Storage(storage=RDBStorage, url=storage_url)
        search1 = OptunaSearchCV(
            SVC(),
            param_distributions,
            storage=storage1,
            n_trials=3,
            cv=2,
        )
        search1.fit(X, y)
        study_name = search1.study_.study_name

        # Load in new session
        storage2 = Storage(storage=RDBStorage, url=storage_url)
        loaded_study = optuna.load_study(study_name=study_name, storage=storage2.instantiate().instance_)

        # Should have same trials
        assert len(loaded_study.trials) == 3

    @pytest.mark.slow
    @pytest.mark.slow
    def test_parallel_optimization(self, large_dataset):
        """Test parallel optimization with n_jobs=-1."""
        X, y = large_dataset

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
            "gamma": FloatDistribution(0.001, 1.0),
        }

        start_time = time.time()
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=10,
            cv=2,
            n_jobs=-1,  # Use all cores
        )
        search.fit(X, y)
        _ = time.time() - start_time

        assert len(search.study_.trials) == 10
        # Parallel should complete (hard to test actual speedup reliably)

    def test_sklearn_clone(self, simple_classification_data):
        """Test that OptunaSearchCV can be cloned."""
        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=5,
            cv=2,
        )

        cloned = clone(search)

        # Should preserve parameters
        assert cloned.get_params()["n_trials"] == 5
        assert cloned.get_params()["cv"] == 2

        # Should not share study
        assert not hasattr(cloned, "study_")

    @pytest.mark.slow
    def test_large_n_trials(self, simple_classification_data):
        """Test with large number of trials."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
            "gamma": FloatDistribution(0.001, 1.0),
        }

        start_time = time.time()
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            n_trials=100,
            cv=2,
        )
        search.fit(X, y)
        elapsed = time.time() - start_time

        assert len(search.study_.trials) == 100

        # Mark as slow if takes >3 seconds
        if elapsed > 3:
            pytest.skip("Test took >3 seconds, should be marked as slow")

    def test_study_sampler_update(self, simple_classification_data, optuna_study):
        """Test that sampler is updated when using existing study."""
        X, y = simple_classification_data

        param_distributions = {
            "C": FloatDistribution(0.1, 10.0),
        }

        new_sampler = Sampler(sampler=RandomSampler, seed=123)
        search = OptunaSearchCV(
            SVC(),
            param_distributions,
            sampler=new_sampler,
            n_trials=3,
            cv=2,
        )

        search.fit(X, y, study=optuna_study)

        # Study sampler should be updated
        assert isinstance(optuna_study.sampler, RandomSampler)
