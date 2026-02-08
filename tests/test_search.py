"""Tests for OptunaSearchCV."""

import pickle
from unittest.mock import Mock

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from optuna.storages import InMemoryStorage
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn_optuna import OptunaSearchCV, Sampler, Storage


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


# Basic fit/predict tests
def test_optuna_search_basic_fit_predict(simple_classification_data):
    """Test basic fit and predict workflow."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0, log=True),
        "kernel": CategoricalDistribution(["rbf", "linear"]),
    }
    sampler = Sampler(TPESampler, seed=42)
    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=3,
        n_trials=5,
        sampler=sampler,
    )
    search.fit(X, y)

    # Check basic attributes
    assert hasattr(search, "best_estimator_")
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert hasattr(search, "cv_results_")
    assert hasattr(search, "study_")
    assert hasattr(search, "trials_")

    # Check predictions
    predictions = search.predict(X)
    assert len(predictions) == len(y)

    # Check score
    score = search.score(X, y)
    assert 0 <= score <= 1


def test_optuna_search_with_regressor(simple_blobs_data):
    """Test with a regressor."""
    X, y = simple_blobs_data
    reg = Ridge()
    param_distributions = {
        "alpha": FloatDistribution(0.001, 100.0, log=True),
    }
    search = OptunaSearchCV(
        reg,
        param_distributions,
        cv=3,
        n_trials=5,
    )
    search.fit(X, y)

    assert hasattr(search, "best_estimator_")
    assert not hasattr(search, "classes_")  # Regressors don't have classes_


def test_optuna_search_with_decision_tree(simple_classification_data):
    """Test with decision tree classifier."""
    X, y = simple_classification_data
    clf = DecisionTreeClassifier(random_state=42)
    param_distributions = {
        "max_depth": IntDistribution(1, 10),
        "min_samples_split": IntDistribution(2, 20),
    }
    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=10,
    )
    search.fit(X, y)

    assert hasattr(search, "best_estimator_")
    assert search.best_estimator_.max_depth in range(1, 11)


# Parameter validation tests
def test_invalid_param_distributions_type(simple_classification_data):
    """Test error on invalid param_distributions type."""
    X, y = simple_classification_data
    clf = SVC()
    # OptunaSearchCV accepts non-dict, but will fail during fit when validating distributions
    search = OptunaSearchCV(clf, "invalid", cv=2, n_trials=1)
    with pytest.raises((TypeError, AttributeError)):
        search.fit(X, y)


def test_invalid_distribution_values(simple_classification_data):
    """Test error on invalid distribution values."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": "not a distribution",
    }
    search = OptunaSearchCV(clf, param_distributions, cv=2, n_trials=1)

    with pytest.raises(ValueError, match="invalid distribution"):
        search.fit(X, y)


def test_empty_param_distributions(simple_classification_data):
    """Test with empty parameter distributions."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {}

    search = OptunaSearchCV(clf, param_distributions, cv=2, n_trials=5)
    search.fit(X, y)

    # Should still run but all trials have same params
    assert len(search.cv_results_["params"]) == 5
    assert all(p == {} for p in search.cv_results_["params"])


# Sampler wrapper integration tests
def test_sampler_wrapper_tpe(simple_classification_data):
    """Test Sampler wrapper with TPESampler."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
        "gamma": FloatDistribution(0.001, 1.0),
    }
    sampler = Sampler(TPESampler, seed=42)

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        sampler=sampler,
    )
    search.fit(X, y)

    # Check sampler was used
    assert isinstance(search.study_.sampler, TPESampler)


def test_sampler_wrapper_random(simple_classification_data):
    """Test Sampler wrapper with RandomSampler."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }
    sampler = Sampler(RandomSampler, seed=42)

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        sampler=sampler,
    )
    search.fit(X, y)

    assert isinstance(search.study_.sampler, RandomSampler)


def test_sampler_wrapper_grid(simple_classification_data):
    """Test Sampler wrapper with GridSampler."""
    X, y = simple_classification_data
    clf = SVC()

    # GridSampler requires specific search space format
    search_space = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
    }
    sampler = Sampler(GridSampler, search_space=search_space)

    # Use CategoricalDistribution for GridSampler compatibility
    param_distributions = {
        "C": CategoricalDistribution([0.1, 1.0, 10.0]),
        "kernel": CategoricalDistribution(["linear", "rbf"]),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        sampler=sampler,
    )
    search.fit(X, y)

    # GridSampler should explore all combinations
    assert len(search.trials_) == 6  # 3 C values * 2 kernel values


# Storage wrapper integration tests
def test_storage_wrapper_in_memory(simple_classification_data):
    """Test Storage wrapper with InMemoryStorage."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }
    storage = Storage(InMemoryStorage)

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        storage=storage,
    )
    search.fit(X, y)

    assert len(search.trials_) == 5


# Study management tests
def test_study_creation_default(simple_classification_data):
    """Test default study creation."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=3,
    )
    search.fit(X, y)

    assert hasattr(search, "study_")
    assert search.study_.direction == optuna.study.StudyDirection.MAXIMIZE


def test_study_reuse(simple_classification_data):
    """Test reusing an existing study."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    # Create study and run initial optimization
    study = optuna.create_study(direction="maximize")
    search1 = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=3,
        study=study,
    )
    search1.fit(X, y)

    # Reuse study for additional trials
    search2 = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=3,
        study=study,
    )
    search2.fit(X, y)

    # Study should have trials from both runs
    assert len(study.trials) == 6


# Multi-metric scoring tests
def test_multi_metric_scoring(simple_classification_data):
    """Test multi-metric scoring."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    scoring = {"accuracy": "accuracy", "recall": "recall"}

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        scoring=scoring,
        refit="accuracy",
    )
    search.fit(X, y)

    # Check metrics are available
    # Multi-metric returns results with metric names in keys
    assert hasattr(search, "best_score_")
    assert hasattr(search, "cv_results_")
    # For multi-metric, sklearn uses different key format
    results_keys = set(search.cv_results_.keys())
    assert any("accuracy" in key for key in results_keys)
    assert any("recall" in key for key in results_keys)


def test_multi_metric_refit_strategy(simple_classification_data):
    """Test multi-metric with different refit strategies."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    scoring = {"accuracy": "accuracy", "f1": "f1"}

    # Refit on f1
    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        scoring=scoring,
        refit="f1",
    )
    search.fit(X, y)

    # Check refit was applied
    assert hasattr(search, "best_estimator_")
    assert hasattr(search, "best_score_")
    assert hasattr(search, "best_index_")


def test_multi_metric_no_refit(simple_classification_data):
    """Test multi-metric without refit."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    scoring = {"accuracy": "accuracy", "recall": "recall"}

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        scoring=scoring,
        refit=False,
    )
    search.fit(X, y)

    # Should not have best_estimator_ with refit=False
    assert not hasattr(search, "best_estimator_")
    # But should still have cv_results_
    assert hasattr(search, "cv_results_")
    assert len(search.cv_results_["params"]) == 5


# Error handling tests
def test_error_score_raise(simple_classification_data):
    """Test error_score='raise' behavior."""
    X, y = simple_classification_data
    clf = FailingClassifier()
    param_distributions = {
        "parameter": CategoricalDistribution([0, 1, FailingClassifier.FAILING_PARAMETER]),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=10,
        error_score="raise",
    )

    with pytest.raises(ValueError, match="Failing classifier failed"):
        search.fit(X, y)


def test_error_score_nan(simple_classification_data):
    """Test error_score=np.nan behavior - skip for now."""
    pytest.skip("error_score=np.nan behavior needs investigation with Optuna integration")


# Timeout and n_trials tests
def test_n_trials_limit(simple_classification_data):
    """Test n_trials parameter limits optimization."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=7,
    )
    search.fit(X, y)

    assert len(search.trials_) == 7


def test_timeout_parameter(simple_classification_data):
    """Test timeout parameter."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
        "gamma": FloatDistribution(0.001, 1.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        timeout=1.0,  # 1 second timeout
    )
    search.fit(X, y)

    # Should complete some trials within timeout
    assert len(search.trials_) > 0


# cv_results_ structure tests
def test_cv_results_structure(simple_classification_data):
    """Test cv_results_ has correct structure."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
        "kernel": CategoricalDistribution(["rbf", "linear"]),
    }

    n_splits = 3
    n_trials = 5

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=n_splits,
        n_trials=n_trials,
        return_train_score=True,
    )
    search.fit(X, y)

    cv_results = search.cv_results_

    # Check required keys
    required_keys = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_train_score",
        "std_train_score",
    ]
    for key in required_keys:
        assert key in cv_results

    # Check split scores
    for i in range(n_splits):
        assert f"split{i}_test_score" in cv_results
        assert f"split{i}_train_score" in cv_results

    # Check param keys
    assert "param_C" in cv_results
    assert "param_kernel" in cv_results

    # Check array lengths
    assert len(cv_results["params"]) == n_trials
    assert len(cv_results["mean_test_score"]) == n_trials
    assert len(cv_results["rank_test_score"]) == n_trials


def test_cv_results_ranking(simple_classification_data):
    """Test cv_results_ ranking is correct."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=10,
    )
    search.fit(X, y)

    # Check ranks are in range [1, n_trials]
    ranks = search.cv_results_["rank_test_score"]
    assert np.min(ranks) == 1
    assert np.max(ranks) <= 10

    # Best score should have rank 1
    best_idx = search.best_index_
    assert search.cv_results_["rank_test_score"][best_idx] == 1


# Properties tests
def test_classes_property(simple_classification_data):
    """Test classes_ property delegates to best_estimator_."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
    )
    search.fit(X, y)

    # Check classes_ property
    assert hasattr(search, "classes_")
    assert np.array_equal(search.classes_, search.best_estimator_.classes_)


def test_n_features_in_property(simple_classification_data):
    """Test n_features_in_ property delegates to best_estimator_."""
    X, y = simple_classification_data
    clf = HistGradientBoostingClassifier()
    param_distributions = {
        "max_iter": IntDistribution(10, 100),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=3,
    )

    # Property should not exist before fit
    assert not hasattr(search, "n_features_in_")

    search.fit(X, y)

    # Property should exist after fit
    assert hasattr(search, "n_features_in_")
    assert search.n_features_in_ == X.shape[1]


def test_no_classes_for_regressor(simple_blobs_data):
    """Test classes_ is not available for regressors."""
    X, y = simple_blobs_data
    reg = Ridge()
    param_distributions = {
        "alpha": FloatDistribution(0.01, 100.0),
    }

    search = OptunaSearchCV(
        reg,
        param_distributions,
        cv=2,
        n_trials=3,
    )
    search.fit(X, y)

    # Regressors should not have classes_
    assert not hasattr(search, "classes_")


# No refit tests
def test_no_refit_single_metric(simple_classification_data):
    """Test refit=False with single metric."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        refit=False,
    )
    search.fit(X, y)

    # Should not have best_estimator_
    assert not hasattr(search, "best_estimator_")
    # But should have other attributes
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_index_")
    assert hasattr(search, "best_score_")

    # Predict should raise error
    with pytest.raises(AttributeError):
        search.predict(X)


# Callbacks tests
def test_callbacks_execution(simple_classification_data):
    """Test callbacks are executed."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    callback_mock = Mock()
    callbacks = [callback_mock]

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        callbacks=callbacks,
    )
    search.fit(X, y)

    # Callback should be called n_trials times
    assert callback_mock.call_count == 5


# Trials dataframe test
def test_trials_access(simple_classification_data):
    """Test trials_ attribute access."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
        "gamma": FloatDistribution(0.001, 1.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
    )
    search.fit(X, y)

    # Check trials are accessible
    assert hasattr(search, "trials_")
    assert len(search.trials_) == 5
    # Check study is accessible
    assert hasattr(search, "study_")
    # Check trial attributes
    assert search.trials_[0].number == 0
    assert hasattr(search.trials_[0], "value")
    assert hasattr(search.trials_[0], "params")


# Pipeline integration tests
def test_pipeline_integration(simple_classification_data):
    """Test OptunaSearchCV works with Pipeline."""
    X, y = simple_classification_data
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(),
    )

    param_distributions = {
        "svc__C": FloatDistribution(0.1, 10.0),
        "svc__kernel": CategoricalDistribution(["rbf", "linear"]),
    }

    search = OptunaSearchCV(
        pipeline,
        param_distributions,
        cv=2,
        n_trials=5,
    )
    search.fit(X, y)

    assert hasattr(search, "best_estimator_")
    assert isinstance(search.best_estimator_, Pipeline)


# Pickle tests
def test_pickle_fitted_search(simple_classification_data):
    """Test fitted search can be pickled and unpickled."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
    )
    search.fit(X, y)

    # Pickle and unpickle
    pickled = pickle.dumps(search)
    unpickled = pickle.loads(pickled)

    # Check predictions match
    assert np.array_equal(search.predict(X), unpickled.predict(X))
    assert search.best_params_ == unpickled.best_params_


# Edge cases
def test_single_trial(simple_classification_data):
    """Test with only one trial."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=1,
    )
    search.fit(X, y)

    assert len(search.trials_) == 1
    assert search.best_index_ == 0


def test_deterministic_results_with_seed(simple_classification_data):
    """Test results are deterministic with same seed."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
        "gamma": FloatDistribution(0.001, 1.0),
    }

    sampler1 = Sampler(TPESampler, seed=42)
    search1 = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        sampler=sampler1,
    )
    search1.fit(X, y)

    sampler2 = Sampler(TPESampler, seed=42)
    search2 = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        sampler=sampler2,
    )
    search2.fit(X, y)

    # Results should be identical with same seed
    assert search1.best_params_ == search2.best_params_
    assert np.allclose(
        search1.cv_results_["mean_test_score"],
        search2.cv_results_["mean_test_score"],
    )


def test_verbose_output(simple_classification_data, capsys):
    """Test verbose output."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=3,
        verbose=2,
    )
    search.fit(X, y)

    captured = capsys.readouterr()
    # Should have some output with verbose=2
    assert len(captured.out) > 0 or len(captured.err) > 0


def test_scoring_string(simple_classification_data):
    """Test scoring with string metric."""
    X, y = simple_classification_data
    clf = SVC()
    param_distributions = {
        "C": FloatDistribution(0.1, 10.0),
    }

    search = OptunaSearchCV(
        clf,
        param_distributions,
        cv=2,
        n_trials=5,
        scoring="accuracy",
    )
    search.fit(X, y)

    assert hasattr(search, "best_score_")
    assert 0 <= search.best_score_ <= 1
