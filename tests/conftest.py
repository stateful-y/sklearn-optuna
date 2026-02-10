"""Test configuration and fixtures for Sklearn-Optuna."""

import numpy as np
import optuna
import pytest
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs, make_classification, make_regression

from sklearn_optuna.optuna import Callback


# Session-scoped fixtures for expensive data generation
@pytest.fixture(scope="session")
def regression_data():
    """Generate regression dataset for testing."""
    return make_regression(n_samples=100, n_features=10, n_informative=5, random_state=42)


@pytest.fixture(scope="session")
def multiclass_data():
    """Generate multi-class classification dataset for testing."""
    return make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )


@pytest.fixture(scope="session")
def large_dataset():
    """Generate large dataset for performance testing."""
    return make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def simple_blobs_data():
    """Simple blobs dataset for clustering and classification."""
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X, y


# Function-scoped fixtures
@pytest.fixture
def optuna_study():
    """Create a reusable Optuna study."""
    return optuna.create_study(direction="maximize")


# Helper classes for testing
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


class BrokenMultiMetricScorer(BaseEstimator, ClassifierMixin):
    """Classifier that returns multi-metric results with empty test_score_keys.

    Used to test the fallback path in OptunaSearchCV._run_search objective function
    where multi-metric scoring returns results without mean_test_* keys.
    """

    def __init__(self, param=1.0):
        self.param = param

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def score(self, X, y):
        return 0.5


# Parametrized fixtures for samplers
@pytest.fixture(params=[TPESampler, RandomSampler, CmaEsSampler])
def sampler_class(request):
    """Parametrize over different Optuna sampler classes."""
    return request.param


# Parametrized fixtures for callbacks
@pytest.fixture(params=[MaxTrialsCallback])
def callback_class(request):
    """Parametrize over different Optuna callback classes, returning Callback wrappers."""
    callback_cls = request.param
    # Return instantiated Callback wrapper with default params
    if callback_cls == MaxTrialsCallback:
        return Callback(callback_cls, n_trials=10)
    return Callback(callback_cls)
