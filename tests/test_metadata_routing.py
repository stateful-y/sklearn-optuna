"""Tests for metadata routing in OptunaSearchCV."""

import numpy as np
import optuna
import pytest
import sklearn
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    get_routing_for_object,
)

from sklearn_optuna import OptunaSearchCV


# Helper classes for testing metadata routing
class WeightedClassifier(BaseEstimator, ClassifierMixin):
    """A simple classifier that records received metadata."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None, metadata=None):
        """Fit the classifier."""
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        self.metadata_ = metadata
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, metadata=None):
        """Predict using the trained classifier."""
        self.predict_metadata_ = metadata
        # Simple prediction: return majority class
        return np.full(X.shape[0], self.classes_[0])

    def score(self, X, y, sample_weight=None):
        """Score the classifier."""
        self.score_sample_weight_ = sample_weight
        y_pred = self.predict(X)
        if sample_weight is not None:
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        return accuracy_score(y, y_pred)


@pytest.fixture
def classification_data_with_metadata():
    """Create classification dataset with metadata."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    sample_weight = np.random.RandomState(42).rand(100)
    groups = np.repeat(np.arange(10), 10)  # 10 groups of 10 samples
    metadata = np.random.RandomState(42).rand(100)
    return X, y, sample_weight, groups, metadata


@pytest.mark.parametrize("enable_routing", [True, False])
def test_metadata_routing_feature_flag(classification_data_with_metadata, enable_routing):
    """Test that metadata routing can be enabled/disabled."""
    X, y, sample_weight, groups, metadata = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=enable_routing):
        lr = LogisticRegression(max_iter=200)
        if enable_routing:
            lr.set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        if enable_routing:
            # Should work with routing enabled
            search.fit(X, y, sample_weight=sample_weight)
            assert hasattr(search, "best_score_")
        else:
            # With routing disabled, passing metadata as kwargs should still work
            # (legacy behavior)
            search.fit(X, y, sample_weight=sample_weight)
            assert hasattr(search, "best_score_")


def test_sample_weight_routing_to_fit(classification_data_with_metadata):
    """Test that sample_weight is routed to estimator's fit method."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        clf = WeightedClassifier().set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            clf,
            {"alpha": FloatDistribution(0.1, 10.0)},
            n_trials=2,
            cv=2,
        )

        search.fit(X, y, sample_weight=sample_weight)

        # Check that the best estimator received sample_weight
        assert hasattr(search.best_estimator_, "sample_weight_")
        # Sample weight should have been passed during cross-validation
        assert search.best_estimator_.sample_weight_ is not None


def test_sample_weight_routing_to_score(classification_data_with_metadata):
    """Test that sample_weight is routed to scorer."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        # Create scorer that accepts sample_weight
        weighted_scorer = make_scorer(accuracy_score).set_score_request(sample_weight=True)

        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
            scoring=weighted_scorer,
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")


def test_groups_routing_with_group_kfold(classification_data_with_metadata):
    """Test that groups are routed to GroupKFold splitter."""
    X, y, _, groups, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=GroupKFold(n_splits=3),
        )

        # GroupKFold requests groups by default
        search.fit(X, y, groups=groups)
        assert hasattr(search, "best_score_")
        assert hasattr(search, "best_params_")


def test_custom_metadata_routing(classification_data_with_metadata):
    """Test routing of custom metadata parameters."""
    X, y, _, _, metadata = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        clf = WeightedClassifier().set_fit_request(metadata=True)

        search = OptunaSearchCV(
            clf,
            {"alpha": FloatDistribution(0.1, 10.0)},
            n_trials=2,
            cv=2,
        )

        search.fit(X, y, metadata=metadata)

        # Check that the best estimator received metadata
        assert hasattr(search.best_estimator_, "metadata_")
        assert search.best_estimator_.metadata_ is not None


def test_metadata_not_requested_raises_error(classification_data_with_metadata):
    """Test that passing unrequested metadata raises an error."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        # Estimator that doesn't request sample_weight
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=None).set_score_request(sample_weight=None)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        # Should raise error because sample_weight is passed but not requested
        with pytest.raises(UnsetMetadataPassedError, match="are passed but are not explicitly set as requested"):
            search.fit(X, y, sample_weight=sample_weight)


def test_metadata_aliasing(classification_data_with_metadata):
    """Test metadata aliasing (different names for fit and score)."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        # Use different aliases for fit and score
        lr = (
            LogisticRegression(max_iter=200)
            .set_fit_request(sample_weight="fit_weight")
            .set_score_request(sample_weight="score_weight")
        )

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        # Pass both aliases
        search.fit(X, y, fit_weight=sample_weight, score_weight=sample_weight)
        assert hasattr(search, "best_score_")


def test_multi_metric_with_metadata_routing(classification_data_with_metadata):
    """Test metadata routing with multiple scoring metrics."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        # Create multiple scorers with different metadata requirements
        weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
        unweighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=False)

        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        scoring = {
            "weighted_accuracy": weighted_acc,
            "unweighted_accuracy": unweighted_acc,
        }

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
            scoring=scoring,
            refit="weighted_accuracy",
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")
        # cv_results_ keys are prefixed with 'mean_test_' and 'std_test_'
        assert "mean_test_weighted_accuracy" in search.cv_results_
        assert "mean_test_unweighted_accuracy" in search.cv_results_


def test_get_metadata_routing_structure():
    """Test that get_metadata_routing returns correct structure."""
    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression()
        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
        )

        routing = search.get_metadata_routing()

        # Check that routing has the expected structure
        assert isinstance(routing, MetadataRouter)
        assert "estimator" in routing._serialize()
        assert "scorer" in routing._serialize()
        assert "splitter" in routing._serialize()


def test_metadata_routing_with_pipeline(classification_data_with_metadata):
    """Test metadata routing through pipeline."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([
            # StandardScaler doesn't use sample_weight, so explicitly set to False
            ("scaler", StandardScaler().set_fit_request(sample_weight=False)),
            (
                "clf",
                LogisticRegression(max_iter=200)
                .set_fit_request(sample_weight=True)
                .set_score_request(sample_weight=True),
            ),
        ])

        search = OptunaSearchCV(
            pipe,
            {"clf__C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")


def test_score_method_with_metadata(classification_data_with_metadata):
    """Test that score() method correctly forwards metadata."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    # Split data
    X_train, X_test = X[:80], X[20:]
    y_train, y_test = y[:80], y[20:]
    sw_train, sw_test = sample_weight[:80], sample_weight[20:]

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        search.fit(X_train, y_train, sample_weight=sw_train)

        # Score should also accept sample_weight
        score = search.score(X_test, y_test, sample_weight=sw_test)
        assert isinstance(score, int | float)
        assert 0 <= score <= 1


def test_metadata_routing_with_refit_false(classification_data_with_metadata):
    """Test metadata routing when refit=False."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
            refit=False,
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")
        # When refit=False, best_estimator_ should not exist
        assert not hasattr(search, "best_estimator_")


def test_metadata_with_categorical_distribution(classification_data_with_metadata):
    """Test metadata routing with categorical parameter distributions."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            lr,
            {
                "C": FloatDistribution(0.1, 1.0),
                # Use solver that supports different penalties
                "solver": CategoricalDistribution(["lbfgs", "saga"]),
            },
            n_trials=3,
            cv=2,
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")
        assert "solver" in search.best_params_


def test_none_metadata_should_not_raise(classification_data_with_metadata):
    """Test that passing None as metadata when not requested doesn't raise."""
    X, y, _, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200)
        # Don't request sample_weight

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        # Passing None should not raise even though sample_weight is not requested
        search.fit(X, y, sample_weight=None)
        assert hasattr(search, "best_score_")


def test_metadata_routing_request_values():
    """Test that metadata request values can be queried."""
    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression().set_fit_request(sample_weight=True).set_score_request(sample_weight=False)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
        )

        routing = search.get_metadata_routing()
        # Check that estimator routing is included
        estimator_routing = routing._serialize()["estimator"]["router"]
        assert "fit" in estimator_routing
        assert "score" in estimator_routing


def test_weighted_fit_unweighted_score(classification_data_with_metadata):
    """Test weighted fit with unweighted scoring."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        # Request sample_weight for fit but not for score
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=False)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")


def test_metadata_routing_preserves_best_estimator_state(classification_data_with_metadata):
    """Test that best_estimator_ preserves metadata routing configuration."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
        )

        search.fit(X, y, sample_weight=sample_weight)

        # Check that best_estimator_ has metadata routing configured
        best_routing = get_routing_for_object(search.best_estimator_)
        assert best_routing.fit.requests["sample_weight"] is True
        assert best_routing.score.requests["sample_weight"] is True


def test_metadata_routing_with_study_object(classification_data_with_metadata):
    """Test metadata routing when providing a pre-existing study."""
    X, y, sample_weight, _, _ = classification_data_with_metadata

    with sklearn.config_context(enable_metadata_routing=True):
        lr = LogisticRegression(max_iter=200).set_fit_request(sample_weight=True).set_score_request(sample_weight=True)

        # Create a study
        study = optuna.create_study(direction="maximize")

        search = OptunaSearchCV(
            lr,
            {"C": FloatDistribution(0.1, 1.0)},
            n_trials=2,
            cv=2,
            study=study,
        )

        search.fit(X, y, sample_weight=sample_weight)
        assert hasattr(search, "best_score_")
        # Study should have trials
        assert len(study.trials) == 2
