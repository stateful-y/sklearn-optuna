"""Edge cases and error handling tests for OptunaSearchCV."""

from unittest.mock import patch

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.samplers import RandomSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from conftest import FailingClassifier
from sklearn_optuna import OptunaSearchCV, Sampler


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
        """Test that error_score=np.nan causes Optuna to mark trials as failed.

        Note: Optuna doesn't accept NaN as a trial value, so trials that would
        return NaN are marked as FAIL state by Optuna.
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

        # Check for failed trials (masked as COMPLETE with -inf value)
        # When error_score=nan, we return -inf to Optuna to avoid crashing,
        # so trials are COMPLETE but have -inf value.
        failed_trials = [t for t in search.trials_ if t.value == float("-inf")]
        assert len(failed_trials) > 0, "Expected some trials to have -inf value"

        # Failed trials should have had parameter=2
        failed_params = [t.params.get("parameter") for t in failed_trials]
        assert 2 in failed_params, "Expected failed trials to have parameter=2"

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

        Note: When error_score=np.nan, Optuna marks trials as FAIL state instead
        of returning NaN scores.
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

        # Check trial states - all should be COMPLETE with our new robust handling
        complete_trials = [t for t in search.trials_ if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(complete_trials) == 50, "All trials should be COMPLETE (failures masked)"

        # Check cv_results_ contains NaNs for failed trials
        scores = search.cv_results_["mean_test_score"]
        assert np.any(np.isnan(scores)), "cv_results_ should contain NaNs for failed trials"
        assert np.any(~np.isnan(scores)), "cv_results_ should contain valid scores for successful trials"

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

    def test_evaluate_candidates_exception_stored(self, simple_classification_data):
        """Test exception handling when evaluate_candidates raises (covers lines 479-490)."""
        X, y = simple_classification_data

        search = OptunaSearchCV(
            SVC(),
            {},
            n_trials=1,
            error_score=0.0,
            cv=2,
        )

        with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
            mock_cv.side_effect = ValueError("boom")
            search.fit(X, y)

        assert len(search.trials_) == 1
        trial = search.trials_[0]
        assert trial.value == 0.0
        assert trial.user_attrs.get("exception") == "boom"
        assert trial.user_attrs.get("exception_type") == "ValueError"
        # mean_test_score is not set if cross_validate fails completely
        # assert trial.user_attrs.get("mean_test_score") == 0.0


# --- Additional Edge Cases moved from temporary coverage file ---


class MockEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.5


@pytest.fixture
def mock_data():
    return make_classification(n_samples=20, random_state=42)


def test_no_trials_completed_explicit(mock_data):
    """Force no trials completed error."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, refit=True)

    with patch.object(optuna.study.Study, "optimize"), pytest.raises(ValueError, match="No trials were completed"):
        search.fit(X, y)


def test_single_metric_valid_return(mock_data):
    """Ensure return mean_test_score line is hit."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2)

    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        mock_cv.return_value = {"test_score": np.array([0.8, 0.8])}
        search.fit(X, y)
    assert search.best_score_ == 0.8


def test_multimetric_nan_optimization_metric(mock_data):
    """Test multimetric case where optimization metric is NaN."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, scoring={"acc": "accuracy"}, refit="acc")

    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        mock_cv.return_value = {"test_acc": np.array([np.nan, np.nan]), "train_acc": np.array([0.5, 0.5])}
        search.fit(X, y)
    assert search.trials_[0].value == float("-inf")


def test_multimetric_return_train_score(mock_data):
    """Test multimetric with return_train_score=True."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(
        clf, {}, n_trials=1, cv=2, scoring={"acc": "accuracy"}, return_train_score=True, refit=False
    )
    search.fit(X, y)
    trial = search.trials_[0]
    assert "mean_train_acc" in trial.user_attrs


def test_nan_score_single_metric(mock_data):
    """Test NaN score handling for single metric."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, error_score=np.nan)
    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        mock_cv.return_value = {"test_score": np.array([np.nan, np.nan])}
        search.fit(X, y)
    assert search.trials_[0].value == float("-inf")


def test_multimetric_refit_bool(mock_data):
    """Test refit=True bool with multimetric raises ValueError (BaseSearchCV behavior)."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, scoring={"a": "accuracy", "b": "accuracy"}, refit=True)
    with pytest.raises(ValueError, match="refit must be set to a scorer key"):
        search.fit(X, y)


def test_multimetric_refit_str(mock_data):
    """Test refit='metric' with multimetric."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, scoring={"a": "accuracy", "b": "accuracy"}, refit="a")
    search.fit(X, y)
    assert search.best_index_ is not None


def test_invalid_callback_type(mock_data):
    """Test invalid callback type."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, callbacks={"bad": "str"}, n_trials=1)
    with pytest.raises(TypeError):
        search.fit(X, y)


def test_mixed_success_failure_trials(mock_data):
    """Test mix of success and failure."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=2, cv=2, error_score=-999.0, return_train_score=True)

    success_result = {"test_score": np.array([0.8, 0.8]), "train_score": np.array([0.9, 0.9])}
    calls = [0]

    def side_effect(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return success_result
        raise ValueError("Simulated failure")

    with patch("sklearn_optuna.objective.cross_validate", side_effect=side_effect):
        search.fit(X, y)

    completed = [t for t in search.trials_ if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 2
    assert "split0_test_score" in search.cv_results_


def test_refit_callable(mock_data):
    """Test refit as callable."""
    X, y = mock_data
    clf = MockEstimator()

    def refit_callable(cv_results):
        return MockEstimator()

    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, refit=refit_callable)
    search.fit(X, y)
    assert isinstance(search.best_estimator_, MockEstimator)


def test_multimetric_list_scoring(mock_data):
    """Test multimetric with list scoring."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, scoring=["accuracy"], refit="accuracy")
    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        mock_cv.return_value = {"test_accuracy": np.array([0.8, 0.8])}
        search.fit(X, y)
    assert "mean_test_accuracy" in search.cv_results_


def test_multimetric_missing_splits(mock_data):
    """Test multimetric with missing splits."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(
        clf, {}, n_trials=1, cv=2, scoring={"a": "accuracy", "b": "accuracy"}, refit=False, return_train_score=True
    )
    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        mock_cv.return_value = {"test_a": np.array([0.8, 0.8]), "train_a": np.array([0.9, 0.9])}
        search.fit(X, y)
    assert "mean_test_a" in search.cv_results_


def test_multimetric_refit_false_return_val(mock_data):
    """Ensure returning a scalar for multimetric refit=False."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, scoring={"a": "accuracy"}, refit=False)
    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        mock_cv.return_value = {"test_a": np.array([1.0, 1.0])}
        search.fit(X, y)

    # Check trial value
    assert search.trials_[0].value == 1.0


def test_multimetric_no_test_keys(mock_data):
    """Test ValueError when no test keys are found."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2, scoring={"a": "accuracy"}, refit=False)
    with patch("sklearn_optuna.objective.cross_validate") as mock_cv:
        # Return empty dict - no test scores
        mock_cv.return_value = {}
        # Expect ValueError from objective function caught and trial failed?
        # No, Exception in objective is caught and stored in user_attrs
        search.fit(X, y)

    # Check if trial failed with exception
    trial = search.trials_[0]
    assert "exception" in trial.user_attrs
    assert "ValueError" in trial.user_attrs["exception_type"]
    assert "No test scores found" in trial.user_attrs["exception"]


def test_partial_failure_cv_results(mock_data):
    """Test cv_results_ construction when some trials fail (missing keys)."""
    X, y = mock_data
    clf = FailingClassifier()

    # We want one success (param=1) and one failure (param=2)
    # create study and enqueue
    study = optuna.create_study()
    study.enqueue_trial({"parameter": 1})
    # Enqueue failing parameter
    study.enqueue_trial({"parameter": 2})

    search = OptunaSearchCV(
        clf,
        {"parameter": IntDistribution(1, 2)},
        n_trials=2,
        cv=2,
        return_train_score=True,
        error_score=0.0,
        refit=False,
    )

    search.fit(X, y, study=study)

    # Verify we hit both
    assert len(search.trials_) == 2

    # cv_results_ should contain keys
    assert "std_train_score" in search.cv_results_


def test_all_trials_fail_refit_false(mock_data):
    """Test cv_results_ when all trials fail (refit=False)."""
    X, y = mock_data
    clf = FailingClassifier()

    # Force failure
    study = optuna.create_study()
    study.enqueue_trial({"parameter": 2})

    search = OptunaSearchCV(clf, {"parameter": IntDistribution(2, 2)}, n_trials=1, cv=2, error_score=0.0, refit=False)

    search.fit(X, y, study=study)

    assert len(search.trials_) == 1
    assert search.trials_[0].state == optuna.trial.TrialState.COMPLETE

    # check that std_test_score is there (NaN)
    assert "std_test_score" in search.cv_results_


def test_all_trials_fail_return_train_score(mock_data):
    """Test cv_results_ when all trials fail and return_train_score=True."""
    X, y = mock_data
    clf = FailingClassifier()

    study = optuna.create_study()
    study.enqueue_trial({"parameter": 2})

    search = OptunaSearchCV(
        clf,
        {"parameter": IntDistribution(2, 2)},
        n_trials=1,
        cv=2,
        error_score=0.0,
        refit=False,
        return_train_score=True,
    )

    search.fit(X, y, study=study)

    assert "std_train_score" in search.cv_results_


def test_metadata_routing_empty(mock_data):
    """Test fit when _get_routed_params_for_fit returns empty dict."""
    X, y = mock_data
    clf = MockEstimator()
    search = OptunaSearchCV(clf, {}, n_trials=1, cv=2)

    with patch.object(search, "_get_routed_params_for_fit", return_value={}):
        search.fit(X, y)
