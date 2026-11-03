"""Tests for fold-level trial pruning, intermediate-value reporting, and the related cv_results_ rules."""

import warnings
from typing import ClassVar

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import MedianPruner, NopPruner, PercentilePruner, ThresholdPruner, WilcoxonPruner
from optuna.samplers import RandomSampler
from optuna.trial import TrialState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit
from sklearn.svm import SVC

from conftest import FailingClassifier
from sklearn_optuna import OptunaSearchCV, Pruner, Sampler

GOOD = 0.9
BAD = 0.2


class LevelClassifier(BaseEstimator, ClassifierMixin):
    """Classifier whose score is exactly its ``level`` parameter; records every fit."""

    fits: ClassVar[list[float]] = []

    def __init__(self, level=GOOD):
        self.level = level

    def fit(self, X, y):
        LevelClassifier.fits.append(self.level)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return self.level


class FirstFitFailsClassifier(BaseEstimator, ClassifierMixin):
    """Classifier whose first ``fit`` call after the per-test reset raises; later calls succeed."""

    failed_once: ClassVar[bool] = False

    def __init__(self, dummy=0):
        self.dummy = dummy

    def fit(self, X, y):
        if not FirstFitFailsClassifier.failed_once:
            FirstFitFailsClassifier.failed_once = True
            raise ValueError("first fit fails as required")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return 0.5


class PrunedFitClassifier(BaseEstimator, ClassifierMixin):
    """Classifier whose ``fit`` raises Optuna's TrialPruned."""

    def __init__(self, dummy=0):
        self.dummy = dummy

    def fit(self, X, y):
        raise optuna.TrialPruned("pruned from inside fit")

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class WarmupOnlyMedianPruner(MedianPruner):
    """MedianPruner subclass whose constructor names its warm-up differently."""

    def __init__(self, warmup=0):
        super().__init__(n_warmup_steps=warmup)


class ThresholdOnlyWilcoxonPruner(WilcoxonPruner):
    """WilcoxonPruner subclass whose constructor names its threshold differently."""

    def __init__(self, threshold=0.1):
        super().__init__(p_threshold=threshold)


@pytest.fixture(autouse=True)
def _reset_counters():
    LevelClassifier.fits = []
    FirstFitFailsClassifier.failed_once = False
    yield
    LevelClassifier.fits = []
    FirstFitFailsClassifier.failed_once = False


@pytest.fixture
def threshold_search():
    """A search where every trial drawing ``BAD`` scores below the threshold pruner's floor."""

    def make(**overrides):
        kwargs = {
            "estimator": LevelClassifier(),
            "param_distributions": {"level": CategoricalDistribution([GOOD, BAD])},
            "pruner": Pruner(pruner=ThresholdPruner, lower=0.5),
            "sampler": Sampler(sampler=RandomSampler, seed=0),
            "n_trials": 8,
            "cv": 5,
        }
        kwargs.update(overrides)
        return OptunaSearchCV(**kwargs)

    return make


def _states(search):
    return [t.state for t in search.trials_]


# ---------------------------------------------------------------------------
# Finite fold scores are reported to the trial
# ---------------------------------------------------------------------------


class TestReporting:
    def test_reported_values_without_a_pruner(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, n_trials=2, cv=5)
        search.fit(X, y)

        for trial in search.trials_:
            assert sorted(trial.intermediate_values) == [0, 1, 2, 3, 4]
            for step, value in trial.intermediate_values.items():
                assert value == trial.user_attrs[f"split{step}_test_score"]

    def test_reported_metric_when_refit_names_a_metric(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(
            SVC(),
            {"C": FloatDistribution(0.1, 10.0)},
            scoring={"acc": "accuracy", "f1": "f1_macro"},
            refit="f1",
            n_trials=2,
            cv=3,
        )
        search.fit(X, y)

        for trial in search.trials_:
            for step, value in trial.intermediate_values.items():
                assert value == trial.user_attrs[f"split{step}_test_f1"]

    @pytest.mark.parametrize("refit", [False, lambda cv_results: 0], ids=["refit-false", "refit-callable"])
    def test_reported_metric_when_refit_does_not_name_a_metric(self, simple_classification_data, refit):
        X, y = simple_classification_data
        search = OptunaSearchCV(
            SVC(),
            {"C": FloatDistribution(0.1, 10.0)},
            scoring={"acc": "accuracy", "f1": "f1_macro"},
            refit=refit,
            n_trials=2,
            cv=3,
        )
        search.fit(X, y)

        for trial in search.trials_:
            assert len(trial.intermediate_values) == 3
            for step, value in trial.intermediate_values.items():
                assert value == trial.user_attrs[f"split{step}_test_acc"]

    def test_failed_fold_with_numeric_error_score_is_reported(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(FailingClassifier(parameter=2), {}, n_trials=1, cv=3, error_score=0.25, refit=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        trial = search.trials_[0]
        assert trial.state == TrialState.COMPLETE
        assert trial.intermediate_values == {0: 0.25, 1: 0.25, 2: 0.25}

    def test_failed_fold_with_nan_error_score_is_not_reported(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(FirstFitFailsClassifier(), {}, n_trials=1, cv=3, error_score=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        trial = search.trials_[0]
        assert trial.state == TrialState.COMPLETE
        assert sorted(trial.intermediate_values) == [1, 2]
        assert np.isnan(trial.user_attrs["split0_test_score"])
        assert trial.user_attrs["split1_test_score"] == 0.5


# ---------------------------------------------------------------------------
# Pruner parameter on the search
# ---------------------------------------------------------------------------


class TestPrunerParameter:
    def test_new_study_uses_the_given_pruner(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(
            SVC(),
            {"C": FloatDistribution(0.1, 10.0)},
            pruner=Pruner(pruner=MedianPruner, n_warmup_steps=1),
            n_trials=1,
            cv=3,
        )
        search.fit(X, y)
        assert isinstance(search.study_.pruner, MedianPruner)
        assert search.study_.pruner._n_warmup_steps == 1

    def test_passed_study_takes_the_given_pruner(self, simple_classification_data):
        X, y = simple_classification_data
        study = optuna.create_study(direction="maximize", pruner=NopPruner())
        search = OptunaSearchCV(
            SVC(), {"C": FloatDistribution(0.1, 10.0)}, pruner=Pruner(pruner=MedianPruner), n_trials=1, cv=3
        )
        search.fit(X, y, study=study)
        assert isinstance(study.pruner, MedianPruner)

    def test_passed_study_keeps_its_pruner_when_none_given(self, simple_classification_data):
        X, y = simple_classification_data
        own = NopPruner()
        study = optuna.create_study(direction="maximize", pruner=own)
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, n_trials=1, cv=3)
        search.fit(X, y, study=study)
        assert study.pruner is own

    def test_raw_pruner_instance_is_rejected_before_any_trial(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, pruner=MedianPruner(), n_trials=1, cv=3)
        with pytest.raises(TypeError, match="pruner must be a Pruner instance"):
            search.fit(X, y)
        assert not hasattr(search, "study_")


# ---------------------------------------------------------------------------
# The search stops trials only when a pruner is given
# ---------------------------------------------------------------------------


class TestStopping:
    def test_no_pruning_without_a_pruner(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        study = optuna.create_study(direction="maximize")  # Optuna's default MedianPruner
        assert isinstance(study.pruner, MedianPruner)
        search = threshold_search(pruner=None)
        search.fit(X, y, study=study)

        assert all(state == TrialState.COMPLETE for state in _states(search))
        assert search.n_pruned_trials_ == 0
        assert BAD in [t.params["level"] for t in search.trials_]

    def test_pruning_with_a_pruner(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search()
        search.fit(X, y)

        bad = [t for t in search.trials_ if t.params["level"] == BAD]
        good = [t for t in search.trials_ if t.params["level"] == GOOD]
        assert bad and good
        assert all(t.state == TrialState.PRUNED for t in bad)
        assert all(t.state == TrialState.COMPLETE for t in good)
        # Each bad trial fitted exactly its first fold; each good one all five.
        assert LevelClassifier.fits.count(BAD) == len(bad)
        assert LevelClassifier.fits.count(GOOD) == 5 * len(good) + 1  # + the refit

    def test_pruned_trial_is_not_an_error(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search(error_score="raise")
        search.fit(X, y)

        pruned = [t for t in search.trials_ if t.state == TrialState.PRUNED]
        assert pruned
        for trial in pruned:
            assert "exception" not in trial.user_attrs
            assert "exception_type" not in trial.user_attrs

    def test_user_code_raising_trial_pruned_with_error_score_raise(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(PrunedFitClassifier(), {}, n_trials=1, cv=3, error_score="raise", refit=False)
        search.fit(X, y)

        assert _states(search) == [TrialState.PRUNED]
        assert search.n_pruned_trials_ == 1
        assert "exception" not in search.trials_[0].user_attrs

    def test_user_code_raising_trial_pruned_with_numeric_error_score(self, simple_classification_data):
        """scikit-learn's fit-failure policy absorbs any exception from fit, TrialPruned included."""
        X, y = simple_classification_data
        search = OptunaSearchCV(PrunedFitClassifier(), {}, n_trials=1, cv=3, error_score=0.0, refit=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        assert _states(search) == [TrialState.COMPLETE]
        assert search.n_pruned_trials_ == 0


# ---------------------------------------------------------------------------
# Pruned trials keep their partial results, are counted, and never refit
# ---------------------------------------------------------------------------


class TestPrunedTrials:
    def test_partial_results_on_a_pruned_trial(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search(pruner=Pruner(pruner=ThresholdPruner, lower=0.5, n_warmup_steps=2))
        search.fit(X, y)

        pruned = [t for t in search.trials_ if t.state == TrialState.PRUNED]
        assert pruned
        for trial in pruned:
            attrs = trial.user_attrs
            assert attrs["param_level"] == BAD
            for fold in range(3):
                assert attrs[f"split{fold}_test_score"] == BAD
                assert f"split{fold}_fit_time" in attrs
            assert "split3_test_score" not in attrs
            assert "mean_test_score" not in attrs
            assert sorted(trial.intermediate_values) == [0, 1, 2]

    def test_count_with_pruning(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search()
        search.fit(X, y)
        assert search.n_pruned_trials_ == sum(state == TrialState.PRUNED for state in _states(search))
        assert search.n_pruned_trials_ > 0

    def test_count_without_pruning(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, n_trials=2, cv=3)
        search.fit(X, y)
        assert search.n_pruned_trials_ == 0

    def test_best_params_come_from_a_completed_trial(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search()
        search.fit(X, y)
        assert search.best_params_ == {"level": GOOD}
        assert search.best_score_ == GOOD
        completed = [t for t in search.trials_ if t.state == TrialState.COMPLETE]
        assert search.best_params_ in [t.params for t in completed]

    def test_everything_pruned_with_refit_raises(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search(param_distributions={"level": CategoricalDistribution([BAD])}, n_trials=3)
        with pytest.raises(ValueError, match="No trials were completed"):
            search.fit(X, y)

    def test_count_covers_the_whole_resumed_study(self, threshold_search, simple_classification_data):
        """n_pruned_trials_ counts every PRUNED trial in the study, not only this fit's."""
        X, y = simple_classification_data
        study = optuna.create_study(direction="maximize")
        first = threshold_search(n_trials=6)
        first.fit(X, y, study=study)
        after_first = first.n_pruned_trials_
        assert after_first > 0

        second = threshold_search(n_trials=6, sampler=Sampler(sampler=RandomSampler, seed=1))
        second.fit(X, y, study=study)
        assert len(second.trials_) == 12
        assert second.n_pruned_trials_ == sum(t.state == TrialState.PRUNED for t in study.trials)
        assert second.n_pruned_trials_ >= after_first

    def test_passed_study_takes_both_sampler_and_pruner(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        study = optuna.create_study(direction="maximize", pruner=NopPruner())
        search = threshold_search(n_trials=4)
        search.fit(X, y, study=study)
        assert isinstance(study.sampler, RandomSampler)
        assert isinstance(study.pruner, ThresholdPruner)
        assert search.n_pruned_trials_ > 0

    def test_pruning_under_threads(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search(n_jobs=2, n_trials=12)
        search.fit(X, y)

        states = _states(search)
        assert search.n_pruned_trials_ > 0
        assert states.count(TrialState.PRUNED) + states.count(TrialState.COMPLETE) == 12
        for trial in search.trials_:
            if trial.state == TrialState.PRUNED:
                assert len(trial.intermediate_values) < search.n_splits_
        assert search.best_params_ == {"level": GOOD}


# ---------------------------------------------------------------------------
# Fit-time warnings
# ---------------------------------------------------------------------------


INERT = "cannot stop any trial"


def _fit_and_collect_warnings(search, X, y, substring):
    """Fit and return the messages of non-experimental UserWarnings containing ``substring``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(X, y)
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning)
        and not issubclass(w.category, ExperimentalWarning)
        and substring in str(w.message)
    ]


class TestInertPrunerWarning:
    def test_wilcoxon_with_too_few_folds(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, pruner=Pruner(), n_trials=2, cv=3)
        messages = _fit_and_collect_warnings(search, X, y, INERT)
        assert len(messages) == 1
        assert "at least 4 folds" in messages[0]
        assert "cv yields 3" in messages[0]
        assert search.n_pruned_trials_ == 0

    def test_wilcoxon_with_zero_threshold_never_prunes(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(
            SVC(), {"C": FloatDistribution(0.1, 10.0)}, pruner=Pruner(p_threshold=0.0), n_trials=2, cv=10
        )
        messages = _fit_and_collect_warnings(search, X, y, INERT)
        assert len(messages) == 1
        assert "never prunes" in messages[0]

    @pytest.mark.parametrize(
        ("pruner", "n_folds"),
        [
            pytest.param(Pruner(pruner=MedianPruner, n_warmup_steps=5), 5, id="median"),
            pytest.param(Pruner(pruner=PercentilePruner, percentile=25.0, n_warmup_steps=3), 3, id="percentile"),
        ],
    )
    def test_warmup_covers_every_fold(self, simple_classification_data, pruner, n_folds):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, pruner=pruner, n_trials=2, cv=n_folds)
        messages = _fit_and_collect_warnings(search, X, y, INERT)
        assert len(messages) == 1
        assert f"n_warmup_steps={pruner.params['n_warmup_steps']}" in messages[0]

    @pytest.mark.parametrize(
        "pruner",
        [
            pytest.param(Pruner(), id="wilcoxon-five-folds"),
            pytest.param(Pruner(pruner=MedianPruner, n_warmup_steps=4), id="median-warmup-below-folds"),
            pytest.param(Pruner(pruner=NopPruner), id="unrecognised-class"),
            pytest.param(Pruner(pruner=WarmupOnlyMedianPruner, warmup=9), id="median-subclass-own-parameters"),
            pytest.param(
                Pruner(pruner=ThresholdOnlyWilcoxonPruner, threshold=0.01), id="wilcoxon-subclass-own-parameters"
            ),
        ],
    )
    def test_no_warning_when_the_check_cannot_tell(self, simple_classification_data, pruner):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, pruner=pruner, n_trials=2, cv=5)
        assert _fit_and_collect_warnings(search, X, y, INERT) == []


def _test_folds_seen_per_trial(search, X, y):
    """Fit and return, per trial, the tuple of test-index sets in fold order."""
    seen = []

    def scorer(estimator, X_test, y_test):
        seen.append(frozenset(X_test[:, 0].round(12).tolist()))
        return estimator.score(X_test, y_test)

    search.set_params(scoring=scorer)
    search.fit(X, y)
    n = search.n_splits_
    return [tuple(seen[i : i + n]) for i in range(0, len(seen), n)][: len(search.trials_)]


class TestFoldsFixedPerFit:
    @pytest.mark.parametrize(
        "cv",
        [
            pytest.param(KFold(n_splits=4, shuffle=True), id="kfold-shuffle-unseeded"),
            pytest.param(
                KFold(n_splits=4, shuffle=True, random_state=np.random.RandomState(0)), id="kfold-randomstate"
            ),
            pytest.param(ShuffleSplit(n_splits=4, test_size=0.25), id="shufflesplit-unseeded"),
            pytest.param(RepeatedKFold(n_splits=2, n_repeats=2), id="repeatedkfold-unseeded"),
        ],
    )
    def test_every_trial_is_scored_on_the_same_folds(self, simple_classification_data, cv):
        """Splits are computed once per fit, so fold i is the same data in every trial."""
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, n_trials=3, cv=cv)
        folds = _test_folds_seen_per_trial(search, X, y)
        assert len(folds) == 3
        assert len(set(folds)) == 1, "trials were scored on different folds"
        assert len(set(folds[0])) == 4, "the folds within one trial should differ"


# ---------------------------------------------------------------------------
# cv_results_: rows and columns
# ---------------------------------------------------------------------------

TIMING_KEYS = ("mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time")


class TestCvResultsRows:
    def test_pruned_trials_are_absent(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search()
        search.fit(X, y)

        n_complete = sum(state == TrialState.COMPLETE for state in _states(search))
        assert search.n_pruned_trials_ > 0
        assert len(search.cv_results_["params"]) == n_complete
        assert all(p == {"level": GOOD} for p in search.cv_results_["params"])
        assert BAD not in search.cv_results_["param_level"].compressed()

    def test_error_scored_trials_are_present_and_rank_last(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(
            FailingClassifier(),
            {"parameter": CategoricalDistribution([0, FailingClassifier.FAILING_PARAMETER])},
            sampler=Sampler(sampler=RandomSampler, seed=0),
            n_trials=6,
            cv=3,
            error_score=np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        assert len(search.cv_results_["params"]) == 6
        params = [p["parameter"] for p in search.cv_results_["params"]]
        assert FailingClassifier.FAILING_PARAMETER in params and 0 in params
        ranks = search.cv_results_["rank_test_score"]
        means = search.cv_results_["mean_test_score"]
        worst_rank = ranks.max()
        for mean, rank in zip(means, ranks, strict=True):
            if np.isnan(mean):
                assert rank == worst_rank
            else:
                assert rank < worst_rank

    def test_no_completed_trials_with_refit_false(self, threshold_search, simple_classification_data):
        X, y = simple_classification_data
        search = threshold_search(
            param_distributions={"level": CategoricalDistribution([BAD])}, n_trials=3, refit=False
        )
        search.fit(X, y)
        assert search.cv_results_ == {"params": []}
        assert search.n_pruned_trials_ == 3


class TestTimingColumns:
    def test_timing_columns_present(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, n_trials=3, cv=4)
        search.fit(X, y)

        for key in TIMING_KEYS:
            column = search.cv_results_[key]
            assert column.shape == (3,)
            assert np.all(np.isfinite(column))
            assert np.all(column >= 0.0)
        assert "split0_fit_time" not in search.cv_results_

    def test_timing_on_an_error_scored_fold(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(FirstFitFailsClassifier(), {}, n_trials=1, cv=3, error_score=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        trial = search.trials_[0]
        assert trial.user_attrs["split0_score_time"] == 0.0
        assert trial.user_attrs["split0_fit_time"] >= 0.0
        for key in TIMING_KEYS:
            assert np.isfinite(search.cv_results_[key][0])

    def test_timing_when_every_fold_failed(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(FailingClassifier(parameter=2), {}, n_trials=1, cv=3, error_score=0.0, refit=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        assert search.trials_[0].state == TrialState.COMPLETE
        assert len(search.cv_results_["params"]) == 1
        for key in TIMING_KEYS:
            assert np.isfinite(search.cv_results_[key][0])

    def test_score_columns_unchanged(self, simple_classification_data):
        X, y = simple_classification_data
        search = OptunaSearchCV(SVC(), {"C": FloatDistribution(0.1, 10.0)}, n_trials=2, cv=5, return_train_score=True)
        search.fit(X, y)

        expected = {
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_train_score",
            "std_train_score",
            "params",
            "param_C",
        }
        expected |= {f"split{i}_test_score" for i in range(5)}
        expected |= {f"split{i}_train_score" for i in range(5)}
        expected |= set(TIMING_KEYS)
        assert set(search.cv_results_) == expected
        assert search.cv_results_["rank_test_score"].min() == 1
        assert np.ma.isMaskedArray(search.cv_results_["param_C"])


def test_wilcoxon_default_prunes_a_strictly_worse_trial_after_fold_four(simple_classification_data):
    """Pins the four-fold minimum the how-to states for the default pruner."""
    X, y = simple_classification_data
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"level": GOOD})
    study.enqueue_trial({"level": BAD})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        search = OptunaSearchCV(
            LevelClassifier(), {"level": CategoricalDistribution([GOOD, BAD])}, pruner=Pruner(), n_trials=2, cv=6
        )
        search.fit(X, y, study=study)

    good, bad = search.trials_
    assert good.state == TrialState.COMPLETE
    assert bad.state == TrialState.PRUNED
    assert sorted(bad.intermediate_values) == [0, 1, 2, 3]
