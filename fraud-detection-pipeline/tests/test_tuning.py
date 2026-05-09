import pandas as pd
import pytest

from fraud_detection.models import RandomForestFactory, XGBoostFactory


def _minimal_xy():
    """6 rows, clearly separable — fast CV."""
    X = pd.DataFrame(
        {
            "V1": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0],
            "Amount": [1.0, 102.0, 1.1, 101.0, 1.2, 100.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])
    return X, y


def test_tune_random_forest_returns_tuning_result():
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    result = tune_random_forest(X, y, n_iter=2, cv=2, random_state=42)

    assert result.best_params is not None
    assert isinstance(result.best_params, dict)
    assert 0.0 <= result.best_score <= 1.0
    assert result.scoring == "roc_auc"


def test_tune_xgboost_returns_tuning_result():
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    result = tune_xgboost(X, y, n_iter=2, cv=2, random_state=42)

    assert result.best_params is not None
    assert isinstance(result.best_params, dict)
    assert 0.0 <= result.best_score <= 1.0
    assert result.scoring == "roc_auc"


def test_tuning_result_best_factory_compatible_with_model_factory_protocol():
    """best_factory.create(scale_pos_weight=None) returns a Classifier."""
    from fraud_detection.tuning import tune_random_forest
    from fraud_detection.models import Classifier

    X, y = _minimal_xy()
    result = tune_random_forest(X, y, n_iter=2, cv=2, random_state=42)
    model = result.best_factory.create(scale_pos_weight=None)

    assert isinstance(model, Classifier)


def test_tuning_result_best_factory_trains_and_predicts():
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    result = tune_xgboost(X, y, n_iter=2, cv=2, random_state=42)
    model = result.best_factory.create(scale_pos_weight=None)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_tuning_result_best_factory_accepts_scale_pos_weight():
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    result = tune_random_forest(X, y, n_iter=2, cv=2, random_state=42)
    model = result.best_factory.create(scale_pos_weight=2.0)
    model.fit(X, y)

    assert model is not None


def test_tune_random_forest_default_scoring_is_roc_auc():
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    result = tune_random_forest(X, y, n_iter=2, cv=2)

    assert result.scoring == "roc_auc"


def test_tune_xgboost_default_scoring_is_roc_auc():
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    result = tune_xgboost(X, y, n_iter=2, cv=2)

    assert result.scoring == "roc_auc"


# ---------------------------------------------------------------------------
# Wave 3 blocker: invalid n_iter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_n_iter", [0, -1, -100])
def test_tune_random_forest_rejects_invalid_n_iter(bad_n_iter):
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    with pytest.raises(ValueError, match="n_iter"):
        tune_random_forest(X, y, n_iter=bad_n_iter, cv=2)


@pytest.mark.parametrize("bad_n_iter", [0, -1, -100])
def test_tune_xgboost_rejects_invalid_n_iter(bad_n_iter):
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    with pytest.raises(ValueError, match="n_iter"):
        tune_xgboost(X, y, n_iter=bad_n_iter, cv=2)


# ---------------------------------------------------------------------------
# Wave 3 blocker: insufficient minority class
# ---------------------------------------------------------------------------


def test_tune_random_forest_rejects_single_class():
    from fraud_detection.tuning import tune_random_forest

    X = pd.DataFrame({"V1": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0])  # only one class
    with pytest.raises(ValueError, match="2 classes"):
        tune_random_forest(X, y, n_iter=2)


def test_tune_xgboost_rejects_single_class():
    from fraud_detection.tuning import tune_xgboost

    X = pd.DataFrame({"V1": [1.0, 2.0, 3.0]})
    y = pd.Series([1, 1, 1])
    with pytest.raises(ValueError, match="2 classes"):
        tune_xgboost(X, y, n_iter=2)


def test_tune_random_forest_rejects_minority_class_with_one_sample():
    from fraud_detection.tuning import tune_random_forest

    X = pd.DataFrame({"V1": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0, 1])  # minority class has 1 sample
    with pytest.raises(ValueError, match="minority class"):
        tune_random_forest(X, y, n_iter=2)


def test_tune_xgboost_rejects_minority_class_with_one_sample():
    from fraud_detection.tuning import tune_xgboost

    X = pd.DataFrame({"V1": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0, 1])
    with pytest.raises(ValueError, match="minority class"):
        tune_xgboost(X, y, n_iter=2)


# ---------------------------------------------------------------------------
# Wave 3 blocker: n_jobs default is 1
# ---------------------------------------------------------------------------


def test_tune_random_forest_n_jobs_default_is_1():
    """n_jobs=1 keeps CI stable; verify via smoke-run (no error)."""
    from fraud_detection.tuning import tune_random_forest
    import inspect

    sig = inspect.signature(tune_random_forest)
    assert sig.parameters["n_jobs"].default == 1


def test_tune_xgboost_n_jobs_default_is_1():
    from fraud_detection.tuning import tune_xgboost
    import inspect

    sig = inspect.signature(tune_xgboost)
    assert sig.parameters["n_jobs"].default == 1


# ---------------------------------------------------------------------------
# Explicit cv validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_cv", [0, 1, -1])
def test_tune_random_forest_rejects_cv_less_than_2(bad_cv):
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    with pytest.raises(ValueError, match="cv must be >= 2"):
        tune_random_forest(X, y, n_iter=2, cv=bad_cv)


@pytest.mark.parametrize("bad_cv", [0, 1, -1])
def test_tune_xgboost_rejects_cv_less_than_2(bad_cv):
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    with pytest.raises(ValueError, match="cv must be >= 2"):
        tune_xgboost(X, y, n_iter=2, cv=bad_cv)


def test_tune_random_forest_rejects_cv_exceeding_minority_count():
    from fraud_detection.tuning import tune_random_forest

    # 4 majority, 2 minority — cv=3 exceeds minority count of 2
    X = pd.DataFrame({"V1": [0.0, 0.1, 0.2, 0.3, 4.0, 4.1]})
    y = pd.Series([0, 0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="minority class count"):
        tune_random_forest(X, y, n_iter=2, cv=3)


def test_tune_xgboost_rejects_cv_exceeding_minority_count():
    from fraud_detection.tuning import tune_xgboost

    X = pd.DataFrame({"V1": [0.0, 0.1, 0.2, 0.3, 4.0, 4.1]})
    y = pd.Series([0, 0, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="minority class count"):
        tune_xgboost(X, y, n_iter=2, cv=3)


# ---------------------------------------------------------------------------
# StratifiedKFold used for CV — class balance in folds
# ---------------------------------------------------------------------------


def test_tune_random_forest_uses_stratified_kfold():
    from unittest.mock import patch
    from fraud_detection.tuning import tune_random_forest, _make_cv
    from sklearn.model_selection import StratifiedKFold

    X, y = _minimal_xy()
    captured = {}

    original_make_cv = _make_cv

    def spy_make_cv(n_splits, random_state):
        result = original_make_cv(n_splits, random_state)
        captured["cv"] = result
        return result

    with patch("fraud_detection.tuning._make_cv", side_effect=spy_make_cv):
        tune_random_forest(X, y, n_iter=2, cv=2, random_state=42)

    assert "cv" in captured
    assert isinstance(captured["cv"], StratifiedKFold)


def test_tune_xgboost_uses_stratified_kfold():
    from unittest.mock import patch
    from fraud_detection.tuning import tune_xgboost, _make_cv
    from sklearn.model_selection import StratifiedKFold

    X, y = _minimal_xy()
    captured = {}

    original_make_cv = _make_cv

    def spy_make_cv(n_splits, random_state):
        result = original_make_cv(n_splits, random_state)
        captured["cv"] = result
        return result

    with patch("fraud_detection.tuning._make_cv", side_effect=spy_make_cv):
        tune_xgboost(X, y, n_iter=2, cv=2, random_state=42)

    assert "cv" in captured
    assert isinstance(captured["cv"], StratifiedKFold)


# ---------------------------------------------------------------------------
# tune_lightgbm
# ---------------------------------------------------------------------------


def test_tune_lightgbm_returns_tuning_result():
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    result = tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)

    assert result.best_params is not None
    assert isinstance(result.best_params, dict)
    assert 0.0 <= result.best_score <= 1.0
    assert result.scoring == "roc_auc"


def test_tune_lightgbm_best_params_are_lgbm_params():
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    result = tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)

    lgbm_param_keys = {"n_estimators", "max_depth", "learning_rate", "num_leaves", "subsample"}
    assert lgbm_param_keys & set(result.best_params.keys()), (
        f"Expected at least one LightGBM param in {result.best_params.keys()}"
    )


def test_tune_lightgbm_factory_creates_classifier():
    from fraud_detection.tuning import tune_lightgbm
    from fraud_detection.models import Classifier

    X, y = _minimal_xy()
    result = tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)
    model = result.best_factory.create(scale_pos_weight=None)

    assert isinstance(model, Classifier)


def test_tune_lightgbm_factory_trains_and_predicts():
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    result = tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)
    model = result.best_factory.create(scale_pos_weight=None)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_tune_lightgbm_factory_accepts_scale_pos_weight():
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    result = tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)
    model = result.best_factory.create(scale_pos_weight=3.0)
    model.fit(X, y)

    assert model is not None


def test_tune_lightgbm_uses_stratified_kfold():
    from unittest.mock import patch
    from fraud_detection.tuning import tune_lightgbm, _make_cv
    from sklearn.model_selection import StratifiedKFold

    X, y = _minimal_xy()
    captured = {}

    original_make_cv = _make_cv

    def spy_make_cv(n_splits, random_state):
        result = original_make_cv(n_splits, random_state)
        captured["cv"] = result
        return result

    with patch("fraud_detection.tuning._make_cv", side_effect=spy_make_cv):
        tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)

    assert "cv" in captured
    assert isinstance(captured["cv"], StratifiedKFold)


def test_tune_lightgbm_rejects_invalid_n_iter():
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    with pytest.raises(ValueError, match="n_iter"):
        tune_lightgbm(X, y, n_iter=0, cv=2)


def test_tune_lightgbm_n_jobs_default_is_1():
    from fraud_detection.tuning import tune_lightgbm
    import inspect

    sig = inspect.signature(tune_lightgbm)
    assert sig.parameters["n_jobs"].default == 1


# ---------------------------------------------------------------------------
# Review blocker: tuned factories must preserve custom random_state
# ---------------------------------------------------------------------------


def test_tune_random_forest_factory_preserves_custom_random_state():
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    custom_state = 99
    result = tune_random_forest(X, y, n_iter=2, cv=2, random_state=custom_state)
    model = result.best_factory.create(scale_pos_weight=None)

    assert model.random_state == custom_state


def test_tune_xgboost_factory_preserves_custom_random_state():
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    custom_state = 99
    result = tune_xgboost(X, y, n_iter=2, cv=2, random_state=custom_state)
    model = result.best_factory.create(scale_pos_weight=None)

    assert model.random_state == custom_state


def test_tune_lightgbm_factory_preserves_custom_random_state():
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    custom_state = 99
    result = tune_lightgbm(X, y, n_iter=1, cv=2, random_state=custom_state)
    model = result.best_factory.create(scale_pos_weight=None)

    assert model.random_state == custom_state


# ---------------------------------------------------------------------------
# Lazy import isolation: importing tuning must not touch xgboost or lightgbm
# ---------------------------------------------------------------------------


def test_import_tuning_does_not_import_xgboost(monkeypatch):
    """tuning module-level import must not trigger xgboost import."""
    import sys
    import importlib

    # Remove cached module so reimport runs module-level code
    monkeypatch.delitem(sys.modules, "fraud_detection.tuning", raising=False)

    # Block xgboost at sys.modules level
    monkeypatch.setitem(sys.modules, "xgboost", None)  # type: ignore[arg-type]

    # Should not raise even though xgboost is "missing"
    importlib.import_module("fraud_detection.tuning")


def test_import_tuning_does_not_import_lightgbm(monkeypatch):
    """tuning module-level import must not trigger lightgbm import."""
    import sys
    import importlib

    monkeypatch.delitem(sys.modules, "fraud_detection.tuning", raising=False)
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # type: ignore[arg-type]

    importlib.import_module("fraud_detection.tuning")


def test_tune_lightgbm_does_not_import_xgboost(monkeypatch):
    """tune_lightgbm path must not touch xgboost."""
    import sys
    import importlib

    monkeypatch.delitem(sys.modules, "fraud_detection.tuning", raising=False)
    # Allow lightgbm but block xgboost
    monkeypatch.setitem(sys.modules, "xgboost", None)  # type: ignore[arg-type]

    tuning = importlib.import_module("fraud_detection.tuning")
    X, y = _minimal_xy()
    # tune_lightgbm only needs lightgbm; xgboost is blocked but call must succeed
    result = tuning.tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)
    assert result.scoring == "roc_auc"


def test_tune_xgboost_does_not_import_lightgbm(monkeypatch):
    """tune_xgboost path must not touch lightgbm."""
    import sys
    import importlib

    monkeypatch.delitem(sys.modules, "fraud_detection.tuning", raising=False)
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # type: ignore[arg-type]

    tuning = importlib.import_module("fraud_detection.tuning")
    X, y = _minimal_xy()
    result = tuning.tune_xgboost(X, y, n_iter=2, cv=2, random_state=42)
    assert result.scoring == "roc_auc"


# ---------------------------------------------------------------------------
# Optuna TPE replaces HalvingRandomSearchCV
# ---------------------------------------------------------------------------


def test_tuning_module_does_not_import_halving_search_cv():
    """tuning module must NOT expose HalvingRandomSearchCV — replaced by Optuna."""
    import fraud_detection.tuning as tuning_mod

    assert not hasattr(tuning_mod, "HalvingRandomSearchCV"), (
        "HalvingRandomSearchCV must not be imported — replaced by Optuna TPE"
    )


def test_tuning_module_does_not_import_randomized_search_cv():
    """tuning module must not expose RandomizedSearchCV."""
    import fraud_detection.tuning as tuning_mod

    assert not hasattr(tuning_mod, "RandomizedSearchCV"), (
        "RandomizedSearchCV should not be imported in tuning module"
    )


def test_tuning_module_uses_optuna_tpe_sampler():
    """tuning module must import optuna and expose TPESampler."""
    import fraud_detection.tuning as tuning_mod

    assert hasattr(tuning_mod, "optuna"), "optuna not imported in tuning module"


def test_tune_random_forest_uses_optuna_study():
    """tune_random_forest must use optuna.create_study internally."""
    import optuna
    from unittest.mock import patch, MagicMock
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    real_create_study = optuna.create_study
    captured = {}

    def spy_create_study(**kwargs):
        study = real_create_study(**kwargs)
        captured["study"] = study
        captured["direction"] = kwargs.get("direction")
        captured["sampler"] = kwargs.get("sampler")
        return study

    with patch("fraud_detection.tuning.optuna.create_study", side_effect=spy_create_study):
        tune_random_forest(X, y, n_iter=2, cv=2, random_state=42)

    assert "study" in captured, "optuna.create_study was not called"
    assert captured["direction"] == "maximize", (
        f"study direction must be 'maximize' for ROC AUC, got {captured['direction']}"
    )
    assert isinstance(captured["sampler"], optuna.samplers.TPESampler), (
        f"sampler must be TPESampler, got {type(captured['sampler'])}"
    )


def test_tune_lightgbm_uses_optuna_study():
    """tune_lightgbm must use optuna.create_study with TPESampler."""
    import optuna
    from unittest.mock import patch
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    real_create_study = optuna.create_study
    captured = {}

    def spy_create_study(**kwargs):
        study = real_create_study(**kwargs)
        captured["sampler"] = kwargs.get("sampler")
        captured["direction"] = kwargs.get("direction")
        return study

    with patch("fraud_detection.tuning.optuna.create_study", side_effect=spy_create_study):
        tune_lightgbm(X, y, n_iter=1, cv=2, random_state=42)

    assert isinstance(captured.get("sampler"), optuna.samplers.TPESampler), (
        "tune_lightgbm must use TPESampler"
    )
    assert captured.get("direction") == "maximize"


def test_tune_xgboost_uses_optuna_study():
    """tune_xgboost must use optuna.create_study with TPESampler."""
    import optuna
    from unittest.mock import patch
    from fraud_detection.tuning import tune_xgboost

    X, y = _minimal_xy()
    real_create_study = optuna.create_study
    captured = {}

    def spy_create_study(**kwargs):
        study = real_create_study(**kwargs)
        captured["sampler"] = kwargs.get("sampler")
        return study

    with patch("fraud_detection.tuning.optuna.create_study", side_effect=spy_create_study):
        tune_xgboost(X, y, n_iter=2, cv=2, random_state=42)

    assert isinstance(captured.get("sampler"), optuna.samplers.TPESampler), (
        "tune_xgboost must use TPESampler"
    )


def test_tune_random_forest_n_iter_controls_trial_count():
    """n_iter controls how many Optuna trials are run."""
    import optuna
    from unittest.mock import patch
    from fraud_detection.tuning import tune_random_forest

    X, y = _minimal_xy()
    trial_counts = []
    real_optimize = None

    def spy_optimize(study_self_fn, n_trials, **kwargs):
        trial_counts.append(n_trials)
        return real_optimize(study_self_fn, n_trials=n_trials, **kwargs)

    real_study = optuna.create_study(direction="maximize")
    real_optimize = real_study.optimize.__func__  # unbound method

    # Patch at the study instance level via create_study spy
    real_create_study = optuna.create_study

    studies = []

    def capturing_create_study(**kwargs):
        study = real_create_study(**kwargs)
        studies.append(study)
        return study

    with patch("fraud_detection.tuning.optuna.create_study", side_effect=capturing_create_study):
        tune_random_forest(X, y, n_iter=3, cv=2, random_state=42)

    # study.optimize was called; best_trial must be accessible
    assert len(studies) == 1
    assert studies[0].best_trial is not None
    assert len(studies[0].trials) == 3


def test_tune_lightgbm_n_iter_controls_trial_count():
    """n_iter=2 produces exactly 2 Optuna trials for lightgbm."""
    import optuna
    from unittest.mock import patch
    from fraud_detection.tuning import tune_lightgbm

    X, y = _minimal_xy()
    real_create_study = optuna.create_study
    studies = []

    def capturing_create_study(**kwargs):
        study = real_create_study(**kwargs)
        studies.append(study)
        return study

    with patch("fraud_detection.tuning.optuna.create_study", side_effect=capturing_create_study):
        tune_lightgbm(X, y, n_iter=2, cv=2, random_state=42)

    assert len(studies[0].trials) == 2


def test_tune_optuna_uses_full_data_cv_not_subsampled():
    """Optuna CV must use all training rows — no resource subsampling."""
    import optuna
    from unittest.mock import patch
    from fraud_detection.tuning import tune_random_forest
    from sklearn.model_selection import cross_val_score

    X, y = _minimal_xy()
    cv_call_sizes = []
    real_cross_val = cross_val_score

    def spy_cross_val(estimator, X_cv, y_cv, **kwargs):
        cv_call_sizes.append(len(X_cv))
        return real_cross_val(estimator, X_cv, y_cv, **kwargs)

    with patch("fraud_detection.tuning.cross_val_score", side_effect=spy_cross_val):
        tune_random_forest(X, y, n_iter=2, cv=2, random_state=42)

    # Every CV call must see the full training set
    assert all(s == len(X) for s in cv_call_sizes), (
        f"Some CV calls saw subsampled data: {cv_call_sizes}, expected all {len(X)}"
    )


# ---------------------------------------------------------------------------
# _larger_xy helper used by several tests
# ---------------------------------------------------------------------------


def _larger_xy(n: int = 100):
    """Balanced dataset large enough to exercise full-data CV."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"V1": rng.standard_normal(n), "Amount": rng.exponential(10, n)})
    y = pd.Series([i % 2 for i in range(n)])
    return X, y


def test_tune_random_forest_on_larger_balanced_data_no_warnings(recwarn):
    """Optuna TPE on larger balanced data must produce zero single-class warnings."""
    import warnings
    from fraud_detection.tuning import tune_random_forest

    X, y = _larger_xy(100)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tune_random_forest(X, y, n_iter=3, cv=3, random_state=42)

    single_class_warns = [
        str(x.message) for x in w
        if "single class" in str(x.message).lower() or "only one class" in str(x.message).lower()
    ]
    assert not single_class_warns, f"Single-class warnings: {single_class_warns}"
