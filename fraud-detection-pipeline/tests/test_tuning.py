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
    from unittest.mock import patch, MagicMock
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
