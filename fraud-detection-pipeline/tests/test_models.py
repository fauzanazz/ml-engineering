from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from fraud_detection.models import (
    LightGbmFactory,
    LogisticRegressionFactory,
    DecisionTreeFactory,
    RandomForestFactory,
    XGBoostFactory,
    FACTORY_MAP,
)


def test_lightgbm_factory_creates_classifier():
    model = LightGbmFactory().create()

    assert isinstance(model, LGBMClassifier)


def test_lightgbm_factory_sets_scale_pos_weight_when_provided():
    model = LightGbmFactory(scale_pos_weight=4.0).create()

    assert model.get_params()["scale_pos_weight"] == 4.0


def test_lightgbm_factory_omits_scale_pos_weight_when_none():
    model = LightGbmFactory().create()

    assert model.get_params().get("scale_pos_weight") is None


def test_lightgbm_factory_create_accepts_scale_pos_weight_override():
    model = LightGbmFactory().create(scale_pos_weight=7.0)

    assert model.get_params()["scale_pos_weight"] == 7.0


def test_lightgbm_factory_create_override_beats_constructor_value():
    model = LightGbmFactory(scale_pos_weight=2.0).create(scale_pos_weight=9.0)

    assert model.get_params()["scale_pos_weight"] == 9.0


def test_logistic_regression_factory_creates_classifier():
    model = LogisticRegressionFactory().create()

    assert isinstance(model, LogisticRegression)


def test_logistic_regression_factory_sets_class_weight_when_scale_pos_weight_provided():
    model = LogisticRegressionFactory().create(scale_pos_weight=4.0)

    assert model.get_params()["class_weight"] == {0: 1.0, 1: 4.0}


def test_logistic_regression_factory_omits_class_weight_when_none():
    model = LogisticRegressionFactory().create()

    assert model.get_params()["class_weight"] is None


def test_decision_tree_factory_creates_classifier():
    model = DecisionTreeFactory().create()

    assert isinstance(model, DecisionTreeClassifier)


def test_decision_tree_factory_max_depth_is_two():
    model = DecisionTreeFactory().create()

    assert model.get_params()["max_depth"] == 2


def test_decision_tree_factory_sets_class_weight_when_scale_pos_weight_provided():
    model = DecisionTreeFactory().create(scale_pos_weight=3.0)

    assert model.get_params()["class_weight"] == {0: 1.0, 1: 3.0}


def test_random_forest_factory_creates_classifier():
    model = RandomForestFactory().create()

    assert isinstance(model, RandomForestClassifier)


def test_random_forest_factory_sets_class_weight_when_scale_pos_weight_provided():
    model = RandomForestFactory().create(scale_pos_weight=5.0)

    assert model.get_params()["class_weight"] == {0: 1.0, 1: 5.0}


def test_xgboost_factory_creates_classifier():
    model = XGBoostFactory().create()

    assert isinstance(model, XGBClassifier)


def test_xgboost_factory_sets_scale_pos_weight_when_provided():
    model = XGBoostFactory().create(scale_pos_weight=6.0)

    assert model.get_params()["scale_pos_weight"] == 6.0


def test_xgboost_factory_omits_scale_pos_weight_when_none():
    model = XGBoostFactory().create()

    # xgboost get_params() returns None for unset params
    assert model.get_params().get("scale_pos_weight") is None


def test_factory_map_contains_all_models():
    assert set(FACTORY_MAP.keys()) == {"lightgbm", "logistic-regression", "decision-tree", "random-forest", "xgboost"}


def test_factory_map_values_are_factory_classes():
    for name, factory_cls in FACTORY_MAP.items():
        assert callable(factory_cls), f"{name} must be callable"
        instance = factory_cls()
        assert hasattr(instance, "create"), f"{name} factory missing create()"


def test_factory_map_each_instantiation_is_fresh():
    cls = FACTORY_MAP["logistic-regression"]
    assert cls() is not cls()
