from lightgbm import LGBMClassifier

from fraud_detection.models import LightGbmFactory


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
