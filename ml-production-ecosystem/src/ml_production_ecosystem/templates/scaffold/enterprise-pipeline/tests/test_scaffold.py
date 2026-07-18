from {{package_name}}.pipeline import planned_stages
from {{package_name}}.quality_gate import approve


def test_pipeline_has_production_stages() -> None:
    assert planned_stages()[-2:] == ("monitoring", "rollback")


def test_quality_gate_uses_minimum_value() -> None:
    assert approve(0.91, 0.9) is True
    assert approve(0.89, 0.9) is False
