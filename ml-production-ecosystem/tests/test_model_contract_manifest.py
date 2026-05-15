from pathlib import Path
import json

from production_patterns.model_contract_manifest import build_model_contract_manifest


def test_model_contract_manifest_accepts_existing_schemas(tmp_path: Path) -> None:
    input_schema = tmp_path / "input.json"
    output_schema = tmp_path / "output.json"
    input_schema.write_text(json.dumps({"title": "Input"}))
    output_schema.write_text(json.dumps({"title": "Output"}))
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "manifest.json"
    config_path.write_text(
        f"""
model_contract:
  input_schema_uri: {input_schema}
  output_schema_uri: {output_schema}
  task_type: classification
  prediction_key: label
""".strip()
    )

    manifest = build_model_contract_manifest(config_path, output_path)

    assert manifest["status"] == "ready"
    assert manifest["missing_fields"] == []
    assert manifest["contract"]["task_type"] == "classification"
    assert manifest["schemas"]["input"]["valid_json"] is True
    assert json.loads(output_path.read_text()) == manifest


def test_model_contract_manifest_blocks_missing_fields_and_schemas(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_contract:
  input_schema_uri: missing-input.json
  task_type: ranking
""".strip()
    )

    manifest = build_model_contract_manifest(config_path, tmp_path / "manifest.json")

    assert manifest["status"] == "blocked"
    assert manifest["missing_fields"] == ["output_schema_uri", "prediction_key"]
    assert manifest["schemas"]["input"]["exists"] is False
