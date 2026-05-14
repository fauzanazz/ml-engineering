import json

from indonesian_banking_asr.synthetic.pipeline import generate_manifest_rows


def test_generate_manifest_rows_supports_multiple_samples_per_template():
    rows = generate_manifest_rows(seed=123, limit=15, samples_per_template=2)

    assert len(rows) == 15
    assert rows[0]["template_sample_index"] == 0
    assert rows[1]["template_id"] == rows[0]["template_id"]
    assert rows[1]["template_sample_index"] == 1
    assert rows[0]["utterance_id"] != rows[1]["utterance_id"]
    assert rows[0]["text"] != rows[1]["text"]


def test_generate_manifest_rows_renders_catalog_with_sampled_entities():
    rows = generate_manifest_rows(seed=123, limit=3)

    assert len(rows) == 3
    assert rows[0]["utterance_id"].startswith("syn_id_")
    assert rows[0]["template_id"]
    assert rows[0]["intent"]
    assert rows[0]["text"]
    assert rows[0]["split"] in {"train", "validation", "test"}
    assert rows[0]["entities"]
    json.dumps(rows[0])
