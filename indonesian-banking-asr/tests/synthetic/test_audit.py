import json

from indonesian_banking_asr.synthetic.audit import write_jsonl


def test_write_jsonl_creates_parent_and_writes_rows(tmp_path):
    output_path = tmp_path / "audit" / "accepted.jsonl"

    write_jsonl(output_path, [{"a": 1}, {"b": "dua"}])

    assert [json.loads(line) for line in output_path.read_text().splitlines()] == [
        {"a": 1},
        {"b": "dua"},
    ]
