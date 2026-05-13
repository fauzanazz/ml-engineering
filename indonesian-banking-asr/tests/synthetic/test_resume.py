from pathlib import Path

from indonesian_banking_asr.synthetic.resume import filter_pending_rows, read_processed_utterance_ids


def test_read_processed_utterance_ids_reads_accepted_rejected_and_raw(tmp_path):
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    raw = tmp_path / "raw.jsonl"
    accepted.write_text('{"source_utterance_id":"row-1"}\n{"utterance_id":"row-2_p01"}\n')
    rejected.write_text('{"source_utterance_id":"row-3"}\n')
    raw.write_text('{"source_utterance_id":"row-4"}\n')

    processed = read_processed_utterance_ids([accepted, rejected, raw])

    assert processed == {"row-1", "row-2", "row-3", "row-4"}


def test_filter_pending_rows_skips_processed_source_rows():
    rows = [
        {"utterance_id": "row-1"},
        {"utterance_id": "row-2"},
        {"utterance_id": "row-3"},
    ]

    assert filter_pending_rows(rows, processed_utterance_ids={"row-1", "row-3"}) == [
        {"utterance_id": "row-2"}
    ]
