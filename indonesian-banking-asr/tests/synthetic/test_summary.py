from indonesian_banking_asr.synthetic.summary import build_generation_summary


def test_build_generation_summary_counts_batch_outputs():
    summary = build_generation_summary(
        canonical_rows=[{"split": "train"}, {"split": "test"}],
        pending_rows=[{"utterance_id": "row-2"}],
        accepted_rows=[{"utterance_id": "row-2_p01"}, {"utterance_id": "row-2_p02"}],
        rejected_rows=[{"reason": "missing required entity: saldo"}],
        raw_rows=[{"status": "ok"}, {"status": "error"}],
        skipped_count=1,
    )

    assert summary == {
        "canonical_rows": 2,
        "pending_rows": 1,
        "skipped_rows": 1,
        "accepted_rows": 2,
        "rejected_rows": 1,
        "raw_rows": 2,
        "raw_status_counts": {"ok": 1, "error": 1},
        "split_counts": {"train": 1, "test": 1},
    }
