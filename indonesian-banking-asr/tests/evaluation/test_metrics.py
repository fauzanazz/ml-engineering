from indonesian_banking_asr.evaluation.metrics import compute_entity_error_rate, compute_wer


def test_compute_wer_counts_insertions_deletions_and_substitutions():
    assert compute_wer("saya mau cek saldo", "saya ingin cek") == {
        "word_errors": 2,
        "reference_words": 4,
        "wer": 0.5,
    }


def test_compute_wer_handles_empty_reference():
    assert compute_wer("", "saya") == {
        "word_errors": 1,
        "reference_words": 0,
        "wer": 1.0,
    }


def test_compute_entity_error_rate_counts_mismatched_entity_text():
    row = {
        "entities": [
            {"type": "ACCOUNT_NUMBER", "text": "1234567890"},
            {"type": "AMOUNT", "text": "dua juta rupiah"},
        ]
    }

    assert compute_entity_error_rate(row, "rekening 1234567890 sebesar dua ratus rupiah") == {
        "entity_errors": 1,
        "entities": 2,
        "entity_error_rate": 0.5,
        "errors_by_type": {"AMOUNT": 1},
    }


def test_compute_entity_error_rate_handles_no_entities():
    assert compute_entity_error_rate({"entities": []}, "saya mau cek saldo") == {
        "entity_errors": 0,
        "entities": 0,
        "entity_error_rate": 0.0,
        "errors_by_type": {},
    }
