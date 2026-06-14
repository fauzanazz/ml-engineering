from indonesian_banking_asr.evaluation.postprocess_cli import _postprocess_prediction_row


def test_postprocess_prediction_row_preserves_raw_hypothesis():
    row = {"utterance_id": "u1", "hypothesis": "saldo BIFAS nomor 0 4 3 3"}

    processed = _postprocess_prediction_row(row)

    assert processed["raw_hypothesis"] == "saldo BIFAS nomor 0 4 3 3"
    assert processed["hypothesis"] == "saldo BI-FAST nomor 0433"
    assert processed["postprocess"] == "banking_entity_v2"
    assert processed["utterance_id"] == "u1"
