import pytest

from indonesian_banking_asr.evaluation import router


def prediction(utterance_id: str, hypothesis: str, model: str = "model") -> dict:
    return {
        "utterance_id": utterance_id,
        "hypothesis": hypothesis,
        "model": model,
        "language": "id",
    }


def test_route_prediction_rows_contract_and_postprocess_before_match():
    rows = router.route_prediction_rows(
        [prediction("u1", "saldo BIFAS nomor 0 4 3 3", "general")],
        [prediction("u1", "saldo BI fast nomor 0433", "banking")],
    )

    assert rows == [
        {
            "utterance_id": "u1",
            "hypothesis": "saldo BI-FAST nomor 0433",
            "raw_hypothesis": "saldo BI fast nomor 0433",
            "model": "banking",
            "language": "id",
            "postprocess": "banking_entity_v2",
            "router": "keyword_router_v1",
            "route": "banking",
            "matched_keyword": "saldo",
            "baseline_raw_hypothesis": "saldo BIFAS nomor 0 4 3 3",
            "baseline_hypothesis": "saldo BI-FAST nomor 0433",
        }
    ]


def test_postprocessing_happens_before_keyword_match():
    rows = router.route_prediction_rows(
        [prediction("u1", "BIFAS")],
        [prediction("u1", "BI-FAST", "banking")],
    )

    assert rows[0]["route"] == "banking"
    assert rows[0]["matched_keyword"] == "BI-FAST"


def test_empty_baseline_and_extra_banking_rows_are_ignored():
    assert router.route_prediction_rows([], [prediction("extra", "saldo")]) == []


def test_missing_banking_prediction_for_match_is_an_error():
    with pytest.raises(ValueError, match="^missing banking prediction for routed utterance: u1$"):
        router.route_prediction_rows([prediction("u1", "cicilan")], [])


def test_transcribe_routed_audio_general_calls_only_baseline(monkeypatch):
    calls = []

    def fake_transcribe(rows, *, model, language):
        calls.append((rows, model, language))
        return [prediction(rows[0]["utterance_id"], "halo dunia", model)]

    monkeypatch.setattr(router, "transcribe_manifest_rows", fake_transcribe)

    result = router.transcribe_routed_audio("audio/sample.wav", general_model="general", banking_model="banking")

    assert result["route"] == "general"
    assert result["hypothesis"] == "halo dunia"
    assert calls == [([{"utterance_id": "sample", "audio_path": "audio/sample.wav"}], "general", "id")]


def test_transcribe_routed_audio_banking_calls_baseline_then_step37(monkeypatch):
    calls = []

    def fake_transcribe(rows, *, model, language):
        calls.append(model)
        text = "Saya cek cicilan RP11070000" if model == "general" else "Saya cek cicilan Rp11.070.000"
        return [prediction(rows[0]["utterance_id"], text, model)]

    monkeypatch.setattr(router, "transcribe_manifest_rows", fake_transcribe)

    result = router.transcribe_routed_audio("sample.wav", general_model="general", banking_model="step37")

    assert calls == ["general", "step37"]
    assert result["route"] == "banking"
    assert result["matched_keyword"] == "cicilan"
    assert result["model"] == "step37"
    assert result["hypothesis"] == "Saya cek cicilan Rp11.070.000"
