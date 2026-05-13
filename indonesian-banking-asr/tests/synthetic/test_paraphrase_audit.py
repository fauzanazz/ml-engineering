from indonesian_banking_asr.synthetic.paraphrase import paraphrase_rows_with_audit


class MixedParaphraser:
    def __init__(self):
        self.calls = 0

    def generate_paraphrases(self, prompt):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Gemini temporary failure")
        return ["Saya mau cek saldo rekening tabungan nomor 1234567890."]


def _row(utterance_id, text):
    return {
        "utterance_id": utterance_id,
        "template_id": "check_balance_001",
        "intent": "check_balance",
        "text": text,
        "language_mix": "id",
        "source": "template",
        "generator": {"template_version": "v1", "llm": None, "prompt_version": None},
        "entities": [
            {"type": "BANKING_TERM", "text": "saldo", "start_char": 13, "end_char": 18},
            {"type": "PRODUCT_NAME", "text": "rekening tabungan", "start_char": 19, "end_char": 36},
            {"type": "ACCOUNT_NUMBER", "text": "1234567890", "start_char": 43, "end_char": 53},
        ],
        "split_group": "check_balance_001",
        "split": "train",
    }


def test_paraphrase_rows_with_audit_continues_after_row_error():
    rows = [
        _row("syn_id_check_balance_001_000001_p00", "Saya mau cek saldo rekening tabungan nomor 1234567890."),
        _row("syn_id_check_balance_001_000002_p00", "Saya mau cek saldo rekening tabungan nomor 1234567890."),
    ]

    accepted, rejected, raw = paraphrase_rows_with_audit(
        rows,
        paraphraser=MixedParaphraser(),
        variant_count=1,
        continue_on_error=True,
    )

    assert len(accepted) == 1
    assert rejected == [
        {
            "source_utterance_id": "syn_id_check_balance_001_000001_p00",
            "text": "",
            "reason": "paraphrase generation failed: Gemini temporary failure",
        }
    ]
    assert raw[0]["status"] == "error"
    assert raw[1]["status"] == "ok"
    assert raw[1]["raw_variants"] == ["Saya mau cek saldo rekening tabungan nomor 1234567890."]
