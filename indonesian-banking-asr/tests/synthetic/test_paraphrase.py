from indonesian_banking_asr.synthetic.paraphrase import paraphrase_rows


class FakeParaphraser:
    def __init__(self):
        self.prompts = []

    def generate_paraphrases(self, prompt):
        self.prompts.append(prompt)
        return [
            "Saya mau tanya cicilan kartu kredit sebesar Rp1.250.000.",
            "Saya mau tanya angsuran kartu kredit sebesar Rp1.250.000.",
        ]


def test_paraphrase_rows_validates_and_audits_accepted_rejected_variants():
    rows = [
        {
            "utterance_id": "syn_id_check_installment_001_000001_p00",
            "template_id": "check_installment_001",
            "intent": "check_installment",
            "text": "Saya mau cek cicilan kartu kredit sebesar Rp1.250.000.",
            "language_mix": "id",
            "source": "template",
            "generator": {"template_version": "v1", "llm": None, "prompt_version": None},
            "entities": [
                {"type": "BANKING_TERM", "text": "cicilan", "start_char": 13, "end_char": 20},
                {"type": "PRODUCT_NAME", "text": "kartu kredit", "start_char": 21, "end_char": 33},
                {"type": "AMOUNT", "text": "Rp1.250.000", "start_char": 42, "end_char": 53},
            ],
            "split_group": "check_installment_001",
            "split": "train",
        }
    ]

    accepted, rejected = paraphrase_rows(rows, paraphraser=FakeParaphraser(), variant_count=2)

    assert len(accepted) == 1
    assert accepted[0]["utterance_id"] == "syn_id_check_installment_001_000001_p01"
    assert accepted[0]["source"] == "template_gemini"
    assert accepted[0]["generator"]["llm"] == "gemini"
    assert accepted[0]["text"] == "Saya mau tanya cicilan kartu kredit sebesar Rp1.250.000."
    assert accepted[0]["entities"][0]["start_char"] == 15

    assert len(rejected) == 1
    assert rejected[0]["source_utterance_id"] == "syn_id_check_installment_001_000001_p00"
    assert rejected[0]["reason"] == "missing required entity: cicilan"
