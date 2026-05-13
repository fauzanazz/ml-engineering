import json

from indonesian_banking_asr.synthetic.manifest import build_manifest_row
from indonesian_banking_asr.synthetic.generator import RenderedUtterance


def test_build_manifest_row_keeps_generation_metadata():
    rendered = RenderedUtterance(
        template_id="check_installment_001",
        intent="check_installment",
        text="Saya mau cek cicilan kartu kredit sebesar Rp1.250.000.",
        entities=[
            {"type": "BANKING_TERM", "text": "cicilan", "start_char": 13, "end_char": 20},
            {"type": "PRODUCT_NAME", "text": "kartu kredit", "start_char": 21, "end_char": 33},
            {"type": "AMOUNT", "text": "Rp1.250.000", "start_char": 42, "end_char": 54},
        ],
    )

    row = build_manifest_row(
        rendered,
        utterance_id="syn_id_check_installment_001_000001_p00",
        language_mix="id",
        source="template",
    )

    assert row == {
        "utterance_id": "syn_id_check_installment_001_000001_p00",
        "template_id": "check_installment_001",
        "intent": "check_installment",
        "text": "Saya mau cek cicilan kartu kredit sebesar Rp1.250.000.",
        "language_mix": "id",
        "source": "template",
        "generator": {
            "template_version": "v1",
            "llm": None,
            "prompt_version": None,
        },
        "entities": rendered.entities,
        "split_group": "check_installment_001",
    }

    json.dumps(row)
