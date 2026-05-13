from indonesian_banking_asr.synthetic.validation import validate_paraphrases


def test_validate_paraphrases_accepts_exact_entity_preservation():
    variants = [
        "Saya mau tanya cicilan kartu kredit bulan ini Rp1.250.000.",
        "Saya ingin check cicilan kartu kredit bulan ini amount-nya Rp1.250.000.",
    ]

    accepted, rejected = validate_paraphrases(
        variants,
        required_entities=["cicilan", "kartu kredit", "Rp1.250.000"],
        source_text="Saya mau cek cicilan kartu kredit bulan ini Rp1.250.000.",
    )

    assert accepted == variants
    assert rejected == []


def test_validate_paraphrases_rejects_missing_or_changed_entities():
    variants = [
        "Saya mau tanya cicilan kartu kredit bulan ini Rp1.250.000.",
        "Saya mau tanya angsuran kartu kredit bulan ini Rp1.250.000.",
        "Saya mau tanya cicilan kartu debit bulan ini Rp1.250.000.",
        "Saya mau tanya cicilan kartu kredit bulan ini Rp2.000.000.",
    ]

    accepted, rejected = validate_paraphrases(
        variants,
        required_entities=["cicilan", "kartu kredit", "Rp1.250.000"],
        source_text="Saya mau cek cicilan kartu kredit bulan ini Rp1.250.000.",
    )

    assert accepted == ["Saya mau tanya cicilan kartu kredit bulan ini Rp1.250.000."]
    assert [item.text for item in rejected] == variants[1:]
    assert [item.reason for item in rejected] == [
        "missing required entity: cicilan",
        "missing required entity: kartu kredit",
        "missing required entity: Rp1.250.000",
    ]


def test_validate_paraphrases_rejects_extra_numbers():
    accepted, rejected = validate_paraphrases(
        ["Saya mau cek rekening 1234567890 dan nomor 9876543210."],
        required_entities=["1234567890"],
        source_text="Saya mau cek rekening 1234567890.",
    )

    assert accepted == []
    assert rejected[0].reason == "extra numeric token: 9876543210"
