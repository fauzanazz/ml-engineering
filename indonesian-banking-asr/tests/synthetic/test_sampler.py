from indonesian_banking_asr.synthetic.sampler import EntitySampler


def test_entity_sampler_is_deterministic_for_same_seed():
    first = EntitySampler(seed=42).sample_values()
    second = EntitySampler(seed=42).sample_values()

    assert first == second


def test_entity_sampler_generates_generic_banking_values_in_expected_ranges():
    values = EntitySampler(seed=7).sample_values()

    assert values["product_name"] in {
        "rekening tabungan",
        "rekening giro",
        "deposito",
        "kartu kredit",
        "kartu debit",
        "KPR",
        "KTA",
        "pinjaman multiguna",
        "cicilan kendaraan",
        "virtual account",
        "BI-FAST",
        "mobile banking",
        "internet banking",
        "QRIS",
        "paylater",
    }
    assert values["account_number"].isdigit()
    assert 10 <= len(values["account_number"]) <= 14
    assert values["amount"].startswith("Rp")
    assert values["card_last4"].isdigit()
    assert len(values["card_last4"]) == 4
    assert values["tenor"].endswith(" bulan")
