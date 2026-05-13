from indonesian_banking_asr.synthetic.paraphrase import DryRunParaphraser


def test_dry_run_paraphraser_returns_source_preserving_variants():
    paraphraser = DryRunParaphraser()

    variants = paraphraser.generate_paraphrases(
        'Input:\n"Saya mau cek cicilan kartu kredit sebesar Rp1.250.000."\n\nEntities that must stay exact:\n- cicilan\n- kartu kredit\n- Rp1.250.000\n'
    )

    assert variants == ["Saya mau cek cicilan kartu kredit sebesar Rp1.250.000."]
