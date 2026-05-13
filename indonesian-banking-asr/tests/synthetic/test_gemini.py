from indonesian_banking_asr.synthetic.gemini import build_paraphrase_prompt, parse_gemini_json_array


def test_build_paraphrase_prompt_contains_entity_preservation_rules():
    prompt = build_paraphrase_prompt(
        text="Saya mau cek cicilan kartu kredit sebesar Rp1.250.000.",
        required_entities=["cicilan", "kartu kredit", "Rp1.250.000"],
        variant_count=5,
    )

    assert "Rewrite this Indonesian banking call-center utterance into 5 natural variants." in prompt
    assert "Keep exact entity strings unchanged." in prompt
    assert "Do not add new account numbers." in prompt
    assert "Output JSON array only." in prompt
    assert "- cicilan" in prompt
    assert "- kartu kredit" in prompt
    assert "- Rp1.250.000" in prompt


def test_parse_gemini_json_array_accepts_string_array_only():
    assert parse_gemini_json_array('["a", "b"]') == ["a", "b"]

    try:
        parse_gemini_json_array('{"text": "a"}')
    except ValueError as error:
        assert "JSON array" in str(error)
    else:
        raise AssertionError("non-array Gemini output should fail")
