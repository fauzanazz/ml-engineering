from indonesian_banking_asr.synthetic.gemini import parse_gemini_json_array


def test_parse_gemini_json_array_extracts_array_from_partial_fence_text():
    raw = '```json\n["satu", "dua"]'

    assert parse_gemini_json_array(raw) == ["satu", "dua"]
