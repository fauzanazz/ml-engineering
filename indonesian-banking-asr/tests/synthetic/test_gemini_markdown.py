from indonesian_banking_asr.synthetic.gemini import parse_gemini_json_array


def test_parse_gemini_json_array_strips_markdown_code_fence():
    raw = '```json\n["satu", "dua"]\n```'

    assert parse_gemini_json_array(raw) == ["satu", "dua"]
