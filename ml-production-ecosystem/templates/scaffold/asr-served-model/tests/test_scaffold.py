from {{package_name}}.metrics import word_error_rate
from {{package_name}}.transcribe import transcribe


def test_transcribe_returns_text() -> None:
    assert transcribe("sample.wav") == "transcript for sample.wav"


def test_word_error_rate_counts_word_edits() -> None:
    assert word_error_rate("cek saldo saya", "cek mutasi saya") == 1 / 3
