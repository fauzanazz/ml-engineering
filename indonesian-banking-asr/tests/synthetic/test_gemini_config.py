import os

from indonesian_banking_asr.synthetic.gemini import GeminiConfig, load_gemini_config


def test_load_gemini_config_reads_env_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    config = load_gemini_config()

    assert config == GeminiConfig(api_key="secret-key", model="gemini-2.5-flash")
    assert "secret-key" not in repr(config)


def test_load_gemini_config_defaults_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert load_gemini_config().model == "gemini-2.5-flash"


def test_load_gemini_config_fails_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    try:
        load_gemini_config()
    except RuntimeError as error:
        assert "GEMINI_API_KEY" in str(error)
    else:
        raise AssertionError("missing Gemini API key should fail")
