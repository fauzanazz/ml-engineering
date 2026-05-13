import json

from indonesian_banking_asr.synthetic.gemini import GeminiClient, GeminiConfig


class FakeTransport:
    def __init__(self):
        self.request = None

    def post_json(self, url, payload, timeout_seconds):
        self.request = {
            "url": url,
            "payload": payload,
            "timeout_seconds": timeout_seconds,
        }
        return {
            "candidates": [
                {"content": {"parts": [{"text": '["variant satu", "variant dua"]'}]}}
            ]
        }


def test_gemini_client_calls_generate_content_without_logging_key():
    transport = FakeTransport()
    client = GeminiClient(
        config=GeminiConfig(api_key="secret-key", model="gemini-2.5-flash"),
        transport=transport,
    )

    variants = client.generate_paraphrases("prompt text")

    assert variants == ["variant satu", "variant dua"]
    assert "gemini-2.5-flash:generateContent" in transport.request["url"]
    assert "secret-key" in transport.request["url"]
    assert transport.request["payload"]["contents"][0]["parts"][0]["text"] == "prompt text"
    assert transport.request["payload"]["generationConfig"]["temperature"] == 0.7
    assert transport.request["payload"]["generationConfig"]["maxOutputTokens"] == 2048
