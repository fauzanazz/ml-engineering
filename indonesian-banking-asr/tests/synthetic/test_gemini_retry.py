from indonesian_banking_asr.synthetic.gemini import GeminiClient, GeminiConfig, RetryableGeminiError


class FlakyTransport:
    def __init__(self):
        self.calls = 0

    def post_json(self, url, payload, timeout_seconds):
        self.calls += 1
        if self.calls < 3:
            raise RetryableGeminiError("temporary quota")
        return {
            "candidates": [
                {"content": {"parts": [{"text": '["variant after retry"]'}]}}
            ]
        }


def test_gemini_client_retries_retryable_errors_with_backoff():
    sleeps = []
    transport = FlakyTransport()
    client = GeminiClient(
        config=GeminiConfig(api_key="secret-key", model="gemini-2.5-flash"),
        transport=transport,
        max_retries=3,
        sleep=sleeps.append,
    )

    assert client.generate_paraphrases("prompt") == ["variant after retry"]
    assert transport.calls == 3
    assert sleeps == [1.0, 2.0]
