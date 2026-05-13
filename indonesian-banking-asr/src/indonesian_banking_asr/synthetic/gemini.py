from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str = "gemini-2.5-flash"

    def __repr__(self) -> str:
        return f"GeminiConfig(api_key='***', model={self.model!r})"


def load_gemini_config() -> GeminiConfig:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required")
    return GeminiConfig(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    )


class RetryableGeminiError(RuntimeError):
    pass


class GeminiTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]: ...


class UrllibGeminiTransport:
    def post_json(self, url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in {429, 500, 502, 503, 504}:
                raise RetryableGeminiError(f"Gemini HTTP {error.code}") from error
            raise


class GeminiClient:
    def __init__(
        self,
        config: GeminiConfig,
        transport: GeminiTransport | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibGeminiTransport()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep = sleep

    def generate_paraphrases(self, prompt: str) -> list[str]:
        response = self._post_with_retry(
            self._generate_content_url(),
            payload={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
            },
            timeout_seconds=self.timeout_seconds,
        )
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        return parse_gemini_json_array(text)

    def _post_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.transport.post_json(url, payload, timeout_seconds)
            except RetryableGeminiError:
                if attempt == self.max_retries:
                    raise
                self.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError("unreachable retry state")

    def _generate_content_url(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={self.config.api_key}"
        )


def build_paraphrase_prompt(
    text: str,
    required_entities: list[str],
    variant_count: int,
) -> str:
    entities = "\n".join(f"- {entity}" for entity in required_entities)
    return f"""Rewrite this Indonesian banking call-center utterance into {variant_count} natural variants.

Rules:
- Keep exact entity strings unchanged.
- Do not change numbers.
- Do not add new account numbers.
- Do not add new money amounts.
- Keep same meaning.
- Include casual spoken Indonesian.
- Include 1 variant with Indonesian-English code-switching.
- Output JSON array only.

Input:
"{text}"

Entities that must stay exact:
{entities}
"""


def parse_gemini_json_array(raw_output: str) -> list[str]:
    parsed = json.loads(_strip_markdown_fence(raw_output))
    if not isinstance(parsed, list):
        raise ValueError("Gemini output must be a JSON array")
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError("Gemini output JSON array must contain strings only")
    return parsed


def _strip_markdown_fence(raw_output: str) -> str:
    stripped = raw_output.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()
