from __future__ import annotations

import re
from dataclasses import dataclass


_NUMERIC_TOKEN_PATTERN = re.compile(r"(?:Rp)?\d(?:[\d.,]*\d)?%?")


@dataclass(frozen=True)
class RejectedParaphrase:
    text: str
    reason: str


def validate_paraphrases(
    variants: list[str],
    required_entities: list[str],
    source_text: str,
) -> tuple[list[str], list[RejectedParaphrase]]:
    accepted: list[str] = []
    rejected: list[RejectedParaphrase] = []
    source_numeric_tokens = set(_numeric_tokens(source_text))

    for variant in variants:
        rejection_reason = _first_rejection_reason(
            variant,
            required_entities,
            source_numeric_tokens,
        )
        if rejection_reason is None:
            accepted.append(variant)
        else:
            rejected.append(RejectedParaphrase(text=variant, reason=rejection_reason))

    return accepted, rejected


def _first_rejection_reason(
    text: str,
    required_entities: list[str],
    source_numeric_tokens: set[str],
) -> str | None:
    for entity in required_entities:
        if entity not in text:
            return f"missing required entity: {entity}"

    for token in _numeric_tokens(text):
        if token not in source_numeric_tokens:
            return f"extra numeric token: {token}"

    return None


def _numeric_tokens(text: str) -> list[str]:
    return _NUMERIC_TOKEN_PATTERN.findall(text)
