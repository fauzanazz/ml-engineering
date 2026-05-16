from __future__ import annotations

import re


LEXICON_REPLACEMENTS = {
    "bi fast": "BI-FAST",
    "beifast": "BI-FAST",
    "bifas": "BI-FAST",
    "bi fas": "BI-FAST",
    "bevas": "BI-FAST",
    "krisaya": "QRIS saya",
    "chris": "QRIS",
    "kris": "QRIS",
    "keris": "QRIS",
    "bloket": "blocked",
    "pelater": "paylater",
    "pilater": "paylater",
    "sopiko": "Shopee",
    "sopi": "Shopee",
}


def postprocess_transcript(transcript: str) -> str:
    processed = _normalize_rupiah_amounts(transcript)
    processed = _collapse_hyphenated_digits(processed)
    processed = _collapse_grouped_digits(processed)
    processed = _collapse_spaced_digits(processed)
    processed = _normalize_rupiah_amounts(processed)
    processed = _replace_lexicon(processed)
    return processed


def _collapse_hyphenated_digits(text: str) -> str:
    pattern = re.compile(r"(?<!\d)(?:\d+-)+\d+(?!\d)")
    return pattern.sub(lambda match: match.group(0).replace("-", ""), text)


def _collapse_grouped_digits(text: str) -> str:
    card_last4_pattern = re.compile(r"(?<!\w)\d[.]\d{3}(?!\w)")
    processed = card_last4_pattern.sub(lambda match: match.group(0).replace(".", ""), text)
    pattern = re.compile(r"(?<!\w)(?:\d{1,4}[\s.-]+){2,}\d{2,6}(?!\w)")
    return pattern.sub(lambda match: re.sub(r"[\s.-]+", "", match.group(0)), processed)


def _collapse_spaced_digits(text: str) -> str:
    pattern = re.compile(r"(?<!\d)(?:\d\s+){3,}\d(?!\d)")
    return pattern.sub(lambda match: re.sub(r"\s+", "", match.group(0)), text)


def _normalize_rupiah_amounts(text: str) -> str:
    processed = re.sub(r"\bRp\s+", "Rp", text, flags=re.IGNORECASE)
    compact_pattern = re.compile(r"\bRP(\d{5,})\b", flags=re.IGNORECASE)
    processed = compact_pattern.sub(lambda match: f"Rp{_format_rupiah_digits(match.group(1))}", processed)
    spoken_pattern = re.compile(r"(?<!\w)(\d{1,3}(?:\.\d{3})+)\s+rupiah\b", flags=re.IGNORECASE)
    return spoken_pattern.sub(lambda match: f"Rp{match.group(1)}", processed)


def _format_rupiah_digits(digits: str) -> str:
    groups = []
    remaining = digits
    while remaining:
        groups.append(remaining[-3:])
        remaining = remaining[:-3]
    return ".".join(reversed(groups))


def _replace_lexicon(text: str) -> str:
    processed = text
    processed = re.sub(r"\bchris(?=\d)", "QRIS ", processed, flags=re.IGNORECASE)
    for source, target in LEXICON_REPLACEMENTS.items():
        processed = re.sub(rf"\b{re.escape(source)}\b", target, processed, flags=re.IGNORECASE)
    return processed
