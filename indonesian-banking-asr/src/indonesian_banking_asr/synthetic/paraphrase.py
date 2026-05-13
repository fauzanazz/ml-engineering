from __future__ import annotations

from typing import Protocol

from indonesian_banking_asr.synthetic.gemini import build_paraphrase_prompt
from indonesian_banking_asr.synthetic.validation import validate_paraphrases


class Paraphraser(Protocol):
    def generate_paraphrases(self, prompt: str) -> list[str]: ...


class DryRunParaphraser:
    def generate_paraphrases(self, prompt: str) -> list[str]:
        return [_extract_input_text(prompt)]


class RateLimitedParaphraser:
    def __init__(self, paraphraser: Paraphraser, rate_limiter) -> None:
        self.paraphraser = paraphraser
        self.rate_limiter = rate_limiter

    def generate_paraphrases(self, prompt: str) -> list[str]:
        self.rate_limiter.wait_before_request()
        return self.paraphraser.generate_paraphrases(prompt)


def paraphrase_rows(
    rows: list[dict],
    paraphraser: Paraphraser,
    variant_count: int,
) -> tuple[list[dict], list[dict]]:
    accepted, rejected, _raw = paraphrase_rows_with_audit(
        rows,
        paraphraser=paraphraser,
        variant_count=variant_count,
        continue_on_error=False,
    )
    return accepted, rejected


def paraphrase_rows_with_audit(
    rows: list[dict],
    paraphraser: Paraphraser,
    variant_count: int,
    continue_on_error: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    raw_rows: list[dict] = []

    for row in rows:
        required_entities = [entity["text"] for entity in row["entities"]]
        prompt = build_paraphrase_prompt(
            text=row["text"],
            required_entities=required_entities,
            variant_count=variant_count,
        )
        try:
            variants = paraphraser.generate_paraphrases(prompt)
        except Exception as error:
            if not continue_on_error:
                raise
            reason = f"paraphrase generation failed: {error}"
            rejected_rows.append(
                {
                    "source_utterance_id": row["utterance_id"],
                    "text": "",
                    "reason": reason,
                }
            )
            raw_rows.append(
                {
                    "source_utterance_id": row["utterance_id"],
                    "status": "error",
                    "prompt": prompt,
                    "error": str(error),
                }
            )
            continue

        raw_rows.append(
            {
                "source_utterance_id": row["utterance_id"],
                "status": "ok",
                "prompt": prompt,
                "raw_variants": variants,
            }
        )
        accepted, rejected = validate_paraphrases(
            variants,
            required_entities=required_entities,
            source_text=row["text"],
        )

        for index, text in enumerate(accepted, start=1):
            accepted_rows.append(_build_paraphrase_row(row, text, index))

        for rejected_item in rejected:
            rejected_rows.append(
                {
                    "source_utterance_id": row["utterance_id"],
                    "text": rejected_item.text,
                    "reason": rejected_item.reason,
                }
            )

    return accepted_rows, rejected_rows, raw_rows


def _extract_input_text(prompt: str) -> str:
    marker = 'Input:\n"'
    start = prompt.index(marker) + len(marker)
    end = prompt.index('"', start)
    return prompt[start:end]


def _build_paraphrase_row(source_row: dict, text: str, index: int) -> dict:
    row = dict(source_row)
    row["utterance_id"] = source_row["utterance_id"].replace("_p00", f"_p{index:02d}")
    row["text"] = text
    row["source"] = "template_gemini"
    row["generator"] = {
        **source_row["generator"],
        "llm": "gemini",
        "prompt_version": "v1",
    }
    row["entities"] = [_relabel_entity(text, entity) for entity in source_row["entities"]]
    return row


def _relabel_entity(text: str, entity: dict) -> dict:
    entity_text = entity["text"]
    start = text.index(entity_text)
    return {
        **entity,
        "start_char": start,
        "end_char": start + len(entity_text),
    }
