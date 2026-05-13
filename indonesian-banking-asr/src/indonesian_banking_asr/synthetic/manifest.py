from __future__ import annotations

from typing import Any

from indonesian_banking_asr.synthetic.generator import RenderedUtterance


def build_manifest_row(
    rendered: RenderedUtterance,
    utterance_id: str,
    language_mix: str,
    source: str,
    template_version: str = "v1",
    llm: str | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    return {
        "utterance_id": utterance_id,
        "template_id": rendered.template_id,
        "intent": rendered.intent,
        "text": rendered.text,
        "language_mix": language_mix,
        "source": source,
        "generator": {
            "template_version": template_version,
            "llm": llm,
            "prompt_version": prompt_version,
        },
        "entities": rendered.entities,
        "split_group": rendered.template_id,
    }
