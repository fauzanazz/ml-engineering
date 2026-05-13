from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from indonesian_banking_asr.synthetic.generator import EntitySpec, TemplateSpec


def load_template_catalog(path: Path) -> list[TemplateSpec]:
    raw_templates = yaml.safe_load(path.read_text())
    if not isinstance(raw_templates, list):
        raise ValueError("template catalog must be a YAML list")

    return [_parse_template(raw_template) for raw_template in raw_templates]


def _parse_template(raw_template: dict[str, Any]) -> TemplateSpec:
    return TemplateSpec(
        template_id=raw_template["template_id"],
        intent=raw_template["intent"],
        text=raw_template["text"],
        entities=tuple(_parse_entity(raw_entity) for raw_entity in raw_template["entities"]),
    )


def _parse_entity(raw_entity: dict[str, Any]) -> EntitySpec:
    return EntitySpec(
        type=raw_entity["type"],
        slot=raw_entity["slot"],
        value=raw_entity.get("value"),
    )
