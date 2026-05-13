from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntitySpec:
    type: str
    slot: str
    value: str | None = None


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    intent: str
    text: str
    entities: tuple[EntitySpec, ...]


@dataclass(frozen=True)
class RenderedUtterance:
    template_id: str
    intent: str
    text: str
    entities: list[dict[str, int | str]]


def render_template(template: TemplateSpec, values: dict[str, str]) -> RenderedUtterance:
    replacements = _build_replacements(template, values)
    text = template.text.format(**replacements)
    entities = [_label_entity(text, spec, replacements) for spec in template.entities]

    return RenderedUtterance(
        template_id=template.template_id,
        intent=template.intent,
        text=text,
        entities=entities,
    )


def _build_replacements(template: TemplateSpec, values: dict[str, str]) -> dict[str, str]:
    replacements = dict(values)
    for entity in template.entities:
        if entity.value is not None:
            replacements[entity.slot] = entity.value
        elif entity.slot not in replacements:
            raise ValueError(f"missing value for slot: {entity.slot}")
    return replacements


def _label_entity(
    text: str,
    entity: EntitySpec,
    replacements: dict[str, str],
) -> dict[str, int | str]:
    entity_text = replacements[entity.slot]
    start = text.index(entity_text)
    end = start + len(entity_text)
    return {
        "type": entity.type,
        "text": entity_text,
        "start_char": start,
        "end_char": end,
    }
