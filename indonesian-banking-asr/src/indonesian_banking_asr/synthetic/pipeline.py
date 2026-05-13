from __future__ import annotations

from pathlib import Path

from indonesian_banking_asr.synthetic.catalog import load_template_catalog
from indonesian_banking_asr.synthetic.generator import render_template
from indonesian_banking_asr.synthetic.manifest import build_manifest_row
from indonesian_banking_asr.synthetic.sampler import EntitySampler
from indonesian_banking_asr.synthetic.split import assign_split

DEFAULT_CATALOG_PATH = Path("data/templates/banking_intents.yaml")


def generate_manifest_rows(
    seed: int,
    limit: int | None = None,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
) -> list[dict]:
    templates = load_template_catalog(catalog_path)
    rows: list[dict] = []

    for index, template in enumerate(templates):
        if limit is not None and len(rows) >= limit:
            break
        values = EntitySampler(seed=seed + index).sample_values()
        rendered = render_template(template, values=values)
        row = build_manifest_row(
            rendered,
            utterance_id=f"syn_id_{template.template_id}_{index + 1:06d}_p00",
            language_mix="id",
            source="template",
        )
        row["split"] = assign_split(
            template.template_id,
            values["account_number"],
            values["amount"],
        )
        rows.append(row)

    return rows
