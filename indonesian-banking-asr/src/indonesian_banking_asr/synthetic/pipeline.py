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
    samples_per_template: int = 1,
) -> list[dict]:
    templates = load_template_catalog(catalog_path)
    rows: list[dict] = []

    if samples_per_template < 1:
        raise ValueError("samples_per_template must be at least 1")

    row_index = 0
    for template_index, template in enumerate(templates):
        for sample_index in range(samples_per_template):
            if limit is not None and len(rows) >= limit:
                break
            row_index += 1
            values = EntitySampler(seed=seed + template_index * samples_per_template + sample_index).sample_values()
            rendered = render_template(template, values=values)
            row = build_manifest_row(
                rendered,
                utterance_id=f"syn_id_{template.template_id}_{row_index:06d}_s{sample_index:02d}_p00",
                language_mix="id",
                source="template",
            )
            row["template_sample_index"] = sample_index
            row["split"] = assign_split(
                template.template_id,
                values["account_number"],
                values["amount"],
            )
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break

    return rows
