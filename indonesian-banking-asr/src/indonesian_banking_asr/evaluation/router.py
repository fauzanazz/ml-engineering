from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from indonesian_banking_asr.evaluation.metrics import evaluate_predictions
from indonesian_banking_asr.evaluation.postprocess import postprocess_prediction_row
from indonesian_banking_asr.evaluation.whisper import DEFAULT_MODEL, transcribe_manifest_rows

ROUTER_NAME = "keyword_router_v1"
DEFAULT_BANKING_MODEL = "models/mlx-whisper-large-v3-fullmix-200step-lora-last8-r4-a8-lr1e-4-merged"
BANKING_KEYWORD_PATTERN = re.compile(
    r"\b(saldo|rekening|kartu|debit|kredit|pinjaman|kta|paylater|cicilan|qris|bi-fast|bifast|rtgs|transfer|virtual account|bpjs|bunga|interest rate|blokir|terblokir|pin|rupiah|rp\d|juta)\b",
    re.IGNORECASE,
)


def match_banking_keyword(transcript: str) -> str | None:
    match = BANKING_KEYWORD_PATTERN.search(transcript)
    return match.group(0) if match else None


def route_prediction_rows(baseline_rows: list[dict], banking_rows: list[dict]) -> list[dict]:
    banking_by_id = {row["utterance_id"]: row for row in banking_rows}
    routed_rows = []

    for baseline_row in baseline_rows:
        baseline = postprocess_prediction_row(baseline_row)
        matched_keyword = match_banking_keyword(baseline["hypothesis"])
        selected = baseline
        route = "general"

        if matched_keyword:
            utterance_id = baseline["utterance_id"]
            if utterance_id not in banking_by_id:
                raise ValueError(f"missing banking prediction for routed utterance: {utterance_id}")
            selected = postprocess_prediction_row(banking_by_id[utterance_id])
            route = "banking"

        routed_rows.append(
            {
                **selected,
                "router": ROUTER_NAME,
                "route": route,
                "matched_keyword": matched_keyword,
                "baseline_raw_hypothesis": baseline["raw_hypothesis"],
                "baseline_hypothesis": baseline["hypothesis"],
            }
        )

    return routed_rows


def transcribe_routed_audio(
    audio_path: str | Path,
    *,
    general_model: str = DEFAULT_MODEL,
    banking_model: str = DEFAULT_BANKING_MODEL,
    language: str = "id",
) -> dict:
    manifest_row = {"utterance_id": Path(audio_path).stem, "audio_path": str(audio_path)}
    baseline_rows = transcribe_manifest_rows([manifest_row], model=general_model, language=language)
    baseline = postprocess_prediction_row(baseline_rows[0])
    banking_rows = (
        transcribe_manifest_rows([manifest_row], model=banking_model, language=language)
        if match_banking_keyword(baseline["hypothesis"])
        else []
    )
    return route_prediction_rows(baseline_rows, banking_rows)[0]


def evaluate_router(
    manifest_rows: list[dict],
    banking_manifest_rows: list[dict],
    real_manifest_rows: list[dict],
    baseline_prediction_rows: list[dict],
    banking_prediction_rows: list[dict],
    *,
    split: str,
) -> dict:
    if any(row.get("split") != split for rows in (manifest_rows, banking_manifest_rows, real_manifest_rows) for row in rows):
        raise ValueError("manifest rows do not match --split")

    all_ids = {row["utterance_id"] for row in manifest_rows}
    banking_ids = {row["utterance_id"] for row in banking_manifest_rows}
    real_ids = {row["utterance_id"] for row in real_manifest_rows}
    if banking_ids & real_ids or banking_ids | real_ids != all_ids:
        raise ValueError("banking and real manifests must partition the combined manifest")

    baseline_ids = {row["utterance_id"] for row in baseline_prediction_rows}
    missing_ids = sorted(all_ids - baseline_ids)
    if missing_ids:
        raise ValueError(f"missing baseline predictions for: {','.join(missing_ids)}")

    baseline_v2 = [postprocess_prediction_row(row) for row in baseline_prediction_rows if row["utterance_id"] in all_ids]
    routed_v2 = route_prediction_rows(
        [row for row in baseline_prediction_rows if row["utterance_id"] in all_ids],
        banking_prediction_rows,
    )
    baseline_by_id = {row["utterance_id"]: row for row in baseline_v2}
    routed_by_id = {row["utterance_id"]: row for row in routed_v2}

    def prediction_slice(rows: list[dict], predictions: dict[str, dict]) -> list[dict]:
        return [predictions[row["utterance_id"]] for row in rows]

    def metrics(predictions: dict[str, dict]) -> dict:
        return {
            "all": evaluate_predictions(manifest_rows, prediction_slice(manifest_rows, predictions)),
            "banking_synthetic": evaluate_predictions(
                banking_manifest_rows, prediction_slice(banking_manifest_rows, predictions)
            ),
            "real_general": evaluate_predictions(real_manifest_rows, prediction_slice(real_manifest_rows, predictions)),
        }

    banking_routes = [routed_by_id[utterance_id]["route"] for utterance_id in banking_ids]
    real_routes = [routed_by_id[utterance_id]["route"] for utterance_id in real_ids]
    return {
        "router": ROUTER_NAME,
        "split": split,
        "datasets": {
            "all": {"rows": len(manifest_rows)},
            "banking_synthetic": {
                "rows": len(banking_manifest_rows),
                "sources": sorted({row["source"] for row in banking_manifest_rows}),
            },
            "real_general": {
                "rows": len(real_manifest_rows),
                "sources": sorted({row["source"] for row in real_manifest_rows}),
            },
        },
        "routing": {
            "banking_routed": banking_routes.count("banking"),
            "banking_missed": banking_routes.count("general"),
            "real_routed": real_routes.count("banking"),
            "real_kept_general": real_routes.count("general"),
        },
        "pipelines": {
            "baseline_v2": metrics(baseline_by_id),
            "keyword_router_v1_v2": metrics(routed_by_id),
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the keyword-routed banking ASR pipeline.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--banking-manifest-path", required=True, type=Path)
    parser.add_argument("--real-manifest-path", required=True, type=Path)
    parser.add_argument("--baseline-predictions-path", required=True, type=Path)
    parser.add_argument("--banking-predictions-path", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    report = evaluate_router(
        _read_jsonl(args.manifest_path),
        _read_jsonl(args.banking_manifest_path),
        _read_jsonl(args.real_manifest_path),
        _read_jsonl(args.baseline_predictions_path),
        _read_jsonl(args.banking_predictions_path),
        split=args.split,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
