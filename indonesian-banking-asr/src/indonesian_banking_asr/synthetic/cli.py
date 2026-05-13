from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from indonesian_banking_asr.synthetic.audio_qa import validate_audio_manifest_rows
from indonesian_banking_asr.synthetic.audit import write_jsonl
from indonesian_banking_asr.synthetic.augmentation import build_augmented_manifest_rows
from indonesian_banking_asr.synthetic.dataset import (
    build_dataset_summary,
    merge_audio_manifest_rows,
    validate_dataset_manifest_rows,
)
from indonesian_banking_asr.synthetic.gemini import GeminiClient, load_gemini_config
from indonesian_banking_asr.synthetic.paraphrase import DryRunParaphraser, RateLimitedParaphraser, paraphrase_rows_with_audit
from indonesian_banking_asr.synthetic.pipeline import generate_manifest_rows
from indonesian_banking_asr.synthetic.rate_limit import RateLimiter
from indonesian_banking_asr.synthetic.resume import filter_pending_rows, read_processed_utterance_ids
from indonesian_banking_asr.synthetic.summary import build_generation_summary
from indonesian_banking_asr.synthetic.tts import GeminiTts, SyntheticToneTts, build_audio_manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic banking ASR dataset artifacts.")
    subparsers = parser.add_subparsers(dest="command")

    tts_parser = subparsers.add_parser("tts", description="Generate TTS audio manifest.")
    tts_parser.add_argument("--input-path", required=True, type=Path)
    tts_parser.add_argument("--output-path", required=True, type=Path)
    tts_parser.add_argument("--audio-dir", required=True, type=Path)
    tts_parser.add_argument("--sample-rate", default=8000, type=int)
    tts_parser.add_argument("--duration-sec", default=1.0, type=float)
    tts_parser.add_argument("--provider", choices=("synthetic-tone", "gemini"), default="synthetic-tone")
    tts_parser.add_argument("--voice", default="Kore")
    tts_parser.add_argument("--model", default=os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"))
    tts_parser.add_argument("--resume", action="store_true")
    tts_parser.add_argument("--seconds-per-request", default=0.0, type=float)

    audio_qa_parser = subparsers.add_parser("audio-qa", description="Validate TTS audio manifest.")
    audio_qa_parser.add_argument("--input-path", required=True, type=Path)
    audio_qa_parser.add_argument("--output-path", required=True, type=Path)

    augment_parser = subparsers.add_parser("augment-audio", description="Generate augmented audio manifest.")
    augment_parser.add_argument("--input-path", required=True, type=Path)
    augment_parser.add_argument("--output-path", required=True, type=Path)
    augment_parser.add_argument("--output-dir", required=True, type=Path)
    augment_parser.add_argument("--gain", type=float)
    augment_parser.add_argument("--noise-amplitude", default=0, type=int)
    augment_parser.add_argument("--seed", default=42, type=int)
    augment_parser.add_argument("--profile", action="append", default=[])

    merge_parser = subparsers.add_parser("merge-audio-manifests", description="Merge clean and augmented audio manifests.")
    merge_parser.add_argument("--clean-input-path", required=True, type=Path)
    merge_parser.add_argument("--augmented-input-path", required=True, type=Path)
    merge_parser.add_argument("--output-path", required=True, type=Path)

    dataset_summary_parser = subparsers.add_parser("dataset-summary", description="Summarize dataset manifest.")
    dataset_summary_parser.add_argument("--input-path", required=True, type=Path)
    dataset_summary_parser.add_argument("--output-path", required=True, type=Path)

    dataset_qa_parser = subparsers.add_parser("dataset-qa", description="Validate dataset manifest.")
    dataset_qa_parser.add_argument("--input-path", required=True, type=Path)
    dataset_qa_parser.add_argument("--output-path", required=True, type=Path)

    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--paraphrase-mode", choices=("none", "dry-run", "live"), default="none")
    parser.add_argument("--accepted-output-path", type=Path)
    parser.add_argument("--rejected-output-path", type=Path)
    parser.add_argument("--raw-output-path", type=Path)
    parser.add_argument("--summary-output-path", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seconds-per-request", default=0.0, type=float)
    parser.add_argument("--variant-count", default=5, type=int)
    args = parser.parse_args()

    if args.command == "tts":
        rows = _read_jsonl(args.input_path)
        existing_rows = _read_jsonl(args.output_path) if args.resume and args.output_path.exists() else []
        processed = read_processed_utterance_ids([args.output_path]) if args.resume else set()
        audio_rows = build_audio_manifest_rows(
            rows,
            audio_dir=args.audio_dir,
            tts=_build_tts(args),
            processed_utterance_ids=processed,
            rate_limiter=_build_rate_limiter(args.seconds_per_request),
        )
        write_jsonl(args.output_path, [*existing_rows, *audio_rows])
        return

    if args.command == "audio-qa":
        rows = _read_jsonl(args.input_path)
        report = validate_audio_manifest_rows(rows)
        write_jsonl(args.output_path, [report])
        return

    if args.command == "augment-audio":
        rows = _read_jsonl(args.input_path)
        profiles = [_parse_augmentation_profile(profile) for profile in args.profile]
        augmented_rows = build_augmented_manifest_rows(
            rows,
            output_dir=args.output_dir,
            gain=args.gain,
            noise_amplitude=args.noise_amplitude,
            seed=args.seed,
            profiles=profiles or None,
        )
        write_jsonl(args.output_path, augmented_rows)
        return

    if args.command == "merge-audio-manifests":
        clean_rows = _read_jsonl(args.clean_input_path)
        augmented_rows = _read_jsonl(args.augmented_input_path)
        dataset_rows = merge_audio_manifest_rows(clean_rows, augmented_rows)
        write_jsonl(args.output_path, dataset_rows)
        return

    if args.command == "dataset-summary":
        rows = _read_jsonl(args.input_path)
        summary = build_dataset_summary(rows)
        write_jsonl(args.output_path, [summary])
        return

    if args.command == "dataset-qa":
        rows = _read_jsonl(args.input_path)
        report = validate_dataset_manifest_rows(rows)
        write_jsonl(args.output_path, [report])
        return

    if args.output_path is None:
        raise SystemExit("--output-path is required")

    rows = generate_manifest_rows(seed=args.seed, limit=args.limit)
    write_jsonl(args.output_path, rows)

    if args.paraphrase_mode != "none":
        if args.accepted_output_path is None or args.rejected_output_path is None:
            raise SystemExit("--accepted-output-path and --rejected-output-path are required for paraphrase mode")
        pending_rows = rows
        if args.resume:
            processed = read_processed_utterance_ids(
                [
                    args.accepted_output_path,
                    args.rejected_output_path,
                    *([args.raw_output_path] if args.raw_output_path else []),
                ]
            )
            pending_rows = filter_pending_rows(rows, processed)
        paraphraser = _build_paraphraser(args.paraphrase_mode, args.seconds_per_request)
        accepted, rejected, raw = paraphrase_rows_with_audit(
            pending_rows,
            paraphraser=paraphraser,
            variant_count=args.variant_count,
            continue_on_error=args.continue_on_error,
        )
        write_jsonl(args.accepted_output_path, accepted)
        write_jsonl(args.rejected_output_path, rejected)
        if args.raw_output_path is not None:
            write_jsonl(args.raw_output_path, raw)
        if args.summary_output_path is not None:
            summary = build_generation_summary(
                canonical_rows=rows,
                pending_rows=pending_rows,
                accepted_rows=accepted,
                rejected_rows=rejected,
                raw_rows=raw,
                skipped_count=len(rows) - len(pending_rows),
            )
            write_jsonl(args.summary_output_path, [summary])


def _build_rate_limiter(seconds_per_request: float):
    if seconds_per_request <= 0:
        return None
    return RateLimiter(seconds_per_request=seconds_per_request)


def _build_tts(args):
    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required for --provider gemini")
        return GeminiTts(
            api_key=api_key,
            model=args.model,
            voice_name=args.voice,
        )
    return SyntheticToneTts(
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _parse_augmentation_profile(value: str) -> dict:
    name, gain, noise_amplitude, seed = value.split(":")
    return {
        "name": name,
        "gain": float(gain),
        "noise_amplitude": int(noise_amplitude),
        "seed": int(seed),
    }


def _build_paraphraser(mode: str, seconds_per_request: float):
    if mode == "dry-run":
        paraphraser = DryRunParaphraser()
    else:
        paraphraser = GeminiClient(load_gemini_config())
    if seconds_per_request <= 0:
        return paraphraser
    return RateLimitedParaphraser(
        paraphraser,
        RateLimiter(seconds_per_request=seconds_per_request),
    )


if __name__ == "__main__":
    main()
