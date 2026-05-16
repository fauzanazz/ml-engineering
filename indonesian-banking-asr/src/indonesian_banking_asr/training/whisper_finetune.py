from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from indonesian_banking_asr.synthetic.audit import write_jsonl

DEFAULT_MODEL = "openai/whisper-tiny"


class ManifestSpeechDataset(Dataset):
    def __init__(self, rows: list[dict], processor: WhisperProcessor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio, sampling_rate = sf.read(row["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(row["text"], return_tensors="pt").input_ids[0]
        return {"input_features": features, "labels": labels, "utterance_id": row["utterance_id"]}


def fine_tune_whisper(
    rows: list[dict],
    *,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    language: str = "indonesian",
    task: str = "transcribe",
    max_steps: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    limit: int | None = None,
) -> dict:
    training_rows = rows[:limit]
    if not training_rows:
        raise ValueError("training manifest is empty")

    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    device = _training_device()
    model.to(device)
    model.train()

    dataset = ManifestSpeechDataset(training_rows, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = []
    completed_steps = 0
    while completed_steps < max_steps:
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            losses.append(float(loss.detach().cpu()))
            if completed_steps >= max_steps:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    return {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "rows_seen": len(training_rows),
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device.type,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a project manifest.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="indonesian")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    summary = fine_tune_whisper(
        _read_jsonl(args.manifest_path),
        output_dir=args.output_dir,
        model_name=args.model_name,
        language=args.language,
        task=args.task,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
    )
    write_jsonl(args.summary_path, [summary])


def _collate_batch(items: list[dict]) -> dict:
    input_features = torch.stack([item["input_features"] for item in items])
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in items],
        batch_first=True,
        padding_value=-100,
    )
    return {"input_features": input_features, "labels": labels}


def _training_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
