from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from huggingface_hub import snapshot_download
from mlx_whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer

from indonesian_banking_asr.synthetic.audit import write_jsonl

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def fine_tune_mlx_whisper(
    rows: list[dict],
    *,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    language: str = "id",
    task: str = "transcribe",
    max_steps: int = 1,
    learning_rate: float = 1e-6,
    limit: int | None = None,
    train_scope: str = "decoder",
    optimizer_name: str = "adamw",
    warmup_steps: int = 0,
    max_grad_norm: float | None = None,
    lr_schedule: str = "constant",
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
) -> dict:
    training_rows = rows[:limit]
    if not training_rows:
        raise ValueError("training manifest is empty")

    model = load_model(model_name, dtype=mx.float32)
    _set_train_scope(model, train_scope)
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=language, task=task)
    optimizer = _build_optimizer(
        optimizer_name,
        learning_rate,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
    )
    loss_and_grad = nn.value_and_grad(model, _loss)

    losses = []
    completed_steps = 0
    while completed_steps < max_steps:
        for row in training_rows:
            mel, decoder_inputs, targets = _build_training_example(row, model.dims.n_mels, tokenizer)
            current_step = completed_steps + 1
            optimizer.learning_rate = _learning_rate_for_step(learning_rate, current_step, warmup_steps, max_steps, lr_schedule)
            loss, gradients = loss_and_grad(model, mel, decoder_inputs, targets)
            grad_norm = None
            if max_grad_norm is not None:
                gradients, grad_norm = optim.clip_grad_norm(gradients, max_grad_norm)
            optimizer.update(model, gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            losses.append(float(loss))
            if grad_norm is not None:
                mx.eval(grad_norm)
            completed_steps += 1
            if completed_steps >= max_steps:
                break

    _save_mlx_checkpoint(model, model_name, output_dir)
    return {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "rows_seen": len(training_rows),
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "lr_schedule": lr_schedule,
        "weight_decay": weight_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_eps": adam_eps,
        "optimizer": optimizer_name,
        "train_scope": train_scope,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MLX Whisper on a project manifest.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="id")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-scope", choices=("decoder", "decoder_last_4", "decoder_last_8", "full"), default="decoder")
    parser.add_argument("--optimizer", choices=("adamw", "sgd", "muon", "muon_sgd"), default="adamw")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_linear_decay"), default="constant")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    args = parser.parse_args()

    summary = fine_tune_mlx_whisper(
        _read_jsonl(args.manifest_path),
        output_dir=args.output_dir,
        model_name=args.model_name,
        language=args.language,
        task=args.task,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        limit=args.limit,
        train_scope=args.train_scope,
        optimizer_name=args.optimizer,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )
    write_jsonl(args.summary_path, [summary])


def _loss(model, mel: mx.array, decoder_inputs: mx.array, targets: mx.array) -> mx.array:
    logits = model(mel, decoder_inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    return nn.losses.cross_entropy(flat_logits, flat_targets, reduction="mean")


def _build_training_example(row: dict, n_mels: int, tokenizer) -> tuple[mx.array, mx.array, mx.array]:
    audio = load_audio(row["audio_path"])
    mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=n_mels)
    tokens = list(tokenizer.sot_sequence_including_notimestamps) + tokenizer.encode(row["text"]) + [tokenizer.eot]
    return mx.expand_dims(mel, 0), mx.array([tokens[:-1]]), mx.array([tokens[1:]])


def _set_train_scope(model, train_scope: str) -> None:
    if train_scope == "decoder":
        model.freeze(recurse=True)
        model.decoder.unfreeze(recurse=True)
        return
    if train_scope == "decoder_last_4":
        _unfreeze_last_decoder_blocks(model, 4)
        return
    if train_scope == "decoder_last_8":
        _unfreeze_last_decoder_blocks(model, 8)
        return
    if train_scope == "full":
        model.unfreeze(recurse=True)
        return
    raise ValueError(f"unsupported train scope: {train_scope}")


def _unfreeze_last_decoder_blocks(model, block_count: int) -> None:
    model.freeze(recurse=True)
    for block in model.decoder.blocks[-block_count:]:
        block.unfreeze(recurse=True)
    model.decoder.ln.unfreeze(recurse=True)


def _build_optimizer(
    optimizer_name: str,
    learning_rate: float,
    *,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
):
    if optimizer_name == "adamw":
        return optim.AdamW(
            learning_rate=learning_rate,
            betas=[adam_beta1, adam_beta2],
            eps=adam_eps,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        return optim.SGD(learning_rate=learning_rate)
    if optimizer_name == "muon":
        return optim.Muon(learning_rate=learning_rate, weight_decay=0.0)
    if optimizer_name == "muon_sgd":
        return optim.MultiOptimizer(
            [
                optim.Muon(learning_rate=learning_rate, weight_decay=0.0),
                optim.SGD(learning_rate=learning_rate),
            ],
            filters=[_is_muon_weight],
        )
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def _is_muon_weight(path: str, weight: mx.array) -> bool:
    if weight.ndim < 2:
        return False
    if path.endswith("token_embedding.weight"):
        return False
    if path.endswith("positional_embedding"):
        return False
    return True


def _learning_rate_for_step(
    base_learning_rate: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_schedule: str,
) -> float:
    if warmup_steps <= 0:
        warmup_scale = 1.0
    else:
        warmup_scale = min(step / warmup_steps, 1.0)
    if lr_schedule == "constant":
        return base_learning_rate * warmup_scale
    if lr_schedule == "warmup_linear_decay":
        if step <= warmup_steps:
            return base_learning_rate * warmup_scale
        decay_steps = max(total_steps - warmup_steps, 1)
        decay_progress = min((step - warmup_steps) / decay_steps, 1.0)
        return base_learning_rate * (1.0 - decay_progress)
    raise ValueError(f"unsupported lr schedule: {lr_schedule}")


def _save_mlx_checkpoint(model, model_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_dir / "weights.safetensors"))
    source_dir = Path(model_name)
    if not source_dir.exists():
        source_dir = Path(snapshot_download(repo_id=model_name))
    for filename in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, output_dir / filename)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
