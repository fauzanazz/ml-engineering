from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_TYPE = "{{task_type}}"
PREDICTION_KEY = "{{prediction_key}}"
PACKAGE_NAME = "{{package_name}}"

FEATURE_DIM = 4
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.12
DEFAULT_SEED = 17


@dataclass(frozen=True)
class RunConfig:
    epochs: int
    learning_rate: float
    seed: int
    model_state: Path | None


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))




def _load_model_state(path: Path) -> tuple[list[float], float] | None:
    if not path.exists():
        raise ValueError(f"model-state path not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("model-state must be a JSON object")

    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, list) or len(raw_weights) != FEATURE_DIM:
        raise ValueError("model-state.weights must be a numeric list of length 4")
    weights: list[float] = []
    for item in raw_weights:
        if not isinstance(item, int | float):
            raise ValueError("model-state.weights must contain only numbers")
        weights.append(float(item))

    bias_raw = payload.get("bias", 0.0)
    if not isinstance(bias_raw, int | float):
        raise ValueError("model-state.bias must be a number")

    return weights, float(bias_raw)


def _resolve_initial_weights(seed: int, model_state: Path | None) -> tuple[list[float], float]:
    if model_state is not None:
        return _load_model_state(model_state)

    rng = random.Random(seed)
    weights = [rng.uniform(-0.2, 0.2) for _ in range(FEATURE_DIM)]
    return weights, 0.0


def _build_dataset(task_type: str, seed: int, samples: int = 64) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    true_weights = [0.9, -0.5, 0.2, 0.1]
    true_bias = -0.35
    features: list[list[float]] = []
    targets: list[float] = []

    for index in range(samples):
        vector = [rng.uniform(-1.5, 1.5) for _ in range(FEATURE_DIM)]
        score = sum(w * x for w, x in zip(true_weights, vector, strict=True)) + true_bias
        if task_type in {"classification", "object_detection", "segmentation"}:
            noise = (rng.random() - 0.5) * 0.3
            target = 1.0 if (_sigmoid(score + noise) > 0.5) else 0.0
        elif task_type == "text_generation":
            noise = (rng.random() - 0.5) * 0.2
            target = _sigmoid((score + noise) / 2.5)
        else:
            target = (score + 0.5) / 3.0 + (rng.random() - 0.5) * 0.05
        features.append(vector)
        targets.append(float(target))

    return features, targets


def _classification_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    predicted = [1.0 if value >= 0.5 else 0.0 for value in predictions]
    gold = [1.0 if value >= 0.5 else 0.0 for value in targets]

    true_pos = sum(1 for p, g in zip(predicted, gold, strict=True) if p == 1.0 and g == 1.0)
    false_pos = sum(1 for p, g in zip(predicted, gold, strict=True) if p == 1.0 and g == 0.0)
    false_neg = sum(1 for p, g in zip(predicted, gold, strict=True) if p == 0.0 and g == 1.0)

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g) / len(gold)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _segmentation_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    binary_pred = [1.0 if p >= 0.5 else 0.0 for p in predictions]
    binary_true = [1.0 if t >= 0.5 else 0.0 for t in targets]
    intersection = sum(p * t for p, t in zip(binary_pred, binary_true, strict=True))
    pred_count = sum(binary_pred)
    true_count = sum(binary_true)
    union = pred_count + true_count - intersection
    mean_iou = float(intersection / union) if union > 0 else 1.0
    dice = (2.0 * intersection / (pred_count + true_count)) if (pred_count + true_count) > 0 else 1.0

    return {
        "mean_iou": mean_iou,
        "dice_coefficient": dice,
    }


def _regression_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    errors = [abs(p - t) for p, t in zip(predictions, targets, strict=True)]
    mae = sum(errors) / len(errors)
    var_true = sum((t - sum(targets) / len(targets)) ** 2 for t in targets) / len(targets)
    var_pred = sum((p - sum(predictions) / len(predictions)) ** 2 for p in predictions) / len(predictions)
    corr_like = max(1.0 - (sum((p - t) ** 2 for p, t in zip(predictions, targets, strict=True)) / max(var_true, 1e-9)), 0.0)

    return {
        "mae": mae,
        "rmse": math.sqrt(sum((p - t) ** 2 for p, t in zip(predictions, targets, strict=True)) / len(predictions)),
        "r2": corr_like,
        "stability": var_pred,
    }


def _resolve_summary_path(summary_path: Path) -> Path:
    if summary_path.exists() and summary_path.is_dir():
        return summary_path / "training-summary.json"
    return summary_path


def _run_mini_loop(features: list[list[float]], targets: list[float], config: RunConfig) -> tuple[list[float], float, list[float], dict[str, float]]:
    weights, bias = _resolve_initial_weights(config.seed, config.model_state)
    loss_history: list[float] = []

    classification_task = TASK_TYPE in {"classification", "object_detection", "segmentation"}

    for _ in range(config.epochs):
        epoch_loss = 0.0
        for x, target in zip(features, targets, strict=True):
            raw_score = sum(w * xi for w, xi in zip(weights, x, strict=True)) + bias
            if TASK_TYPE == "text_generation":
                prediction = raw_score
                gradient_scale = prediction - target
                loss = 0.5 * (prediction - target) ** 2
            elif classification_task:
                prediction = _sigmoid(raw_score)
                prediction = min(max(prediction, 1e-7), 1.0 - 1e-7)
                gradient_scale = prediction - target
                loss = -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))
            else:
                prediction = raw_score
                gradient_scale = prediction - target
                loss = 0.5 * (prediction - target) ** 2

            epoch_loss += loss
            for index, feature_value in enumerate(x):
                weights[index] -= config.learning_rate * gradient_scale * feature_value
            bias -= config.learning_rate * gradient_scale

        epoch_loss /= max(len(features), 1)
        loss_history.append(round(epoch_loss, 6))

    raw_scores = [sum(w * xi for w, xi in zip(weights, x, strict=True)) + bias for x in features]
    predictions = [_sigmoid(score) for score in raw_scores] if classification_task else raw_scores
    last_loss = loss_history[-1] if loss_history else 0.0

    metrics = {
        "loss": last_loss,
        "loss_history": loss_history,
    }

    if classification_task:
        metrics.update(_classification_metrics(predictions, targets))
        if TASK_TYPE in {"segmentation", "object_detection"}:
            metrics.update(_segmentation_metrics(predictions, targets))
    else:
        metrics.update(_regression_metrics(predictions, targets))

    if TASK_TYPE == "text_generation":
        metrics.update(
            {
                "perplexity": max(math.exp(min(last_loss, 20.0)), 0.0),
                "bleu": min(1.0, max(0.0, 1.0 - last_loss / 4.0)),
                "latency_ms": 12.5,
            }
        )

    return weights, bias, loss_history, metrics




def _write_summary(summary_path: Path, metrics: dict[str, Any]) -> dict[str, str]:
    metrics_path = summary_path.parent / "metrics.json"

    metrics_uri = str(metrics_path)
    try:
        metrics_uri = str(metrics_path.relative_to(summary_path.parent))
    except ValueError:
        metrics_uri = str(metrics_path)

    summary = {
        "model_name": PACKAGE_NAME,
        "version": "bootstrap",
        "artifact_uri": "artifacts/model",
        "metrics_uri": metrics_uri,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return summary





def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic mini-loop training example.")
    parser.add_argument("--summary-path", type=Path, default=Path("reports/training-summary.json"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-state", type=Path, default=None)
    args = parser.parse_args()

    config = RunConfig(
        epochs=max(args.epochs, 1),
        learning_rate=max(args.learning_rate, 1e-6),
        seed=max(args.seed, 1),
        model_state=args.model_state,
    )

    features, targets = _build_dataset(TASK_TYPE, config.seed)
    _, _, _, metrics = _run_mini_loop(features, targets, config)
    summary_path = _resolve_summary_path(args.summary_path)
    summary = _write_summary(summary_path, metrics)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
