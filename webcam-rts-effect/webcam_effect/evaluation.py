from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

from webcam_effect.state import PoseStateMachine
from webcam_effect.yolo_models import YoloFrameClassifier, YoloPersonSegmenter


@dataclass
class ModelMetrics:
    model: str
    true_positive_frames: int = 0
    false_positive_frames: int = 0
    false_negative_frames: int = 0
    true_negative_frames: int = 0
    false_activations: int = 0
    negative_seconds: float = 0.0
    inference_seconds: float = 0.0
    end_to_end_seconds: float = 0.0
    processed_frames: int = 0
    activation_latencies: list[float] | None = None

    def __post_init__(self) -> None:
        self.activation_latencies = []

    def as_dict(self) -> dict:
        precision = self.true_positive_frames / max(1, self.true_positive_frames + self.false_positive_frames)
        recall = self.true_positive_frames / max(1, self.true_positive_frames + self.false_negative_frames)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        latencies = self.activation_latencies or []
        return {
            "model": self.model,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_activation_per_minute": round(self.false_activations / max(self.negative_seconds / 60, 1e-9), 4),
            "activation_latency_ms": round(1000 * sum(latencies) / len(latencies), 2) if latencies else None,
            "inference_fps": round(self.processed_frames / max(self.inference_seconds, 1e-9), 2),
            "end_to_end_fps": round(self.processed_frames / max(self.end_to_end_seconds, 1e-9), 2),
            "processed_frames": self.processed_frames,
            "confusion_frames": {
                "tp": self.true_positive_frames,
                "fp": self.false_positive_frames,
                "fn": self.false_negative_frames,
                "tn": self.true_negative_frames,
            },
        }


def trained_session_ids(dataset_root: Path) -> set[str]:
    root = dataset_root / "classifier_frames"
    if not root.exists():
        return set()
    return {path.stem.split("_", 1)[0] for path in root.rglob("*.jpg")}


def holdout_clips(dataset_root: Path) -> list[tuple[Path, str]]:
    trained = trained_session_ids(dataset_root)
    clips = []
    for label in ("kicau", "none"):
        for path in sorted((dataset_root / "clips" / label).glob("*.mp4")):
            if path.stem not in trained:
                clips.append((path, label))
    return clips


def evaluate_models(
    model_paths: list[Path],
    dataset_root: Path = Path("datasets/kicau_mania"),
    detector_path: str = "yolo26n-seg.pt",
    device: str = "mps",
    output_path: Path = Path("outputs/webcam-evaluation.json"),
    export_path: Path = Path("models/kicau-classifier/best.pt"),
) -> dict:
    import cv2

    clips = holdout_clips(dataset_root)
    if not clips:
        raise RuntimeError(f"no held-out webcam clips under {dataset_root / 'clips'}")
    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing classifier model(s): " + ", ".join(missing))

    segmenter = YoloPersonSegmenter(detector_path, device=device)
    classifiers = [YoloFrameClassifier(str(path), device=device) for path in model_paths]
    metrics = [ModelMetrics(str(path)) for path in model_paths]

    sessions = []
    for clip_path, expected_label in clips:
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open holdout clip: {clip_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sessions.append({"path": str(clip_path), "label": expected_label, "frames": frame_count, "fps": round(fps, 2)})
        states = [PoseStateMachine() for _ in model_paths]
        windows = [deque(maxlen=3) for _ in model_paths]
        first_active = [None for _ in model_paths]
        previous_active = [False for _ in model_paths]
        index = 0
        session_started = time.perf_counter()
        inference_before = [metric.inference_seconds for metric in metrics]
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            segmented = segmenter.segment(frame, segmentation_input="masked-crop")
            crop = segmented.crop if segmented is not None else None
            for model_index, (classifier, model_metric) in enumerate(zip(classifiers, metrics)):
                inference_started = time.perf_counter()
                if crop is not None:
                    windows[model_index].append(classifier.predict(crop))
                    active = states[model_index].update(list(windows[model_index])) if len(windows[model_index]) == 3 else states[model_index].active
                else:
                    active = states[model_index].active
                model_metric.inference_seconds += time.perf_counter() - inference_started
                model_metric.processed_frames += 1
                expected_active = expected_label == "kicau"
                if active and expected_active:
                    model_metric.true_positive_frames += 1
                elif active:
                    model_metric.false_positive_frames += 1
                elif expected_active:
                    model_metric.false_negative_frames += 1
                else:
                    model_metric.true_negative_frames += 1
                if active and not previous_active[model_index]:
                    if expected_active and first_active[model_index] is None:
                        first_active[model_index] = index / fps
                    elif not expected_active:
                        model_metric.false_activations += 1
                previous_active[model_index] = active
            index += 1
        elapsed = time.perf_counter() - session_started
        capture.release()
        clip_inference = [metric.inference_seconds - inference_before[index] for index, metric in enumerate(metrics)]
        shared_seconds = max(0.0, elapsed - sum(clip_inference))
        for model_index, model_metric in enumerate(metrics):
            model_metric.end_to_end_seconds += shared_seconds + clip_inference[model_index]
            if expected_label == "none":
                model_metric.negative_seconds += index / fps
            elif first_active[model_index] is not None:
                model_metric.activation_latencies.append(first_active[model_index])

    results = [metric.as_dict() for metric in metrics]
    winner = max(results, key=lambda item: (item["f1"], -item["false_activation_per_minute"], -(item["activation_latency_ms"] if item["activation_latency_ms"] is not None else float("inf"))))
    report = {
        "holdout": "recorded webcam sessions excluded from classifier training frames",
        "sessions": sessions,
        "models": results,
        "winner": winner["model"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["model"], export_path)
    return report
