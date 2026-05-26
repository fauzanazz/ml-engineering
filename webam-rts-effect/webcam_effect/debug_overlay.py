from dataclasses import dataclass

from webcam_effect.state import PosePrediction, smoothed_kicau_score


@dataclass(frozen=True)
class DebugInfo:
    predictions: list[PosePrediction]
    active: bool
    segmenter_backend: str
    classifier_backend: str
    fps: float
    effect_active: bool
    crop_visible: bool


def prediction_summary(predictions: list[PosePrediction]) -> str:
    if not predictions:
        return "no predictions"
    best = max(predictions, key=lambda prediction: prediction.confidence)
    return f"{best.label} {best.confidence:.2f} score={smoothed_kicau_score(predictions):.2f}"


def draw_debug_overlay(frame, info: DebugInfo):
    import cv2

    output = frame
    lines = [
        prediction_summary(info.predictions),
        f"active={info.active} effect={info.effect_active}",
        f"segmentation={'visible' if info.crop_visible else 'missing'}",
        f"segmenter={info.segmenter_backend} classifier={info.classifier_backend}",
        f"fps={info.fps:.1f}",
    ]
    for index, line in enumerate(lines):
        y = 28 + index * 24
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 120), 1, cv2.LINE_AA)
    return output
