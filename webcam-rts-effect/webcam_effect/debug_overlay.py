from dataclasses import dataclass

from webcam_effect.hand_tracking import HandTrackFrame
from webcam_effect.state import PosePrediction, smoothed_kicau_score

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class DebugInfo:
    predictions: list[PosePrediction]
    active: bool
    segmenter_backend: str
    classifier_backend: str
    fps: float
    effect_active: bool
    crop_visible: bool
    segmented_crop: object | None = None
    hands: HandTrackFrame | None = None


def prediction_summary(predictions: list[PosePrediction]) -> str:
    if not predictions:
        return "no predictions"
    best = max(predictions, key=lambda prediction: prediction.confidence)
    return f"classifier={best.label} {best.confidence:.2f} score={smoothed_kicau_score(predictions):.2f}"

def prediction_details(predictions: list[PosePrediction]) -> str:
    if not predictions:
        return "classifier window=[]"
    parts = [f"{prediction.label}:{prediction.confidence:.2f}" for prediction in predictions]
    return f"classifier window=[{', '.join(parts)}]"


def draw_debug_overlay(frame, info: DebugInfo):
    import cv2

    output = frame
    lines = [
        prediction_summary(info.predictions),
        f"active={info.active} effect={info.effect_active}",
        f"segmentation={'visible' if info.crop_visible else 'missing'}",
        prediction_details(info.predictions),
        f"segmenter={info.segmenter_backend} classifier={info.classifier_backend}",
        f"fps={info.fps:.1f}",
    ]
    for index, line in enumerate(lines):
        y = 28 + index * 24
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 120), 1, cv2.LINE_AA)
    if info.segmented_crop is not None:
        draw_segmented_crop(output, info.segmented_crop)
    if info.hands is not None:
        draw_hand_skeletons(output, info.hands)
    return output

def draw_hand_skeletons(frame, hands: HandTrackFrame) -> None:
    import cv2

    frame_height, frame_width = frame.shape[:2]
    for hand in hands.hands:
        points = [normalized_point(landmark, frame_width, frame_height) for landmark in hand.landmarks]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (255, 190, 40), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (40, 255, 120), -1, cv2.LINE_AA)
        label_point = (hand.box.x1, max(18, hand.box.y1 - 8))
        cv2.putText(
            frame,
            f"{hand.label} {hand.confidence:.2f}",
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{hand.label} {hand.confidence:.2f}",
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 255, 120),
            1,
            cv2.LINE_AA,
        )

def normalized_point(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    x = int(max(0.0, min(1.0, landmark.x)) * frame_width)
    y = int(max(0.0, min(1.0, landmark.y)) * frame_height)
    return x, y

def draw_segmented_crop(frame, segmented_crop):
    import cv2

    crop_height, crop_width = segmented_crop.shape[:2]
    if crop_height == 0 or crop_width == 0:
        return

    frame_height, frame_width = frame.shape[:2]
    max_width = min(240, frame_width // 4)
    max_height = min(180, frame_height // 4)
    scale = min(max_width / crop_width, max_height / crop_height)
    preview_width = max(1, int(crop_width * scale))
    preview_height = max(1, int(crop_height * scale))
    preview = cv2.resize(segmented_crop, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

    x = 16
    y = frame_height - preview_height - 32
    cv2.rectangle(frame, (x - 2, y - 22), (x + preview_width + 2, y + preview_height + 2), (0, 0, 0), -1)
    cv2.putText(frame, "segmented", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 120), 1, cv2.LINE_AA)
    frame[y : y + preview_height, x : x + preview_width] = preview
