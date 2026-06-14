from dataclasses import dataclass
import math
from pathlib import Path

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import landmark_point, load_overlay, overlay_centered


@dataclass
class FaceStickerFilter:
    provider: object
    sticker_path: Path

    spec = FilterSpec(
        key="8",
        name="face_sticker",
        description="Attach sticker to forehead or cheeks, follow head rotation and scale.",
        requires=("face", "assets"),
        heavy=True,
    )

    def __post_init__(self) -> None:
        self._sticker = load_overlay(self.sticker_path)

    def process(self, frame, timestamp_ms: int):
        landmarks = self.provider.face_landmarks(frame, timestamp_ms)
        if len(landmarks) < 2:
            return frame.copy()

        height, width = frame.shape[:2]
        forehead = landmark_point(landmarks[10] if len(landmarks) > 10 else landmarks[0], width, height)
        left = landmark_point(landmarks[234] if len(landmarks) > 234 else landmarks[0], width, height)
        right = landmark_point(landmarks[454] if len(landmarks) > 454 else landmarks[-1], width, height)
        sticker_width = max(32, int(abs(right[0] - left[0]) * 0.45))
        center = (forehead[0], max(0, forehead[1] - sticker_width // 3))
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        return overlay_centered(frame, self._sticker, center, sticker_width, angle_degrees=angle)
