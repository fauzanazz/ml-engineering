from dataclasses import dataclass
from pathlib import Path

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import landmark_point, load_overlay, overlay_centered


@dataclass
class VirtualGlassesFilter:
    provider: object
    glasses_path: Path

    spec = FilterSpec(
        key="2",
        name="virtual_glasses",
        description="Track face landmarks, place glasses asset over both eyes, scale with face width.",
        requires=("face", "assets"),
        heavy=True,
    )

    def __post_init__(self) -> None:
        self._glasses = load_overlay(self.glasses_path, transparent_white=True)

    def process(self, frame, timestamp_ms: int):
        import cv2

        landmarks = self.provider.face_landmarks(frame, timestamp_ms)
        if len(landmarks) < 2:
            return frame.copy()

        height, width = frame.shape[:2]
        left = landmark_point(landmarks[33] if len(landmarks) > 263 else landmarks[0], width, height)
        right = landmark_point(landmarks[263] if len(landmarks) > 263 else landmarks[-1], width, height)
        center = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
        glasses_width = max(32, int(abs(right[0] - left[0]) * 1.9))

        if self._glasses is not None:
            return overlay_centered(frame, self._glasses, center, glasses_width)

        output = frame.copy()
        radius = max(8, glasses_width // 8)
        cv2.circle(output, left, radius, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.circle(output, right, radius, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.line(output, left, right, (20, 20, 20), 3, cv2.LINE_AA)
        return output
