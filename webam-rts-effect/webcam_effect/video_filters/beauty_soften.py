from dataclasses import dataclass

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import filled_landmark_mask
from webcam_effect.video_filters.neon_face_mesh import FACE_OVAL, LEFT_EYE, RIGHT_EYE


LIPS = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
LEFT_EYEBROW = (70, 63, 105, 66, 107)
RIGHT_EYEBROW = (336, 296, 334, 293, 300)


@dataclass
class BeautySoftenFilter:
    provider: object

    spec = FilterSpec(
        key="4",
        name="beauty_soften",
        description="Smooth skin area inside face oval, preserve eyes, lips, eyebrows.",
        requires=("face",),
        heavy=True,
    )

    def process(self, frame, timestamp_ms: int):
        import cv2
        import numpy as np

        landmarks = self.provider.face_landmarks(frame, timestamp_ms)
        if not landmarks:
            return frame.copy()

        face_mask = filled_landmark_mask(frame.shape, landmarks, FACE_OVAL)
        protected = sum(
            filled_landmark_mask(frame.shape, landmarks, indexes)
            for indexes in (LEFT_EYE, RIGHT_EYE, LIPS, LEFT_EYEBROW, RIGHT_EYEBROW)
        )
        skin_mask = np.clip(face_mask - protected, 0.0, 1.0)
        skin_mask = cv2.GaussianBlur(skin_mask, (31, 31), 0)[:, :, None]
        smoothed = cv2.bilateralFilter(frame, 9, 60, 60)
        return (skin_mask * smoothed + (1.0 - skin_mask) * frame).astype(frame.dtype)
