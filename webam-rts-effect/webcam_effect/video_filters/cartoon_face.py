from dataclasses import dataclass

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import blend_with_mask


@dataclass
class CartoonFaceFilter:
    provider: object

    spec = FilterSpec(
        key="5",
        name="cartoon_face",
        description="Posterize colors, add black edge outlines around face and body.",
        requires=("segmentation",),
        heavy=True,
    )

    def process(self, frame, timestamp_ms: int):
        import cv2
        import numpy as np

        posterized = ((frame // 48) * 48 + 24).astype(frame.dtype)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
        cartoon = posterized.copy()
        cartoon[edges > 0] = (0, 0, 0)
        mask = self.provider.person_mask(frame, timestamp_ms)
        return blend_with_mask(cartoon, frame, mask) if mask is not None else cartoon
