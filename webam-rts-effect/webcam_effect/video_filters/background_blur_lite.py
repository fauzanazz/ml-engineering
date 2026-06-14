from dataclasses import dataclass

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import blend_with_mask


@dataclass
class BackgroundBlurLiteFilter:
    provider: object
    scale: float = 0.25

    spec = FilterSpec(
        key="0",
        name="background_blur_lite",
        description="Optimized low-cost blur mode for webcam input.",
        requires=("segmentation",),
        optimized=True,
        target_fps=24.0,
    )

    def process(self, frame, timestamp_ms: int):
        import cv2

        height, width = frame.shape[:2]
        small_width = max(1, int(width * self.scale))
        small_height = max(1, int(height * self.scale))
        small = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
        small_blur = cv2.GaussianBlur(small, (15, 15), 0)
        blurred = cv2.resize(small_blur, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = self.provider.person_mask(frame, timestamp_ms)
        return blend_with_mask(frame, blurred, mask)
