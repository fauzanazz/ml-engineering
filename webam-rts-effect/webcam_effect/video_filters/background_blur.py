from dataclasses import dataclass

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import blend_with_mask


@dataclass
class BackgroundBlurFilter:
    provider: object
    blur_kernel: int = 55

    spec = FilterSpec(
        key="1",
        name="background_blur",
        description="Segment person with MediaPipe, blur background only, keep person sharp.",
        requires=("segmentation",),
        heavy=True,
    )

    def process(self, frame, timestamp_ms: int):
        import cv2

        kernel = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
        blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
        mask = self.provider.person_mask(frame, timestamp_ms)
        return blend_with_mask(frame, blurred, mask)
