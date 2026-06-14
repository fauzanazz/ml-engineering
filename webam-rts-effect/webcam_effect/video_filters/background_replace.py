from dataclasses import dataclass
from pathlib import Path

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.effects import load_sticker_frames
from webcam_effect.video_filters.drawing import blend_with_mask


@dataclass
class BackgroundReplaceFilter:
    provider: object
    background_path: Path

    spec = FilterSpec(
        key="9",
        name="background_replace",
        description="Replace background with image/video while preserving person mask.",
        requires=("segmentation", "assets"),
        heavy=True,
    )

    def __post_init__(self) -> None:
        if not self.background_path.exists():
            self._frames = ()
            return
        try:
            self._frames = tuple(load_sticker_frames(self.background_path))
        except RuntimeError:
            self._frames = ()

    def process(self, frame, timestamp_ms: int):
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        if not self._frames:
            background = np.full_like(frame, (28, 36, 52))
        else:
            index = (timestamp_ms // 33) % len(self._frames)
            background = cv2.resize(self._frames[index][:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
        mask = self.provider.person_mask(frame, timestamp_ms)
        return blend_with_mask(frame, background, mask)
