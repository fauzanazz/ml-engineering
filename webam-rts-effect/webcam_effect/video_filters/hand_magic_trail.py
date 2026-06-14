from dataclasses import dataclass, field

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import landmark_point


INDEX_TIP = 8


@dataclass
class HandMagicTrailFilter:
    provider: object
    max_points: int = 40
    points: list[tuple[int, int, int]] = field(default_factory=list)

    spec = FilterSpec(
        key="6",
        name="hand_magic_trail",
        description="Track index fingertips, draw fading particle trail while hand moves.",
        requires=("hand",),
        heavy=True,
    )

    def process(self, frame, timestamp_ms: int):
        import cv2

        output = frame.copy()
        height, width = frame.shape[:2]
        for hand in self.provider.hand_landmarks(frame, timestamp_ms):
            if len(hand) > INDEX_TIP:
                x, y = landmark_point(hand[INDEX_TIP], width, height)
                self.points.append((x, y, self.max_points))

        faded = []
        for x, y, life in self.points[-self.max_points :]:
            radius = max(2, life // 6)
            color = (255 - life * 3 % 255, 80 + life * 4 % 175, 255)
            cv2.circle(output, (x, y), radius, color, -1, cv2.LINE_AA)
            if life > 1:
                faded.append((x, y, life - 1))
        self.points = faded
        return output
