from dataclasses import dataclass

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import landmark_point


POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))


@dataclass
class PoseAuraFilter:
    provider: object

    spec = FilterSpec(
        key="7",
        name="pose_aura",
        description="Track body pose, draw animated glow around torso, arms, and head.",
        requires=("pose",),
        heavy=True,
    )

    def process(self, frame, timestamp_ms: int):
        import cv2

        landmarks = self.provider.pose_landmarks(frame, timestamp_ms)
        if not landmarks:
            return frame.copy()
        height, width = frame.shape[:2]
        output = frame.copy()
        glow = output.copy()
        points = [landmark_point(landmark, width, height) for landmark in landmarks]
        pulse = 8 + int((timestamp_ms // 90) % 6)
        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(glow, points[start], points[end], (255, 120, 40), pulse, cv2.LINE_AA)
        if points:
            cv2.circle(glow, points[0], pulse * 2, (255, 120, 40), 3, cv2.LINE_AA)
        return cv2.addWeighted(glow, 0.45, output, 0.55, 0)
