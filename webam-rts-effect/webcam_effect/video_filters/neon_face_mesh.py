from dataclasses import dataclass

from webcam_effect.video_filters.base import FilterSpec
from webcam_effect.video_filters.drawing import draw_polyline_glow, landmark_points


FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
FALLBACK_CONNECTIONS = tuple(zip(FACE_OVAL, FACE_OVAL[1:] + FACE_OVAL[:1]))


@dataclass
class NeonFaceMeshFilter:
    provider: object

    spec = FilterSpec(
        key="3",
        name="neon_face_mesh",
        description="Draw glowing face mesh lines using Face Landmarker points.",
        requires=("face",),
        heavy=True,
    )

    def process(self, frame, timestamp_ms: int):
        landmarks = self.provider.face_landmarks(frame, timestamp_ms)
        if not landmarks:
            return frame.copy()
        height, width = frame.shape[:2]
        points = landmark_points(landmarks, width, height)
        output = self._draw_mesh_glow(frame, points)
        for indexes, color in ((FACE_OVAL, (255, 80, 220)), (LEFT_EYE, (40, 255, 255)), (RIGHT_EYE, (40, 255, 255))):
            selected = [landmarks[index] for index in indexes if index < len(landmarks)]
            output = draw_polyline_glow(output, landmark_points(selected, width, height), color, closed=True)
        return output

    def _draw_mesh_glow(self, frame, points: list[tuple[int, int]]):
        import cv2

        output = frame.copy()
        connections = _face_mesh_connections()
        for thickness, alpha in ((3, 0.25), (1, 0.9)):
            layer = output.copy()
            for start, end in connections:
                if start < len(points) and end < len(points):
                    cv2.line(layer, points[start], points[end], (255, 60, 180), thickness, cv2.LINE_AA)
            output = cv2.addWeighted(layer, alpha, output, 1.0 - alpha, 0)
        return output


def _face_mesh_connections() -> tuple[tuple[int, int], ...]:
    try:
        import mediapipe as mp
    except Exception:
        return FALLBACK_CONNECTIONS
    connections = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    return tuple((connection.start, connection.end) for connection in connections) or FALLBACK_CONNECTIONS
