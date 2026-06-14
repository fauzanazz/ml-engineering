from pathlib import Path

from webcam_effect.effects import load_sticker_frames, overlay_image


def load_overlay(path: Path, transparent_white: bool = False):
    if not path.exists():
        return None
    try:
        frame = load_sticker_frames(path)[0]
    except RuntimeError:
        return None
    if transparent_white:
        return white_to_alpha(frame)
    return frame


def white_to_alpha(frame):
    import cv2
    import numpy as np

    if frame.shape[2] == 4:
        output = frame.copy()
    else:
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    white = np.all(output[:, :, :3] > 245, axis=2)
    output[:, :, 3] = np.where(white, 0, output[:, :, 3])
    return output


def landmark_xy(landmark) -> tuple[float, float]:
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return float(landmark.x), float(landmark.y)
    return float(landmark[0]), float(landmark[1])


def landmark_point(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    x, y = landmark_xy(landmark)
    return int(max(0.0, min(1.0, x)) * frame_width), int(max(0.0, min(1.0, y)) * frame_height)


def landmark_points(landmarks, frame_width: int, frame_height: int) -> list[tuple[int, int]]:
    return [landmark_point(landmark, frame_width, frame_height) for landmark in landmarks]


def blend_with_mask(foreground, background, mask):
    import cv2
    import numpy as np

    if mask is None:
        return foreground.copy()
    alpha = np.asarray(mask, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    height, width = foreground.shape[:2]
    if alpha.shape[:2] != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(np.clip(alpha, 0.0, 1.0), (9, 9), 0)[:, :, None]
    return (alpha * foreground + (1.0 - alpha) * background).astype(foreground.dtype)


def overlay_centered(frame, overlay, center: tuple[int, int], width: int, angle_degrees: float = 0.0):
    import cv2

    if overlay is None or width <= 0:
        return frame.copy()
    aspect_ratio = overlay.shape[0] / overlay.shape[1]
    resized_width = max(1, width)
    resized_height = max(1, int(resized_width * aspect_ratio))
    resized = cv2.resize(overlay, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    if angle_degrees:
        matrix = cv2.getRotationMatrix2D((resized_width / 2, resized_height / 2), angle_degrees, 1.0)
        resized = cv2.warpAffine(resized, matrix, (resized_width, resized_height), borderMode=cv2.BORDER_TRANSPARENT)
    x = center[0] - resized_width // 2
    y = center[1] - resized_height // 2
    return overlay_image(frame, resized, x, y)


def draw_polyline_glow(frame, points: list[tuple[int, int]], color: tuple[int, int, int], closed: bool = False):
    import cv2

    output = frame.copy()
    if len(points) < 2:
        return output
    for thickness, scale in ((10, 0.2), (5, 0.45), (2, 1.0)):
        glow = output.copy()
        for index in range(len(points) - 1):
            cv2.line(glow, points[index], points[index + 1], color, thickness, cv2.LINE_AA)
        if closed:
            cv2.line(glow, points[-1], points[0], color, thickness, cv2.LINE_AA)
        output = cv2.addWeighted(glow, scale, output, 1.0 - scale, 0)
    return output


def draw_text(frame, text: str, origin: tuple[int, int] = (16, 28)):
    import cv2

    output = frame.copy()
    cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 140), 1, cv2.LINE_AA)
    return output


def filled_landmark_mask(frame_shape, landmarks, indexes: tuple[int, ...]):
    import cv2
    import numpy as np

    height, width = frame_shape[:2]
    points = [landmark_point(landmarks[index], width, height) for index in indexes if index < len(landmarks)]
    mask = np.zeros((height, width), dtype=np.float32)
    if len(points) >= 3:
        cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 1.0)
    return mask
