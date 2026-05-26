from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class AnimatedSticker:
    path: Path
    scale: float = 0.25
    chroma_key_green: bool = False
    chroma_tolerance: int = 80

    def __post_init__(self):
        self.frames = load_sticker_frames(self.path)
        if self.chroma_key_green:
            self.frames = [remove_green_screen(frame, tolerance=self.chroma_tolerance) for frame in self.frames]
        self.frame_index = 0
        self._resized_frame_cache = {}

    def next_frame(self):
        frame = self.frames[self.frame_index]
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        return frame

    def next_resized_frame(self, frame_width: int):
        import cv2

        index = self.frame_index
        frame = self.next_frame()
        sticker_width = max(1, int(frame_width * self.scale))
        aspect_ratio = frame.shape[0] / frame.shape[1]
        sticker_height = max(1, int(sticker_width * aspect_ratio))
        cache_key = (index, sticker_width, sticker_height)
        if cache_key not in self._resized_frame_cache:
            self._resized_frame_cache[cache_key] = cv2.resize(frame, (sticker_width, sticker_height))
        return self._resized_frame_cache[cache_key]


@dataclass
class StickerEffect:
    right_sticker_path: Path
    left_sticker_path: Path | None = None
    scale: float = 0.25
    right_x: float | None = None
    right_y: float = 0.08
    left_x: float | None = None
    left_y: float = 0.08
    left_chroma_key_green: bool = True

    def __post_init__(self):
        self.right_sticker = AnimatedSticker(self.right_sticker_path, scale=self.scale)
        self.left_sticker = (
            AnimatedSticker(self.left_sticker_path, scale=self.scale, chroma_key_green=self.left_chroma_key_green)
            if self.left_sticker_path
            else None
        )

    def apply(self, frame):
        output = apply_sticker(frame, self.right_sticker, x_position="right", x_ratio=self.right_x, y_ratio=self.right_y)
        if self.left_sticker is not None:
            output = apply_sticker(output, self.left_sticker, x_position="left", x_ratio=self.left_x, y_ratio=self.left_y)
        return output

    def set_scale(self, scale: float) -> None:
        self.right_sticker.scale = scale
        if self.left_sticker is not None:
            self.left_sticker.scale = scale


@dataclass(frozen=True)
class EffectDefinition:
    name: str = "kicau mania"
    right_sticker: str = "assets/nick.gif"
    left_sticker: str | None = "assets/cat.gif"
    audio: str = "assets/Kicau Mania Cutted.mp3"
    scale: float = 0.25
    right_x: float = 0.72
    right_y: float = 0.12
    left_x: float = 0.04
    left_y: float = 0.12


def load_effect_definition(path: Path | None) -> EffectDefinition:
    if path is None or not path.exists():
        return EffectDefinition()

    data = json.loads(path.read_text())
    allowed = EffectDefinition.__dataclass_fields__
    return EffectDefinition(**{key: data[key] for key in allowed if key in data})


def save_effect_definition(path: Path, definition: EffectDefinition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition.__dict__, indent=2) + "\n")

def apply_sticker(frame, sticker: AnimatedSticker, x_position: str, x_ratio: float | None = None, y_ratio: float = 0.08):
    frame_height, frame_width = frame.shape[:2]
    resized_sticker = sticker.next_resized_frame(frame_width)
    sticker_width = resized_sticker.shape[1]

    x = int(frame_width * x_ratio) if x_ratio is not None else 24 if x_position == "left" else frame_width - sticker_width - 24
    y = int(frame_height * y_ratio)
    return overlay_image(frame, resized_sticker, x=x, y=y)


def load_sticker_frames(path: Path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None and path.suffix.lower() != ".gif":
        return [image]

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"could not load sticker: {path}")
    return frames


def remove_green_screen(frame, tolerance: int = 80):
    import cv2
    import numpy as np

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    bgr = frame[:, :, :3]
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    green_dominance = green - np.maximum(red, blue)
    green_mask = (green > 80) & (green_dominance > tolerance)

    output = frame.copy()
    output[:, :, 3] = np.where(green_mask, 0, output[:, :, 3])
    return output


def overlay_image(frame, overlay, x: int, y: int):
    import numpy as np

    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + overlay_width)
    y2 = min(frame_height, y + overlay_height)
    if x1 >= x2 or y1 >= y2:
        return output

    overlay_crop = overlay[y1 - y : y2 - y, x1 - x : x2 - x]
    target = output[y1:y2, x1:x2]

    if len(overlay_crop.shape) == 3 and overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3:4] / 255.0
        output[y1:y2, x1:x2] = (alpha * overlay_crop[:, :, :3] + (1 - alpha) * target).astype(np.uint8)
    else:
        output[y1:y2, x1:x2] = overlay_crop[:, :, :3]

    return output
