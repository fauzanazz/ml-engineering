from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re


@dataclass
class AnimatedSticker:
    path: Path
    scale: float = 0.25
    chroma_key_green: bool = False
    chroma_tolerance: int = 80
    opacity: float = 1.0

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

    def next_resized_frame(self, frame_width: int, scale_multiplier: float = 1.0, rotation: float = 0.0):
        import cv2

        index = self.frame_index
        frame = self.next_frame()
        sticker_width = max(1, int(frame_width * self.scale * scale_multiplier))
        aspect_ratio = frame.shape[0] / frame.shape[1]
        sticker_height = max(1, int(sticker_width * aspect_ratio))
        cache_key = (index, sticker_width, sticker_height, round(rotation, 2), round(self.opacity, 3))
        if cache_key not in self._resized_frame_cache:
            resized = cv2.resize(frame, (sticker_width, sticker_height))
            if rotation:
                center = (sticker_width / 2, sticker_height / 2)
                matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                resized = cv2.warpAffine(resized, matrix, (sticker_width, sticker_height), borderMode=cv2.BORDER_TRANSPARENT)
            if self.opacity < 1:
                if len(resized.shape) == 3 and resized.shape[2] == 4:
                    resized = resized.copy()
                    resized[:, :, 3] = (resized[:, :, 3] * self.opacity).astype(resized.dtype)
                else:
                    resized = (resized * self.opacity).astype(resized.dtype)
            self._resized_frame_cache[cache_key] = resized
        return self._resized_frame_cache[cache_key]

@dataclass(frozen=True)
class StickerLayer:
    id: str = "right"
    name: str = "Sticker"
    asset_path: str = "assets/nick.gif"
    x: float = 0.72
    y: float = 0.12
    scale: float = 0.25
    rotation: float = 0.0
    opacity: float = 1.0
    hidden: bool = False
    chroma_key_green: bool = False
    chroma_tolerance: int = 80
    enter_animation: str = "none"
    loop_animation: str = "none"
    animation_speed: float = 1.0

@dataclass(frozen=True)
class AudioTrack:
    id: str = "track-1"
    name: str = "Audio"
    path: str = "assets/Kicau Mania Cutted.mp3"
    volume: float = 1.0
    loop: bool = True
    muted: bool = False

    def __eq__(self, other):
        if isinstance(other, str):
            return self.path == other
        return super().__eq__(other)


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
    layers: tuple[StickerLayer, ...] | None = None

    def __post_init__(self):
        layer_definitions = self.layers or tuple(
            layer
            for layer in [
                StickerLayer(id="right", name="Gif 1", asset_path=str(self.right_sticker_path), x=self.right_x or 0.72, y=self.right_y, scale=self.scale),
                StickerLayer(
                    id="left",
                    name="Gif 2",
                    asset_path=str(self.left_sticker_path) if self.left_sticker_path else "",
                    x=self.left_x or 0.04,
                    y=self.left_y,
                    scale=self.scale,
                    chroma_key_green=self.left_chroma_key_green,
                ),
            ]
            if layer.asset_path
        )
        self.layer_definitions = layer_definitions
        self.stickers = [
            AnimatedSticker(
                Path(layer.asset_path),
                scale=layer.scale,
                chroma_key_green=layer.chroma_key_green,
                chroma_tolerance=layer.chroma_tolerance,
                opacity=layer.opacity,
            )
            for layer in layer_definitions
            if not layer.hidden
        ]
        self.active_layers = [layer for layer in layer_definitions if not layer.hidden]
        self.right_sticker = self.stickers[0] if self.stickers else AnimatedSticker(self.right_sticker_path, scale=self.scale)
        self.left_sticker = self.stickers[1] if len(self.stickers) > 1 else None

    def apply(self, frame, elapsed_active_time: float = 0.0):
        output = frame
        for layer, sticker in zip(self.active_layers, self.stickers):
            transform = layer_runtime_transform(layer, elapsed_active_time)
            output = apply_sticker(
                output,
                sticker,
                x_position="left",
                x_ratio=transform["x"],
                y_ratio=transform["y"],
                scale_multiplier=transform["scale_multiplier"],
                rotation=transform["rotation"],
            )
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
    audio_tracks: tuple[AudioTrack, ...] = (AudioTrack(),)
    selected_audio: str = "assets/Kicau Mania Cutted.mp3"
    audio_volume: float = 1.0
    audio_loop: bool = True
    scale: float = 0.25
    right_x: float = 0.72
    right_y: float = 0.12
    left_x: float = 0.04
    left_y: float = 0.12
    layers: tuple[StickerLayer, ...] = field(default_factory=tuple)
    trigger_labels: tuple[str, ...] = field(default_factory=tuple)
    activate_threshold: float = 0.7
    deactivate_threshold: float = 0.45

@dataclass(frozen=True)
class EffectLibrary:
    selected_id: str
    effects: dict[str, EffectDefinition]

def effect_id_from_name(name: str, existing_ids: set[str] | None = None) -> str:
    existing_ids = existing_ids or set()
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "effect"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def load_effect_definition(path: Path | None) -> EffectDefinition:
    library = load_effect_library(path)
    return library.effects[library.selected_id]


def save_effect_definition(path: Path, definition: EffectDefinition) -> None:
    effect_id = effect_id_from_name(definition.name)
    save_effect_library(path, EffectLibrary(selected_id=effect_id, effects={effect_id: definition}))

def load_effect_library(path: Path | None) -> EffectLibrary:
    default = EffectDefinition()
    if path is None or not path.exists():
        return EffectLibrary(selected_id="kicau-mania", effects={"kicau-mania": default})

    data = json.loads(path.read_text())
    if "effects" not in data:
        effect = effect_from_dict(data)
        effect_id = effect_id_from_name(effect.name)
        return EffectLibrary(selected_id=effect_id, effects={effect_id: effect})

    effects = {effect_id: effect_from_dict(effect_data) for effect_id, effect_data in data["effects"].items()}
    if not effects:
        effects = {"kicau-mania": default}
    selected_id = data.get("selected_id") if data.get("selected_id") in effects else next(iter(effects))
    return EffectLibrary(selected_id=selected_id, effects=effects)

def save_effect_library(path: Path, library: EffectLibrary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "selected_id": library.selected_id,
        "effects": {effect_id: effect_to_dict(effect) for effect_id, effect in library.effects.items()},
    }
    path.write_text(json.dumps(data, indent=2) + "\n")

def effect_to_dict(effect: EffectDefinition) -> dict:
    return asdict(effect)

def effect_from_dict(data: dict) -> EffectDefinition:
    allowed = EffectDefinition.__dataclass_fields__
    normalized = {key: data[key] for key in allowed if key in data}
    if normalized.get("left_sticker") == "":
        normalized["left_sticker"] = None
    normalized["audio_tracks"] = normalize_audio_tracks(normalized.get("audio_tracks"), normalized.get("audio"))
    if not normalized.get("selected_audio"):
        normalized["selected_audio"] = normalized.get("audio") or (normalized["audio_tracks"][0] if normalized["audio_tracks"] else "")
    if not normalized.get("audio"):
        normalized["audio"] = normalized.get("selected_audio", "")
    if normalized.get("selected_audio") and isinstance(normalized["selected_audio"], AudioTrack):
        normalized["selected_audio"] = normalized["selected_audio"].path
    normalized["layers"] = normalize_layers(normalized.get("layers"), normalized)
    normalized["trigger_labels"] = tuple(normalized.get("trigger_labels") or ())
    return EffectDefinition(**normalized)

def normalize_audio_tracks(raw_tracks, legacy_audio: str | None = None) -> tuple[AudioTrack, ...]:
    values = raw_tracks if raw_tracks is not None else tuple(track for track in [legacy_audio] if track)
    tracks = []
    for index, item in enumerate(values or []):
        if isinstance(item, AudioTrack):
            track = item
        elif isinstance(item, str):
            track = AudioTrack(id=f"track-{index + 1}", name=Path(item).stem or f"Track {index + 1}", path=item)
        else:
            track = AudioTrack(
                id=str(item.get("id") or f"track-{index + 1}"),
                name=str(item.get("name") or Path(item.get("path", "")).stem or f"Track {index + 1}"),
                path=str(item.get("path") or ""),
                volume=float(item.get("volume", 1.0)),
                loop=bool(item.get("loop", True)),
                muted=bool(item.get("muted", False)),
            )
        if track.path:
            tracks.append(track)
    return tuple(tracks)

def normalize_layers(raw_layers, effect_data: dict) -> tuple[StickerLayer, ...]:
    if raw_layers:
        return tuple(layer_from_dict(index, layer) for index, layer in enumerate(raw_layers))
    layers = [
        StickerLayer(
            id="right",
            name="Gif 1",
            asset_path=effect_data.get("right_sticker") or "",
            x=float(effect_data.get("right_x", 0.72)),
            y=float(effect_data.get("right_y", 0.12)),
            scale=float(effect_data.get("scale", 0.25)),
        ),
        StickerLayer(
            id="left",
            name="Gif 2",
            asset_path=effect_data.get("left_sticker") or "",
            x=float(effect_data.get("left_x", 0.04)),
            y=float(effect_data.get("left_y", 0.12)),
            scale=float(effect_data.get("scale", 0.25)),
            chroma_key_green=True,
        ),
    ]
    return tuple(layer for layer in layers if layer.asset_path)

def layer_from_dict(index: int, raw_layer) -> StickerLayer:
    if isinstance(raw_layer, StickerLayer):
        return raw_layer
    return StickerLayer(
        id=str(raw_layer.get("id") or f"layer-{index + 1}"),
        name=str(raw_layer.get("name") or f"Layer {index + 1}"),
        asset_path=str(raw_layer.get("asset_path") or raw_layer.get("path") or ""),
        x=float(raw_layer.get("x", 0.5)),
        y=float(raw_layer.get("y", 0.12)),
        scale=float(raw_layer.get("scale", 0.25)),
        rotation=float(raw_layer.get("rotation", 0.0)),
        opacity=float(raw_layer.get("opacity", 1.0)),
        hidden=bool(raw_layer.get("hidden", False)),
        chroma_key_green=bool(raw_layer.get("chroma_key_green", False)),
        chroma_tolerance=int(raw_layer.get("chroma_tolerance", 80)),
        enter_animation=str(raw_layer.get("enter_animation") or "none"),
        loop_animation=str(raw_layer.get("loop_animation") or "none"),
        animation_speed=float(raw_layer.get("animation_speed", 1.0)),
    )

def layer_runtime_transform(layer: StickerLayer, elapsed_active_time: float) -> dict[str, float]:
    time_value = max(0.0, elapsed_active_time) * max(0.01, layer.animation_speed)
    x = layer.x
    y = layer.y
    scale_multiplier = 1.0
    rotation = layer.rotation
    progress = min(time_value / 0.35, 1.0)

    if layer.enter_animation == "pop":
        scale_multiplier *= 0.2 + 0.8 * progress
    elif layer.enter_animation == "slide":
        x -= (1.0 - progress) * 0.12
    elif layer.enter_animation == "bounce":
        y -= math.sin(progress * math.pi) * 0.08
    elif layer.enter_animation == "spin":
        rotation += (1.0 - progress) * -180

    wave = math.sin(time_value * math.tau)
    if layer.loop_animation == "pulse":
        scale_multiplier *= 1.0 + 0.06 * wave
    elif layer.loop_animation == "bob":
        y += 0.025 * wave
    elif layer.loop_animation == "shake":
        x += 0.012 * math.sin(time_value * math.tau * 4)

    return {"x": x, "y": y, "scale_multiplier": scale_multiplier, "rotation": rotation}

def apply_sticker(
    frame,
    sticker: AnimatedSticker,
    x_position: str,
    x_ratio: float | None = None,
    y_ratio: float = 0.08,
    scale_multiplier: float = 1.0,
    rotation: float = 0.0,
):
    frame_height, frame_width = frame.shape[:2]
    resized_sticker = sticker.next_resized_frame(frame_width, scale_multiplier=scale_multiplier, rotation=rotation)
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
