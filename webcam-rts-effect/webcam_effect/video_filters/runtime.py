from dataclasses import dataclass
from pathlib import Path
import time

from webcam_effect.camera import CameraSource, parse_resolution
from webcam_effect.video_filters.base import FilterAssets, asset_path
from webcam_effect.video_filters.drawing import draw_text
from webcam_effect.video_filters.mediapipe_tasks import MediaPipeDetectionProvider, MediaPipeTaskPaths
from webcam_effect.video_filters.registry import build_filters


@dataclass
class FpsMeter:
    started_at: float = 0.0
    last_frame_at: float = 0.0
    frame_count: int = 0
    fps: float = 0.0

    def update(self, now: float) -> None:
        if self.started_at == 0.0:
            self.started_at = now
        if self.last_frame_at > 0.0:
            elapsed = now - self.last_frame_at
            if elapsed > 0:
                self.fps = 1.0 / elapsed
        self.last_frame_at = now
        self.frame_count += 1

    def average(self, now: float) -> float:
        elapsed = max(now - self.started_at, 1e-9)
        return self.frame_count / elapsed


def run_video_filter_app(
    camera: str = "0",
    resolution: str = "640x480",
    assets_dir: str = "assets",
    background: str = "example.png",
    glasses: str = "example.png",
    sticker: str = "nick.gif",
    face_model: str = "assets/face_landmarker.task",
    hand_model: str = "assets/hand_landmarker.task",
    pose_model: str = "assets/pose_landmarker_lite.task",
    segmenter_model: str = "assets/selfie_segmenter.tflite",
    start_filter: str = "0",
    frame_skip: int = 1,
    inference_scale: float = 0.5,
    video_output: str = "preview",
    record_output: str = "",
    benchmark_seconds: int = 0,
) -> None:
    import cv2

    width, height = parse_resolution(resolution)
    try:
        capture = CameraSource(camera, width=width, height=height).open()
    except RuntimeError as exc:
        print(exc)
        return

    assets_root = Path(assets_dir)
    assets = FilterAssets(
        assets_dir=assets_root,
        background=asset_path(assets_root, background),
        glasses=asset_path(assets_root, glasses),
        sticker=asset_path(assets_root, sticker),
    )
    paths = MediaPipeTaskPaths(face=Path(face_model), hand=Path(hand_model), pose=Path(pose_model), segmenter=Path(segmenter_model))
    provider = MediaPipeDetectionProvider(paths=paths, frame_skip=frame_skip, inference_scale=inference_scale)
    filters = build_filters(provider, assets)
    selected_key = start_filter if start_filter in filters else next(iter(filters))
    writer = create_video_writer(record_output, width, height) if record_output else None
    fps_meter = FpsMeter()
    started_at = time.monotonic()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("camera frame read failed")
                break

            now = time.monotonic()
            fps_meter.update(now)
            timestamp_ms = int((now - started_at) * 1000)
            active_filter = filters[selected_key]
            output = active_filter.process(frame, timestamp_ms)
            output = draw_runtime_overlay(output, selected_key, active_filter.spec.name, fps_meter, provider, now)

            if writer is not None:
                writer.write(output)
            if video_output == "preview":
                cv2.imshow("MediaPipe video filters", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            pressed = chr(key) if key else ""
            if pressed in filters:
                selected_key = pressed
            provider.next_frame()

            if benchmark_seconds and now - started_at >= benchmark_seconds:
                break
    finally:
        average_fps = fps_meter.average(time.monotonic())
        if benchmark_seconds:
            print(f"average fps={average_fps:.1f} frames={fps_meter.frame_count} seconds={benchmark_seconds}")
        if writer is not None:
            writer.release()
        provider.close()
        capture.release()
        cv2.destroyAllWindows()


def create_video_writer(path: str, width: int, height: int):
    import cv2

    writer_path = Path(path)
    writer_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        print(f"could not open video writer: {writer_path}")
        return None
    return writer


def draw_runtime_overlay(frame, key: str, name: str, fps_meter: FpsMeter, provider, now: float):
    average_fps = fps_meter.average(now)
    target = " target>=24" if name == "background_blur_lite" else ""
    output = draw_text(frame, f"{key}: {name} fps={fps_meter.fps:.1f} avg={average_fps:.1f}{target}")
    for index, message in enumerate(provider.status_messages()[:2]):
        output = draw_text(output, message, origin=(16, 56 + index * 24))
    return output
