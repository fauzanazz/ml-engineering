import json
from pathlib import Path
import time

from webcam_effect.camera import CameraSource, parse_resolution
from webcam_effect.effects import StickerEffect, load_effect_definition
from webcam_effect.mediapipe_assets import require_models
from webcam_effect.runtime_status import RuntimeStatus, RuntimeStatusWriter
from webcam_effect.state import PoseStateMachine
from webcam_effect.video_filters.base import FilterAssets, asset_path
from webcam_effect.video_filters.mediapipe_tasks import MediaPipeDetectionProvider
from webcam_effect.video_filters.registry import build_filters
from webcam_effect.yolo_models import YoloFrameClassifier, YoloPersonSegmenter


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def run_benchmark(
    duration_seconds: int = 60,
    camera: str = "0",
    resolution: str = "640x480",
    classifier_path: str = "models/kicau-classifier/best.pt",
    detector_path: str = "yolo26n-seg.pt",
    device: str = "mps",
    output_dir: Path = Path("outputs/benchmarks"),
    device_name: str = "Apple M5",
) -> list[dict]:
    if duration_seconds <= 0:
        raise ValueError("duration must be positive")
    classifier_file = Path(classifier_path)
    if not classifier_file.is_file():
        raise FileNotFoundError(f"missing classifier model: {classifier_file}. Run evaluation and export its winner first.")
    require_models()

    width, height = parse_resolution(resolution)
    capture = CameraSource(camera, width=width, height=height).open()
    segmenter = YoloPersonSegmenter(detector_path, device=device)
    classifier = YoloFrameClassifier(classifier_path, device=device)
    state = PoseStateMachine()
    definition = load_effect_definition(Path("assets/effect.json"))
    effect = StickerEffect(
        right_sticker_path=definition.right_sticker,
        left_sticker_path=definition.left_sticker,
        scale=definition.scale,
        right_x=definition.right_x,
        right_y=definition.right_y,
        left_x=definition.left_x,
        left_y=definition.left_y,
        layers=definition.layers,
    )
    provider = MediaPipeDetectionProvider(frame_skip=1, inference_scale=0.5)
    assets = FilterAssets(
        assets_dir=Path("assets"),
        background=asset_path(Path("assets"), "example.png"),
        glasses=asset_path(Path("assets"), "glasses.ppm"),
        sticker=asset_path(Path("assets"), "nick.gif"),
    )
    filters = build_filters(provider, assets)

    def baseline(frame, timestamp_ms):
        return frame

    recent_predictions = []

    def classifier_effect(frame, timestamp_ms):
        segmented = segmenter.segment(frame, segmentation_input="masked-crop")
        if segmented is not None:
            recent_predictions.append(classifier.predict(segmented.crop))
            del recent_predictions[:-3]
        active = state.update(recent_predictions) if len(recent_predictions) == 3 else state.active
        return effect.apply(frame, timestamp_ms / 1000) if active else frame

    scenarios = [
        ("no_effect_baseline", baseline, None),
        ("classifier_effect", classifier_effect, None),
        ("background_blur_lite", filters["0"].process, filters["0"].spec.name),
        ("heaviest_filter", filters["1"].process, filters["1"].spec.name),
    ]
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    status_writer = RuntimeStatusWriter()
    try:
        for scenario_name, process, filter_name in scenarios:
            frame_times = []
            dropped_frames = 0
            frames = 0
            started = time.monotonic()
            status = RuntimeStatus(connected=True, active_filter=filter_name)
            status_writer.update(status, force=True)
            while time.monotonic() - started < duration_seconds:
                frame_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    dropped_frames += 1
                    continue
                timestamp_ms = int((time.monotonic() - started) * 1000)
                process(frame, timestamp_ms)
                provider.next_frame()
                frame_times.append(time.perf_counter() - frame_started)
                frames += 1
                status.processing_fps = 1 / max(frame_times[-1], 1e-9)
                status.dropped_frames = dropped_frames
                status_writer.update(status)
            elapsed = time.monotonic() - started
            result = {
                "scenario": scenario_name,
                "device": device_name,
                "resolution": resolution,
                "duration_seconds": duration_seconds,
                "average_fps": round(frames / max(elapsed, 1e-9), 2),
                "p95_frame_time_ms": round(percentile(frame_times, 0.95) * 1000, 2),
                "dropped_frames": dropped_frames,
                "frames": frames,
                "passed_24_fps": frames / max(elapsed, 1e-9) >= 24,
            }
            (output_dir / f"{scenario_name}.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
    finally:
        provider.close()
        capture.release()
        status_writer.close(RuntimeStatus())
    (output_dir / "summary.json").write_text(json.dumps({"benchmarks": results}, indent=2) + "\n")
    return results
