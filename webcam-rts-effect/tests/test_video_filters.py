import inspect
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from webcam_effect.camera import parse_resolution
from webcam_effect.video_filters.base import FilterAssets, asset_path
from webcam_effect.video_filters.background_replace import BackgroundReplaceFilter
from webcam_effect.video_filters.beauty_soften import LEFT_EYEBROW, RIGHT_EYEBROW
from webcam_effect.video_filters.drawing import load_overlay
from webcam_effect.video_filters.face_sticker import FaceStickerFilter
from webcam_effect.video_filters.mediapipe_tasks import MediaPipeDetectionProvider, MediaPipeTaskPaths, StaticDetectionProvider
from webcam_effect.video_filters.neon_face_mesh import _face_mesh_connections
from webcam_effect.video_filters.registry import build_filters, filter_names


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def gradient_frame(width: int = 160, height: int = 120):
    y, x = np.indices((height, width))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = x % 255
    frame[:, :, 1] = y % 255
    frame[:, :, 2] = (x + y) % 255
    return frame


def fake_provider():
    mask = np.zeros((120, 160), dtype=np.float32)
    mask[25:95, 50:110] = 1.0
    face = [Point(0.5, 0.5) for _ in range(478)]
    for index, point in {
        10: Point(0.5, 0.22),
        33: Point(0.35, 0.42),
        133: Point(0.43, 0.42),
        234: Point(0.25, 0.52),
        263: Point(0.65, 0.42),
        362: Point(0.57, 0.42),
        454: Point(0.75, 0.52),
    }.items():
        face[index] = point
    hand = tuple(Point(0.2 + index * 0.01, 0.3 + index * 0.01) for index in range(21))
    pose = tuple(Point(0.5, 0.2 + index * 0.02) for index in range(25))
    return StaticDetectionProvider(mask=mask, face=tuple(face), hands=(hand,), pose=pose)


class VideoFiltersTest(unittest.TestCase):
    def test_registry_exposes_ten_keyboard_filters(self):
        filters = build_filters(fake_provider(), FilterAssets())

        self.assertEqual(list(filters), list("1234567890"))
        self.assertEqual(
            filter_names(filters),
            [
                "background_blur",
                "virtual_glasses",
                "neon_face_mesh",
                "beauty_soften",
                "cartoon_face",
                "hand_magic_trail",
                "pose_aura",
                "face_sticker",
                "background_replace",
                "background_blur_lite",
            ],
        )

    def test_filters_share_process_frame_timestamp_interface(self):
        filters = build_filters(fake_provider(), FilterAssets())

        for filter_item in filters.values():
            self.assertEqual(list(inspect.signature(filter_item.process).parameters), ["frame", "timestamp_ms"])

    def test_each_filter_processes_frame_without_changing_shape(self):
        frame = gradient_frame()
        filters = build_filters(fake_provider(), FilterAssets())

        for filter_item in filters.values():
            output = filter_item.process(frame, 100)
            self.assertEqual(output.shape, frame.shape, filter_item.spec.name)
            self.assertEqual(output.dtype, frame.dtype, filter_item.spec.name)

    def test_background_blur_lite_has_optimized_fps_target(self):
        frame = gradient_frame()
        lite_filter = build_filters(fake_provider(), FilterAssets())["0"]

        started_at = time.perf_counter()
        for index in range(120):
            lite_filter.process(frame, index * 33)
        fps = 120 / max(time.perf_counter() - started_at, 1e-9)

        self.assertTrue(lite_filter.spec.optimized)
        self.assertEqual(lite_filter.spec.target_fps, 24.0)
        self.assertGreater(fps, 24.0)

    def test_background_replace_accepts_video_frames(self):
        frame = gradient_frame()
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        first = np.full_like(frame, (12, 24, 36))
        second = np.full_like(frame, (90, 120, 150))

        with patch("webcam_effect.video_filters.background_replace.load_sticker_frames", return_value=(first, second)):
            filter_item = BackgroundReplaceFilter(StaticDetectionProvider(mask=mask), Path("assets/example.png"))

        self.assertTrue(np.all(filter_item.process(frame, 0) == first))
        self.assertTrue(np.all(filter_item.process(frame, 33) == second))

    def test_face_sticker_follows_face_rotation(self):
        provider = fake_provider()
        face = list(provider.face)
        face[454] = Point(0.75, 0.62)
        provider.face = tuple(face)

        with patch("webcam_effect.video_filters.face_sticker.load_overlay", return_value=np.ones((8, 16, 4), dtype=np.uint8)):
            filter_item = FaceStickerFilter(provider, Path("assets/nick.gif"))
        with patch("webcam_effect.video_filters.face_sticker.overlay_centered", side_effect=lambda frame, *args, **kwargs: frame.copy()) as overlay:
            filter_item.process(gradient_frame(), 100)

        self.assertGreater(overlay.call_args.kwargs["angle_degrees"], 0.0)

    def test_beauty_soften_protects_eyebrow_landmarks(self):
        self.assertEqual(LEFT_EYEBROW, (70, 63, 105, 66, 107))
        self.assertEqual(RIGHT_EYEBROW, (336, 296, 334, 293, 300))

    def test_neon_face_mesh_uses_tessellation_connections(self):
        self.assertGreater(len(_face_mesh_connections()), 100)

    def test_filter_modules_stay_separate_and_small(self):
        module_dir = Path("webcam_effect/video_filters")
        module_names = set(filter_names(build_filters(fake_provider(), FilterAssets())))
        filter_files = [module_dir / f"{name}.py" for name in module_names]

        self.assertEqual(len(filter_files), 10)
        for path in filter_files:
            self.assertTrue(path.exists(), path)
            self.assertLessEqual(len(path.read_text().splitlines()), 200, path)

    def test_resolution_config_supports_required_sizes(self):
        self.assertEqual(parse_resolution("640x480"), (640, 480))
        self.assertEqual(parse_resolution("1280x720"), (1280, 720))

    def test_asset_paths_default_to_local_assets_folder(self):
        self.assertEqual(asset_path(Path("assets"), "glasses.png"), Path("assets/glasses.png"))
        self.assertEqual(asset_path(Path("assets"), "assets/example.png"), Path("assets/example.png"))

    def test_glasses_asset_loads_with_transparency(self):
        overlay = load_overlay(Path("assets/glasses.ppm"), transparent_white=True)

        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape[2], 4)
        self.assertEqual(int(overlay[:, :, 3].min()), 0)
        self.assertEqual(int(overlay[:, :, 3].max()), 255)

    def test_missing_model_files_report_status_without_crashing(self):
        provider = MediaPipeDetectionProvider(
            paths=MediaPipeTaskPaths(
                face=Path("assets/missing-face.task"),
                hand=Path("assets/missing-hand.task"),
                pose=Path("assets/missing-pose.task"),
                segmenter=Path("assets/missing-segmenter.tflite"),
            )
        )
        frame = gradient_frame()

        self.assertEqual(provider.face_landmarks(frame, 0), ())
        self.assertEqual(provider.hand_landmarks(frame, 0), ())
        self.assertEqual(provider.pose_landmarks(frame, 0), ())
        self.assertIsNone(provider.person_mask(frame, 0))
        self.assertGreaterEqual(len(provider.status_messages()), 4)

    def test_missing_asset_files_fall_back_without_crashing(self):
        frame = gradient_frame()
        filters = build_filters(
            fake_provider(),
            FilterAssets(
                glasses=Path("assets/missing-glasses.ppm"),
                sticker=Path("assets/missing-sticker.gif"),
                background=Path("assets/missing-background.mp4"),
            ),
        )

        for key in ("2", "8", "9"):
            self.assertEqual(filters[key].process(frame, 100).shape, frame.shape)

    def test_mediapipe_timestamps_are_monotonic(self):
        provider = MediaPipeDetectionProvider()

        self.assertEqual(provider._monotonic_timestamp("pose", 10), 10)
        self.assertEqual(provider._monotonic_timestamp("pose", 10), 11)
        self.assertEqual(provider._monotonic_timestamp("pose", 9), 12)

    def test_project_metadata_supports_python_310(self):
        pyproject = Path("pyproject.toml").read_text()

        self.assertIn('requires-python = ">=3.10"', pyproject)
        self.assertIn("onnxruntime>=1.23.2,<1.26.0; python_version == '3.10'", pyproject)


if __name__ == "__main__":
    unittest.main()
