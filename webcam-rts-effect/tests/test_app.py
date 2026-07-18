import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np

from webcam_effect.analyzer import AnalysisResult
from webcam_effect.app import (
    BenchmarkTimer,
    PEACE_CALIBRATION_KEY,
    apply_peace_sign_blur,
    audio_for_effect_definition,
    handle_peace_calibration_key,
    preview_key_to_code,
    track_hands_for_analysis,
)
from webcam_effect.audio import NullAudio
from webcam_effect.components import ComponentSettings
from webcam_effect.effects import EffectDefinition
from webcam_effect.hand_tracking import HandTrackFrame, PeaceSignCalibration
from webcam_effect.tracking import BoundingBox

class SpyHandTracker:
    def __init__(self):
        self.calls = []

    def track(self, frame):
        self.calls.append(("track", frame.shape))
        return HandTrackFrame(hands=())

    def track_in_box(self, frame, box):
        self.calls.append(("track_in_box", box))
        return HandTrackFrame(hands=())


class AppTest(unittest.TestCase):
    def test_preview_key_to_code_accepts_single_character(self):
        self.assertEqual(preview_key_to_code("p"), ord("p"))

    def test_preview_key_to_code_rejects_multiple_characters(self):
        with self.assertRaises(ValueError):
            preview_key_to_code("preview")

    def test_audio_for_effect_definition_uses_fallback_audio_when_no_tracks(self):
        effect = replace(EffectDefinition(), audio_tracks=(), audio_volume=0.4, audio_loop=False)

        audio = audio_for_effect_definition(effect, fallback_audio="assets/song.mp3")

        self.assertEqual(len(audio.players), 1)
        self.assertEqual(str(audio.players[0].audio_path), "assets/song.mp3")
        self.assertEqual(audio.players[0].volume, 0.4)
        self.assertFalse(audio.players[0].loop)

    def test_audio_for_effect_definition_can_disable_audio_backend(self):
        effect = replace(EffectDefinition(), audio_tracks=())

        audio = audio_for_effect_definition(effect, fallback_audio="assets/song.mp3", audio_enabled=False)

        self.assertIsInstance(audio, NullAudio)
        audio.start()
        audio.stop()
        audio.close()


    def test_track_hands_uses_box_when_segmentation_enabled(self):
        tracker = SpyHandTracker()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        box = BoundingBox(1, 1, 3, 3, 0.9)
        analysis = AnalysisResult(predictions=[], active=False, crop_visible=True, box=box)

        hands, mode = track_hands_for_analysis(tracker, frame, analysis, ComponentSettings(segment=True))

        self.assertEqual(tracker.calls, [("track_in_box", box)])
        self.assertEqual(hands, HandTrackFrame(hands=()))
        self.assertEqual(mode, "bbox")

    def test_track_hands_uses_full_frame_without_segmentation_box(self):
        tracker = SpyHandTracker()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        analysis = AnalysisResult(predictions=[], active=False, crop_visible=False)

        hands, mode = track_hands_for_analysis(tracker, frame, analysis, ComponentSettings(segment=False))

        self.assertEqual(tracker.calls, [("track", frame.shape)])
        self.assertEqual(hands, HandTrackFrame(hands=()))
        self.assertEqual(mode, "full")

    def test_track_hands_can_force_full_frame_with_segmentation_box(self):
        tracker = SpyHandTracker()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        analysis = AnalysisResult(predictions=[], active=False, crop_visible=True, box=BoundingBox(1, 1, 3, 3, 0.9))

        _, mode = track_hands_for_analysis(tracker, frame, analysis, ComponentSettings(segment=True), hand_track_input="full")

        self.assertEqual(tracker.calls, [("track", frame.shape)])
        self.assertEqual(mode, "full")

    def test_track_hands_can_skip_when_forced_bbox_is_missing(self):
        tracker = SpyHandTracker()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        analysis = AnalysisResult(predictions=[], active=False, crop_visible=False)

        hands, mode = track_hands_for_analysis(tracker, frame, analysis, ComponentSettings(segment=True), hand_track_input="bbox")

        self.assertIsNone(hands)
        self.assertEqual(mode, "skipped")
        self.assertEqual(tracker.calls, [])

    @patch("webcam_effect.app.is_peace_sign", return_value=True)
    def test_peace_sign_blurs_frame(self, _is_peace_sign):
        checkerboard = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
        frame = np.repeat(checkerboard[:, :, None], 3, axis=2)
        hands = HandTrackFrame(hands=(object(),))

        blurred = apply_peace_sign_blur(frame, hands)

        self.assertEqual(blurred.shape, frame.shape)
        self.assertIsNot(blurred, frame)
        self.assertEqual(blurred.dtype, frame.dtype)
        self.assertLess(blurred.var(), frame.var())

    def test_peace_sign_blur_reuses_frame_without_hands(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        self.assertIs(apply_peace_sign_blur(frame, HandTrackFrame(hands=())), frame)

    def test_peace_calibration_shortcut_toggles_mode(self):
        calibration = PeaceSignCalibration()

        active, unchanged = handle_peace_calibration_key(
            False,
            calibration,
            HandTrackFrame(hands=()),
            PEACE_CALIBRATION_KEY,
        )

        self.assertTrue(active)
        self.assertIs(unchanged, calibration)

    def test_peace_calibration_shortcuts_label_current_hand(self):
        hand = object()
        for key, expected in (("y", True), ("n", False)):
            with self.subTest(key=key):
                calibration = MagicMock()
                calibration.add.return_value = calibration

                active, updated = handle_peace_calibration_key(
                    True,
                    calibration,
                    HandTrackFrame(hands=(hand,)),
                    ord(key),
                )

                self.assertTrue(active)
                self.assertIs(updated, calibration)
                calibration.add.assert_called_once_with(hand, positive=expected)

    def test_benchmark_timer_reports_average_milliseconds(self):
        timer = BenchmarkTimer(frame_count=2)
        timer.add_analysis(0.02)
        timer.add_hand(0.01, "bbox")
        timer.add_output(0.004)

        report = timer.report(elapsed_seconds=0.1)

        self.assertIn("20.0 fps", report[0])
        self.assertIn("analysis=10.00ms", report[1])
        self.assertIn("hand=5.00ms", report[1])
        self.assertIn("bbox=1", report[2])

if __name__ == "__main__":
    unittest.main()
