import threading
import time
import unittest

import numpy as np

from webcam_effect.analyzer import AnalysisResult, AsyncLatestAnalyzer, EffectAnalyzer
from webcam_effect.components import ComponentSettings
from webcam_effect.frame_window import FrameWindow
from webcam_effect.state import PosePrediction, PoseStateMachine
from webcam_effect.tracking import BoundingBox

class StaticSegmenter:
    box = BoundingBox(1, 1, 3, 3, 0.9)

    def segment(self, frame):
        return type("Segment", (), {"box": self.box})()

    def crop(self, frame, segmentation_input: str):
        return frame

class LabelClassifier:
    def predict_window(self, frames: list) -> list[PosePrediction]:
        return [PosePrediction(label="kicau", confidence=0.9) for _ in frames]

class FailingSegmenter:
    def segment(self, frame):
        raise AssertionError("segmenter should not run")

    def crop(self, frame, segmentation_input: str):
        raise AssertionError("segmenter should not run")

class BlockingAnalyzer:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen_values = []

    def analyze(self, frame, segmentation_input: str):
        self.started.set()
        self.release.wait(timeout=1.0)
        self.seen_values.append(int(frame[0, 0, 0]))
        return AnalysisResult(predictions=[], active=False, crop_visible=False)

class EffectAnalyzerTest(unittest.TestCase):
    def test_analyze_updates_state_when_window_ready(self):
        analyzer = EffectAnalyzer(
            segmenter_backend="mediapipe",
            segmenter=StaticSegmenter(),
            classifier=LabelClassifier(),
            state=PoseStateMachine(),
            frame_window=FrameWindow(size=2),
        )

        analyzer.analyze(np.zeros((2, 2, 3), dtype=np.uint8), "masked-crop")
        result = analyzer.analyze(np.zeros((2, 2, 3), dtype=np.uint8), "masked-crop")

        self.assertTrue(result.active)
        self.assertTrue(result.crop_visible)
        self.assertEqual(result.box, StaticSegmenter.box)
        self.assertEqual(result.predictions, [PosePrediction(label="kicau", confidence=0.9), PosePrediction(label="kicau", confidence=0.9)])

    def test_analyze_uses_full_frame_when_segment_disabled(self):
        analyzer = EffectAnalyzer(
            segmenter_backend="mediapipe",
            segmenter=FailingSegmenter(),
            classifier=LabelClassifier(),
            state=PoseStateMachine(),
            frame_window=FrameWindow(size=1),
            components=ComponentSettings(segment=False, classify=True, hand_track=True),
        )

        result = analyzer.analyze(np.zeros((2, 2, 3), dtype=np.uint8), "masked-crop")

        self.assertTrue(result.active)
        self.assertFalse(result.crop_visible)

    def test_analyze_disables_active_state_when_classify_disabled(self):
        state = PoseStateMachine(active=True)
        analyzer = EffectAnalyzer(
            segmenter_backend="mediapipe",
            segmenter=StaticSegmenter(),
            classifier=LabelClassifier(),
            state=state,
            frame_window=FrameWindow(size=1),
            components=ComponentSettings(segment=True, classify=False, hand_track=True),
        )

        result = analyzer.analyze(np.zeros((2, 2, 3), dtype=np.uint8), "masked-crop")

        self.assertFalse(result.active)
        self.assertEqual(result.predictions, [])
        self.assertTrue(result.crop_visible)

class AsyncLatestAnalyzerTest(unittest.TestCase):
    def test_keeps_latest_frame_when_worker_is_busy(self):
        analyzer = BlockingAnalyzer()
        async_analyzer = AsyncLatestAnalyzer(analyzer, "masked-crop")
        try:
            async_analyzer.submit(np.full((1, 1, 3), 1, dtype=np.uint8))
            self.assertTrue(analyzer.started.wait(timeout=1.0))
            async_analyzer.submit(np.full((1, 1, 3), 2, dtype=np.uint8))
            async_analyzer.submit(np.full((1, 1, 3), 3, dtype=np.uint8))

            analyzer.release.set()
            deadline = time.monotonic() + 1.0
            while len(analyzer.seen_values) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            async_analyzer.close()

        self.assertIn(1, analyzer.seen_values)
        self.assertIn(3, analyzer.seen_values)
        self.assertNotIn(2, analyzer.seen_values)

if __name__ == "__main__":
    unittest.main()
