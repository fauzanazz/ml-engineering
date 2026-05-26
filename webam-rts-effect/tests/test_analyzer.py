import threading
import time
import unittest

import numpy as np

from webcam_effect.analyzer import AnalysisResult, AsyncLatestAnalyzer, EffectAnalyzer
from webcam_effect.frame_window import FrameWindow
from webcam_effect.state import PosePrediction, PoseStateMachine

class StaticSegmenter:
    def crop(self, frame, segmentation_input: str):
        return frame

class LabelClassifier:
    def predict_window(self, frames: list) -> list[PosePrediction]:
        return [PosePrediction(label="kicau", confidence=0.9) for _ in frames]

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
            frame_window=FrameWindow(size=1),
        )

        result = analyzer.analyze(np.zeros((2, 2, 3), dtype=np.uint8), "masked-crop")

        self.assertTrue(result.active)
        self.assertTrue(result.crop_visible)
        self.assertEqual(result.predictions, [PosePrediction(label="kicau", confidence=0.9)])

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
