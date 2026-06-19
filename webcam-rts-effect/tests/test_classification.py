import unittest

from webcam_effect.classification import TemporalKicauClassifier
from webcam_effect.state import PosePrediction


class FakeFrameClassifier:
    def __init__(self, predictions):
        self.predictions = list(predictions)

    def predict(self, frame):
        return self.predictions.pop(0)

class CountingFrameClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, frame):
        self.calls.append(frame)
        return PosePrediction(label=frame, confidence=1.0)


class TemporalKicauClassifierTest(unittest.TestCase):
    def test_classifies_each_frame_in_window(self):
        frame_classifier = FakeFrameClassifier(
            [
                PosePrediction(label="none", confidence=0.8),
                PosePrediction(label="kicau", confidence=0.7),
                PosePrediction(label="kicau", confidence=0.9),
            ]
        )
        classifier = TemporalKicauClassifier(frame_classifier)

        predictions = classifier.predict_window(["t-2", "t-1", "t"])

        self.assertEqual(
            predictions,
            [
                PosePrediction(label="none", confidence=0.8),
                PosePrediction(label="kicau", confidence=0.7),
                PosePrediction(label="kicau", confidence=0.9),
            ],
        )

    def test_reuses_predictions_for_overlapping_window_frames(self):
        frame_classifier = CountingFrameClassifier()
        classifier = TemporalKicauClassifier(frame_classifier)

        first_predictions = classifier.predict_window(["a", "b", "c"])
        second_predictions = classifier.predict_window(["b", "c", "d"])

        self.assertEqual([prediction.label for prediction in first_predictions], ["a", "b", "c"])
        self.assertEqual([prediction.label for prediction in second_predictions], ["b", "c", "d"])
        self.assertEqual(frame_classifier.calls, ["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
