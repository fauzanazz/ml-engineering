from typing import Protocol

from webcam_effect.state import PosePrediction


class FrameClassifier(Protocol):
    def predict(self, frame) -> PosePrediction:
        pass


class WindowClassifier(Protocol):
    def predict_window(self, frames: list) -> list[PosePrediction]:
        pass


class TemporalKicauClassifier:
    def __init__(self, frame_classifier: FrameClassifier):
        self.frame_classifier = frame_classifier
        self._prediction_by_frame_id = {}

    def predict_window(self, frames: list) -> list[PosePrediction]:
        live_frame_ids = {id(frame) for frame in frames}
        self._prediction_by_frame_id = {
            frame_id: prediction
            for frame_id, prediction in self._prediction_by_frame_id.items()
            if frame_id in live_frame_ids
        }

        predictions = []
        for frame in frames:
            frame_id = id(frame)
            if frame_id not in self._prediction_by_frame_id:
                self._prediction_by_frame_id[frame_id] = self.frame_classifier.predict(frame)
            predictions.append(self._prediction_by_frame_id[frame_id])
        return predictions


class StaticWindowClassifier:
    def __init__(self, prediction: PosePrediction):
        self.prediction = prediction

    def predict_window(self, frames: list) -> list[PosePrediction]:
        return [self.prediction for _ in frames]
