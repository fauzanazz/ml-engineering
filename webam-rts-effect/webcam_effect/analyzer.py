from dataclasses import dataclass
import threading

from webcam_effect.frame_window import FrameWindow
from webcam_effect.state import PosePrediction, PoseStateMachine
from webcam_effect.tracking import best_person_box, crop_box
from webcam_effect.yolo_models import YoloPersonDetector

@dataclass(frozen=True)
class AnalysisResult:
    predictions: list[PosePrediction]
    active: bool
    crop_visible: bool

class EffectAnalyzer:
    def __init__(self, segmenter_backend: str, segmenter, classifier, state: PoseStateMachine, frame_window: FrameWindow):
        self.segmenter_backend = segmenter_backend
        self.segmenter = segmenter
        self.classifier = classifier
        self.state = state
        self.frame_window = frame_window

    def analyze(self, frame, segmentation_input: str) -> AnalysisResult:
        crop = crop_user(frame, self.segmenter_backend, self.segmenter, segmentation_input)
        if crop is not None:
            self.frame_window.append(crop)

        predictions = []
        if self.frame_window.ready:
            predictions = self.classifier.predict_window(self.frame_window.frames())
            active = self.state.update(predictions)
        else:
            active = self.state.active

        return AnalysisResult(predictions=predictions, active=active, crop_visible=crop is not None)

class AsyncLatestAnalyzer:
    def __init__(self, analyzer: EffectAnalyzer, segmentation_input: str):
        self.analyzer = analyzer
        self.segmentation_input = segmentation_input
        self._condition = threading.Condition()
        self._latest_frame = None
        self._result = AnalysisResult(predictions=[], active=False, crop_visible=False)
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        with self._condition:
            if self._closed:
                return
            self._latest_frame = frame.copy()
            self._condition.notify()

    def result(self) -> AnalysisResult:
        with self._condition:
            return self._result

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest_frame is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                frame = self._latest_frame
                self._latest_frame = None

            result = self.analyzer.analyze(frame, self.segmentation_input)
            with self._condition:
                self._result = result

def crop_user(frame, backend: str, segmenter, segmentation_input: str):
    if backend == "yolo":
        return _crop_best_person(frame, segmenter)
    if backend == "mediapipe":
        return segmenter.crop(frame, segmentation_input=segmentation_input)
    raise ValueError(f"unknown segmenter backend: {backend}")

def _crop_best_person(frame, detector: YoloPersonDetector):
    box = best_person_box(detector.detect(frame))
    if box is None:
        return None
    return crop_box(frame, box)
