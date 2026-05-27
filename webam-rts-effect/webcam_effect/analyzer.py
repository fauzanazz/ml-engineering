from dataclasses import dataclass
import threading

from webcam_effect.components import ComponentSettings
from webcam_effect.frame_window import FrameWindow
from webcam_effect.state import PosePrediction, PoseStateMachine
from webcam_effect.tracking import BoundingBox, best_person_box, crop_box
from webcam_effect.yolo_models import YoloPersonDetector

@dataclass(frozen=True)
class AnalysisResult:
    predictions: list[PosePrediction]
    active: bool
    crop_visible: bool
    crop: object | None = None
    box: BoundingBox | None = None

@dataclass(frozen=True)
class SegmentedUser:
    crop: object
    box: BoundingBox

class EffectAnalyzer:
    def __init__(
        self,
        segmenter_backend: str,
        segmenter,
        classifier,
        state: PoseStateMachine,
        frame_window: FrameWindow,
        components: ComponentSettings | None = None,
    ):
        self.segmenter_backend = segmenter_backend
        self.segmenter = segmenter
        self.classifier = classifier
        self.state = state
        self.frame_window = frame_window
        self.components = components or ComponentSettings()

    def analyze(self, frame, segmentation_input: str) -> AnalysisResult:
        crop = None
        box = None
        classifier_input = frame
        if self.components.segment:
            segmented_user = segment_user(frame, self.segmenter_backend, self.segmenter, segmentation_input)
            if segmented_user is not None:
                crop = segmented_user.crop
                box = segmented_user.box
            classifier_input = crop

        if self.components.classify and classifier_input is not None:
            self.frame_window.append(classifier_input)

        predictions = []
        if self.components.classify and self.frame_window.ready:
            frames = self.frame_window.frames()
            predictions = self.classifier.predict_window(frames)
            active = self.state.update(predictions)
        else:
            active = self.state.active if self.components.classify else False

        return AnalysisResult(
            predictions=predictions,
            active=active,
            crop_visible=crop is not None,
            crop=crop,
            box=box,
        )

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

def segment_user(frame, backend: str, segmenter, segmentation_input: str) -> SegmentedUser | None:
    if backend == "yolo":
        return _segment_best_person(frame, segmenter)
    if backend == "yolo-seg":
        result = segmenter.segment(frame, segmentation_input=segmentation_input)
        if result is None:
            return None
        return SegmentedUser(crop=result.crop, box=result.box)
    if backend == "mediapipe":
        result = segmenter.segment(frame)
        if result is None:
            return None
        if segmentation_input == "crop":
            crop = crop_box(frame, result.box)
        elif segmentation_input == "masked-crop":
            crop = segmenter.crop(frame, segmentation_input=segmentation_input)
        else:
            raise ValueError(f"unknown segmentation input: {segmentation_input}")
        if crop is None:
            return None
        return SegmentedUser(crop=crop, box=result.box)
    raise ValueError(f"unknown segmenter backend: {backend}")

def crop_user(frame, backend: str, segmenter, segmentation_input: str):
    result = segment_user(frame, backend, segmenter, segmentation_input)
    if result is None:
        return None
    return result.crop

def _segment_best_person(frame, detector: YoloPersonDetector) -> SegmentedUser | None:
    box = best_person_box(detector.detect(frame))
    if box is None:
        return None
    crop = crop_box(frame, box)
    if crop is None:
        return None
    return SegmentedUser(crop=crop, box=box)
