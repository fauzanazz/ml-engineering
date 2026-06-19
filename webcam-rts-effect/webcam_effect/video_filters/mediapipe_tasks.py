from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MediaPipeTaskPaths:
    face: Path = Path("assets/face_landmarker.task")
    hand: Path = Path("assets/hand_landmarker.task")
    pose: Path = Path("assets/pose_landmarker_lite.task")
    segmenter: Path = Path("assets/selfie_segmenter.tflite")


@dataclass
class StaticDetectionProvider:
    mask: object | None = None
    face: tuple = ()
    hands: tuple = ()
    pose: tuple = ()

    def person_mask(self, frame, timestamp_ms: int):
        return self.mask

    def face_landmarks(self, frame, timestamp_ms: int):
        return self.face

    def hand_landmarks(self, frame, timestamp_ms: int):
        return self.hands

    def pose_landmarks(self, frame, timestamp_ms: int):
        return self.pose

    def status_messages(self) -> list[str]:
        return []

    def close(self) -> None:
        return None


@dataclass
class MediaPipeDetectionProvider:
    paths: MediaPipeTaskPaths = field(default_factory=MediaPipeTaskPaths)
    frame_skip: int = 0
    inference_scale: float = 1.0

    def __post_init__(self) -> None:
        self._tasks = {}
        self._cache = {}
        self._missing = {}
        self._last_frame_index = {}
        self._last_timestamp_ms = {}
        self._frame_index = 0

    def person_mask(self, frame, timestamp_ms: int):
        pose_result = self._result("pose", frame, timestamp_ms)
        if pose_result is not None and getattr(pose_result, "segmentation_masks", None):
            return pose_result.segmentation_masks[0].numpy_view()

        segment_result = self._result("segmenter", frame, timestamp_ms)
        if segment_result is None or segment_result.category_mask is None:
            return None
        return (segment_result.category_mask.numpy_view() == 1).astype("float32")

    def face_landmarks(self, frame, timestamp_ms: int):
        result = self._result("face", frame, timestamp_ms)
        if result is None or not result.face_landmarks:
            return ()
        return tuple(result.face_landmarks[0])

    def hand_landmarks(self, frame, timestamp_ms: int):
        result = self._result("hand", frame, timestamp_ms)
        if result is None or not result.hand_landmarks:
            return ()
        return tuple(tuple(hand) for hand in result.hand_landmarks)

    def pose_landmarks(self, frame, timestamp_ms: int):
        result = self._result("pose", frame, timestamp_ms)
        if result is None or not result.pose_landmarks:
            return ()
        return tuple(result.pose_landmarks[0])

    def status_messages(self) -> list[str]:
        return [message for message in self._missing.values() if message]

    def close(self) -> None:
        for task in self._tasks.values():
            task.close()

    def next_frame(self) -> None:
        self._frame_index += 1

    def _result(self, kind: str, frame, timestamp_ms: int):
        if not self._should_run(kind):
            return self._cache.get(kind)
        task = self._task(kind)
        if task is None:
            return None
        timestamp = self._monotonic_timestamp(kind, timestamp_ms)
        image = self._mp_image(frame)
        try:
            if kind == "segmenter":
                result = task.segment_for_video(image, timestamp)
            else:
                result = task.detect_for_video(image, timestamp)
        except Exception as exc:  # MediaPipe can fail per model/frame; keep preview alive.
            self._missing[kind] = f"{kind} detector failed: {exc}"
            return self._cache.get(kind)
        self._cache[kind] = result
        self._last_frame_index[kind] = self._frame_index
        return result

    def _should_run(self, kind: str) -> bool:
        if kind not in self._cache:
            return True
        skip = max(0, self.frame_skip)
        return self._frame_index - self._last_frame_index.get(kind, -1) > skip

    def _task(self, kind: str):
        if kind in self._tasks:
            return self._tasks[kind]
        path = getattr(self.paths, kind)
        if not path.exists():
            self._missing[kind] = f"missing {kind} model: {path}"
            return None
        try:
            task = self._create_task(kind, path)
        except Exception as exc:
            self._missing[kind] = f"could not load {kind} model: {exc}"
            return None
        self._tasks[kind] = task
        self._missing.pop(kind, None)
        return task

    def _create_task(self, kind: str, path: Path):
        import mediapipe as mp

        base_options = mp.tasks.BaseOptions(model_asset_path=str(path))
        running_mode = mp.tasks.vision.RunningMode.VIDEO
        if kind == "face":
            options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options, running_mode=running_mode, num_faces=1)
            return mp.tasks.vision.FaceLandmarker.create_from_options(options)
        if kind == "hand":
            options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, running_mode=running_mode, num_hands=2)
            return mp.tasks.vision.HandLandmarker.create_from_options(options)
        if kind == "pose":
            options = mp.tasks.vision.PoseLandmarkerOptions(base_options=base_options, running_mode=running_mode, num_poses=1, output_segmentation_masks=True)
            return mp.tasks.vision.PoseLandmarker.create_from_options(options)
        options = mp.tasks.vision.ImageSegmenterOptions(base_options=base_options, running_mode=running_mode, output_category_mask=True)
        return mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def _mp_image(self, frame):
        import cv2
        import mediapipe as mp

        scale = min(1.0, max(0.1, self.inference_scale))
        source = frame if scale == 1.0 else cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _monotonic_timestamp(self, kind: str, timestamp_ms: int) -> int:
        previous = self._last_timestamp_ms.get(kind, -1)
        current = max(int(timestamp_ms), previous + 1)
        self._last_timestamp_ms[kind] = current
        return current
