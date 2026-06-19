from dataclasses import dataclass
from pathlib import Path

from webcam_effect.state import PosePrediction
from webcam_effect.tracking import BoundingBox, crop_box


@dataclass(frozen=True)
class Landmark2D:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class PoseSnapshot:
    nose: Landmark2D | None
    left_wrist: Landmark2D | None
    right_wrist: Landmark2D | None


@dataclass(frozen=True)
class SegmentationResult:
    box: BoundingBox
    mask: object


@dataclass
class MediaPipeUserSegmenter:
    model_path: str
    threshold: float = 0.3
    padding: int = 24

    def __post_init__(self):
        import mediapipe as mp

        self._validate_model_path()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame) -> list[BoundingBox]:
        result = self.segment(frame)
        if result is None:
            return []
        return [result.box]

    def crop(self, frame, segmentation_input: str = "masked-crop"):
        result = self.segment(frame)
        if result is None:
            return None
        if segmentation_input == "crop":
            return crop_box(frame, result.box)
        if segmentation_input == "masked-crop":
            return masked_crop(frame, result.box, result.mask, threshold=self.threshold)
        raise ValueError(f"unknown segmentation input: {segmentation_input}")

    def segment(self, frame) -> SegmentationResult | None:
        import cv2
        import numpy as np

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _media_pipe_image(rgb_frame)
        result = self.landmarker.detect(mp_image)
        if not result.segmentation_masks:
            return None

        mask = normalize_segmentation_mask(result.segmentation_masks[0].numpy_view())

        rows, cols = np.where(mask > self.threshold)
        if len(rows) == 0 or len(cols) == 0:
            return None

        height, width = frame.shape[:2]
        x1 = max(0, int(cols.min()) - self.padding)
        y1 = max(0, int(rows.min()) - self.padding)
        x2 = min(width, int(cols.max()) + self.padding)
        y2 = min(height, int(rows.max()) + self.padding)
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=1.0)
        return SegmentationResult(box=box, mask=mask)

    def _validate_model_path(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}. "
                "Pass --mediapipe-model with a Pose Landmarker .task file."
            )


@dataclass
class MediaPipeKicauWindowClassifier:
    model_path: str
    nose_distance_threshold: float = 0.16
    flap_threshold: float = 0.06

    def __post_init__(self):
        import mediapipe as mp

        self._validate_model_path()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def predict_window(self, frames: list) -> list[PosePrediction]:
        snapshots = [self._pose_snapshot(frame) for frame in frames]
        prediction = classify_kicau_pose(
            snapshots,
            nose_distance_threshold=self.nose_distance_threshold,
            flap_threshold=self.flap_threshold,
        )
        return [prediction for _ in frames]

    def _pose_snapshot(self, frame) -> PoseSnapshot:
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _media_pipe_image(rgb_frame)
        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return PoseSnapshot(nose=None, left_wrist=None, right_wrist=None)

        landmarks = result.pose_landmarks[0]
        return PoseSnapshot(
            nose=_landmark(landmarks[0]),
            left_wrist=_landmark(landmarks[15]),
            right_wrist=_landmark(landmarks[16]),
        )

    def _validate_model_path(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}. "
                "Pass --mediapipe-model with a Pose Landmarker .task file."
            )


def classify_kicau_pose(
    snapshots: list[PoseSnapshot],
    nose_distance_threshold: float = 0.16,
    flap_threshold: float = 0.06,
) -> PosePrediction:
    if not snapshots:
        return PosePrediction(label="none", confidence=0.0)

    latest = snapshots[-1]
    if latest.nose is None:
        return PosePrediction(label="none", confidence=0.0)

    left_closes_nose = is_close(latest.nose, latest.left_wrist, nose_distance_threshold)
    right_closes_nose = is_close(latest.nose, latest.right_wrist, nose_distance_threshold)
    left_flaps = has_vertical_motion([snapshot.left_wrist for snapshot in snapshots], flap_threshold)
    right_flaps = has_vertical_motion([snapshot.right_wrist for snapshot in snapshots], flap_threshold)

    is_kicau = (left_closes_nose and right_flaps) or (right_closes_nose and left_flaps)
    confidence = 0.85 if is_kicau else 0.0
    return PosePrediction(label="kicau" if is_kicau else "none", confidence=confidence)


def is_close(first: Landmark2D, second: Landmark2D | None, threshold: float) -> bool:
    if second is None or second.visibility < 0.4:
        return False
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5 <= threshold


def has_vertical_motion(landmarks: list[Landmark2D | None], threshold: float) -> bool:
    visible_y = [landmark.y for landmark in landmarks if landmark is not None and landmark.visibility >= 0.4]
    if len(visible_y) < 2:
        return False
    return max(visible_y) - min(visible_y) >= threshold


def crop_segmented_user(frame, segmenter: MediaPipeUserSegmenter):
    return segmenter.crop(frame)


def masked_crop(frame, box: BoundingBox, mask, threshold: float = 0.3):
    import numpy as np

    frame_crop = crop_box(frame, box)
    if frame_crop is None:
        return None

    mask_crop = crop_box(normalize_segmentation_mask(mask), box)
    if mask_crop is None:
        return frame_crop

    alpha = np.clip(mask_crop, 0.0, 1.0)
    alpha = (alpha >= threshold).astype(frame_crop.dtype)[:, :, None]
    return frame_crop * alpha

def normalize_segmentation_mask(mask):
    import numpy as np

    normalized = np.asarray(mask)
    if normalized.ndim == 3 and normalized.shape[-1] == 1:
        return normalized[:, :, 0]
    if normalized.ndim == 2:
        return normalized
    raise ValueError(f"expected 2D segmentation mask, got shape {normalized.shape}")


def _landmark(landmark) -> Landmark2D:
    return Landmark2D(x=float(landmark.x), y=float(landmark.y), visibility=float(landmark.visibility))


def _media_pipe_image(rgb_frame):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
