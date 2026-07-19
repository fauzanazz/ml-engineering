from dataclasses import dataclass
import json
from math import dist
from pathlib import Path

from webcam_effect.tracking import BoundingBox

WRIST_LANDMARK = 0
INDEX_MCP_LANDMARK = 5
INDEX_PIP_LANDMARK = 6
THUMB_TIP_LANDMARK = 4
THUMB_IP_LANDMARK = 3
INDEX_TIP_LANDMARK = 8
MIDDLE_MCP_LANDMARK = 9
MIDDLE_PIP_LANDMARK = 10
MIDDLE_TIP_LANDMARK = 12
RING_MCP_LANDMARK = 13
RING_PIP_LANDMARK = 14
RING_TIP_LANDMARK = 16
PINKY_MCP_LANDMARK = 17
PINKY_PIP_LANDMARK = 18
PINKY_TIP_LANDMARK = 20
FINGERTIP_LANDMARKS = (
    THUMB_TIP_LANDMARK,
    INDEX_TIP_LANDMARK,
    MIDDLE_TIP_LANDMARK,
    RING_TIP_LANDMARK,
    PINKY_TIP_LANDMARK,
)
PEACE_EXTENSION_RATIO = 1.1
PEACE_SEPARATION_RATIO = 0.35
PEACE_THUMB_RATIO = 1.1
PEACE_FEATURE_COUNT = 6


@dataclass(frozen=True)
class HandLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class TrackedHand:
    label: str
    confidence: float
    landmarks: tuple[HandLandmark, ...]
    box: BoundingBox

    @property
    def center(self) -> HandLandmark:
        return hand_center(self.landmarks)


@dataclass(frozen=True)
class HandTrackFrame:
    hands: tuple[TrackedHand, ...]

    def by_label(self, label: str) -> TrackedHand | None:
        for hand in self.hands:
            if hand.label == label:
                return hand
        return None


@dataclass
class MediaPipeHandTracker:
    model_path: str
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        import mediapipe as mp

        self._validate_model_path()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def track(self, frame) -> HandTrackFrame:
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _media_pipe_image(rgb_frame)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return HandTrackFrame(hands=())

        height, width = frame.shape[:2]
        handedness = result.handedness or [None] * len(result.hand_landmarks)
        hands = tuple(
            tracked_hand_from_mediapipe(hand_landmarks, handedness[index], width, height)
            for index, hand_landmarks in enumerate(result.hand_landmarks)
        )
        return HandTrackFrame(hands=hands)

    def track_in_box(self, frame, box: BoundingBox) -> HandTrackFrame:
        from webcam_effect.tracking import crop_box

        clipped_box = clip_box(box, frame_width=frame.shape[1], frame_height=frame.shape[0])
        crop = crop_box(frame, clipped_box)
        if crop is None:
            return HandTrackFrame(hands=())
        return remap_hand_track_frame(
            self.track(crop),
            box=clipped_box,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )

    def close(self) -> None:
        self._landmarker.close()

    def _validate_model_path(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {self.model_path}. "
                "Pass --hand-model with a Hand Landmarker .task file."
            )


def tracked_hand_from_mediapipe(hand_landmarks, handedness, frame_width: int, frame_height: int) -> TrackedHand:
    category = handedness[0] if handedness else None
    landmarks = tuple(
        HandLandmark(x=float(landmark.x), y=float(landmark.y), z=float(landmark.z))
        for landmark in hand_landmarks
    )
    label = handedness_label(category)
    confidence = float(category.score) if category is not None else 0.0
    return TrackedHand(
        label=label,
        confidence=confidence,
        landmarks=landmarks,
        box=hand_box(landmarks, frame_width=frame_width, frame_height=frame_height, confidence=confidence),
    )


def hand_box(
    landmarks: tuple[HandLandmark, ...],
    frame_width: int,
    frame_height: int,
    confidence: float = 1.0,
) -> BoundingBox:
    if not landmarks:
        return BoundingBox(x1=0, y1=0, x2=0, y2=0, confidence=0.0)

    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]
    x1 = int(max(0.0, min(xs)) * frame_width)
    y1 = int(max(0.0, min(ys)) * frame_height)
    x2 = int(min(1.0, max(xs)) * frame_width)
    y2 = int(min(1.0, max(ys)) * frame_height)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def hand_center(landmarks: tuple[HandLandmark, ...]) -> HandLandmark:
    if not landmarks:
        return HandLandmark(x=0.0, y=0.0, z=0.0)

    count = len(landmarks)
    return HandLandmark(
        x=sum(landmark.x for landmark in landmarks) / count,
        y=sum(landmark.y for landmark in landmarks) / count,
        z=sum(landmark.z for landmark in landmarks) / count,
    )


def hand_flapped(frames: list[HandTrackFrame], label: str, threshold: float = 0.06) -> bool:
    hands = [hand for frame in frames if (hand := frame.by_label(label)) is not None]
    centers = [hand.center for hand in hands]
    if len(centers) < 2:
        return False
    return max(center.y for center in centers) - min(center.y for center in centers) >= threshold


def fingertip_spread(hand: TrackedHand) -> float:
    fingertips = [hand.landmarks[index] for index in FINGERTIP_LANDMARKS if index < len(hand.landmarks)]
    if len(fingertips) < 2:
        return 0.0

    xs = [landmark.x for landmark in fingertips]
    ys = [landmark.y for landmark in fingertips]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5


@dataclass(frozen=True)
class PeaceSignCalibration:
    positive_samples: tuple[tuple[float, ...], ...] = ()
    negative_samples: tuple[tuple[float, ...], ...] = ()

    @property
    def ready(self) -> bool:
        return bool(self.positive_samples and self.negative_samples)

    def add(self, hand: TrackedHand, positive: bool) -> "PeaceSignCalibration":
        features = peace_sign_features(hand)
        if features is None:
            return self
        if positive:
            return PeaceSignCalibration(self.positive_samples + (features,), self.negative_samples)
        return PeaceSignCalibration(self.positive_samples, self.negative_samples + (features,))

    def matches(self, hand: TrackedHand) -> bool:
        features = peace_sign_features(hand)
        if features is None:
            return False
        if not self.ready:
            return _matches_default_peace_thresholds(features)

        positive_center = _feature_center(self.positive_samples)
        negative_center = _feature_center(self.negative_samples)
        return _squared_distance(features, positive_center) <= _squared_distance(features, negative_center)


def load_peace_sign_calibration(path: Path) -> PeaceSignCalibration:
    if not path.exists():
        return PeaceSignCalibration()

    data = json.loads(path.read_text())
    try:
        return PeaceSignCalibration(
            positive_samples=_calibration_samples(data.get("positive", [])),
            negative_samples=_calibration_samples(data.get("negative", [])),
        )
    except ValueError:
        return PeaceSignCalibration()


def save_peace_sign_calibration(path: Path, calibration: PeaceSignCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "positive": calibration.positive_samples,
                "negative": calibration.negative_samples,
            },
            indent=2,
        )
        + "\n"
    )


def peace_sign_features(hand: TrackedHand) -> tuple[float, ...] | None:
    if len(hand.landmarks) < 21:
        return None

    landmarks = hand.landmarks

    def point(index: int) -> tuple[float, float, float]:
        landmark = landmarks[index]
        return landmark.x, landmark.y, landmark.z

    wrist = point(WRIST_LANDMARK)
    palm_width = dist(point(INDEX_MCP_LANDMARK), point(PINKY_MCP_LANDMARK))
    pip_distances = tuple(
        dist(point(index), wrist)
        for index in (INDEX_PIP_LANDMARK, MIDDLE_PIP_LANDMARK, RING_PIP_LANDMARK, PINKY_PIP_LANDMARK)
    )
    thumb_ip_distance = dist(point(THUMB_IP_LANDMARK), wrist)
    if palm_width == 0 or 0 in pip_distances or thumb_ip_distance == 0:
        return None

    return (
        dist(point(INDEX_TIP_LANDMARK), wrist) / pip_distances[0],
        dist(point(MIDDLE_TIP_LANDMARK), wrist) / pip_distances[1],
        dist(point(RING_TIP_LANDMARK), wrist) / pip_distances[2],
        dist(point(PINKY_TIP_LANDMARK), wrist) / pip_distances[3],
        dist(point(INDEX_TIP_LANDMARK), point(MIDDLE_TIP_LANDMARK)) / palm_width,
        dist(point(THUMB_TIP_LANDMARK), wrist) / thumb_ip_distance,
    )


def _calibration_samples(samples) -> tuple[tuple[float, ...], ...]:
    parsed = tuple(tuple(float(value) for value in sample) for sample in samples)
    if any(len(sample) != PEACE_FEATURE_COUNT for sample in parsed):
        raise ValueError(f"peace calibration samples must contain {PEACE_FEATURE_COUNT} features")
    return parsed


def _feature_center(samples: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(sum(values) / len(samples) for values in zip(*samples))


def _squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))


def _matches_default_peace_thresholds(features: tuple[float, ...]) -> bool:
    index, middle, ring, pinky, separation, thumb = features
    return (
        index > PEACE_EXTENSION_RATIO
        and middle > PEACE_EXTENSION_RATIO
        and ring <= PEACE_EXTENSION_RATIO
        and pinky <= PEACE_EXTENSION_RATIO
        and separation >= PEACE_SEPARATION_RATIO
        and thumb <= PEACE_THUMB_RATIO
    )


def is_peace_sign(hand: TrackedHand, calibration: PeaceSignCalibration | None = None) -> bool:
    if calibration is not None:
        return calibration.matches(hand)
    features = peace_sign_features(hand)
    return features is not None and _matches_default_peace_thresholds(features)


def remap_hand_track_frame(hands: HandTrackFrame, box: BoundingBox, frame_width: int, frame_height: int) -> HandTrackFrame:
    return HandTrackFrame(
        hands=tuple(remap_tracked_hand(hand, box, frame_width=frame_width, frame_height=frame_height) for hand in hands.hands)
    )


def remap_tracked_hand(hand: TrackedHand, box: BoundingBox, frame_width: int, frame_height: int) -> TrackedHand:
    crop_width = max(1, box.x2 - box.x1)
    crop_height = max(1, box.y2 - box.y1)
    landmarks = tuple(
        HandLandmark(
            x=(box.x1 + landmark.x * crop_width) / frame_width,
            y=(box.y1 + landmark.y * crop_height) / frame_height,
            z=landmark.z,
        )
        for landmark in hand.landmarks
    )
    return TrackedHand(
        label=hand.label,
        confidence=hand.confidence,
        landmarks=landmarks,
        box=BoundingBox(
            x1=box.x1 + hand.box.x1,
            y1=box.y1 + hand.box.y1,
            x2=box.x1 + hand.box.x2,
            y2=box.y1 + hand.box.y2,
            confidence=hand.box.confidence,
        ),
    )


def clip_box(box: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
    return BoundingBox(
        x1=max(0, min(frame_width, box.x1)),
        y1=max(0, min(frame_height, box.y1)),
        x2=max(0, min(frame_width, box.x2)),
        y2=max(0, min(frame_height, box.y2)),
        confidence=box.confidence,
    )


def handedness_label(category) -> str:
    if category is None:
        return "unknown"
    label = getattr(category, "category_name", None) or getattr(category, "label", None) or getattr(category, "display_name", None)
    return str(label or "unknown").lower()


def _media_pipe_image(rgb_frame):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
