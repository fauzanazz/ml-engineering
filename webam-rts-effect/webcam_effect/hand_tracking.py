from dataclasses import dataclass
from pathlib import Path

from webcam_effect.tracking import BoundingBox

WRIST_LANDMARK = 0
THUMB_TIP_LANDMARK = 4
INDEX_TIP_LANDMARK = 8
MIDDLE_TIP_LANDMARK = 12
RING_TIP_LANDMARK = 16
PINKY_TIP_LANDMARK = 20
FINGERTIP_LANDMARKS = (
    THUMB_TIP_LANDMARK,
    INDEX_TIP_LANDMARK,
    MIDDLE_TIP_LANDMARK,
    RING_TIP_LANDMARK,
    PINKY_TIP_LANDMARK,
)


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
