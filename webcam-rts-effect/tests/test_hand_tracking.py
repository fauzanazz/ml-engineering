import unittest

from webcam_effect.hand_tracking import (
    FINGERTIP_LANDMARKS,
    HandLandmark,
    HandTrackFrame,
    MediaPipeHandTracker,
    TrackedHand,
    fingertip_spread,
    remap_hand_track_frame,
    hand_box,
    hand_center,
    hand_flapped,
    handedness_label,
)
from webcam_effect.tracking import BoundingBox


class HandTrackingTest(unittest.TestCase):
    def test_hand_tracker_requires_task_model(self):
        with self.assertRaises(FileNotFoundError):
            MediaPipeHandTracker(model_path="missing.task")

    def test_handedness_label_reads_tasks_category_name(self):
        category = type("Category", (), {"category_name": "Right", "score": 0.9})()

        self.assertEqual(handedness_label(category), "right")

    def test_hand_box_converts_normalized_landmarks_to_pixels(self):
        landmarks = (
            HandLandmark(0.25, 0.2),
            HandLandmark(0.75, 0.8),
        )

        box = hand_box(landmarks, frame_width=640, frame_height=480, confidence=0.9)

        self.assertEqual(box, BoundingBox(x1=160, y1=96, x2=480, y2=384, confidence=0.9))

    def test_hand_box_clamps_to_frame(self):
        landmarks = (
            HandLandmark(-0.2, -0.1),
            HandLandmark(1.2, 1.1),
        )

        box = hand_box(landmarks, frame_width=100, frame_height=50)

        self.assertEqual(box, BoundingBox(x1=0, y1=0, x2=100, y2=50, confidence=1.0))

    def test_hand_center_averages_landmarks(self):
        center = hand_center((HandLandmark(0.2, 0.4, 0.1), HandLandmark(0.6, 0.8, 0.3)))

        self.assertEqual(center, HandLandmark(x=0.4, y=0.6000000000000001, z=0.2))

    def test_track_frame_finds_hand_by_label(self):
        right = hand_with_center("right", 0.5)
        frame = HandTrackFrame(hands=(right,))

        self.assertEqual(frame.by_label("right"), right)
        self.assertIsNone(frame.by_label("left"))

    def test_hand_flapped_detects_vertical_center_motion(self):
        frames = [
            HandTrackFrame(hands=(hand_with_center("right", 0.2),)),
            HandTrackFrame(hands=(hand_with_center("right", 0.28),)),
            HandTrackFrame(hands=(hand_with_center("right", 0.18),)),
        ]

        self.assertTrue(hand_flapped(frames, "right", threshold=0.06))

    def test_hand_flapped_rejects_small_motion(self):
        frames = [
            HandTrackFrame(hands=(hand_with_center("left", 0.2),)),
            HandTrackFrame(hands=(hand_with_center("left", 0.22),)),
        ]

        self.assertFalse(hand_flapped(frames, "left", threshold=0.06))

    def test_fingertip_spread_uses_tip_extents(self):
        landmarks = [HandLandmark(0.2, 0.2) for _ in range(max(FINGERTIP_LANDMARKS) + 1)]
        landmarks[4] = HandLandmark(0.1, 0.1)
        landmarks[8] = HandLandmark(0.4, 0.5)
        hand = TrackedHand("right", 0.9, tuple(landmarks), BoundingBox(0, 0, 1, 1, 0.9))

        self.assertAlmostEqual(fingertip_spread(hand), 0.5)

    def test_remap_hand_track_frame_moves_crop_landmarks_to_full_frame(self):
        crop_hand = TrackedHand(
            "right",
            0.9,
            (HandLandmark(0.5, 0.5),),
            BoundingBox(10, 20, 30, 40, 0.9),
        )
        hands = HandTrackFrame(hands=(crop_hand,))

        remapped = remap_hand_track_frame(
            hands,
            box=BoundingBox(100, 50, 300, 250, 0.8),
            frame_width=400,
            frame_height=300,
        )

        hand = remapped.hands[0]
        self.assertEqual(hand.landmarks[0], HandLandmark(0.5, 0.5))
        self.assertEqual(hand.box, BoundingBox(110, 70, 130, 90, 0.9))


def hand_with_center(label: str, y: float) -> TrackedHand:
    landmarks = (HandLandmark(0.4, y), HandLandmark(0.6, y))
    return TrackedHand(label=label, confidence=0.9, landmarks=landmarks, box=BoundingBox(0, 0, 1, 1, 0.9))


if __name__ == "__main__":
    unittest.main()
