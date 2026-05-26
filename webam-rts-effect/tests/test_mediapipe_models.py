import unittest

import numpy as np

from webcam_effect.mediapipe_models import (
    Landmark2D,
    MediaPipeKicauWindowClassifier,
    MediaPipeUserSegmenter,
    PoseSnapshot,
    classify_kicau_pose,
    masked_crop,
    normalize_segmentation_mask,
)
from webcam_effect.tracking import BoundingBox


class MediaPipeKicauHeuristicTest(unittest.TestCase):
    def test_segmenter_requires_task_model(self):
        with self.assertRaises(FileNotFoundError):
            MediaPipeUserSegmenter(model_path="missing.task")

    def test_classifier_requires_task_model(self):
        with self.assertRaises(FileNotFoundError):
            MediaPipeKicauWindowClassifier(model_path="missing.task")

    def test_masked_crop_keeps_user_pixels_and_blacks_background(self):
        frame = np.array(
            [
                [[10, 10, 10], [20, 20, 20], [30, 30, 30]],
                [[40, 40, 40], [50, 50, 50], [60, 60, 60]],
                [[70, 70, 70], [80, 80, 80], [90, 90, 90]],
            ],
            dtype=np.uint8,
        )
        mask = np.array(
            [
                [0.0, 0.9, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        )

        crop = masked_crop(frame, BoundingBox(0, 0, 3, 3, 1.0), mask, threshold=0.3)

        self.assertEqual(crop[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(crop[0, 1].tolist(), [20, 20, 20])
        self.assertEqual(crop[1, 1].tolist(), [50, 50, 50])
        self.assertEqual(crop[2, 1].tolist(), [0, 0, 0])

    def test_masked_crop_accepts_single_channel_3d_mask(self):
        frame = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.array([[[0.9], [0.0]], [[0.0], [0.9]]], dtype=np.float32)

        crop = masked_crop(frame, BoundingBox(0, 0, 2, 2, 1.0), mask, threshold=0.3)

        self.assertEqual(crop[0, 0].tolist(), [100, 100, 100])
        self.assertEqual(crop[0, 1].tolist(), [0, 0, 0])
        self.assertEqual(crop[1, 1].tolist(), [100, 100, 100])

    def test_normalize_segmentation_mask_rejects_unexpected_shape(self):
        with self.assertRaises(ValueError):
            normalize_segmentation_mask(np.zeros((2, 2, 2), dtype=np.float32))

    def test_detects_one_hand_near_nose_and_other_hand_flapping(self):
        snapshots = [
            PoseSnapshot(
                nose=Landmark2D(0.5, 0.2),
                left_wrist=Landmark2D(0.51, 0.22),
                right_wrist=Landmark2D(0.8, 0.2),
            ),
            PoseSnapshot(
                nose=Landmark2D(0.5, 0.2),
                left_wrist=Landmark2D(0.51, 0.22),
                right_wrist=Landmark2D(0.8, 0.32),
            ),
            PoseSnapshot(
                nose=Landmark2D(0.5, 0.2),
                left_wrist=Landmark2D(0.51, 0.22),
                right_wrist=Landmark2D(0.8, 0.18),
            ),
        ]

        prediction = classify_kicau_pose(snapshots)

        self.assertEqual(prediction.label, "kicau")
        self.assertGreaterEqual(prediction.confidence, 0.7)

    def test_rejects_when_same_hand_closes_nose_without_other_hand_flap(self):
        snapshots = [
            PoseSnapshot(
                nose=Landmark2D(0.5, 0.2),
                left_wrist=Landmark2D(0.51, 0.22),
                right_wrist=Landmark2D(0.8, 0.2),
            ),
            PoseSnapshot(
                nose=Landmark2D(0.5, 0.2),
                left_wrist=Landmark2D(0.51, 0.21),
                right_wrist=Landmark2D(0.8, 0.21),
            ),
            PoseSnapshot(
                nose=Landmark2D(0.5, 0.2),
                left_wrist=Landmark2D(0.51, 0.22),
                right_wrist=Landmark2D(0.8, 0.2),
            ),
        ]

        prediction = classify_kicau_pose(snapshots)

        self.assertEqual(prediction.label, "none")
        self.assertEqual(prediction.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
