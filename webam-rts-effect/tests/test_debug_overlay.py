import unittest

import numpy as np

from webcam_effect.debug_overlay import draw_hand_skeletons, normalized_point
from webcam_effect.hand_tracking import HandLandmark, HandTrackFrame, TrackedHand
from webcam_effect.tracking import BoundingBox


class DebugOverlayTest(unittest.TestCase):
    def test_normalized_point_clamps_to_frame(self):
        point = normalized_point(HandLandmark(1.2, -0.5), frame_width=100, frame_height=80)

        self.assertEqual(point, (100, 0))

    def test_draw_hand_skeletons_marks_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = tuple(HandLandmark(index / 20, index / 20) for index in range(21))
        hands = HandTrackFrame(
            hands=(
                TrackedHand(
                    label="right",
                    confidence=0.9,
                    landmarks=landmarks,
                    box=BoundingBox(10, 10, 90, 90, 0.9),
                ),
            )
        )

        draw_hand_skeletons(frame, hands)

        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
