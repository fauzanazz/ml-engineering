import unittest

from webcam_effect.frame_window import FrameWindow


class FrameWindowTest(unittest.TestCase):
    def test_keeps_latest_frames_in_oldest_to_newest_order(self):
        window = FrameWindow(size=3)

        window.append("t-3")
        window.append("t-2")
        window.append("t-1")
        window.append("t")

        self.assertEqual(window.frames(), ["t-2", "t-1", "t"])

    def test_reports_ready_only_when_full(self):
        window = FrameWindow(size=3)

        window.append("t-1")
        window.append("t")

        self.assertFalse(window.ready)

        window.append("t+1")

        self.assertTrue(window.ready)


if __name__ == "__main__":
    unittest.main()
