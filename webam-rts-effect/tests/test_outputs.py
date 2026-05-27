import unittest

from webcam_effect.outputs import FfmpegVideoOutput, NullVideoOutput, PreviewOutput, create_video_output


class OutputsTest(unittest.TestCase):
    def test_creates_selected_video_outputs(self):
        self.assertIsInstance(create_video_output("preview"), PreviewOutput)
        self.assertIsInstance(create_video_output("none"), NullVideoOutput)
        self.assertIsInstance(create_video_output("ffmpeg", "ffmpeg -f rawvideo -"), FfmpegVideoOutput)

    def test_ffmpeg_requires_command(self):
        with self.assertRaises(ValueError):
            create_video_output("ffmpeg")


if __name__ == "__main__":
    unittest.main()
