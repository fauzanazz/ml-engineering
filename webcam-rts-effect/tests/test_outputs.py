import unittest
from unittest import mock

from webcam_effect.outputs import FfmpegVideoOutput, NullVideoOutput, PreviewOutput, create_video_output


class OutputsTest(unittest.TestCase):
    def test_creates_selected_video_outputs(self):
        self.assertIsInstance(create_video_output("preview"), PreviewOutput)
        self.assertIsInstance(create_video_output("none"), NullVideoOutput)
        self.assertIsInstance(create_video_output("ffmpeg", "ffmpeg -f rawvideo -"), FfmpegVideoOutput)

    def test_ffmpeg_requires_command(self):
        with self.assertRaises(ValueError):
            create_video_output("ffmpeg")

    def test_ffmpeg_reports_closed_pipe(self):
        output = FfmpegVideoOutput("ffmpeg -f rawvideo -")
        frame = mock.Mock()
        frame.shape = (2, 3, 3)
        frame.tobytes.return_value = b"frame"
        stdin = mock.Mock()
        stdin.write.side_effect = BrokenPipeError()
        process = mock.Mock()
        process.stdin = stdin
        process.poll.return_value = None

        with mock.patch("webcam_effect.outputs.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg video output pipe closed"):
                output.write(frame)

    def test_ffmpeg_reports_exited_process(self):
        output = FfmpegVideoOutput("ffmpeg -f rawvideo -")
        frame = mock.Mock()
        frame.shape = (2, 3, 3)
        process = mock.Mock()
        process.poll.return_value = 1
        process.returncode = 1
        process.stdin = mock.Mock()

        with mock.patch("webcam_effect.outputs.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg video output exited"):
                output.write(frame)


if __name__ == "__main__":
    unittest.main()
