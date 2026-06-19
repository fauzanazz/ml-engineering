import unittest
from pathlib import Path

from webcam_effect.audio import LoopingAudio


class AudioTest(unittest.TestCase):
    def test_afplay_command_includes_volume(self):
        audio = LoopingAudio(Path("assets/song.mp3"), volume=0.35)

        self.assertEqual(audio._command(), ["afplay", "-v", "0.35", "assets/song.mp3"])

    def test_non_afplay_command_omits_volume_flag(self):
        audio = LoopingAudio(Path("assets/song.mp3"), player="play", volume=0.35)

        self.assertEqual(audio._command(), ["play", "assets/song.mp3"])


if __name__ == "__main__":
    unittest.main()
