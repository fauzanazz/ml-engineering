import unittest
from pathlib import Path

from webcam_effect.assets import cached_asset_path, is_remote_source, resolve_asset_path, youtube_audio_command


class AssetsTest(unittest.TestCase):
    def test_detects_remote_sources(self):
        self.assertTrue(is_remote_source("https://example.com/effect.gif"))
        self.assertFalse(is_remote_source("assets/effect.gif"))

    def test_maps_url_to_cache_path(self):
        path = cached_asset_path("https://example.com/stickers/kicau.gif", Path("assets/cache"))

        self.assertEqual(path, Path("assets/cache/kicau.gif"))

    def test_local_source_resolves_without_download(self):
        self.assertEqual(resolve_asset_path("assets/nick.gif"), Path("assets/nick.gif"))

    def test_youtube_audio_command_extracts_mp3(self):
        command = youtube_audio_command("https://youtu.be/example", Path("assets/song.mp3"))

        self.assertIn("--extract-audio", command)
        self.assertIn("--audio-format", command)
        self.assertIn("mp3", command)
        self.assertEqual(command[-1], "https://youtu.be/example")


if __name__ == "__main__":
    unittest.main()
