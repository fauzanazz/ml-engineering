import unittest
from dataclasses import replace

from webcam_effect.app import audio_for_effect_definition, preview_key_to_code
from webcam_effect.effects import EffectDefinition


class AppTest(unittest.TestCase):
    def test_preview_key_to_code_accepts_single_character(self):
        self.assertEqual(preview_key_to_code("p"), ord("p"))

    def test_preview_key_to_code_rejects_multiple_characters(self):
        with self.assertRaises(ValueError):
            preview_key_to_code("preview")

    def test_audio_for_effect_definition_uses_fallback_audio_when_no_tracks(self):
        effect = replace(EffectDefinition(), audio_tracks=(), audio_volume=0.4, audio_loop=False)

        audio = audio_for_effect_definition(effect, fallback_audio="assets/song.mp3")

        self.assertEqual(len(audio.players), 1)
        self.assertEqual(str(audio.players[0].audio_path), "assets/song.mp3")
        self.assertEqual(audio.players[0].volume, 0.4)
        self.assertFalse(audio.players[0].loop)


if __name__ == "__main__":
    unittest.main()
