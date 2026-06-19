import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from webcam_effect.effect_editor import MISSING_WEBUI_MESSAGE, export_effect_pack, import_effect_pack, list_assets, load_asset_tags, parse_effect_payload, referenced_assets, safe_asset_name, save_asset_tags, validate_effect
from webcam_effect.effects import EffectDefinition, EffectLibrary, load_effect_library, save_effect_library


class EffectEditorTest(unittest.TestCase):
    def test_missing_webui_message_points_to_vite_build(self):
        self.assertIn(b"npm run build", MISSING_WEBUI_MESSAGE)
        self.assertIn(b"5173", MISSING_WEBUI_MESSAGE)

    def test_parse_effect_payload_normalizes_empty_left_sticker(self):
        payload = b'{"name":"test","right_sticker":"assets/a.gif","left_sticker":"","audio":"assets/a.mp3","scale":0.4}'

        definition = parse_effect_payload(payload)

        self.assertEqual(definition.name, "test")
        self.assertIsNone(definition.left_sticker)
        self.assertEqual(definition.scale, 0.4)

    def test_parse_effect_payload_keeps_placement(self):
        payload = b'{"right_x":0.6,"right_y":0.2,"left_x":0.1,"left_y":0.3}'

        definition = parse_effect_payload(payload)

        self.assertEqual(definition.right_x, 0.6)
        self.assertEqual(definition.right_y, 0.2)
        self.assertEqual(definition.left_x, 0.1)
        self.assertEqual(definition.left_y, 0.3)

    def test_parse_effect_payload_keeps_song_settings(self):
        payload = b'{"audio_tracks":["assets/a.mp3","assets/b.mp3"],"selected_audio":"assets/b.mp3","audio_volume":0.35,"audio_loop":false}'

        definition = parse_effect_payload(payload)

        self.assertEqual(definition.audio_tracks, ("assets/a.mp3", "assets/b.mp3"))
        self.assertEqual(definition.selected_audio, "assets/b.mp3")
        self.assertEqual(definition.audio, "assets/b.mp3")
        self.assertEqual(definition.audio_volume, 0.35)
        self.assertFalse(definition.audio_loop)

    def test_effect_library_round_trips_multiple_effects(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "effects.json"
            save_effect_library(
                path,
                EffectLibrary(
                    selected_id="custom",
                    effects={
                        "kicau": EffectDefinition(name="kicau mania"),
                        "custom": EffectDefinition(name="Custom Drop", right_sticker="assets/user/custom.gif"),
                    },
                ),
            )

            library = load_effect_library(path)

            self.assertEqual(library.selected_id, "custom")
            self.assertEqual(library.effects["custom"].right_sticker, "assets/user/custom.gif")
            self.assertEqual(library.effects["kicau"].name, "kicau mania")

    def test_safe_asset_name_removes_path_segments(self):
        self.assertEqual(safe_asset_name("../../My Sound!.mp3"), "My-Sound-.mp3")

    def test_validation_rejects_bad_scale_name_and_paths(self):
        definition = parse_effect_payload(b'{"name":"","scale":2,"layers":[{"asset_path":"../bad.gif"}],"audio_tracks":[{"path":"/bad.mp3"}]}')

        errors = validate_effect(definition)

        self.assertIn("name", {error["field"] for error in errors})
        self.assertIn("scale", {error["field"] for error in errors})
        self.assertIn("layers.0.asset_path", {error["field"] for error in errors})
        self.assertIn("audio_tracks.0.path", {error["field"] for error in errors})

    def test_referenced_assets_includes_layers_and_tracks(self):
        definition = parse_effect_payload(b'{"layers":[{"asset_path":"assets/a.gif"}],"audio_tracks":[{"path":"assets/a.mp3"}]}')
        library = EffectLibrary(selected_id="one", effects={"one": definition})

        self.assertIn("assets/a.gif", referenced_assets(library))
        self.assertIn("assets/a.mp3", referenced_assets(library))

    def test_export_import_effect_pack_round_trip(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asset = root / "assets" / "user" / "sample.gif"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"gif")
            path = root / "assets" / "effect.json"
            library = EffectLibrary(selected_id="custom", effects={"custom": EffectDefinition(name="Custom", right_sticker="assets/user/sample.gif")})
            current = Path.cwd()
            try:
                import os
                os.chdir(root)
                payload = export_effect_pack(library)
                summary = import_effect_pack(path, payload)
            finally:
                os.chdir(current)

        self.assertEqual(summary["imported_effects"], ["custom"])

    def test_asset_listing_includes_user_tags(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            asset = root / "assets" / "user" / "sample.gif"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"gif")
            current = Path.cwd()
            try:
                import os
                os.chdir(root)
                save_asset_tags({"assets/user/sample.gif": "meme"})
                listed = list_assets()
            finally:
                os.chdir(current)

        self.assertIn({"path": "assets/user/sample.gif", "name": "sample.gif", "type": "image", "tag": "meme"}, listed)


if __name__ == "__main__":
    unittest.main()
