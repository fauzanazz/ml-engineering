import unittest
from pathlib import Path

import numpy as np

from webcam_effect.effects import AnimatedSticker, StickerEffect, load_sticker_frames, overlay_image, remove_green_screen


class EffectsTest(unittest.TestCase):
    def test_gif_loader_reads_multiple_frames(self):
        frames = load_sticker_frames(Path("assets/nick.gif"))

        self.assertGreater(len(frames), 1)

    def test_animated_sticker_advances_frames(self):
        sticker = AnimatedSticker(Path("assets/nick.gif"))

        first = sticker.next_frame()
        second = sticker.next_frame()

        self.assertEqual(first.shape, second.shape)
        self.assertEqual(sticker.frame_index, 2)

    def test_animated_sticker_caches_resized_frames(self):
        sticker = AnimatedSticker(Path("assets/nick.gif"))

        first = sticker.next_resized_frame(frame_width=320)
        second = sticker.next_resized_frame(frame_width=320)
        sticker.frame_index = 0
        repeated = sticker.next_resized_frame(frame_width=320)

        self.assertEqual(first.shape, second.shape)
        self.assertIs(first, repeated)

    def test_sticker_effect_draws_left_and_right(self):
        effect = StickerEffect(right_sticker_path=Path("assets/nick.gif"), left_sticker_path=Path("assets/cat.gif"), scale=0.2)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        output = effect.apply(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(int(output.sum()), 0)

    def test_remove_green_screen_turns_green_pixels_transparent(self):
        frame = np.array(
            [
                [[0, 255, 0], [0, 20, 255]],
                [[10, 240, 10], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )

        output = remove_green_screen(frame, tolerance=80)

        self.assertEqual(output.shape[2], 4)
        self.assertEqual(output[0, 0, 3], 0)
        self.assertEqual(output[1, 0, 3], 0)
        self.assertEqual(output[0, 1, 3], 255)
        self.assertEqual(output[1, 1, 3], 255)

    def test_cat_gif_chroma_key_has_transparent_pixels(self):
        sticker = AnimatedSticker(Path("assets/cat.gif"), chroma_key_green=True)

        frame = sticker.next_frame()

        self.assertEqual(frame.shape[2], 4)
        self.assertLess(int(frame[:, :, 3].min()), 255)

    def test_overlay_clips_to_frame(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        overlay = np.full((4, 4, 3), 255, dtype=np.uint8)

        output = overlay_image(frame, overlay, x=-2, y=-2)

        self.assertGreater(int(output.sum()), 0)


    def test_sticker_effect_accepts_normalized_positions(self):
        effect = StickerEffect(
            right_sticker_path=Path("assets/nick.gif"),
            left_sticker_path=None,
            scale=0.2,
            right_x=0.1,
            right_y=0.2,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        output = effect.apply(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(int(output.sum()), 0)

if __name__ == "__main__":
    unittest.main()
