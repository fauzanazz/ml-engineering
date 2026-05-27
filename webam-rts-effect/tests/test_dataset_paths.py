import unittest
from pathlib import Path

from webcam_effect.dataset_creator import DatasetPaths


class DatasetPathsTest(unittest.TestCase):
    def test_builds_class_scoped_clip_and_frame_paths(self):
        paths = DatasetPaths(root=Path("datasets/kicau_mania"), label="kicau")

        self.assertEqual(paths.clip_dir, Path("datasets/kicau_mania/clips/kicau"))
        self.assertEqual(paths.frame_dir, Path("datasets/kicau_mania/frames/kicau"))

    def test_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            DatasetPaths(root=Path("datasets/kicau_mania"), label="bird")


if __name__ == "__main__":
    unittest.main()
