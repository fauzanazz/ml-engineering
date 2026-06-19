import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from webcam_effect.training import has_near_duplicate, hamming_distance, prepare_classifier_split

class TrainingDatasetTest(unittest.TestCase):
    def test_hamming_distance_counts_different_fingerprint_bits(self):
        self.assertEqual(hamming_distance(0b1010, 0b0011), 2)

    def test_near_duplicate_accepts_small_hash_distance(self):
        fingerprint = 0b11110000
        existing = [0b11110001]

        self.assertTrue(has_near_duplicate(fingerprint, existing))

    def test_near_duplicate_rejects_large_hash_distance(self):
        fingerprint = 0b11111111
        existing = [0b00000000]

        self.assertFalse(has_near_duplicate(fingerprint, existing))

    def test_prepare_classifier_split_rebuilds_derived_split(self):
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "prepared"
            for label in ["kicau", "none"]:
                label_dir = source_dir / label
                label_dir.mkdir(parents=True)
                for index in range(5):
                    (label_dir / f"{index}.jpg").write_bytes(b"image")
            stale_path = source_dir.with_name("prepared_split") / "train" / "kicau" / "stale.jpg"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_bytes(b"stale")

            split_dir = prepare_classifier_split(source_dir, seed=1)

            self.assertFalse(stale_path.exists())
            self.assertEqual(count_images(split_dir / "train" / "kicau"), 4)
            self.assertEqual(count_images(split_dir / "val" / "kicau"), 1)
            self.assertEqual(count_images(split_dir / "train" / "none"), 4)
            self.assertEqual(count_images(split_dir / "val" / "none"), 1)

def count_images(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.suffix == ".jpg")

if __name__ == "__main__":
    unittest.main()
