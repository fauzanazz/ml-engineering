import unittest

from webcam_effect.cli import build_parser


class CliTest(unittest.TestCase):
    def test_run_command_has_defaults(self):
        args = build_parser().parse_args(["run"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.camera, "0")
        self.assertEqual(args.detector, "yolo26n.pt")
        self.assertEqual(args.classifier, "yolo26n.pt")
        self.assertEqual(args.data, "coco8.yaml")
        self.assertEqual(args.segmenter, "mediapipe")
        self.assertEqual(args.classifier_backend, "yolo")
        self.assertEqual(args.mediapipe_model, "assets/pose_landmarker_lite.task")
        self.assertEqual(args.preview_key, "p")
        self.assertEqual(args.segmentation_input, "masked-crop")
        self.assertEqual(args.left_sticker, "assets/cat.gif")
        self.assertEqual(args.sticker, "assets/nick.gif")
        self.assertEqual(args.audio, "assets/Kicau Mania Cutted.mp3")
        self.assertEqual(args.device, "mps")
        self.assertFalse(args.debug)
        self.assertFalse(args.sync_analysis)
        self.assertEqual(args.benchmark_frames, 0)
        self.assertIsNone(args.runtime_config)
        self.assertIsNone(args.effect_config)
        self.assertEqual(args.video_output, "preview")

    def test_parses_run_command(self):
        args = build_parser().parse_args(
            [
                "run",
                "--camera",
                "1",
                "--detector",
                "yolo26n.pt",
                "--classifier",
                "runs/classify/kicau/weights/best.pt",
                "--data",
                "coco8.yaml",
                "--segmenter",
                "mediapipe",
                "--classifier-backend",
                "mediapipe",
                "--mediapipe-model",
                "assets/pose_landmarker_lite.task",
                "--preview-key",
                "k",
                "--segmentation-input",
                "crop",
                "--left-sticker",
                "assets/cat.gif",
                "--sticker",
                "assets/nick.gif",
                "--audio",
                "assets/Kicau Mania Cutted.mp3",
                "--device",
                "mps",
                "--debug",
                "--sync-analysis",
                "--benchmark-frames",
                "120",
                "--runtime-config",
                "runtime.json",
                "--effect-config",
                "effect.json",
                "--video-output",
                "ffmpeg",
                "--ffmpeg-video-command",
                "ffmpeg -f rawvideo -",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.camera, "1")
        self.assertEqual(args.data, "coco8.yaml")
        self.assertEqual(args.segmenter, "mediapipe")
        self.assertEqual(args.classifier_backend, "mediapipe")
        self.assertEqual(args.mediapipe_model, "assets/pose_landmarker_lite.task")
        self.assertEqual(args.preview_key, "k")
        self.assertEqual(args.segmentation_input, "crop")
        self.assertEqual(args.left_sticker, "assets/cat.gif")
        self.assertEqual(args.audio, "assets/Kicau Mania Cutted.mp3")
        self.assertEqual(args.device, "mps")
        self.assertTrue(args.debug)
        self.assertTrue(args.sync_analysis)
        self.assertEqual(args.benchmark_frames, 120)
        self.assertEqual(args.runtime_config, "runtime.json")
        self.assertEqual(args.effect_config, "effect.json")
        self.assertEqual(args.video_output, "ffmpeg")
        self.assertEqual(args.ffmpeg_video_command, "ffmpeg -f rawvideo -")

    def test_parses_dataset_command(self):
        args = build_parser().parse_args(["dataset", "--camera", "0", "--label", "none"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.label, "none")

    def test_dataset_command_has_defaults(self):
        args = build_parser().parse_args(["dataset"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.camera, "0")
        self.assertEqual(args.label, "kicau")
        self.assertEqual(args.root, "datasets/kicau_mania")

    def test_classify_train_command_has_defaults(self):
        args = build_parser().parse_args(["classify", "train"])

        self.assertEqual(args.command, "classify")
        self.assertEqual(args.classify_command, "train")
        self.assertEqual(args.model, "yolo26n-cls.pt")
        self.assertEqual(args.data, "datasets/kicau_mania/frames")
        self.assertEqual(args.epochs, 80)
        self.assertEqual(args.imgsz, 224)
        self.assertEqual(args.batch, 16)
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.project, "runs/classify")
        self.assertEqual(args.name, "kicau_aug")

    def test_classify_train_command_accepts_overrides(self):
        args = build_parser().parse_args(
            ["classify", "train", "--epochs", "10", "--device", "cpu", "--name", "smoke"]
        )

        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.name, "smoke")

    def test_editor_command_has_defaults(self):
        args = build_parser().parse_args(["editor"])

        self.assertEqual(args.command, "editor")
        self.assertEqual(args.effect_config, "assets/effect.json")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_youtube_audio_command_parses_url_and_output(self):
        args = build_parser().parse_args(["youtube-audio", "--url", "https://youtu.be/example", "--output", "assets/song.mp3"])

        self.assertEqual(args.command, "youtube-audio")
        self.assertEqual(args.url, "https://youtu.be/example")
        self.assertEqual(args.output, "assets/song.mp3")


if __name__ == "__main__":
    unittest.main()
