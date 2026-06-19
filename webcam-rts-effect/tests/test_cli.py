import unittest

from webcam_effect.cli import build_parser


class CliTest(unittest.TestCase):
    def test_run_command_has_defaults(self):
        args = build_parser().parse_args(["run"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.camera, "0")
        self.assertEqual(args.detector, "yolo26n-seg.pt")
        self.assertEqual(args.classifier, "runs/classify/kicau_yolo26s_masked_aug/weights/best.pt")
        self.assertEqual(args.data, "coco8.yaml")
        self.assertEqual(args.segmenter, "yolo-seg")
        self.assertEqual(args.classifier_backend, "yolo")
        self.assertEqual(args.mediapipe_model, "assets/pose_landmarker_lite.task")
        self.assertEqual(args.hand_model, "assets/hand_landmarker.task")
        self.assertEqual(args.hand_track_input, "auto")
        self.assertEqual(args.preview_key, "p")
        self.assertEqual(args.segmentation_input, "masked-crop")
        self.assertEqual(args.left_sticker, "assets/cat.gif")
        self.assertEqual(args.sticker, "assets/nick.gif")
        self.assertEqual(args.audio, "assets/Kicau Mania Cutted.mp3")
        self.assertFalse(args.no_audio)
        self.assertEqual(args.device, "mps")
        self.assertFalse(args.debug)
        self.assertEqual(args.components.segment, True)
        self.assertEqual(args.components.classify, True)
        self.assertEqual(args.components.hand_track, True)
        self.assertIsNone(args.components_tui)
        self.assertFalse(args.sync_analysis)
        self.assertEqual(args.benchmark_frames, 0)
        self.assertIsNone(args.runtime_config)
        self.assertEqual(args.effect_config, "assets/effect.json")
        self.assertEqual(args.video_output, "preview")
        self.assertEqual(args.ffmpeg_video_command, "")

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
                "--hand-model",
                "assets/hand_landmarker.task",
                "--hand-track-input",
                "full",
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
                "--no-audio",
                "--device",
                "mps",
                "--debug",
                "--components",
                "segment,hand_track",
                "--components-tui",
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
        self.assertEqual(args.hand_model, "assets/hand_landmarker.task")
        self.assertEqual(args.hand_track_input, "full")
        self.assertEqual(args.preview_key, "k")
        self.assertEqual(args.segmentation_input, "crop")
        self.assertEqual(args.left_sticker, "assets/cat.gif")
        self.assertEqual(args.audio, "assets/Kicau Mania Cutted.mp3")
        self.assertTrue(args.no_audio)
        self.assertEqual(args.device, "mps")
        self.assertTrue(args.debug)
        self.assertTrue(args.components.segment)
        self.assertFalse(args.components.classify)
        self.assertTrue(args.components.hand_track)
        self.assertTrue(args.components_tui)
        self.assertTrue(args.sync_analysis)
        self.assertEqual(args.benchmark_frames, 120)
        self.assertEqual(args.runtime_config, "runtime.json")
        self.assertEqual(args.effect_config, "effect.json")
        self.assertEqual(args.video_output, "ffmpeg")
        self.assertEqual(args.ffmpeg_video_command, "ffmpeg -f rawvideo -")

    def test_parses_yolo_seg_segmenter(self):
        args = build_parser().parse_args(
            ["run", "--segmenter", "yolo-seg", "--detector", "yolo26n-seg.pt", "--segmentation-input", "masked-crop"]
        )

        self.assertEqual(args.segmenter, "yolo-seg")
        self.assertEqual(args.detector, "yolo26n-seg.pt")
        self.assertEqual(args.segmentation_input, "masked-crop")

    def test_filters_command_has_mediapipe_filter_defaults(self):
        args = build_parser().parse_args(["filters"])

        self.assertEqual(args.command, "filters")
        self.assertEqual(args.camera, "0")
        self.assertEqual(args.resolution, "640x480")
        self.assertEqual(args.assets_dir, "assets")
        self.assertEqual(args.glasses, "glasses.ppm")
        self.assertEqual(args.face_model, "assets/face_landmarker.task")
        self.assertEqual(args.hand_model, "assets/hand_landmarker.task")
        self.assertEqual(args.pose_model, "assets/pose_landmarker_lite.task")
        self.assertEqual(args.segmenter_model, "assets/selfie_segmenter.tflite")
        self.assertEqual(args.start_filter, "0")
        self.assertEqual(args.frame_skip, 1)
        self.assertEqual(args.inference_scale, 0.5)
        self.assertEqual(args.video_output, "preview")
        self.assertEqual(args.record_output, "")

    def test_components_tui_can_be_disabled(self):
        args = build_parser().parse_args(["run", "--no-components-tui"])

        self.assertFalse(args.components_tui)

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
        self.assertEqual(args.model, "yolo26s-cls.pt")
        self.assertEqual(args.data, "datasets/kicau_mania/frames")
        self.assertEqual(args.prepared_data, "datasets/kicau_mania/classifier_frames/yolo_masked_crop_dedup")
        self.assertEqual(args.segmenter, "yolo-seg")
        self.assertEqual(args.detector, "yolo26n-seg.pt")
        self.assertEqual(args.mediapipe_model, "assets/pose_landmarker_lite.task")
        self.assertEqual(args.segmentation_input, "masked-crop")
        self.assertEqual(args.epochs, 80)
        self.assertEqual(args.imgsz, 224)
        self.assertEqual(args.batch, 16)
        self.assertEqual(args.device, "mps")
        self.assertEqual(args.project, "runs/classify")
        self.assertEqual(args.name, "kicau_yolo26s_masked_aug")

    def test_classify_train_command_accepts_overrides(self):
        args = build_parser().parse_args(
            [
                "classify",
                "train",
                "--epochs",
                "10",
                "--device",
                "cpu",
                "--name",
                "smoke",
                "--segmenter",
                "mediapipe",
                "--detector",
                "other.pt",
            ]
        )

        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.name, "smoke")
        self.assertEqual(args.segmenter, "mediapipe")
        self.assertEqual(args.detector, "other.pt")

    def test_train_command_is_short_alias(self):
        args = build_parser().parse_args(["train"])

        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "yolo26s-cls.pt")
        self.assertEqual(args.data, "datasets/kicau_mania/frames")
        self.assertEqual(args.prepared_data, "datasets/kicau_mania/classifier_frames/yolo_masked_crop_dedup")
        self.assertEqual(args.segmenter, "yolo-seg")
        self.assertEqual(args.detector, "yolo26n-seg.pt")
        self.assertEqual(args.segmentation_input, "masked-crop")

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
