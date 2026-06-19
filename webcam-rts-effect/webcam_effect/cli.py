import argparse
import sys

from webcam_effect.components import ComponentSettings, format_components, parse_components, select_components_tui
from webcam_effect.dataset_creator import DATASET_LABELS
from webcam_effect.training import CLASSIFIER_TRAIN_DEFAULTS


DEFAULT_CAMERA = "0"
DEFAULT_CLASSIFIER = "runs/classify/kicau_yolo26s_masked_aug/weights/best.pt"
DEFAULT_DATA = "coco8.yaml"
DEFAULT_DATASET_ROOT = "datasets/kicau_mania"
DEFAULT_DETECTOR = "yolo26n-seg.pt"
DEFAULT_DEVICE = "mps"
DEFAULT_EFFECT_AUDIO = "assets/Kicau Mania Cutted.mp3"
DEFAULT_EFFECT_CONFIG = "assets/effect.json"
DEFAULT_LEFT_STICKER = "assets/cat.gif"
DEFAULT_LABEL = "kicau"
DEFAULT_CLASSIFIER_BACKEND = "yolo"
DEFAULT_COMPONENTS = format_components(ComponentSettings())
DEFAULT_HAND_MODEL = "assets/hand_landmarker.task"
DEFAULT_HAND_TRACK_INPUT = "auto"
DEFAULT_MEDIAPIPE_MODEL = "assets/pose_landmarker_lite.task"
DEFAULT_PREVIEW_KEY = "p"
DEFAULT_SEGMENTATION_INPUT = "masked-crop"
DEFAULT_SEGMENTER_BACKEND = "yolo-seg"
DEFAULT_STICKER = "assets/nick.gif"
DEFAULT_VIDEO_OUTPUT = "preview"
DEFAULT_FFMPEG_VIDEO_COMMAND = ""
DEFAULT_FILTER_BACKGROUND = "example.png"
DEFAULT_FILTER_FACE_MODEL = "assets/face_landmarker.task"
DEFAULT_FILTER_GLASSES = "glasses.ppm"
DEFAULT_FILTER_RECORD_OUTPUT = ""
DEFAULT_FILTER_RESOLUTION = "640x480"
DEFAULT_FILTER_SEGMENTER_MODEL = "assets/selfie_segmenter.tflite"
DEFAULT_FILTER_START = "0"
DEFAULT_FILTER_STICKER = "nick.gif"
CLASSIFIER_BACKENDS = ("yolo", "mediapipe")
SEGMENTER_BACKENDS = ("yolo", "yolo-seg", "mediapipe")
SEGMENTATION_INPUTS = ("crop", "masked-crop")
VIDEO_OUTPUTS = ("preview", "ffmpeg", "none")
FILTER_VIDEO_OUTPUTS = ("preview", "none")
FILTER_KEYS = tuple("1234567890")
HAND_TRACK_INPUTS = ("auto", "bbox", "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="webcam-effect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--camera", default=DEFAULT_CAMERA)
    run_parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    run_parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    run_parser.add_argument("--data", default=DEFAULT_DATA)
    run_parser.add_argument("--segmenter", choices=SEGMENTER_BACKENDS, default=DEFAULT_SEGMENTER_BACKEND)
    run_parser.add_argument("--classifier-backend", choices=CLASSIFIER_BACKENDS, default=DEFAULT_CLASSIFIER_BACKEND)
    run_parser.add_argument("--mediapipe-model", default=DEFAULT_MEDIAPIPE_MODEL)
    run_parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    run_parser.add_argument("--hand-track-input", choices=HAND_TRACK_INPUTS, default=DEFAULT_HAND_TRACK_INPUT)
    run_parser.add_argument("--preview-key", default=DEFAULT_PREVIEW_KEY)
    run_parser.add_argument("--segmentation-input", choices=SEGMENTATION_INPUTS, default=DEFAULT_SEGMENTATION_INPUT)
    run_parser.add_argument("--left-sticker", default=DEFAULT_LEFT_STICKER)
    run_parser.add_argument("--sticker", default=DEFAULT_STICKER)
    run_parser.add_argument("--audio", default=DEFAULT_EFFECT_AUDIO)
    run_parser.add_argument("--no-audio", action="store_true")
    run_parser.add_argument("--device", default=DEFAULT_DEVICE)
    run_parser.add_argument("--debug", action="store_true")
    run_parser.add_argument("--components", type=component_settings, default=ComponentSettings())
    run_parser.add_argument("--components-tui", action="store_true", default=None)
    run_parser.add_argument("--no-components-tui", action="store_false", dest="components_tui")
    run_parser.add_argument("--sync-analysis", action="store_true")
    run_parser.add_argument("--benchmark-frames", type=int, default=0)
    run_parser.add_argument("--runtime-config")
    run_parser.add_argument("--effect-config", default=DEFAULT_EFFECT_CONFIG)
    run_parser.add_argument("--video-output", choices=VIDEO_OUTPUTS, default=DEFAULT_VIDEO_OUTPUT)
    run_parser.add_argument("--ffmpeg-video-command", default=DEFAULT_FFMPEG_VIDEO_COMMAND)

    filters_parser = subparsers.add_parser("filters")
    filters_parser.add_argument("--camera", default=DEFAULT_CAMERA)
    filters_parser.add_argument("--resolution", choices=("640x480", "1280x720"), default=DEFAULT_FILTER_RESOLUTION)
    filters_parser.add_argument("--assets-dir", default="assets")
    filters_parser.add_argument("--background", default=DEFAULT_FILTER_BACKGROUND)
    filters_parser.add_argument("--glasses", default=DEFAULT_FILTER_GLASSES)
    filters_parser.add_argument("--sticker", default=DEFAULT_FILTER_STICKER)
    filters_parser.add_argument("--face-model", default=DEFAULT_FILTER_FACE_MODEL)
    filters_parser.add_argument("--hand-model", default=DEFAULT_HAND_MODEL)
    filters_parser.add_argument("--pose-model", default=DEFAULT_MEDIAPIPE_MODEL)
    filters_parser.add_argument("--segmenter-model", default=DEFAULT_FILTER_SEGMENTER_MODEL)
    filters_parser.add_argument("--start-filter", choices=FILTER_KEYS, default=DEFAULT_FILTER_START)
    filters_parser.add_argument("--frame-skip", type=int, default=1)
    filters_parser.add_argument("--inference-scale", type=float, default=0.5)
    filters_parser.add_argument("--video-output", choices=FILTER_VIDEO_OUTPUTS, default=DEFAULT_VIDEO_OUTPUT)
    filters_parser.add_argument("--record-output", default=DEFAULT_FILTER_RECORD_OUTPUT)
    filters_parser.add_argument("--benchmark-seconds", type=int, default=0)

    dataset_parser = subparsers.add_parser("dataset")
    dataset_parser.add_argument("--camera", default=DEFAULT_CAMERA)
    dataset_parser.add_argument("--label", choices=sorted(DATASET_LABELS), default=DEFAULT_LABEL)
    dataset_parser.add_argument("--root", default=DEFAULT_DATASET_ROOT)

    add_train_arguments(subparsers.add_parser("train"))

    classify_parser = subparsers.add_parser("classify")
    classify_subparsers = classify_parser.add_subparsers(dest="classify_command", required=True)
    add_train_arguments(classify_subparsers.add_parser("train"))

    editor_parser = subparsers.add_parser("editor")
    editor_parser.add_argument("--effect-config", default=DEFAULT_EFFECT_CONFIG)
    editor_parser.add_argument("--host", default="127.0.0.1")
    editor_parser.add_argument("--port", type=int, default=8765)

    editor_api_parser = subparsers.add_parser("editor-api")
    editor_api_parser.add_argument("--effect-config", default=DEFAULT_EFFECT_CONFIG)
    editor_api_parser.add_argument("--host", default="127.0.0.1")
    editor_api_parser.add_argument("--port", type=int, default=8765)

    youtube_parser = subparsers.add_parser("youtube-audio")
    youtube_parser.add_argument("--url", required=True)
    youtube_parser.add_argument("--output", default="assets/youtube-audio.mp3")

    return parser

def add_train_arguments(train_parser: argparse.ArgumentParser) -> None:
    train_parser.add_argument("--model", default=CLASSIFIER_TRAIN_DEFAULTS["model"])
    train_parser.add_argument("--data", default=CLASSIFIER_TRAIN_DEFAULTS["data"])
    train_parser.add_argument("--prepared-data", default=CLASSIFIER_TRAIN_DEFAULTS["prepared_data"])
    train_parser.add_argument("--segmenter", choices=SEGMENTER_BACKENDS, default=CLASSIFIER_TRAIN_DEFAULTS["segmenter"])
    train_parser.add_argument("--detector", default=CLASSIFIER_TRAIN_DEFAULTS["detector"])
    train_parser.add_argument("--mediapipe-model", default=CLASSIFIER_TRAIN_DEFAULTS["mediapipe_model"])
    train_parser.add_argument(
        "--segmentation-input",
        choices=SEGMENTATION_INPUTS,
        default=CLASSIFIER_TRAIN_DEFAULTS["segmentation_input"],
    )
    train_parser.add_argument("--epochs", type=int, default=CLASSIFIER_TRAIN_DEFAULTS["epochs"])
    train_parser.add_argument("--imgsz", type=int, default=CLASSIFIER_TRAIN_DEFAULTS["imgsz"])
    train_parser.add_argument("--batch", type=int, default=CLASSIFIER_TRAIN_DEFAULTS["batch"])
    train_parser.add_argument("--device", default=CLASSIFIER_TRAIN_DEFAULTS["device"])
    train_parser.add_argument("--project", default=CLASSIFIER_TRAIN_DEFAULTS["project"])
    train_parser.add_argument("--name", default=CLASSIFIER_TRAIN_DEFAULTS["name"])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        from webcam_effect.app import run_live_effect

        sticker = "" if args.effect_config and args.sticker == DEFAULT_STICKER else args.sticker
        left_sticker = "" if args.effect_config and args.left_sticker == DEFAULT_LEFT_STICKER else args.left_sticker
        audio = "" if args.effect_config and args.audio == DEFAULT_EFFECT_AUDIO else args.audio
        use_components_tui = args.components_tui if args.components_tui is not None else sys.stdin.isatty()
        components = select_components_tui(args.components) if use_components_tui else args.components
        run_live_effect(
            camera=args.camera,
            detector_path=args.detector,
            classifier_path=args.classifier,
            data=args.data,
            segmenter_backend=args.segmenter,
            classifier_backend=args.classifier_backend,
            mediapipe_model=args.mediapipe_model,
            hand_model=args.hand_model,
            preview_key=args.preview_key,
            segmentation_input=args.segmentation_input,
            left_sticker_path=left_sticker,
            sticker_path=sticker,
            audio_path=audio,
            device=args.device,
            debug=args.debug,
            runtime_config_path=args.runtime_config,
            effect_config_path=args.effect_config,
            video_output=args.video_output,
            ffmpeg_video_command=args.ffmpeg_video_command,
            async_analysis=not args.sync_analysis,
            benchmark_frames=args.benchmark_frames,
            components=components,
            hand_track_input=args.hand_track_input,
            audio_enabled=not args.no_audio,
        )
        return

    if args.command == "filters":
        from webcam_effect.video_filters.runtime import run_video_filter_app

        run_video_filter_app(
            camera=args.camera,
            resolution=args.resolution,
            assets_dir=args.assets_dir,
            background=args.background,
            glasses=args.glasses,
            sticker=args.sticker,
            face_model=args.face_model,
            hand_model=args.hand_model,
            pose_model=args.pose_model,
            segmenter_model=args.segmenter_model,
            start_filter=args.start_filter,
            frame_skip=args.frame_skip,
            inference_scale=args.inference_scale,
            video_output=args.video_output,
            record_output=args.record_output,
            benchmark_seconds=args.benchmark_seconds,
        )
        return

    if args.command == "dataset":
        from webcam_effect.dataset_creator import run_dataset_creator

        run_dataset_creator(camera=args.camera, root=args.root, label=args.label)
        return

    if args.command == "train" or (args.command == "classify" and args.classify_command == "train"):
        from webcam_effect.training import run_classifier_training

        run_classifier_training(**train_overrides(args))
        return

    if args.command in {"editor", "editor-api"}:
        from pathlib import Path

        from webcam_effect.effect_editor import run_effect_editor

        run_effect_editor(effect_path=Path(args.effect_config), host=args.host, port=args.port)
        return

    if args.command == "youtube-audio":
        from pathlib import Path

        from webcam_effect.assets import download_youtube_audio

        download_youtube_audio(url=args.url, output=Path(args.output))
        return

    raise ValueError(f"unknown command: {args.command}")

def component_settings(value: str) -> ComponentSettings:
    try:
        return parse_components(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

def train_overrides(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "data": args.data,
        "prepared_data": args.prepared_data,
        "segmenter": args.segmenter,
        "detector": args.detector,
        "mediapipe_model": args.mediapipe_model,
        "segmentation_input": args.segmentation_input,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
    }
