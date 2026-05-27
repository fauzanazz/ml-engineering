from dataclasses import replace
from pathlib import Path
import time

from webcam_effect.analyzer import AnalysisResult, AsyncLatestAnalyzer, EffectAnalyzer
from webcam_effect.audio import AudioTrackConfig, MultiTrackAudio
from webcam_effect.assets import resolve_asset_path
from webcam_effect.camera import CameraSource
from webcam_effect.classification import TemporalKicauClassifier
from webcam_effect.components import ComponentSettings, format_components
from webcam_effect.config import apply_runtime_config, load_runtime_config, update_runtime_config
from webcam_effect.debug_overlay import DebugInfo, draw_debug_overlay
from webcam_effect.effects import StickerEffect, load_effect_definition, load_effect_library
from webcam_effect.frame_window import FrameWindow
from webcam_effect.hand_tracking import MediaPipeHandTracker
from webcam_effect.mediapipe_models import MediaPipeKicauWindowClassifier, MediaPipeUserSegmenter
from webcam_effect.outputs import create_video_output
from webcam_effect.state import PoseStateMachine
from webcam_effect.yolo_models import YoloFrameClassifier, YoloPersonDetector, YoloPersonSegmenter


def run_live_effect(
    camera: str,
    detector_path: str,
    classifier_path: str,
    data: str,
    segmenter_backend: str,
    classifier_backend: str,
    mediapipe_model: str,
    hand_model: str,
    preview_key: str,
    segmentation_input: str,
    left_sticker_path: str,
    sticker_path: str,
    audio_path: str,
    device: str,
    debug: bool = False,
    runtime_config_path: str | None = None,
    effect_config_path: str | None = None,
    video_output: str = "preview",
    ffmpeg_video_command: str = "",
    async_analysis: bool = True,
    benchmark_frames: int = 0,
    components: ComponentSettings | None = None,
) -> None:
    import cv2

    components = components or ComponentSettings()
    print(f"components={format_components(components) or 'none'}")
    capture = CameraSource(camera).open()
    segmenter = create_segmenter(segmenter_backend, detector_path, device, data, mediapipe_model) if components.segment else None
    classifier = create_classifier(classifier_backend, classifier_path, device, data, mediapipe_model) if components.classify else None
    state = PoseStateMachine()
    frame_window = FrameWindow(size=3)
    runtime_config_file = Path(runtime_config_path) if runtime_config_path else None
    runtime_config = load_runtime_config(runtime_config_file)
    if debug:
        runtime_config = replace(runtime_config, debug=True)
    apply_runtime_config(state, runtime_config)
    analyzer = EffectAnalyzer(
        segmenter_backend=segmenter_backend,
        segmenter=segmenter,
        classifier=classifier,
        state=state,
        frame_window=frame_window,
        components=components,
    )
    async_analyzer = AsyncLatestAnalyzer(analyzer, segmentation_input) if async_analysis and needs_analysis(components) else None
    effect_config_file = Path(effect_config_path) if effect_config_path else None
    effect_library = load_effect_library(effect_config_file)
    effect_definition = effect_library.effects[effect_library.selected_id]
    right_sticker = sticker_path if sticker_path else effect_definition.right_sticker
    left_sticker = left_sticker_path if left_sticker_path else effect_definition.left_sticker
    effect_audio = audio_path if audio_path else effect_definition.selected_audio or effect_definition.audio
    sticker_scale = runtime_config.sticker_scale if runtime_config_path else effect_definition.scale
    effect = StickerEffect(
        right_sticker_path=resolve_asset_path(right_sticker),
        left_sticker_path=resolve_asset_path(left_sticker) if left_sticker else None,
        scale=sticker_scale,
        right_x=effect_definition.right_x,
        right_y=effect_definition.right_y,
        left_x=effect_definition.left_x,
        left_y=effect_definition.left_y,
        layers=effect_definition.layers,
    )
    audio = audio_for_effect_definition(effect_definition, fallback_audio=effect_audio)
    output_sink = create_video_output(video_output, ffmpeg_video_command)
    preview_active = False
    preview_key_code = preview_key_to_code(preview_key)
    effect_was_active = False
    active_started_at = time.monotonic()
    last_frame_time = time.monotonic()
    benchmark_start_time = last_frame_time
    frame_count = 0
    missed_frame_count = 0
    fps = 0.0
    analysis = AnalysisResult(predictions=[], active=False, crop_visible=False, crop=None)
    hand_tracker = None
    hand_tracker_failed = False

    try:
        while True:
            now = time.monotonic()
            elapsed = now - last_frame_time
            last_frame_time = now
            if elapsed > 0:
                fps = 1.0 / elapsed

            ok, frame = capture.read()
            if not ok:
                missed_frame_count += 1
                if missed_frame_count >= 30:
                    break
                time.sleep(0.01)
                continue
            missed_frame_count = 0
            frame_count += 1

            if not needs_analysis(components):
                analysis = AnalysisResult(predictions=[], active=False, crop_visible=False, crop=None)
            elif async_analyzer is None:
                analysis = analyzer.analyze(frame, segmentation_input)
            else:
                async_analyzer.submit(frame)
                analysis = async_analyzer.result()

            selected_definition = select_effect_for_analysis(effect_library, analysis, fallback_id=effect_library.selected_id)

            if selected_definition != effect_definition:
                audio.close()
                effect_definition = selected_definition
                effect = StickerEffect(
                    right_sticker_path=resolve_asset_path(effect_definition.right_sticker),
                    left_sticker_path=resolve_asset_path(effect_definition.left_sticker) if effect_definition.left_sticker else None,
                    scale=runtime_config.sticker_scale if runtime_config_path else effect_definition.scale,
                    right_x=effect_definition.right_x,
                    right_y=effect_definition.right_y,
                    left_x=effect_definition.left_x,
                    left_y=effect_definition.left_y,
                    layers=effect_definition.layers,
                )
                audio = audio_for_effect_definition(effect_definition, fallback_audio=effect_audio)

            effect_active = analysis.active or preview_active
            if effect_active and not effect_was_active:
                active_started_at = now
                audio.start()
            if not effect_active and effect_was_active:
                audio.stop()
            effect_was_active = effect_active
            effect.set_scale(runtime_config.sticker_scale)
            output = effect.apply(frame, elapsed_active_time=now - active_started_at) if effect_active else frame
            if runtime_config.debug:
                hands = None
                if components.hand_track and not hand_tracker_failed:
                    try:
                        if hand_tracker is None:
                            hand_tracker = MediaPipeHandTracker(model_path=hand_model)
                        hands = hand_tracker.track(frame)
                    except FileNotFoundError as exc:
                        print(exc)
                        hand_tracker_failed = True
                output = draw_debug_overlay(
                    output,
                    DebugInfo(
                        predictions=analysis.predictions,
                        active=analysis.active,
                        segmenter_backend=segmenter_backend,
                        classifier_backend=classifier_backend,
                        fps=fps,
                        effect_active=effect_active,
                        crop_visible=analysis.crop_visible,
                        segmented_crop=analysis.crop,
                        hands=hands,
                    ),
                )
            output_sink.write(output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == preview_key_code:
                preview_active = not preview_active
            new_runtime_config = update_runtime_config(runtime_config, key)
            if new_runtime_config != runtime_config:
                runtime_config = new_runtime_config
                apply_runtime_config(state, runtime_config)
            if benchmark_frames and frame_count >= benchmark_frames:
                break
    finally:
        if benchmark_frames:
            elapsed = max(time.monotonic() - benchmark_start_time, 1e-9)
            print(f"processed {frame_count} frames in {elapsed:.2f}s ({frame_count / elapsed:.1f} fps)")
        if async_analyzer is not None:
            async_analyzer.close()
        if hand_tracker is not None:
            hand_tracker.close()
        audio.close()
        output_sink.close()
        capture.release()
        cv2.destroyAllWindows()


def create_segmenter(backend: str, detector_path: str, device: str, data: str, mediapipe_model: str):
    if backend == "yolo":
        return YoloPersonDetector(detector_path, device=device, data=data)
    if backend == "yolo-seg":
        return YoloPersonSegmenter(detector_path, device=device)
    if backend == "mediapipe":
        return MediaPipeUserSegmenter(model_path=mediapipe_model)
    raise ValueError(f"unknown segmenter backend: {backend}")

def needs_analysis(components: ComponentSettings) -> bool:
    return components.segment or components.classify


def preview_key_to_code(preview_key: str) -> int:
    if len(preview_key) != 1:
        raise ValueError("preview key must be one character")
    return ord(preview_key)

def select_effect_for_analysis(library, analysis: AnalysisResult, fallback_id: str):
    best_label = ""
    best_confidence = 0.0
    for prediction in analysis.predictions:
        label = getattr(prediction, "label", "")
        confidence = float(getattr(prediction, "confidence", 0.0))
        if confidence > best_confidence:
            best_label = label
            best_confidence = confidence
    for definition in library.effects.values():
        if definition.trigger_labels and best_label in definition.trigger_labels and best_confidence >= definition.activate_threshold:
            return definition
    return library.effects[fallback_id]


def audio_for_effect_definition(effect_definition, fallback_audio: str) -> MultiTrackAudio:
    track_configs = [
        AudioTrackConfig(Path(track.path), volume=track.volume, loop=track.loop, muted=track.muted)
        for track in effect_definition.audio_tracks
        if track.path
    ] or [AudioTrackConfig(Path(fallback_audio), volume=effect_definition.audio_volume, loop=effect_definition.audio_loop)]
    return MultiTrackAudio(track_configs)


def create_classifier(backend: str, classifier_path: str, device: str, data: str, mediapipe_model: str):
    if backend == "yolo":
        frame_classifier = YoloFrameClassifier(classifier_path, device=device, data=data)
        return TemporalKicauClassifier(frame_classifier)
    if backend == "mediapipe":
        return MediaPipeKicauWindowClassifier(model_path=mediapipe_model)
    raise ValueError(f"unknown classifier backend: {backend}")
