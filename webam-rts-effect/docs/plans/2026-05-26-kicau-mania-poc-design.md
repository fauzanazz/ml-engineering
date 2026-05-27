# Kicau Mania POC Design

## Summary

Build a local Python webcam pipeline that detects a single person, crops the tracked person region, classifies the crop as `kicau` or `none`, and applies the kicau effect while the recent temporal window says the pose is active.

## Chosen Approach

Use YOLO detection/tracking plus a YOLO classification adapter.

- Detector/tracker: finds one person and returns a stable crop per frame.
- Classifier: model-agnostic YOLO wrapper. First target model is `YOLO26n-cls` trained on `kicau` and `none`.
- Temporal window: keeps frames `t-2`, `t-1`, and `t`, then smooths classification results before changing effect state.
- Effect renderer: overlays sticker while active. Audio hook exists but no audio playback is required until the asset is available.
- Dataset creator: records webcam clips and extracted frames into class folders.

## Architecture

```text
CameraSource
  -> FrameWindow(t-2,t-1,t)
  -> PersonTracker
  -> PersonCrop
  -> KicauClassifier
  -> PoseStateMachine
  -> EffectRenderer
  -> LocalDisplay
```

Dataset creator flow:

```text
CameraSource
  -> RecordControls(label=kicau|none)
  -> ClipWriter
  -> FrameExtractor
  -> datasets/kicau_mania/{clips,frames}/{kicau,none}
```

## Extension Point For Approach 3

Keep classifier behind a small protocol that accepts a short frame/crop window and returns a pose prediction.

Initial implementation can aggregate three per-frame YOLO classification scores. Later, a clip-fusion classifier can replace that implementation without changing camera, tracker, state machine, effect renderer, or dataset creator code.

## Proposed Modules

- `webcam_effect/camera.py`: webcam/video capture abstraction.
- `webcam_effect/frame_window.py`: rolling `t-2,t-1,t` buffer.
- `webcam_effect/yolo_models.py`: detector/classifier adapters.
- `webcam_effect/tracking.py`: single-person tracking/crop selection.
- `webcam_effect/classification.py`: kicau classifier protocol and YOLO implementation.
- `webcam_effect/state.py`: smoothing and active/inactive effect state.
- `webcam_effect/effects.py`: sticker overlay and future audio hook.
- `webcam_effect/app.py`: live webcam POC loop.
- `webcam_effect/dataset_creator.py`: record/stop dataset workflow.
- `webcam_effect/cli.py`: CLI config and command dispatch.
- `main.py`: CLI entrypoint.

## CLI

```bash
uv run python main.py run --camera 0 --classifier runs/classify/kicau/weights/best.pt --sticker assets/nick.gif --device mps
uv run python main.py dataset --camera 0 --label kicau
uv run python main.py dataset --camera 0 --label none
```

## Dataset Layout

```text
datasets/kicau_mania/
  clips/
    kicau/
    none/
  frames/
    kicau/
    none/
```

## Trigger Logic

- Run detector/tracker on each frame.
- Crop best person candidate.
- Classify each crop.
- Maintain latest three predictions for `t-2`, `t-1`, and `t`.
- Activate effect when smoothed `kicau` confidence is above threshold.
- Deactivate effect when smoothed `kicau` confidence falls below threshold.

## Testing Strategy

- Unit-test frame window ordering.
- Unit-test state machine hysteresis and smoothing.
- Unit-test classifier aggregation with fake predictions.
- Unit-test dataset path generation without webcam dependency.
- Smoke-test CLI argument parsing.

## Out Of Scope For First Build

- Training command and augmentation pipeline.
- Audio playback until audio asset exists.
- Multi-person selection beyond best single person.
- Remote streaming.
- Polished GUI.

## Roadmap

### Next Nice To Have

- Add `--debug` overlay for development: segmentation mask or crop box, classifier label, confidence score, backend names, FPS, and effect state.
- Add realtime tuning config: keyboard hotkeys or local control panel for thresholds, smoothing, selected sticker, and backend settings.
- Add online GIF source support: download/cache remote GIF assets and use them as stickers.
- Add YouTube-to-MP3 helper: download/extract audio into `assets/` for effects.

### V2 Must Have

- Add virtual camera and audio output support so Google Meet and similar apps can receive processed video/audio.
- Add Effect Editor WebGUI for creating effects, selecting sticker/audio assets, editing placement/scale/timing, and saving effect definitions.

### Suggested Build Order

1. Debug overlay: lowest risk, helps tune all later work.
2. Realtime config changes: needs debug values visible first.
3. Online GIF source: asset pipeline without output-device complexity.
4. YouTube-to-MP3: separate audio asset pipeline.
5. Virtual video/audio outputs: larger platform-specific integration.
6. Effect Editor WebGUI: depends on effect definitions and asset pipeline.
