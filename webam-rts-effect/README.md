# Webcam RTS Effect

Local Python POC for a webcam-triggered `kicau mania` effect.

## Run Effect

Default run uses `assets/effect.json`, YOLO segmentation for user masking, YOLO for kicau classification, and `ffmpeg` output to `OBS Virtual Camera`.

```bash
uv run python main.py run
```

Defaults: `--camera 0`, 1280x720 capture, `--effect-config assets/effect.json`, `--segmenter yolo-seg`, `--classifier-backend yolo`, `--mediapipe-model assets/pose_landmarker_lite.task`, `--detector yolo26n-seg.pt`, `--classifier runs/classify/kicau_yolo26s_masked_aug/weights/best.pt`, `--data coco8.yaml`, `--segmentation-input masked-crop`, `--video-output ffmpeg`, `--ffmpeg-video-command "ffmpeg -f rawvideo -pix_fmt bgr24 -s {width}x{height} -i - -f avfoundation 'OBS Virtual Camera'"`, `--device mps`. ML analysis runs asynchronously by default so webcam preview is not blocked by segmentation or classification inference.

Use another trained kicau classifier when available:

```bash
uv run python main.py run --classifier runs/classify/kicau/weights/best.pt
```

Use MediaPipe for both user segmentation and heuristic kicau classification:

```bash
uv run python main.py run \
  --segmenter mediapipe \
  --classifier-backend mediapipe \
  --mediapipe-model assets/pose_landmarker_lite.task
```

MediaPipe segmentation needs a Pose Landmarker `.task` model file at `assets/pose_landmarker_lite.task`, or pass `--mediapipe-model` with another `.task` file.

Use YOLO bbox-only crop instead of segmentation mask:

```bash
uv run python main.py run --segmenter yolo --detector yolo26n.pt --segmentation-input crop
```

Default YOLO segmentation with an explicit trained classifier:

```bash
uv run python main.py run \
  --classifier runs/classify/kicau/weights/best.pt \
  --segmentation-input masked-crop
```

`masked-crop` removes background pixels before classification. Train with the same preprocessing for best results. Use `--segmentation-input crop` when your classifier was trained on normal person crops.

Effect mode animates both GIFs: `--sticker` renders on the right and `--left-sticker` renders on the left. Audio loops from `--audio` while the effect is active.

Press `p` to toggle preview effect without pose detection. Press `q` to quit.

Turn virtual camera output off and use a local preview window:

```bash
uv run python main.py run --video-output preview
```

Disable video output entirely:

```bash
uv run python main.py run --video-output none
```

Development controls:

- `--debug`: show classifier label, confidence score, active state, model backends, and FPS overlay.
- `--sync-analysis`: run MediaPipe and YOLO on the preview thread for latency/FPS comparison.
- `--benchmark-frames N`: stop after `N` frames and print average FPS.
- `d`: toggle debug overlay while running.
- `[` / `]`: decrease/increase activation threshold.
- `-` / `=`: decrease/increase sticker scale.

Persist or seed runtime tuning with JSON:

```bash
uv run python main.py run --runtime-config assets/runtime.json
```

Example `assets/runtime.json`:

```json
{
  "activate_threshold": 0.7,
  "deactivate_threshold": 0.4,
  "sticker_scale": 0.25,
  "debug": true
}
```

Change preview key:

```bash
uv run python main.py run --preview-key k
```

Use an online GIF as a sticker. The URL is cached under `assets/cache/`:

```bash
uv run python main.py run --sticker https://example.com/sticker.gif
```

Extract YouTube audio into `assets/` with `yt-dlp` installed:

```bash
uv run python main.py youtube-audio --url https://youtu.be/VIDEO_ID --output assets/kicau.mp3
```

## Effect Editor

Build and run the local WebGUI for editing effect definitions:

```bash
cd webui
npm run build
cd ..
uv run python main.py editor --effect-config assets/effect.json
```

Open `http://127.0.0.1:8765`. The editor supports live browser camera preview, manual active/inactive simulation, ordered sticker layers, layer animation controls, chroma key tolerance, multi-track audio mixing, trigger thresholds, presets, duplicate/version restore, undo/redo, drafts, asset upload/import, and effect pack import/export. Then run with:

```bash
uv run python main.py run --effect-config assets/effect.json
```

The editor UI is a Vite React app in `webui/`. For frontend development, run the Python API server and Vite dev server separately:

```bash
uv run python main.py editor-api --effect-config assets/effect.json
cd webui
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` and `/assets` to the Python API server on `127.0.0.1:8765`.

Production builds are served by the Python editor command after:

```bash
cd webui
npm run build
```

## Virtual Outputs

The app can select preview, no preview, or an `ffmpeg` raw-video pipe. Use `ffmpeg` with a virtual camera device/plugin that your OS and meeting app can consume.

```bash
uv run python main.py run \
  --video-output ffmpeg \
  --ffmpeg-video-command "ffmpeg -f rawvideo -pix_fmt bgr24 -s {width}x{height} -i - -f avfoundation 'OBS Virtual Camera'"
```

For Google Meet, select the virtual camera or virtual audio device exposed by OBS, BlackHole, v4l2loopback, or equivalent platform tooling.

## Create Dataset

```bash
uv run python main.py dataset
uv run python main.py dataset --camera 0 --label none
```

Defaults: `--camera 0`, `--label kicau`, `--root datasets/kicau_mania`.

Controls:

- `r`: start recording
- `s`: stop recording
- `q`: quit

Dataset output:

```text
datasets/kicau_mania/
  clips/{kicau,none}/
  frames/{kicau,none}/
```

## Train Classifier

Train with the project defaults: YOLO masked-crop preprocessing, MPS, and webcam-friendly augmentation.

```bash
uv run main.py train
```

Training first converts recorded full-frame images into the same classifier input used at runtime: YOLO segmentation with masked person crops.

```text
datasets/kicau_mania/classifier_frames/yolo_masked_crop_dedup/{kicau,none}/
```

Near-duplicate crops are skipped with a perceptual hash so repeated webcam frames do not waste training.

Outputs the best model to:

```text
runs/classify/kicau_yolo26s_masked_aug/weights/best.pt
```

Use it with the same crop input:

```bash
uv run main.py run
```

## Test

```bash
uv run python -m unittest discover -s tests
cd webui
npm run build
npm run test:e2e
```

`npm run test:e2e` uses mocked API/media streams and writes desktop, 768px, and 390px editor screenshots under `scratch/editor-screenshots/`.
