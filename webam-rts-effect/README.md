# Webcam RTS Effect

Local Python POC for a webcam-triggered `kicau mania` effect.

## Run Effect

Default run uses MediaPipe for user segmentation and YOLO for kicau classification. The fallback classifier uses `yolo26n.pt`, so it will normally classify as `none` until you pass a trained kicau classifier.

```bash
uv run python main.py run
```

Defaults: `--camera 0`, 1280x720 capture, `--segmenter mediapipe`, `--classifier-backend yolo`, `--mediapipe-model assets/pose_landmarker_lite.task`, `--detector yolo26n.pt`, `--classifier yolo26n.pt`, `--data coco8.yaml`, `--segmentation-input masked-crop`, `--sticker assets/nick.gif`, `--left-sticker assets/cat.gif`, `--audio "assets/Kicau Mania Cutted.mp3"`, `--device mps`. ML analysis runs asynchronously by default so webcam preview is not blocked by MediaPipe or YOLO inference.

Use a trained kicau classifier when available:

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

Default MediaPipe segmentation with a trained YOLO classifier:

```bash
uv run python main.py run \
  --classifier runs/classify/kicau/weights/best.pt \
  --segmentation-input masked-crop
```

`masked-crop` removes background pixels before classification. Train with the same preprocessing for best results. Use `--segmentation-input crop` when your classifier was trained on normal person crops.

Effect mode animates both GIFs: `--sticker` renders on the right and `--left-sticker` renders on the left. Audio loops from `--audio` while the effect is active.

Press `p` to toggle preview effect without pose detection. Press `q` to quit.

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

Run a local WebGUI for editing effect definitions:

```bash
uv run python main.py editor --effect-config assets/effect.json
```

Open `http://127.0.0.1:8765`, edit sticker/audio paths and scale, then run with:

```bash
uv run python main.py run --effect-config assets/effect.json
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

Train with the project defaults: MPS, recorded frames, and webcam-friendly augmentation.

```bash
uv run python main.py classify train
```

Outputs the best model to:

```text
runs/classify/kicau_aug/weights/best.pt
```

Use it with raw crop input, because the dataset creator records raw webcam frames:

```bash
uv run python main.py run \
  --classifier runs/classify/kicau_aug/weights/best.pt \
  --segmentation-input crop
```

## Test

```bash
uv run python -m unittest discover -s tests
```
