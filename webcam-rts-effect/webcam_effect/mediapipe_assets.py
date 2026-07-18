from pathlib import Path
import urllib.request


MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def missing_models(assets_dir: Path = Path("assets")) -> list[Path]:
    return [assets_dir / name for name in MODEL_URLS if not (assets_dir / name).is_file() or (assets_dir / name).stat().st_size == 0]


def require_models(assets_dir: Path = Path("assets")) -> None:
    missing = missing_models(assets_dir)
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing MediaPipe model assets: {names}. Download them with: uv run python main.py download-models")


def require_model_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing MediaPipe model assets: {names}. Download defaults with: uv run python main.py download-models")


def download_models(assets_dir: Path = Path("assets")) -> list[Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, url in MODEL_URLS.items():
        target = assets_dir / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded.append(target)
    return downloaded


def doctor(classifier: Path = Path("models/kicau-classifier/best.pt"), assets_dir: Path = Path("assets")) -> list[str]:
    problems = []
    for path in missing_models(assets_dir):
        problems.append(f"missing: {path} (run: uv run python main.py download-models)")
    if not classifier.is_file() or classifier.stat().st_size == 0:
        problems.append(f"missing: {classifier} (export the evaluated winner to this stable path)")
    for name in ("effect.json", "nick.gif", "cat.gif", "Kicau Mania Cutted.mp3"):
        path = assets_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing: {path}")
    return problems
