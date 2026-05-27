from pathlib import Path
import random
import shutil

CLASSIFIER_TRAIN_DEFAULTS = {
    "model": "yolo26s-cls.pt",
    "data": "datasets/kicau_mania/frames",
    "prepared_data": "datasets/kicau_mania/classifier_frames/yolo_masked_crop_dedup",
    "segmenter": "yolo-seg",
    "detector": "yolo26n-seg.pt",
    "mediapipe_model": "assets/pose_landmarker_lite.task",
    "segmentation_input": "masked-crop",
    "epochs": 80,
    "imgsz": 224,
    "batch": 16,
    "device": "mps",
    "project": "runs/classify",
    "name": "kicau_yolo26s_masked_aug",
    "auto_augment": "randaugment",
    "erasing": 0.25,
    "hsv_h": 0.02,
    "hsv_s": 0.5,
    "hsv_v": 0.35,
    "degrees": 8,
    "translate": 0.12,
    "scale": 0.35,
    "fliplr": 0.5,
    "cls_pw": 0.4,
}

MAX_DUPLICATE_HASH_DISTANCE = 6
TRAIN_SPLIT_FRACTION = 0.8

def run_classifier_training(**overrides):
    from ultralytics import YOLO

    train_args = CLASSIFIER_TRAIN_DEFAULTS | {key: value for key, value in overrides.items() if value is not None}
    source_dir = Path(train_args.pop("data"))
    prepared_dir = Path(train_args.pop("prepared_data"))
    segmenter_backend = train_args.pop("segmenter")
    detector_path = train_args.pop("detector")
    mediapipe_model = train_args.pop("mediapipe_model")
    segmentation_input = train_args.pop("segmentation_input")
    prepare_classifier_dataset(
        source_dir=source_dir,
        output_dir=prepared_dir,
        segmenter_backend=segmenter_backend,
        detector_path=detector_path,
        device=train_args["device"],
        mediapipe_model=mediapipe_model,
        segmentation_input=segmentation_input,
    )
    split_dir = prepare_classifier_split(prepared_dir)
    train_args["data"] = str(split_dir.resolve())
    train_args["project"] = str(Path(train_args["project"]).resolve())
    model = YOLO(train_args.pop("model"))
    return model.train(**train_args)

def prepare_classifier_dataset(
    source_dir: Path,
    output_dir: Path,
    segmenter_backend: str,
    detector_path: str,
    device: str,
    mediapipe_model: str,
    segmentation_input: str,
) -> None:
    import cv2

    segmenter = create_training_segmenter(segmenter_backend, detector_path, device, mediapipe_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints_by_label = load_existing_fingerprints(output_dir)
    image_paths = sorted(path for path in source_dir.glob("*/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    written = 0
    skipped_duplicates = 0
    skipped_unsegmented = 0
    for image_path in image_paths:
        label_dir = output_dir / image_path.parent.name
        label_dir.mkdir(parents=True, exist_ok=True)
        output_path = label_dir / image_path.name
        if output_path.exists():
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        crop = segmenter.crop(frame, segmentation_input=segmentation_input)
        if crop is None:
            skipped_unsegmented += 1
            continue
        fingerprint = image_fingerprint(crop)
        label_fingerprints = fingerprints_by_label.setdefault(image_path.parent.name, [])
        if has_near_duplicate(fingerprint, label_fingerprints):
            skipped_duplicates += 1
            continue
        cv2.imwrite(str(output_path), crop)
        label_fingerprints.append(fingerprint)
        written += 1
    print(
        f"prepared classifier dataset: wrote={written} "
        f"duplicate_skips={skipped_duplicates} unsegmented_skips={skipped_unsegmented} output={output_dir}"
    )

def load_existing_fingerprints(output_dir: Path) -> dict[str, list[int]]:
    import cv2

    fingerprints_by_label: dict[str, list[int]] = {}
    for image_path in sorted(path for path in output_dir.glob("*/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        fingerprints_by_label.setdefault(image_path.parent.name, []).append(image_fingerprint(image))
    return fingerprints_by_label

def prepare_classifier_split(source_dir: Path, train_fraction: float = TRAIN_SPLIT_FRACTION, seed: int = 0) -> Path:
    split_dir = source_dir.with_name(f"{source_dir.name}_split")
    reset_derived_split_dir(source_dir, split_dir)
    rng = random.Random(seed)
    for label_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        image_paths = sorted(path for path in label_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(image_paths)
        train_count = max(1, round(len(image_paths) * train_fraction)) if image_paths else 0
        for index, image_path in enumerate(image_paths):
            split_name = "train" if index < train_count else "val"
            target_dir = split_dir / split_name / label_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_dir / image_path.name)
    return split_dir

def reset_derived_split_dir(source_dir: Path, split_dir: Path) -> None:
    expected_name = f"{source_dir.name}_split"
    if split_dir.parent != source_dir.parent or split_dir.name != expected_name:
        raise ValueError(f"refusing to reset non-derived split directory: {split_dir}")
    if split_dir.exists():
        shutil.rmtree(split_dir)

def create_training_segmenter(backend: str, detector_path: str, device: str, mediapipe_model: str):
    if backend == "yolo":
        from webcam_effect.analyzer import _crop_best_person
        from webcam_effect.yolo_models import YoloPersonDetector

        class YoloCropper:
            def __init__(self, model_path: str, device: str):
                self.detector = YoloPersonDetector(model_path, device=device)

            def crop(self, frame, segmentation_input: str):
                if segmentation_input != "crop":
                    raise ValueError("yolo detector training only supports segmentation_input='crop'")
                return _crop_best_person(frame, self.detector)

        return YoloCropper(detector_path, device)
    if backend == "yolo-seg":
        from webcam_effect.yolo_models import YoloPersonSegmenter

        return YoloPersonSegmenter(detector_path, device=device)
    if backend == "mediapipe":
        from webcam_effect.mediapipe_models import MediaPipeUserSegmenter

        return MediaPipeUserSegmenter(model_path=mediapipe_model)
    raise ValueError(f"unknown segmenter backend: {backend}")

def image_fingerprint(image) -> int:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    threshold = float(resized.mean())
    bits = resized > threshold
    fingerprint = 0
    for bit in bits.flatten():
        fingerprint = (fingerprint << 1) | int(bit)
    return fingerprint

def has_near_duplicate(fingerprint: int, existing_fingerprints: list[int]) -> bool:
    return any(
        hamming_distance(fingerprint, existing_fingerprint) <= MAX_DUPLICATE_HASH_DISTANCE
        for existing_fingerprint in existing_fingerprints
    )

def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()
