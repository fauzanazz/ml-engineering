from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DATASET_LABELS = frozenset({"kicau", "none"})


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    label: str

    def __post_init__(self):
        if self.label not in DATASET_LABELS:
            labels = ", ".join(sorted(DATASET_LABELS))
            raise ValueError(f"label must be one of: {labels}")

    @property
    def clip_dir(self) -> Path:
        return self.root / "clips" / self.label

    @property
    def frame_dir(self) -> Path:
        return self.root / "frames" / self.label

    def create(self) -> None:
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)


def run_dataset_creator(camera: str, root: str, label: str) -> None:
    import cv2

    from webcam_effect.camera import CameraSource

    paths = DatasetPaths(root=Path(root), label=label)
    paths.create()
    capture = CameraSource(camera).open()
    recording = False
    writer = None
    clip_id = ""
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            display = _draw_controls(frame.copy(), recording)
            cv2.imshow("kicau dataset creator", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not recording:
                clip_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                writer = _create_writer(paths.clip_dir / f"{clip_id}.mp4", capture, frame)
                recording = True
                frame_index = 0
            if key == ord("s") and recording:
                writer.release()
                writer = None
                recording = False

            if recording and writer is not None:
                writer.write(frame)
                frame_path = paths.frame_dir / f"{clip_id}_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_index += 1
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()


def _create_writer(path: Path, capture, frame):
    import cv2

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def _draw_controls(frame, recording: bool):
    import cv2

    status = "REC" if recording else "IDLE"
    color = (0, 0, 255) if recording else (60, 160, 60)
    cv2.rectangle(frame, (16, 16), (180, 64), color, -1)
    cv2.putText(frame, f"{status}  r=record s=stop q=quit", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame
