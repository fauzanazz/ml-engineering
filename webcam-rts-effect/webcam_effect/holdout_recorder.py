import json
from pathlib import Path
import time

from webcam_effect.camera import CameraSource, parse_resolution


SESSION_PLAN = [
    ("kicau", "target, bright, background A"),
    ("none", "non-target, bright, background A"),
    ("kicau", "target, dim, background A"),
    ("none", "non-target, dim, background A"),
    ("kicau", "target, different background"),
    ("none", "non-target object interaction"),
    ("kicau", "target with fast motion"),
    ("kicau", "target with partial occlusion"),
]


def record_holdout(
    camera: str = "0",
    resolution: str = "640x480",
    seconds_per_session: int = 8,
    countdown_seconds: int = 5,
    output_root: Path = Path("datasets/kicau_mania_holdout"),
) -> list[dict]:
    import cv2

    width, height = parse_resolution(resolution)
    capture = CameraSource(camera, width=width, height=height).open()
    output_root.mkdir(parents=True, exist_ok=True)
    sessions = []
    try:
        for index, (label, condition) in enumerate(SESSION_PLAN, start=1):
            session_id = time.strftime("%Y%m%dT%H%M%S") + f"-{index}"
            target = output_root / "clips" / label / f"{session_id}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"could not create holdout clip: {target}")
            for remaining in range(countdown_seconds, 0, -1):
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    draw_instruction(frame, f"Session {index}/8 starts in {remaining}", condition)
                    cv2.imshow("Webcam holdout recorder", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
            started = time.monotonic()
            frames = 0
            while time.monotonic() - started < seconds_per_session:
                ok, frame = capture.read()
                if not ok:
                    continue
                writer.write(frame)
                frames += 1
                draw_instruction(frame, f"RECORDING {index}/8: {label}", condition)
                cv2.imshow("Webcam holdout recorder", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            writer.release()
            sessions.append({"path": str(target), "label": label, "condition": condition, "frames": frames})
    finally:
        capture.release()
        cv2.destroyAllWindows()
    (output_root / "sessions.json").write_text(json.dumps({"sessions": sessions}, indent=2) + "\n")
    return sessions


def draw_instruction(frame, headline: str, condition: str) -> None:
    import cv2

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 82), (0, 0, 0), -1)
    cv2.putText(frame, headline, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, condition, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
