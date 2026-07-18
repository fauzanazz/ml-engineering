from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time


DEFAULT_STATUS_PATH = Path("outputs/runtime-status.json")


@dataclass
class RuntimeStatus:
    connected: bool = False
    active_effect: str | None = None
    label: str = "none"
    confidence: float = 0.0
    processing_fps: float = 0.0
    dropped_frames: int = 0
    active_filter: str | None = None
    recording: bool = False
    updated_at: float = 0.0

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["fps"] = payload["processing_fps"]
        return payload


class RuntimeStatusWriter:
    def __init__(self, path: Path = DEFAULT_STATUS_PATH, interval_seconds: float = 0.5):
        self.path = path
        self.interval_seconds = interval_seconds
        self.last_write = 0.0

    def update(self, status: RuntimeStatus, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_write < self.interval_seconds:
            return
        status.updated_at = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status.as_dict()) + "\n")
        temporary.replace(self.path)
        self.last_write = now

    def close(self, status: RuntimeStatus) -> None:
        status.connected = False
        self.update(status, force=True)


def read_runtime_status(path: Path = DEFAULT_STATUS_PATH) -> dict:
    fallback = RuntimeStatus().as_dict()
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback
    if time.time() - float(payload.get("updated_at", 0)) > 3:
        payload["connected"] = False
    return {**fallback, **payload}
