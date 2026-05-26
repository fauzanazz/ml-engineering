from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading

@dataclass
class LoopingAudio:
    audio_path: Path
    player: str = "afplay"

    def __post_init__(self):
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.audio_path.exists():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._process = subprocess.Popen([self.player, str(self.audio_path)])
            while self._process.poll() is None and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.1)
            if self._process.poll() is None:
                self._process.terminate()
