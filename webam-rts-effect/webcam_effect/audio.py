from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading

@dataclass
class LoopingAudio:
    audio_path: Path
    player: str = "afplay"
    volume: float = 1.0
    loop: bool = True

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
            self._process = subprocess.Popen(self._command())
            while self._process.poll() is None and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.1)
            if self._process.poll() is None:
                self._process.terminate()
            if not self.loop:
                break

    def _command(self) -> list[str]:
        if self.player == "afplay":
            return [self.player, "-v", str(max(0.0, min(1.0, self.volume))), str(self.audio_path)]
        return [self.player, str(self.audio_path)]

class PreloadedAudio:
    def __init__(self, audio_path: Path, volume: float = 1.0, loop: bool = True, sample_rate: int = 44100) -> None:
        self.audio_path = audio_path
        self.volume = max(0.0, min(1.0, volume))
        self.loop = loop
        self.sample_rate = sample_rate
        self._active = False
        self._position = 0
        self._lock = threading.Lock()
        self._samples = self._load_samples()
        self._stream = self._open_stream()
        self._stream.start()

    def start(self) -> None:
        with self._lock:
            self._position = 0
            self._active = True

    def stop(self) -> None:
        with self._lock:
            self._active = False
            self._position = 0

    def close(self) -> None:
        self.stop()
        self._stream.stop()
        self._stream.close()

    def _load_samples(self):
        import numpy as np

        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(self.audio_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(self.sample_rate),
            "-",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
        samples = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 2)
        return samples * self.volume

    def _open_stream(self):
        import sounddevice as sd

        return sd.OutputStream(samplerate=self.sample_rate, channels=2, dtype="float32", callback=self._callback)

    def _callback(self, outdata, frames, time, status) -> None:
        del time, status
        with self._lock:
            if not self._active or len(self._samples) == 0:
                outdata.fill(0)
                return

            written = 0
            while written < frames and self._active:
                remaining_output = frames - written
                remaining_audio = len(self._samples) - self._position
                count = min(remaining_output, remaining_audio)
                outdata[written : written + count] = self._samples[self._position : self._position + count]
                written += count
                self._position += count
                if self._position >= len(self._samples):
                    if self.loop:
                        self._position = 0
                    else:
                        self._active = False
            if written < frames:
                outdata[written:frames].fill(0)

@dataclass
class AudioTrackConfig:
    path: Path
    volume: float = 1.0
    loop: bool = True
    muted: bool = False

class MultiTrackAudio:
    def __init__(self, tracks: list[AudioTrackConfig], player: str = "afplay") -> None:
        self.players = [create_audio_player(track, player=player) for track in tracks if not track.muted]

    def start(self) -> None:
        for player in self.players:
            player.start()

    def stop(self) -> None:
        for player in self.players:
            player.stop()

    def close(self) -> None:
        for player in self.players:
            player.stop()
            close = getattr(player, "close", None)
            if close is not None:
                close()

class NullAudio:
    players = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

def create_audio_player(track: AudioTrackConfig, player: str = "afplay"):
    if track.path.exists():
        try:
            return PreloadedAudio(track.path, volume=track.volume, loop=track.loop)
        except Exception:
            pass
    return LoopingAudio(track.path, player=player, volume=track.volume, loop=track.loop)
