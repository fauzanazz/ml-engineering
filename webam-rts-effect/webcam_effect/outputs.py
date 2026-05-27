from dataclasses import dataclass
import shlex
import subprocess
from typing import Protocol


class VideoOutput(Protocol):
    def write(self, frame) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass
class PreviewOutput:
    window_name: str = "kicau mania"

    def write(self, frame) -> None:
        import cv2

        cv2.imshow(self.window_name, frame)

    def close(self) -> None:
        import cv2

        cv2.destroyAllWindows()


class NullVideoOutput:
    def write(self, frame) -> None:
        return None

    def close(self) -> None:
        return None


class FfmpegVideoOutput:
    def __init__(self, command: str):
        if not command:
            raise ValueError("ffmpeg video output requires --ffmpeg-video-command")
        self.command = command
        self.process: subprocess.Popen | None = None

    def write(self, frame) -> None:
        if self.process is None:
            height, width = frame.shape[:2]
            command = self.command.format(width=width, height=height)
            self.process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE)
        if self.process.stdin is not None:
            self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.terminate()


def create_video_output(kind: str, ffmpeg_command: str = "") -> VideoOutput:
    if kind == "preview":
        return PreviewOutput()
    if kind == "none":
        return NullVideoOutput()
    if kind == "ffmpeg":
        return FfmpegVideoOutput(ffmpeg_command)
    raise ValueError(f"unknown video output: {kind}")
