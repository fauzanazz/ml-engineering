from collections import deque
from dataclasses import dataclass, field
from typing import Generic, TypeVar


Frame = TypeVar("Frame")


@dataclass
class FrameWindow(Generic[Frame]):
    size: int = 3
    _frames: deque[Frame] = field(init=False, repr=False)

    def __post_init__(self):
        if self.size < 1:
            raise ValueError("FrameWindow size must be positive")
        self._frames = deque(maxlen=self.size)

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)

    def frames(self) -> list[Frame]:
        return list(self._frames)

    @property
    def ready(self) -> bool:
        return len(self._frames) == self.size
