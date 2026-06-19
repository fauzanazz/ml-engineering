from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FilterSpec:
    key: str
    name: str
    description: str
    requires: tuple[str, ...] = ()
    heavy: bool = False
    optimized: bool = False
    target_fps: float | None = None


class VideoFilter(Protocol):
    spec: FilterSpec

    def process(self, frame, timestamp_ms: int):
        pass


@dataclass(frozen=True)
class FilterAssets:
    assets_dir: Path = Path("assets")
    background: Path = Path("assets/example.png")
    glasses: Path = Path("assets/glasses.ppm")
    sticker: Path = Path("assets/nick.gif")


def asset_path(assets_dir: Path, source: str | Path) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == assets_dir.name:
        return path
    return assets_dir / path
