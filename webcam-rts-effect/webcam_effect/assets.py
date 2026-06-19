from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import subprocess


def is_remote_source(source: str) -> bool:
    return urlparse(source).scheme in {"http", "https"}


def cached_asset_path(source: str, cache_dir: Path) -> Path:
    parsed = urlparse(source)
    filename = Path(parsed.path).name or "asset.gif"
    if not Path(filename).suffix:
        filename = f"{filename}.gif"
    return cache_dir / filename


def resolve_asset_path(source: str, cache_dir: Path = Path("assets/cache")) -> Path:
    if not is_remote_source(source):
        return Path(source)

    target = cached_asset_path(source, cache_dir)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(source, target)
    return target


def youtube_audio_command(url: str, output: Path) -> list[str]:
    return [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--output",
        str(output),
        url,
    ]


def download_youtube_audio(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(youtube_audio_command(url, output), check=True)
