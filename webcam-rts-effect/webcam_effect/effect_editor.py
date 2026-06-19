from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
import zipfile

from webcam_effect.effects import (
    EffectDefinition,
    EffectLibrary,
    effect_to_dict,
    effect_from_dict,
    effect_id_from_name,
    load_effect_definition,
    load_effect_library,
    save_effect_definition,
    save_effect_library,
)

MISSING_WEBUI_MESSAGE = b"Run `cd webui && npm run build` or use Vite dev server on port 5173.\n"


def parse_effect_payload(payload: bytes) -> EffectDefinition:
    return effect_from_dict(json.loads(payload.decode("utf-8")))

def library_payload(library: EffectLibrary) -> dict:
    return {
        "selected_id": library.selected_id,
        "effects": {effect_id: effect_to_dict(effect) for effect_id, effect in library.effects.items()},
    }

def safe_asset_name(name: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in ".-_" else "-" for character in Path(name).name)
    return cleaned or "asset.bin"

def safe_user_asset_path(name: str) -> Path:
    safe_name = safe_asset_name(name)
    target = (Path("assets/user") / safe_name).resolve()
    root = Path("assets/user").resolve()
    if root != target.parent:
        raise ValueError("invalid asset path")
    return target

def list_assets() -> list[dict]:
    tags = load_asset_tags()
    assets = []
    for root in [Path("assets"), Path("assets/user")]:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name == ".asset-tags.json":
                continue
            relative = path.as_posix()
            if relative not in {asset["path"] for asset in assets}:
                assets.append({"path": relative, "name": path.name, "type": asset_type(path), "tag": tags.get(relative, "user" if "assets/user" in relative else "builtin")})
    return assets

def load_asset_tags() -> dict[str, str]:
    path = Path("assets/user/.asset-tags.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def save_asset_tags(tags: dict[str, str]) -> None:
    path = Path("assets/user/.asset-tags.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tags, indent=2) + "\n")

def asset_type(path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or ""
    if content_type.startswith("audio/"):
        return "audio"
    if content_type.startswith("image/") or path.suffix.lower() == ".gif":
        return "image"
    return "file"

def referenced_assets(library: EffectLibrary) -> set[str]:
    paths = set()
    for effect in library.effects.values():
        paths.update(layer.asset_path for layer in effect.layers if layer.asset_path)
        paths.update(track.path for track in effect.audio_tracks if track.path)
        paths.update(item for item in [effect.right_sticker, effect.left_sticker, effect.audio, effect.selected_audio] if item)
    return paths

def validate_effect(definition: EffectDefinition) -> list[dict]:
    errors = []
    if not definition.name.strip():
        errors.append({"field": "name", "message": "Name is required."})
    if not 0.05 <= definition.scale <= 1.0:
        errors.append({"field": "scale", "message": "Scale must be between 0.05 and 1."})
    if not 0 <= definition.audio_volume <= 1:
        errors.append({"field": "audio_volume", "message": "Audio volume must be between 0 and 1."})
    if not 0 <= definition.deactivate_threshold <= definition.activate_threshold <= 1:
        errors.append({"field": "activate_threshold", "message": "Thresholds must be ordered between 0 and 1."})
    for index, layer in enumerate(definition.layers):
        if layer.asset_path and not valid_asset_reference(layer.asset_path):
            errors.append({"field": f"layers.{index}.asset_path", "message": "Asset path must stay inside assets/."})
        if not 0 <= layer.x <= 1 or not 0 <= layer.y <= 1:
            errors.append({"field": f"layers.{index}.position", "message": "Layer position must be normalized."})
        if not 0.01 <= layer.scale <= 2:
            errors.append({"field": f"layers.{index}.scale", "message": "Layer scale must be between 0.01 and 2."})
    for index, track in enumerate(definition.audio_tracks):
        if track.path and not valid_asset_reference(track.path):
            errors.append({"field": f"audio_tracks.{index}.path", "message": "Audio path must stay inside assets/."})
        if not 0 <= track.volume <= 1:
            errors.append({"field": f"audio_tracks.{index}.volume", "message": "Track volume must be between 0 and 1."})
    return errors

def valid_asset_reference(path: str) -> bool:
    if not path:
        return True
    parsed = urlparse(path)
    if parsed.scheme in {"http", "https"}:
        return True
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts and candidate.parts[:1] == ("assets",)

def save_version(effect_path: Path, effect_id: str, definition: EffectDefinition, limit: int = 20) -> None:
    versions = load_versions(effect_path, effect_id)
    versions.append({"created_at": datetime.now(timezone.utc).isoformat(), "effect": effect_to_dict(definition)})
    versions = versions[-limit:]
    version_path(effect_path, effect_id).write_text(json.dumps(versions, indent=2) + "\n")

def version_path(effect_path: Path, effect_id: str) -> Path:
    root = effect_path.parent / ".effect_versions"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_asset_name(effect_id)}.json"

def load_versions(effect_path: Path, effect_id: str) -> list[dict]:
    path = version_path(effect_path, effect_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def make_handler(effect_path: Path):
    class EffectEditorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/effects":
                self._json(library_payload(load_effect_library(effect_path)))
                return
            if self.path == "/api/assets":
                self._json({"assets": list_assets()})
                return
            if self.path == "/api/runtime/status":
                self._json({"connected": False, "label": "manual", "confidence": 0.0, "fps": 0.0, "active_effect": None})
                return
            if self.path == "/api/effects/export":
                self._send_zip(export_effect_pack(load_effect_library(effect_path)))
                return
            if self.path.startswith("/api/effects/") and self.path.endswith("/versions"):
                effect_id = unquote(self.path.removeprefix("/api/effects/").removesuffix("/versions"))
                self._json({"versions": load_versions(effect_path, effect_id)})
                return
            if self.path == "/api/effect":
                self._json(effect_to_dict(load_effect_definition(effect_path)))
                return
            if self.path.startswith("/assets/"):
                self._asset(Path.cwd(), self.path)
                return
            if self._webui_asset(Path.cwd() / "webui" / "dist", self.path):
                return
            self.send_response(503)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", str(len(MISSING_WEBUI_MESSAGE)))
            self.end_headers()
            self.wfile.write(MISSING_WEBUI_MESSAGE)

        def do_PUT(self) -> None:
            if self.path == "/api/effect":
                length = int(self.headers.get("content-length", "0"))
                definition = parse_effect_payload(self.rfile.read(length))
                errors = validate_effect(definition)
                if errors:
                    self._json_error(400, {"errors": errors})
                    return
                save_effect_definition(effect_path, definition)
                self._json(effect_to_dict(definition))
                return
            if self.path.startswith("/api/effects/"):
                effect_id = unquote(self.path.removeprefix("/api/effects/"))
                length = int(self.headers.get("content-length", "0"))
                definition = parse_effect_payload(self.rfile.read(length))
                library = load_effect_library(effect_path)
                if effect_id not in library.effects:
                    self.send_error(404)
                    return
                errors = validate_effect(definition)
                if errors:
                    self._json_error(400, {"errors": errors})
                    return
                save_version(effect_path, effect_id, library.effects[effect_id])
                effects = {**library.effects, effect_id: definition}
                save_effect_library(effect_path, EffectLibrary(selected_id=effect_id, effects=effects))
                self._json(library_payload(load_effect_library(effect_path)))
                return
            if self.path.startswith("/api/select-effect/"):
                effect_id = unquote(self.path.removeprefix("/api/select-effect/"))
                library = load_effect_library(effect_path)
                if effect_id not in library.effects:
                    self.send_error(404)
                    return
                save_effect_library(effect_path, EffectLibrary(selected_id=effect_id, effects=library.effects))
                self._json(library_payload(load_effect_library(effect_path)))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/api/effects":
                length = int(self.headers.get("content-length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                name = data.get("name") or "New effect"
                library = load_effect_library(effect_path)
                effect_id = effect_id_from_name(name, set(library.effects))
                definition = effect_from_dict({**data.get("effect", {}), "name": name})
                effects = {**library.effects, effect_id: definition}
                save_effect_library(effect_path, EffectLibrary(selected_id=effect_id, effects=effects))
                self._json(library_payload(load_effect_library(effect_path)))
                return
            if self.path.startswith("/api/effects/") and self.path.endswith("/duplicate"):
                effect_id = unquote(self.path.removeprefix("/api/effects/").removesuffix("/duplicate"))
                library = load_effect_library(effect_path)
                if effect_id not in library.effects:
                    self.send_error(404)
                    return
                source = library.effects[effect_id]
                name = f"{source.name} Copy"
                new_id = effect_id_from_name(name, set(library.effects))
                effects = {**library.effects, new_id: replace(source, name=name)}
                save_effect_library(effect_path, EffectLibrary(selected_id=new_id, effects=effects))
                self._json(library_payload(load_effect_library(effect_path)))
                return
            if self.path.startswith("/api/effects/") and self.path.endswith("/restore"):
                effect_id = unquote(self.path.removeprefix("/api/effects/").removesuffix("/restore"))
                length = int(self.headers.get("content-length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                index = int(data.get("index", -1))
                versions = load_versions(effect_path, effect_id)
                if not versions or index >= len(versions):
                    self.send_error(404)
                    return
                library = load_effect_library(effect_path)
                restored = effect_from_dict(versions[index]["effect"])
                save_effect_library(effect_path, EffectLibrary(selected_id=effect_id, effects={**library.effects, effect_id: restored}))
                self._json(library_payload(load_effect_library(effect_path)))
                return
            if self.path == "/api/assets":
                length = int(self.headers.get("content-length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                name = safe_asset_name(data["name"])
                encoded = data["data"].split(",", 1)[-1]
                output = Path("assets/user") / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(base64.b64decode(encoded))
                self._json({"path": str(output)})
                return
            if self.path == "/api/assets/import-url":
                length = int(self.headers.get("content-length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                try:
                    self._json({"path": import_url_asset(data.get("url", ""))})
                except Exception as error:
                    self._json_error(400, {"errors": [{"field": "url", "message": str(error)}]})
                return
            if self.path == "/api/assets/youtube-audio":
                length = int(self.headers.get("content-length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                try:
                    self._json({"path": import_youtube_audio(data.get("url", ""), data.get("name", "youtube-audio.mp3"))})
                except Exception as error:
                    self._json_error(400, {"errors": [{"field": "youtube", "message": str(error)}]})
                return
            if self.path == "/api/effects/import":
                length = int(self.headers.get("content-length", "0"))
                try:
                    summary = import_effect_pack(effect_path, self.rfile.read(length))
                    self._json(summary)
                except Exception as error:
                    self._json_error(400, {"errors": [{"field": "pack", "message": str(error)}]})
                return
            self.send_error(404)

        def do_PATCH(self) -> None:
            if not self.path.startswith("/api/assets/"):
                self.send_error(404)
                return
            old_name = unquote(self.path.removeprefix("/api/assets/"))
            length = int(self.headers.get("content-length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            old_path = safe_user_asset_path(old_name)
            new_path = safe_user_asset_path(data.get("name") or old_name)
            if not old_path.exists():
                self.send_error(404)
                return
            if old_path != new_path:
                old_path.rename(new_path)
            tags = load_asset_tags()
            old_relative = old_path.relative_to(Path.cwd()).as_posix()
            new_relative = new_path.relative_to(Path.cwd()).as_posix()
            if old_relative in tags and old_relative != new_relative:
                tags[new_relative] = tags.pop(old_relative)
            if "tag" in data:
                tags[new_relative] = str(data.get("tag") or "user")
            save_asset_tags(tags)
            self._json({"path": str(new_path)})

        def do_DELETE(self) -> None:
            if self.path.startswith("/api/assets/"):
                raw_name, _, query = self.path.removeprefix("/api/assets/").partition("?")
                name = unquote(raw_name)
                asset_path = safe_user_asset_path(name)
                relative = asset_path.relative_to(Path.cwd()).as_posix() if asset_path.is_absolute() else asset_path.as_posix()
                library = load_effect_library(effect_path)
                if relative in referenced_assets(library) and "confirm=true" not in query:
                    self._json_error(409, {"errors": [{"field": "asset", "message": "Asset is referenced by an effect."}]})
                    return
                if asset_path.exists():
                    asset_path.unlink()
                self._json({"deleted": relative})
                return
            if not self.path.startswith("/api/effects/"):
                self.send_error(404)
                return
            effect_id = unquote(self.path.removeprefix("/api/effects/"))
            library = load_effect_library(effect_path)
            if effect_id not in library.effects or len(library.effects) == 1:
                self.send_error(400)
                return
            effects = dict(library.effects)
            del effects[effect_id]
            selected_id = next(iter(effects)) if library.selected_id == effect_id else library.selected_id
            save_effect_library(effect_path, EffectLibrary(selected_id=selected_id, effects=effects))
            self._json(library_payload(load_effect_library(effect_path)))

        def _json(self, data: dict) -> None:
            self._send("application/json", json.dumps(data).encode("utf-8"))

        def _json_error(self, status: int, data: dict) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_zip(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("content-type", "application/zip")
            self.send_header("content-disposition", "attachment; filename=effect-pack.zip")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send(self, content_type: str, body: bytes) -> None:
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _asset(self, root: Path, request_path: str) -> None:
            relative = Path(unquote(request_path).lstrip("/"))
            target = (root / relative).resolve()
            if root.resolve() not in target.parents or not target.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(content_type, target.read_bytes())

        def _webui_asset(self, dist_root: Path, request_path: str) -> bool:
            if not dist_root.exists():
                return False
            relative_path = Path(unquote(request_path.split("?", 1)[0]).lstrip("/"))
            target = (dist_root / relative_path).resolve() if str(relative_path) else dist_root / "index.html"
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                target = dist_root / "index.html"
            if dist_root.resolve() not in target.resolve().parents:
                self.send_error(404)
                return True
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(content_type, target.read_bytes())
            return True

    return EffectEditorHandler

def import_url_asset(url: str, size_limit: int = 20 * 1024 * 1024) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must use http or https")
    extension = Path(parsed.path).suffix.lower()
    if extension not in {".gif", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".ogg", ".m4a"}:
        raise ValueError("Unsupported asset extension")
    request = Request(url, headers={"user-agent": "webcam-effect-editor/1.0"})
    with urlopen(request, timeout=10) as response:
        content_length = int(response.headers.get("content-length", "0") or 0)
        if content_length > size_limit:
            raise ValueError("Asset is too large")
        data = response.read(size_limit + 1)
    if len(data) > size_limit:
        raise ValueError("Asset is too large")
    output = safe_user_asset_path(Path(parsed.path).name or f"remote{extension}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output.as_posix()

def import_youtube_audio(url: str, name: str) -> str:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not installed")
    output = safe_user_asset_path(name if Path(name).suffix else f"{name}.mp3")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["yt-dlp", "--extract-audio", "--audio-format", "mp3", "--no-playlist", "--max-downloads", "1", "-o", str(output), url], check=True, timeout=120)
    return output.as_posix()

def export_effect_pack(library: EffectLibrary) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("effect.json", json.dumps(library_payload(library), indent=2))
        for path in sorted(referenced_assets(library)):
            asset = Path(path)
            if asset.is_file() and valid_asset_reference(path):
                archive.write(asset, path)
    return buffer.getvalue()

def import_effect_pack(effect_path: Path, payload: bytes) -> dict:
    library = load_effect_library(effect_path)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("zip contains unsafe paths")
        data = json.loads(archive.read("effect.json"))
        imported_library = EffectLibrary(
            selected_id=data.get("selected_id", ""),
            effects={effect_id_from_name(effect_data.get("name", effect_id), set(library.effects)): effect_from_dict(effect_data) for effect_id, effect_data in data.get("effects", {}).items()},
        )
        for name in names:
            if name.startswith("assets/") and not name.endswith("/"):
                output = safe_user_asset_path(Path(name).name)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.read(name))
    effects = {**library.effects, **imported_library.effects}
    selected_id = next(iter(imported_library.effects), library.selected_id)
    save_effect_library(effect_path, EffectLibrary(selected_id=selected_id, effects=effects))
    return {"imported_effects": list(imported_library.effects), "selected_id": selected_id}


def run_effect_editor(effect_path: Path, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(effect_path))
    print(f"Effect editor API: http://{host}:{port}")
    server.serve_forever()
