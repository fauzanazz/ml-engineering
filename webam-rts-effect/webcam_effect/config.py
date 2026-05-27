from dataclasses import dataclass, replace
import json
from pathlib import Path

from webcam_effect.state import PoseStateMachine


@dataclass(frozen=True)
class RuntimeConfig:
    activate_threshold: float = 0.7
    deactivate_threshold: float = 0.4
    sticker_scale: float = 0.25
    debug: bool = False


def load_runtime_config(path: Path | None) -> RuntimeConfig:
    if path is None or not path.exists():
        return RuntimeConfig()

    data = json.loads(path.read_text())
    return RuntimeConfig(**{key: data[key] for key in RuntimeConfig.__dataclass_fields__ if key in data})


def save_runtime_config(path: Path, config: RuntimeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.__dict__, indent=2) + "\n")


def apply_runtime_config(state: PoseStateMachine, config: RuntimeConfig) -> None:
    state.activate_threshold = config.activate_threshold
    state.deactivate_threshold = config.deactivate_threshold


def update_runtime_config(config: RuntimeConfig, key_code: int) -> RuntimeConfig:
    step = 0.05
    if key_code == ord("["):
        return replace(config, activate_threshold=max(0.0, config.activate_threshold - step))
    if key_code == ord("]"):
        return replace(config, activate_threshold=min(1.0, config.activate_threshold + step))
    if key_code == ord("-"):
        return replace(config, sticker_scale=max(0.05, config.sticker_scale - step))
    if key_code == ord("="):
        return replace(config, sticker_scale=min(1.0, config.sticker_scale + step))
    if key_code == ord("d"):
        return replace(config, debug=not config.debug)
    return config
