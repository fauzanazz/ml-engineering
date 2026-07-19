"""Guided launcher wizard: assembles argv for the existing CLI dispatch.

The builders below are the load-bearing logic (unit-tested); the questionary
front-end collects fields and calls them. It never runs the app directly.
"""

from webcam_effect.components import ComponentSettings, format_components
from webcam_effect.dataset_creator import DATASET_LABELS
from webcam_effect.mediapipe_assets import doctor

DEFAULT_CAMERA = "0"
DEFAULT_CLASSIFIER = "models/kicau-classifier/best.pt"


# --- argv builders ---------------------------------------------------------

def build_run_argv(
    components: ComponentSettings,
    camera: str,
    debug: bool,
    audio: bool,
    video_output: str,
) -> list[str]:
    argv = [
        "run",
        "--no-components-tui",
        "--components",
        format_components(components),
        "--camera",
        camera,
        "--video-output",
        video_output,
    ]
    if debug:
        argv.append("--debug")
    if not audio:
        argv.append("--no-audio")
    return argv


def build_filters_argv(camera: str, resolution: str, video_output: str) -> list[str]:
    return ["filters", "--camera", camera, "--resolution", resolution, "--video-output", video_output]


def build_dataset_argv(label: str, camera: str) -> list[str]:
    return ["dataset", "--label", label, "--camera", camera]


def build_editor_argv() -> list[str]:
    return ["editor"]


def build_doctor_argv() -> list[str]:
    return ["doctor"]


def build_download_argv() -> list[str]:
    return ["download-models"]


# --- questionary front-end -------------------------------------------------

def run_wizard() -> list[str] | None:
    """Return argv to execute, or None if the user quits/cancels."""
    try:
        return _drive()
    except (KeyboardInterrupt, EOFError):
        return None


def _drive() -> list[str] | None:
    import questionary

    flows = {
        "Run effect": _flow_run,
        "Video filters": _flow_filters,
        "Effect editor (web UI)": _flow_editor,
        "Record dataset": _flow_dataset,
        "Check setup (doctor)": _flow_doctor,
        "Download models": _flow_download,
    }
    while True:
        action = questionary.select(
            "Webcam RTS Effect — what do you want to run?",
            choices=[*flows, "Quit"],
        ).ask()
        if action is None or action == "Quit":
            return None
        argv = flows[action]()
        if argv is not None:
            return argv
        # cancelled sub-flow → back to main menu


def _flow_run() -> list[str] | None:
    import questionary
    from pathlib import Path

    from webcam_effect.effects import EffectLibrary, load_effect_library, save_effect_library

    config = Path("assets/effect.json")
    library = load_effect_library(config)
    choices = [questionary.Choice(effect.name, value=effect_id) for effect_id, effect in library.effects.items()]
    default = next((c for c in choices if c.value == library.selected_id), None)
    effect_id = questionary.select("Effect", choices=choices, default=default).ask()
    if effect_id is None:
        return None

    camera = questionary.text("Camera", default=DEFAULT_CAMERA).ask()
    if camera is None:
        return None

    video_output = questionary.select("Video output", choices=["preview", "none"]).ask()
    if video_output is None:
        return None

    debug = questionary.confirm("Debug overlay?", default=False).ask()
    if debug is None:
        return None

    audio = questionary.confirm("Audio?", default=True).ask()
    if audio is None:
        return None

    problems = doctor(Path(DEFAULT_CLASSIFIER), Path("assets"))
    if problems:
        choice = questionary.select(
            "Setup problems found:\n  - " + "\n  - ".join(problems),
            choices=["Continue anyway", "Download models", "Back to menu"],
        ).ask()
        if choice is None or choice == "Back to menu":
            return None
        if choice == "Download models":
            return build_download_argv()

    # Full pipeline; the chosen effect drives the visuals, not per-component toggles.
    argv = _confirm(build_run_argv(ComponentSettings(), camera, debug, audio, video_output))
    if argv is None:
        return None
    # ponytail: persist the pick as selected_id (same model as the editor's select-effect).
    if effect_id != library.selected_id:
        save_effect_library(config, EffectLibrary(selected_id=effect_id, effects=library.effects))
    return argv


def _flow_filters() -> list[str] | None:
    import questionary

    camera = questionary.text("Camera", default=DEFAULT_CAMERA).ask()
    if camera is None:
        return None
    resolution = questionary.select("Resolution", choices=["640x480", "1280x720"]).ask()
    if resolution is None:
        return None
    video_output = questionary.select("Video output", choices=["preview", "none"]).ask()
    if video_output is None:
        return None
    return _confirm(build_filters_argv(camera, resolution, video_output))


def _flow_dataset() -> list[str] | None:
    import questionary

    label = questionary.select("Dataset label", choices=sorted(DATASET_LABELS)).ask()
    if label is None:
        return None
    camera = questionary.text("Camera", default=DEFAULT_CAMERA).ask()
    if camera is None:
        return None
    return _confirm(build_dataset_argv(label, camera))


def _flow_editor() -> list[str] | None:
    return _confirm(build_editor_argv(), "Starts a local web server for the effect editor.")


def _flow_doctor() -> list[str] | None:
    return _confirm(build_doctor_argv(), "Checks models and effect assets are present.")


def _flow_download() -> list[str] | None:
    return _confirm(build_download_argv(), "Downloads missing MediaPipe models into assets/.")


def _confirm(argv: list[str], note: str | None = None) -> list[str] | None:
    import questionary

    if note:
        questionary.print(note, style="italic")
    launch = questionary.confirm(
        "Launch: python main.py " + " ".join(argv) + " ?",
        default=True,
    ).ask()
    return argv if launch else None
