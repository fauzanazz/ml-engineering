"""Beginner-friendly local orchestrator for ML Production Ecosystem."""

from pathlib import Path
import argparse
import json
import shutil
import sys
from typing import Callable

from .lifecycle_demo import run_lifecycle_demo
from .lifecycle_status import DEFAULT_OUTPUT_PATH, DEFAULT_REPORT_DIR, build_lifecycle_status
from .scaffold import SUPPORTED_PRESETS, ScaffoldRequest, scaffold_project

DEFAULT_CONFIG_PATH = Path("configs/local-lifecycle-demo.yaml")
LOCAL_LIFECYCLE_OUTPUT_PATH = DEFAULT_REPORT_DIR / "local-lifecycle-demo.json"
LOCAL_LIFECYCLE_GRAPH_PATH = DEFAULT_REPORT_DIR / "local-lifecycle-demo.mmd"
LOCAL_LIFECYCLE_GRAPH_HTML_PATH = DEFAULT_REPORT_DIR / "local-lifecycle-demo.html"
DOCTOR_REQUIRED_PATHS = (
    Path("pyproject.toml"),
    DEFAULT_CONFIG_PATH,
    Path("01-foundation"),
    Path("02-production-patterns"),
    Path("shared"),
)

CommandHandler = Callable[[argparse.Namespace], int]


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def _print_json_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def _status_icon(status: object) -> str:
    status_text = str(status)
    if status_text in {"approved", "completed", "monitor", "passed", "ready", "stable"}:
        return "[ok]"
    if status_text in {"blocked", "failed", "pending", "skipped", "unknown"}:
        return "[wait]"
    if status_text == "missing":
        return "[missing]"
    return "[info]"


def _recommended_next_step(summary: dict[str, object]) -> str:
    steps = summary.get("steps", {})
    step_statuses = {
        name: step.get("status")
        for name, step in steps.items()
        if isinstance(steps, dict) and isinstance(step, dict)
    }
    missing = [str(step) for step in summary.get("missing_steps", [])]
    blocked = [str(step) for step in summary.get("blocked_steps", [])]
    if "training" in missing or "offline_validation" in missing:
        return "uv run mle quickstart"
    if step_statuses.get("approval") == "pending":
        return "uv run mle quickstart --approve"
    if step_statuses.get("deployment_demo") == "skipped" or step_statuses.get("drift") == "skipped":
        return "uv run foundation-serve-recommender --port 18080, then ./scripts/smoke-local-deployment.sh"
    if "continual_learning_summary" in missing:
        return "uv run production-continual-summary"
    if missing:
        return f"generate missing report: {missing[0]}"
    if blocked:
        return f"inspect blocked report: {blocked[0]}"
    return "uv run foundation-serve-recommender --port 18080"


def print_human_status(summary: dict[str, object]) -> None:
    _print_header("ML Production Status")
    print(f"Overall: {summary['status']}")
    print(f"Reports: {summary['report_dir']}")
    print("\nSteps:")
    steps = summary.get("steps", {})
    if isinstance(steps, dict):
        for name, raw_step in steps.items():
            step = raw_step if isinstance(raw_step, dict) else {}
            status = step.get("status", "unknown")
            path = step.get("path", "")
            print(f"  {_status_icon(status)} {name}: {status} ({path})")
    print(f"\nNext: {_recommended_next_step(summary)}")


def run_quickstart(args: argparse.Namespace) -> int:
    _print_header("Beginner Quickstart")
    print("Running local no-download lifecycle demo.")
    summary = run_lifecycle_demo(
        config_path=args.config,
        approve=args.approve,
        set_active=args.set_active,
        output_path=LOCAL_LIFECYCLE_OUTPUT_PATH,
        graph_path=LOCAL_LIFECYCLE_GRAPH_PATH,
        graph_html_path=LOCAL_LIFECYCLE_GRAPH_HTML_PATH,
    )
    print("\nDone.")
    print(f"Lifecycle report: {LOCAL_LIFECYCLE_OUTPUT_PATH}")
    print(f"Lifecycle graph: {LOCAL_LIFECYCLE_GRAPH_HTML_PATH}")
    if args.json:
        _print_json_summary(summary)
        return 0
    status = build_lifecycle_status(DEFAULT_REPORT_DIR, DEFAULT_OUTPUT_PATH)
    print_human_status(status)
    return 0


def run_status(args: argparse.Namespace) -> int:
    summary = build_lifecycle_status(args.report_dir, args.output_path)
    if args.json:
        _print_json_summary(summary)
    else:
        print_human_status(summary)
    return 0


def run_doctor(args: argparse.Namespace) -> int:
    del args
    _print_header("Local Doctor")
    checks = {
        "uv executable": shutil.which("uv") is not None,
        "python executable": shutil.which("python") is not None or shutil.which("python3") is not None,
        "docker executable": shutil.which("docker") is not None,
    }
    for path in DOCTOR_REQUIRED_PATHS:
        checks[str(path)] = path.exists()
    for name, passed in checks.items():
        print(f"  {'[ok]' if passed else '[missing]'} {name}")
    if all(checks.values()):
        print("\nReady: uv run mle quickstart")
        return 0
    print("\nFix missing items, then rerun: uv run mle doctor")
    return 1


def run_explain(args: argparse.Namespace) -> int:
    summary = build_lifecycle_status(args.report_dir, args.output_path)
    missing = summary.get("missing_steps", [])
    blocked = summary.get("blocked_steps", [])
    _print_header("What This Means")
    if summary["status"] == "ready":
        print("Lifecycle ready. Model trained, validated, approved, checked, and graphed locally.")
    else:
        print("Lifecycle incomplete. Some production evidence missing or blocked.")
    if missing:
        print(f"Missing: {', '.join(str(step) for step in missing)}")
    if blocked:
        print(f"Blocked: {', '.join(str(step) for step in blocked)}")
    print(f"Next: {_recommended_next_step(summary)}")
    return 0


def run_new(args: argparse.Namespace) -> int:
    preset = args.preset or _prompt_preset()
    name = args.name or _prompt_required("Project name")
    target = args.target or Path(input(f"Target directory [{name}]: ").strip() or name)
    result = scaffold_project(
        ScaffoldRequest(
            preset=preset,
            name=name,
            target=target,
            force=args.force,
        )
    )
    _print_header("Project Scaffolded")
    print(f"Preset: {result.preset}")
    print(f"Name: {result.name}")
    print(f"Package: {result.package_name}")
    print(f"Target: {result.target}")
    print(f"Files: {len(result.written_paths)}")
    print(f"\nNext: cd {result.target} && uv run pytest")
    return 0


def _prompt_preset() -> str:
    choices = ", ".join(SUPPORTED_PRESETS)
    while True:
        preset = input(f"Preset ({choices}): ").strip().lower()
        if preset in SUPPORTED_PRESETS:
            return preset
        print(f"Choose one of: {choices}")


def _prompt_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print(f"{label} is required.")


def run_menu(args: argparse.Namespace) -> int:
    del args
    actions: list[tuple[str, CommandHandler, argparse.Namespace]] = [
        ("Start beginner demo", run_quickstart, argparse.Namespace(config=DEFAULT_CONFIG_PATH, approve=False, set_active=False, json=False)),
        ("View lifecycle status", run_status, argparse.Namespace(report_dir=DEFAULT_REPORT_DIR, output_path=DEFAULT_OUTPUT_PATH, json=False)),
        ("Run local doctor", run_doctor, argparse.Namespace()),
        ("Explain current state", run_explain, argparse.Namespace(report_dir=DEFAULT_REPORT_DIR, output_path=DEFAULT_OUTPUT_PATH)),
        ("Exit", lambda _: 0, argparse.Namespace()),
    ]
    while True:
        _print_header("ML Production Ecosystem")
        for index, (label, _, _) in enumerate(actions, start=1):
            print(f"{index}. {label}")
        choice = input("Choose: ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(actions):
            print("Invalid choice.")
            continue
        label, handler, namespace = actions[int(choice) - 1]
        if label == "Exit":
            return 0
        result = handler(namespace)
        input("\nPress Enter to continue...")
        if result != 0:
            return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beginner-friendly ML Production Ecosystem orchestrator.")
    subparsers = parser.add_subparsers(dest="command")

    quickstart = subparsers.add_parser("quickstart", help="Run no-download local lifecycle demo.")
    quickstart.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    quickstart.add_argument("--approve", action="store_true")
    quickstart.add_argument("--set-active", action="store_true")
    quickstart.add_argument("--json", action="store_true")
    quickstart.set_defaults(handler=run_quickstart)

    status = subparsers.add_parser("status", help="Show lifecycle status summary.")
    status.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    status.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=run_status)

    doctor = subparsers.add_parser("doctor", help="Check local prerequisites.")
    doctor.set_defaults(handler=run_doctor)

    explain = subparsers.add_parser("explain", help="Explain current lifecycle state.")
    explain.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    explain.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    explain.set_defaults(handler=run_explain)

    new = subparsers.add_parser("new", help="Create a boilerplate ML project scaffold.")
    new.add_argument("--preset", choices=SUPPORTED_PRESETS)
    new.add_argument("--name")
    new.add_argument("--target", type=Path)
    new.add_argument("--force", action="store_true")
    new.set_defaults(handler=run_new)

    parser.set_defaults(handler=run_menu)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.handler(args))


if __name__ == "__main__":
    main()
