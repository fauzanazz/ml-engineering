"""Summarize continual-learning history JSONL into trend report."""

from pathlib import Path
import argparse
import json
from collections import Counter
from typing import Any

DEFAULT_HISTORY_PATH = Path("artifacts/reports/production-patterns/continual-learning-history.jsonl")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/continual-learning-summary.json")


def _read_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in history_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if isinstance(record, dict):
            entries.append(record)
    return entries


def summarize_continual_history(
    history_path: Path = DEFAULT_HISTORY_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    entries = _read_history(history_path)
    actions = Counter(entry.get("action") for entry in entries)
    triggers = Counter(entry.get("trigger") for entry in entries)
    approved = sum(1 for entry in entries if entry.get("approved_for_retraining"))

    summary: dict[str, Any] = {
        "history_path": str(history_path),
        "total_decisions": len(entries),
        "action_counts": {str(action): count for action, count in actions.items()},
        "trigger_counts": {str(trigger): count for trigger, count in triggers.items()},
        "approved_for_retraining": approved,
        "latest_decision": entries[-1] if entries else None,
    }
    if not entries:
        summary["status"] = "empty"
    elif actions.get("retrain", 0) >= 2:
        summary["status"] = "recurring-retrain"
    elif actions.get("investigate", 0) >= 1:
        summary["status"] = "needs-investigation"
    else:
        summary["status"] = "stable"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize continual-learning history.")
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = summarize_continual_history(args.history_path, args.output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
