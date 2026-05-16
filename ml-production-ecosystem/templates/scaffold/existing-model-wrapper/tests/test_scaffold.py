from pathlib import Path
from {{package_name}}.adapter import write_summary

def test_write_summary(tmp_path: Path) -> None:
    assert write_summary(tmp_path / "reports" / "summary.json")["version"] == "external"
