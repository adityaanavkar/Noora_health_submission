"""Regenerate reports from existing artifacts without training or notebook execution."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "version 3"))
    from src.reporting import load_artifacts, write_reports

    data = load_artifacts(root, strict=False)
    write_reports(data)
    state = "pending" if data["pending"] else "measured"
    print(f"Reports regenerated from {state} V3 artifacts.")


if __name__ == "__main__":
    main()
