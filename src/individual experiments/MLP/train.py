"""Command-line entry point for the Version 3 MLP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grouped nested Version 3 MLP experiment")
    parser.add_argument("--pilot", action="store_true", help="Run one bounded fit and write pilot_runtime.json only")
    parser.add_argument("--run", action="store_true", help="Run the complete 54-candidate-per-search experiment")
    args = parser.parse_args()
    if args.pilot == args.run:
        parser.error("choose exactly one of --pilot or --run")
    result = run_experiment(Path(__file__).resolve().parent, pilot=args.pilot)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
