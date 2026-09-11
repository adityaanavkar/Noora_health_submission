"""Run the Version 2 core once, execute the evidence notebook, and export reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true", help="Force the RF core to rebuild its cache.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    version_dir = root / "version 2"
    runtime = version_dir / ".runtime"
    lock = runtime / "rf_execution.lock"
    if lock.exists():
        raise RuntimeError(f"An RF execution is already marked active: {lock}")
    runtime.mkdir(parents=True, exist_ok=True)
    lock.write_text("single executor lock\n", encoding="utf-8")
    try:
        sys.path.insert(0, str(version_dir / "src"))
        from experiment import run_experiment  # supplied by the core worker

        print("Running core experiment once...")
        run_experiment(version_dir, force_recompute=args.force_recompute)

        import nbformat
        from nbclient import NotebookClient

        notebook_path = version_dir / "RANDOM_FOREST.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=1200,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, notebook_path)

        import subprocess

        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--HTMLExporter.exclude_input_prompt=True", "--output", "RANDOM_FOREST.html", str(notebook_path)],
            cwd=version_dir,
            check=True,
        )

        from reporting import load_artifacts, write_reports

        data = load_artifacts(version_dir)
        write_reports(data)
        print(f"Executed notebook: {notebook_path}")
        print(f"Notebook HTML: {version_dir / 'RANDOM_FOREST.html'}")
        print(f"Standalone report HTML: {version_dir / 'RANDOM_FOREST_REPORT.html'}")
        print(f"Markdown report: {version_dir / 'REPORT.md'}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
