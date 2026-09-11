"""Execute the canonical evidence notebook and export HTML.

This script never trains.  Training is performed explicitly by train.py or by
setting RUN_CORE=True in the notebook.  It requires the new V3 outputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    version_dir = root / "version 3"
    notebook_path = version_dir / "NEURAL_NETWORK.ipynb"
    if not notebook_path.is_file():
        raise FileNotFoundError(notebook_path)
    sys.path.insert(0, str(version_dir / "src"))
    from reporting import load_artifacts, write_reports

    data = load_artifacts(root, strict=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--HTMLExporter.exclude_input_prompt=True",
            "--output",
            "NEURAL_NETWORK.html",
            str(notebook_path),
        ],
        cwd=version_dir,
        check=True,
    )
    write_reports(data)
    print("Executed:", notebook_path)
    print("Exported:", version_dir / "NEURAL_NETWORK.html")
    print("Regenerated reports and recommended prediction handoff.")


if __name__ == "__main__":
    main()
