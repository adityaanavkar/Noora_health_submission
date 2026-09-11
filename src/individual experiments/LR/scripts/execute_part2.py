"""Execute LOGISTIC_REGRESSION_PART2.ipynb and export its HTML copy.

The notebook is the single execution entry point.  Its first code cell calls
the reusable core API once, so the executor does not duplicate model training.
"""

from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager


PART2_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PART2_DIR / "LOGISTIC_REGRESSION_PART2.ipynb"
HTML_PATH = PART2_DIR / "LOGISTIC_REGRESSION_PART2.html"
RUNTIME_DIR = PART2_DIR / ".runtime" / "ipython"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ["IPYTHONDIR"] = str(RUNTIME_DIR)

notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
nbformat.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"LOGISTIC_REGRESSION_PART2_cell_{index}", "exec")

manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [
    sys.executable,
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
client = NotebookClient(
    notebook,
    km=manager,
    timeout=1200,
    resources={"metadata": {"path": str(PART2_DIR)}},
)


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)


client.on_cell_start = progress
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)

nbformat.validate(notebook)
errors = [
    output
    for cell in notebook.cells
    if cell.cell_type == "code"
    for output in cell.get("outputs", [])
    if output.output_type == "error"
]
if errors:
    raise RuntimeError(f"Notebook produced {len(errors)} error output(s)")

nbformat.write(notebook, NOTEBOOK_PATH)
exporter = HTMLExporter()
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
body, _ = exporter.from_notebook_node(notebook)
HTML_PATH.write_text(body, encoding="utf-8")

print(
    {
        "cells": len(notebook.cells),
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "markdown_cells": sum(cell.cell_type == "markdown" for cell in notebook.cells),
        "errors": len(errors),
        "html": str(HTML_PATH),
    },
    flush=True,
)
