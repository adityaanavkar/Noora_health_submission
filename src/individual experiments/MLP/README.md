# Version 3: beginner MLP router

This spaced version 3 folder is the canonical Version 3 notebook and report folder. The old no-space version3 directory is preserved and appears only as an exploratory fairness audit. The core is version 3/src/experiment.py; the reporting API is version 3/src/reporting.py.

The core uses the shared Version 1 Part 2 / Version 2 grouped nested-validation contract: the same labeled rows, duplicate groups, outer and inner split IDs, A/B/C embeddings, frozen training LLM labels, three threshold policies, and no blind-test labels. The notebook reads actual settings from summary.json, including regularization, iteration controls, convergence fields, and hardware. It does not guess those values.

## Run from the repository root

Use the current root virtual environment in PowerShell:

~~~powershell
.\.venv\Scripts\python.exe "version 3\train.py" --pilot
.\.venv\Scripts\python.exe "version 3\train.py" --run
.\.venv\Scripts\python.exe "version 3\scripts\execute_notebook.py"
~~~

The pilot performs one fit for a runtime check. The full core run writes machine-readable artifacts under version 3/outputs, including the declared grid, split assignments, fold choices, outer predictions, metrics, threshold sweep, final model, final source-order test predictions, settings, convergence diagnostics, and environment.

The predecessor outputs in version 1/part2/outputs and version 2/outputs, plus the quantitative EDA files in analysis_outputs, are prerequisites for the coordinated comparison and assignment response. The core verifies the predecessor split contract before fitting.

## One notebook switch

The first notebook code cell contains one switch:

~~~python
RUN_CORE = True
~~~

With True, the notebook trains only when new V3 outputs are missing and reuses complete outputs when they are present. With False, it only reads saved outputs. The default is True so Run All follows the expected train-on-first-run/reuse-existing behavior. If outputs are missing and the switch is False, the notebook stays readable and writes pending report drafts rather than inventing results.

## Fast cached rerender

After a successful core run, regenerate the executed notebook, HTML, charts, reports, and the recommended local handoff without training:

~~~powershell
.\.venv\Scripts\python.exe "version 3\scripts\execute_notebook.py"
~~~

This reads version 3/outputs, executes NEURAL_NETWORK.ipynb with the repository root as its working directory, exports NEURAL_NETWORK.html, and regenerates REPORT.md, COMPARISON_REPORT.md, ASSIGNMENT_RESPONSE.md, recommended_test_predictions.csv, and its metadata.

## Handoff and interpretation

The core final prediction file is a blind inference artifact. Its labels cannot be scored because test truth is unavailable. The reporting API applies the declared local handoff rule—minimum Medical false negatives, then relevant false negatives, then false positives, then maximum macro-F1—to measured tuned LR/RF/MLP policies and copies that existing model's predictions into recommended_test_predictions.csv. This local assignment artifact is separate from the production recommendation. If the supplied LLM still dominates after the new MLP run, production can prefer the LLM behind human fallback and monitoring, subject to its unknown encoder cost, latency, privacy fit, and drift behavior.

The charts cover training/splits, tuning rank, ROC/precision-recall, development threshold rate and count, three aggregate policy confusion matrices, calibration with bin counts, and compact aggregate comparison. Fold variation is separate. Reports distinguish held-out development evidence, assumptions, and the absence of blind-test quality.

Astra handled planning, coordination, and review. Luna implemented the core-facing reporting API, canonical notebook, charts, reports, and checks.
