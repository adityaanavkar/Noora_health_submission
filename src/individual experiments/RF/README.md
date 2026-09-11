# Version 2: nested Random Forest

This folder evaluates a Random Forest router under the same grouped nested-validation protocol used by Version 1 Part 2. The notebook is a reader of saved evidence. The model-fitting implementation belongs to the core worker in `src/experiment.py`; this folder does not reuse or execute the old logistic-regression notebook.

## What the forest search means

A Random Forest is a committee of decision trees. Each tree learns a set of simple if/then splits, and the committee combines the trees into a probability-like score. `n_estimators` is fixed at 200. `max_depth` limits how many decisions a tree can make; `min_samples_leaf` prevents leaves from becoming too specific; `max_features` controls how many embedding coordinates are considered at each split; and `class_weight=balanced` gives extra influence to the smaller class.

The search tests the complete bounded grid of 24 forest settings across embedding spaces A, B, and C: 72 candidates for each outer search. The grid is fixed before outer evaluation. The selected result is the best configuration within this bounded search, not a global maximum over every possible forest.

## Run from the repository root

The executor uses the repository root virtual environment, runs the core exactly once, reuses a valid cache when available, executes the notebook with the root as its working directory, writes the executed notebook and `RANDOM_FOREST.html`, creates the charts, and writes `REPORT.md` plus `RANDOM_FOREST_REPORT.html`.

```powershell
.\.venv\Scripts\python.exe "version 2\scripts\execute_rf.py"
```

To intentionally rebuild the core cache:

```powershell
.\.venv\Scripts\python.exe "version 2\scripts\execute_rf.py" --force-recompute
```

Run only one executor at a time. The script leaves all generated RF artifacts under `version 2`; it does not write the Version 1 output directory or the root `outputs` directory.

## Artifact contract used by the notebook

The reporting layer fails loudly when an artifact or required field is absent. The core writes `outputs/metrics.csv`, `comparative_metrics.csv`, `outer_predictions.csv`, `candidate_results.csv`, `actual_grid.csv`, `threshold_sweep.csv`, `fold_choices.csv`, `splits.csv`, `summary.json`, `configuration.json`, and `test_predictions.csv`.

`candidate_results.csv` includes `embedding`, `config_id`, JSON `forest_params`, the expanded depth/leaf/features/weight fields, and `mean_inner_average_precision`, with 72 candidate rows per outer search. `outer_predictions.csv` includes raw and selected probabilities plus fold-specific thresholds and decisions for `default_0_5`, `balanced_macro_f1`, and `recall_first`. `metrics.csv` contains aggregate and outer-fold rows for RF, along with aggregate LR and label-only LLM comparison rows. The LLM must have `score_available=False` or an equivalent missing probability; no LLM ROC or AP curve is drawn.

The final model recipe is read from the core's all-training selection in `summary.json`. A pooled average of outer-fold candidate scores is used only for a tuning display and is never described as the final recipe.

## Charts and interpretation

The notebook writes held-out ROC/precision-recall curves, a readable top-12 forest-parameter tuning rank from the full grid, development-only threshold rate and count panels, three aggregate confusion matrices for the policies over the 903 saved fold-specific decisions, calibration and probability-bin plots, and an aggregate RF/LR/LLM comparison. Fold variation appears in a separate table. Percentages are shown with one or two decimal places; probability metrics such as AP and Brier retain a compact decimal format.

Precision asks: of the rows referred, how many were truly relevant? Recall asks: of all truly relevant rows, how many were caught? A false positive is an unnecessary referral; a false negative is a missed relevant row. The supplied LLM has labels but no probability score. It can be compared on accuracy, precision, recall, subgroup counts, and workload; ROC AUC and average precision are undefined. The blind test has no labels and therefore cannot support test performance or safety claims.
