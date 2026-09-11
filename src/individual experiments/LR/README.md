# Part 2: nested logistic-regression evaluation

This folder contains the beginner-friendly notebook and the execution wrapper for Part 2. The reusable training code belongs to the core worker in `src/experiment.py`; the notebook is an evidence reader and visualization layer.

The experiment compares logistic regression across embedding spaces A, B, and C. It uses five grouped outer folds and three grouped inner folds. Each inner choice considers `C` in `{0.01, 0.1, 1, 10, 100}` and `class_weight` in `{None, balanced}` and selects by inner average precision. The core compares raw and sigmoid probabilities by inner-held-out Brier score; this run selected sigmoid for the final all-training recipe. The final threshold policy includes the default `0.5` operating point and development diagnostics for balanced macro-F1 and recall-first targets of 98% relevant recall and 99% Medical recall.

Those recall targets are assumptions for a conservative development analysis. They are not guarantees about future messages, production traffic, or clinical safety. Medical is a source intent used for subgroup measurement; it is not a feature available to the classifier at inference time.

## Run

The core API and `outputs/artifact_schema.json` are ready. The notebook calls the core exactly once, and the current source/config fingerprints allow the core to reuse its completed cache:

```powershell
.\.venv\Scripts\python.exe "version 1\part2\scripts\execute_part2.py"
```

The same command can be launched while the working directory is `version 1\part2`:

```powershell
..\..\.venv\Scripts\python.exe scripts\execute_part2.py
```

The executor uses `version 1/part2/.runtime/ipython` for notebook runtime state, runs with `version 1/part2` as the notebook working directory, writes the executed notebook, and exports `LOGISTIC_REGRESSION_PART2.html`. The normal path should report a cache hit and avoid a second training run.

## Core output contract

After a successful run, the core writes these files under `version 1/part2/outputs/`:

```text
summary.json
metrics.csv
outer_predictions.csv
fold_choices.csv
splits.csv
candidate_results.csv
threshold_sweep.csv
test_predictions.csv
```

The notebook now enforces the agreed core schema before drawing any chart. Missing files or columns raise a clear error. The required fields are: `metrics.csv` (`system,embedding,policy,evaluation,n,accuracy,macro_f1,precision,recall,specificity,tp,tn,fp,fn,medical_fn,medical_n,medical_recall,referral_fraction,roc_auc,average_precision,score_available,note`); `outer_predictions.csv` (`row_index,query_id,outer_fold,group_id,truth,true_label,intent,selected_embedding,selected_C,selected_class_weight,probability_method,p_raw,p_chosen,threshold_default_0_5,threshold_balanced_macro_f1,threshold_recall_first,label_default_0_5,label_balanced_macro_f1,label_recall_first,llm_predicted_label`); `fold_choices.csv` (the saved outer choices, probability method, three thresholds, feasibility, and leakage checks); `splits.csv` (`split_level,outer_fold,role,row_index,query_id,group_id,truth,label,intent`); `candidate_results.csv` (outer fold, embedding, `C`, class weight, mean inner AP, row count, rank); `threshold_sweep.csv` (inner development scope, threshold, precision/recall, Medical recall, workload, errors, and policy flags); and `test_predictions.csv` (`query_id,predicted_label,p_relevant`). These names are also recorded in `outputs/artifact_schema.json`; the notebook fails clearly if a required artifact or field is absent.

The outer predictions must be out-of-fold predictions: each row is scored by a model that did not fit on that row and whose candidate and policy choices did not use that row. The test file has 904 rows in source order and no labels. It can be checked for shape, ID uniqueness, label validity, and finite probabilities, but it cannot support test accuracy or recall.

## Notebook map

The notebook explains the data and split, defines folds, precision, recall, ROC/AUC, and `C`, shows label and fold counts, plots the inner tuning results, measures selected outer ROC and precision-recall behavior, sweeps inner-development thresholds, compares the three saved fold-specific policies with confusion matrices, checks raw versus selected calibration and Brier score, and finishes with a score-only comparison table including the supplied LLM label-only baseline. The confusion matrices read saved outer decisions for `default_0_5`, `balanced_macro_f1`, and `recall_first`; they never apply one final threshold to pooled outer probabilities.

The old V1 single holdout used a different selection protocol and is not a direct comparison. The current ABC logistic-regression rows are the fair untuned outer baselines inside this nested experiment. SVM and MLP experiments are future work and are not implemented in this folder. No external model API is used by Part 2.

## Current coordinated run

The saved run reports tuned default accuracy of **87.26%** and fair fixed B baseline accuracy of **87.38%** on the same 903 outer-held-out rows. Tuned default therefore does not beat fixed B on accuracy. Average precision is the search and ranking measure: tuned default AP is **0.9584**, versus **0.9625** for fixed B. AP should be read separately from accuracy; it evaluates score ranking across thresholds.

The three tuned policies show the workload trade-off. Default has 46 relevant misses, 69 false positives, and 14 Medical misses out of 334. Balanced macro-F1 has 54 relevant misses, 67 false positives, and 18 Medical misses out of 334; its outer macro-F1 is 85.7216%, below default's 86.3479%, so it is not called an outer winner. Recall-first has 98.0251% relevant recall, 11 relevant misses, 134 false positives, and 1 Medical miss out of 334. These are observations from this development experiment and do not guarantee future recall or improvement.

The supplied LLM has labels only. On the same 903 rows it measures **97.90%** accuracy, 7 relevant misses, 12 false positives, and 0 Medical misses out of 334. It has no probability score, so ROC-AUC and AP are undefined for it.

## Evidence policy

`REPORT.md` records the current measured threshold trade-offs and subgroup counts from the saved outputs. It does not claim test performance, calibration guarantees, future recall, or clinical improvement.
