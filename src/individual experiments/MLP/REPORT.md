# Version 3 report: a regularized MLP router

This report is generated from the executed notebook and the new V3 core artifacts. Values below are measured on held-out labeled development rows unless stated otherwise.

## Scope and procedure

The data has 903 labeled rows, 902 exact-duplicate groups, five grouped outer folds, and three grouped inner folds. The bounded search tests 54 MLP candidates per inner search across embeddings A, B, and C. The outer holdout is excluded from model, calibration, and threshold selection.

The final all-training recipe selected embedding C, configuration mlp_c_00, hidden layers [32], alpha 0.01, learning rate 0.0005, and mean inner average precision 0.9677. The saved configuration records the actual solver, activation, batch size, max iterations, tolerance, and early-stopping setting: {'architectures': [[32], [64], [32, 16]], 'alphas': [0.01, 1.0, 100.0], 'learning_rates': [0.0005, 0.001], 'max_iter': 300, 'early_stopping': False, 'batch_size': 64, 'tol': 0.0001, 'solver': 'adam', 'activation': 'relu', 'n_jobs': 1, 'pilot_max_fits': 1}. The model uses fit-partition standardization.

## Held-out development results

| Policy | Accuracy | Macro-F1 | Precision | Relevant recall | Relevant missed | Medical missed / total | Medical recall | Total referrals | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default 0.50 | 87.26% | 86.45% | 88.91% | 90.66% | 52 | 11 / 334 | 96.71% | 568 | 0.500000 |
| Balanced macro-F1 | 87.38% | 86.62% | 89.48% | 90.13% | 55 | 14 / 334 | 95.81% | 561 | 0.710575 |
| Recall-first | 86.05% | 84.28% | 83.20% | 96.95% | 17 | 1 / 334 | 99.70% | 649 | 0.005897 |

Selected-score ranking on held-out development rows: ROC AUC 0.9273, average precision 0.9464, and Brier score 0.1011. These are ranking and probability diagnostics, not blind-test quality or safety guarantees.

## Fold variation

| Policy | Recall range | Macro-F1 range | Relevant FN range | Medical FN range |
|---|---:|---:|---:|---:|
| Balanced macro-F1 | 81.25%–93.69% | 82.32%–88.96% | 7–21 | 1–8 |
| Default 0.50 | 85.71%–93.69% | 83.95%–88.11% | 7–16 | 1–5 |
| Recall-first | 92.86%–100.00% | 78.11%–88.78% | 0–8 | 0–1 |

## Blind prediction artifact

The final prediction file contains 904 rows and 904 unique IDs in source order. Blind labels are unavailable, so no blind-test quality metric is calculated.

## Recommendation and limitations

The recorded recommended local policy is recall_first at threshold 0.005897. It is a conditional operating choice when the illustrative 98% relevant and 99% Medical recall targets are prioritized. It changes referral workload and misses as shown above; it does not create a future guarantee. Keep the MLP only when its measured local-head trade-off fits the operating policy. Exact duplicate grouping does not test patient, conversation, time, language, or future distribution shift, and Medical is a measurement subgroup rather than an input feature.

Astra handled planning, coordination, and review. Luna implemented the core-facing API, notebook, reports, charts, and checks. All reported values are read from saved artifacts.
