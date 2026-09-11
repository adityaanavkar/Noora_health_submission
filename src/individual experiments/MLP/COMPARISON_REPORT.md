# Comparison report: V1 LR, V2 RF, V3 MLP, and LLM

This report reads actual predecessor CSV/JSON artifacts. New V3 rows are added only after the new V3 core outputs exist; the old no-space version3 attempt is never substituted for the new run.

| System | Policy | Protocol | Accuracy | Macro-F1 | Precision | Relevant recall | Medical recall | AP | FN | FP | Referrals | Threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 Part2 tuned LR | default_0_5 | V1 Part2 nested grouped | 87.26% | 86.35% | 88.10% | 91.74% | 95.81% | 0.9584 | 46 | 69 | 580 | 0.500000 |
| V1 Part2 tuned LR | balanced_macro_f1 | V1 Part2 nested grouped | 86.60% | 85.72% | 88.25% | 90.31% | 94.61% | 0.9584 | 54 | 67 | 570 | 0.434508 |
| V1 Part2 tuned LR | recall_first | V1 Part2 nested grouped | 83.94% | 81.40% | 80.29% | 98.03% | 99.70% | 0.9584 | 11 | 134 | 680 | 0.240482 |
| V1 Part2 fixed B LR | default_0_5 | V1 Part2 nested grouped | 87.38% | 86.22% | 86.49% | 94.25% | 98.80% | 0.9625 | 32 | 82 | 607 | 0.500000 |
| V2 tuned RF | default_0_5 | V2 nested grouped | 87.04% | 86.11% | 87.93% | 91.56% | 97.90% | 0.9654 | 47 | 70 | 580 | 0.500000 |
| V2 tuned RF | balanced_macro_f1 | V2 nested grouped | 86.82% | 86.01% | 88.83% | 89.95% | 97.01% | 0.9654 | 56 | 63 | 564 | 0.593978 |
| V2 tuned RF | recall_first | V2 nested grouped | 84.83% | 82.65% | 81.53% | 97.49% | 100.00% | 0.9654 | 14 | 123 | 666 | 0.207687 |
| Supplied LLM | frozen_label | V2 shared-row label-only reference | 97.90% | 97.77% | 97.86% | 98.74% | 100.00% | undefined | 7 | 12 | 562 | — |
| V3 tuned MLP | default_0_5 | V3 nested grouped | 87.26% | 86.45% | 88.91% | 90.66% | 96.71% | 0.9464 | 52 | 63 | 568 | 0.500000 |
| V3 tuned MLP | balanced_macro_f1 | V3 nested grouped | 87.38% | 86.62% | 89.48% | 90.13% | 95.81% | 0.9464 | 55 | 59 | 561 | 0.710575 |
| V3 tuned MLP | recall_first | V3 nested grouped | 86.05% | 84.28% | 83.20% | 96.95% | 99.70% | 0.9464 | 17 | 109 | 649 | 0.005897 |

V1 Part 2, V2, and the planned new V3 use the shared five-fold grouped outer / three-fold grouped inner design when their coordinated artifacts are complete. The original V1 single approximately 80/20 holdout is a different protocol and is kept in the separate table below. Different protocols are descriptive evidence, not a causal ranking.

Type I error is an unnecessary referral: a human spends time reviewing an irrelevant query. Type II error is a missed relevant query: the nursing workflow may fail to receive a query that should have been routed. The second error can carry greater service or safety consequence, so the threshold rule must be agreed with operations rather than chosen from accuracy alone.

The LLM has labels only and no probability score, so ROC AUC and average precision are undefined for it. Local heads and an LLM also differ in encoder, latency, privacy, and cost assumptions.

## Original V1 single-holdout audit (181 labeled rows)

| Method | Accuracy | Macro-F1 | Relevant recall | Relevant FN | Medical FN / total |
|---|---:|---:|---:|---:|---:|
| embedding_a | 82.87% | 81.20% | 91.89% | 9 | 2 / 72 |
| embedding_b | 84.53% | 83.20% | 91.89% | 9 | 1 / 72 |
| embedding_c | 81.77% | 79.68% | 92.79% | 8 | 0 / 72 |
| llm_train | 99.45% | 99.42% | 100.00% | 0 | 0 / 72 |
| always_relevant | 61.33% | 38.01% | 100.00% | 0 | 0 / 72 |

This 181-row table is not merged into the nested comparison because its split, selection, and evaluation scope differ.

## Preserved old version3 NN audit (exploratory only)

The old no-space attempt selected c and nn_medium. It has 36 candidate rows and 4515 saved split rows. Its fairness is unverified against the current contract because it used a six-candidate search, did not persist the full inner split table, and lacked the required calibration/Brier and provenance evidence. Its measured rows are excluded from the main comparison.

Among default-policy rows, V3 tuned MLP has the highest measured macro-F1 (86.45%). The declared local handoff rule is minimum Medical FN, then relevant FN, then FP, then maximum macro-F1; it selects V2 tuned RF / Recall-first for the assignment prediction artifact. This is a measured development choice, not a production guarantee.
