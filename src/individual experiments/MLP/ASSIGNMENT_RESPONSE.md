# Assignment response

## 1. Embedding structure

The first two PCA components explain 18.27% of A, 19.24% of B, and 7.44% of C. Reaching 95% variance requires 125, 155, and 457 components. Binary silhouette is 0.0487, 0.0726, and 0.0292 for A, B, and C. At k=5, neighborhood label agreement is 82.24%, 85.29%, and 84.32%. These checks show local label structure but weak global separation.

PCA variance is not predictive signal. Silhouette depends on the distance and labels, and neighborhood agreement can reflect duplicates or local concentration. Grouped nested evaluation is the model-selection evidence. New V3 status: complete; see the measured V3 report.

The label count is 557 relevant and 346 irrelevant rows. The EDA found 2 duplicated query IDs represented in 4 embedding-space rows, 19 LLM routing errors (7 missed-relevant and 12 unnecessary-referral errors), and 3 errors involving intents outside the frozen routing vocabulary. A simple diagnostic flag of k=15 neighborhood label agreement at or below 0.20 identifies 59 query IDs; this is an investigation queue, not a ground-truth outlier label.

The 2D PCA and t-SNE figures are useful for orientation, but a 2D projection can hide separation, distort distances, and change apparent clusters. Use the saved high-dimensional geometry and grouped held-out results for decisions.

### Measured quality snapshot

| System | Policy | Accuracy | Recall | Medical FN / total | FN | FP | Referrals |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 Part2 tuned LR | default_0_5 | 87.26% | 91.74% | 14 / 334 | 46 | 69 | 580 |
| V1 Part2 tuned LR | balanced_macro_f1 | 86.60% | 90.31% | 18 / 334 | 54 | 67 | 570 |
| V1 Part2 tuned LR | recall_first | 83.94% | 98.03% | 1 / 334 | 11 | 134 | 680 |
| V1 Part2 fixed B LR | default_0_5 | 87.38% | 94.25% | 4 / 334 | 32 | 82 | 607 |
| V2 tuned RF | default_0_5 | 87.04% | 91.56% | 7 / 334 | 47 | 70 | 580 |
| V2 tuned RF | balanced_macro_f1 | 86.82% | 89.95% | 10 / 334 | 56 | 63 | 564 |
| V2 tuned RF | recall_first | 84.83% | 97.49% | 0 / 334 | 14 | 123 | 666 |
| Supplied LLM | frozen_label | 97.90% | 98.74% | 0 / 334 | 7 | 12 | 562 |
| V3 tuned MLP | default_0_5 | 87.26% | 90.66% | 11 / 334 | 52 | 63 | 568 |
| V3 tuned MLP | balanced_macro_f1 | 87.38% | 90.13% | 14 / 334 | 55 | 59 | 561 |
| V3 tuned MLP | recall_first | 86.05% | 96.95% | 1 / 334 | 17 | 109 | 649 |

For an individual subgroup, the saved outer predictions show:

| System | Intent | Rows | Relevant | Type II FN | Type I FP |
|---|---|---:|---:|---:|---:|
| V1 Part2 tuned LR | Medical | 334 | 334 | 14 | 0 |
| V1 Part2 tuned LR | Other | 51 | 51 | 15 | 0 |
| V1 Part2 tuned LR | Feedback | 34 | 34 | 10 | 0 |
| V1 Part2 tuned LR | Acknowledgment | 208 | 0 | 0 | 29 |
| V2 tuned RF | Medical | 334 | 334 | 7 | 0 |
| V2 tuned RF | Other | 51 | 51 | 17 | 0 |
| V2 tuned RF | Feedback | 34 | 34 | 16 | 0 |
| V2 tuned RF | Acknowledgment | 208 | 0 | 0 | 20 |
| Supplied LLM | Medical | 334 | 334 | 0 | 0 |
| Supplied LLM | Other | 51 | 51 | 5 | 0 |
| Supplied LLM | Feedback | 34 | 34 | 2 | 0 |
| Supplied LLM | Acknowledgment | 208 | 0 | 0 | 8 |
| V3 tuned MLP | Medical | 334 | 334 | 11 | 0 |
| V3 tuned MLP | Other | 51 | 51 | 14 | 0 |
| V3 tuned MLP | Feedback | 34 | 34 | 17 | 0 |
| V3 tuned MLP | Acknowledgment | 208 | 0 | 0 | 23 |

Type I error means an irrelevant query is referred unnecessarily, consuming human capacity and possibly delaying higher-priority work. Type II error means a relevant query is missed, so the intended nursing workflow may never see it. Type II consequences can be more serious, so the threshold and fallback policy must be agreed with operations.

## 2. Pipeline

Validate IDs, labels, shapes, finiteness, and source order; align A/B/C embeddings; build exact duplicate groups from the union of embedding rows; and keep intent for subgroup measurement only. Use five stratified grouped outer folds and three grouped inner folds. Select the MLP by inner average precision, fit scaling inside each fit partition, compare raw and sigmoid-selected probabilities by inner Brier score, and choose default, balanced macro-F1, and recall-first thresholds inside development data. Evaluate frozen outer decisions, then refit the frozen recipe on all labeled rows and emit source-order blind predictions.

## 3. Cost and recommendation

The supplied scenario is 500 USD base plus 40 USD usage, or **540 USD/month**, for 1,000,000 monthly LLM-routed queries at its stated token assumptions. This is the requested hybrid 540 USD/month scenario and excludes embedding generation. Encoder cost and latency are unknown because embeddings are precomputed and no encoder benchmark was supplied.

The supplied LLM has the measured aggregate accuracy 97.90%; compare it with the measured local choice in COMPARISON_REPORT.md. Recommend the LLM behind a human fallback if its advantage survives prospective validation and its latency, privacy, encoder cost, and drift controls are acceptable. Otherwise deploy the exported local artifact under the declared miss-priority rule.

The declared local handoff rule is minimum Medical FN, then relevant FN, then FP, then maximum macro-F1. It is applied only to measured local rows. The measured local handoff is V2 tuned RF with Recall-first under the declared rule (Medical FN 0, relevant FN 14, FP 123, macro-F1 82.65%); its existing test probabilities are exported to recommended_test_predictions.csv. New V3 is complete; see the measured V3 report.

## 4. Failure modes, monitoring, and deliverables

Monitor embedding validity and norms, score and referral-rate drift, class prevalence, duplicate rates, per-intent and Medical outcomes, false-negative samples, queue capacity, latency, and model/version fingerprints. Watch for new intents, language or channel shift, encoder changes, mislabeled queries, threshold drift, and overloaded human review. Recompute quality only when delayed human labels arrive.

Deliverables are the beginner MLP notebook and exported HTML, split/tuning/ROC-PR/threshold/confusion/calibration/comparison charts, REPORT.md, COMPARISON_REPORT.md, this response, src/reporting.py, execution scripts, and the core-produced final recommended-classifier prediction file. Blind labels are absent, so no blind-test quality is claimed.
