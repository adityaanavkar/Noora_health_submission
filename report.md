# Intent classification: analysis and production recommendation

## 1. Understand the embedding spaces

We have **903 labeled queries**: 557 relevant and 346 irrelevant. The 904 test queries have no supplied ground truth, so all quality figures below refer to labeled development data. An **embedding** is a numeric representation of a query; A, B and C represent the same queries in different spaces.

| Space | Dimensions | Variance shown by 2D PCA | Label silhouette | Neighbour agreement, averaged equally across labels |
|---|---:|---:|---:|---:|
| A | 384 | 18.3% | 0.049 | 77.9% |
| B | 768 | 19.2% | 0.073 | 81.3% |
| C | 1,536 | 7.4% | 0.029 | 79.9% |

**PCA** compresses the vectors into fewer coordinates while preserving as much variation as possible. These two-dimensional views retain little of the original variation, especially for C: apparent clusters or overlaps cannot establish classification quality. **Silhouette** measures separation between label groups; values near zero indicate substantial overlap. **Neighbour agreement** measures how often nearby queries share a label, here using 15 neighbours.

B shows the strongest descriptive separation, although none of the spaces separates labels cleanly. More dimensions do not automatically help. Fine-grained intent silhouette is also low, indicating overlapping intent groups. Queries surrounded by opposite-label neighbours are candidates for investigating unusual examples or ambiguous labels; without query text, we cannot determine their cause. Two Other queries are exact duplicates in A and B, but not C. They are kept together during validation to avoid an overly optimistic estimate.

## 2. Build the classification pipeline

We compare **logistic regression (LR)**, a weighted combination of embedding coordinates; **random forest (RF)**, a collection of decision trees; and **MLP**, a small neural network that learns nonlinear combinations. All use supplied embeddings, not query IDs or known source intents, as inputs.

We use five **outer folds**: each labeled query is held out once to measure performance. Three **inner folds** inside each outer training partition select model settings, embedding space, confidence calibration and thresholds. Duplicate groups remain together, and scaling is learned only from the corresponding training rows. This separates model choices from their evaluation.

Local models provide a probability of relevance. **Calibration** checks whether a probability such as 0.8 corresponds to roughly 80% relevant examples. Raw and sigmoid-adjusted probabilities are compared using **Brier score**, the average squared probability error; lower is better. A **threshold** turns that probability into a routing decision.

We examine default 0.50, balanced class performance, and recall-first thresholds. Recall-first uses assumed inner-development targets of 98% relevant recall and 99% Medical recall; these are not guaranteed on unseen queries. Final models are refitted on all labeled rows after development choices, then produce blind-test predictions.

## 3. Evaluate production trade-offs

**Precision** is the proportion of referrals that are truly relevant. **Recall** is the proportion of relevant queries caught. A **false positive (Type I error)** unnecessarily refers an irrelevant query, increasing workload. A **false negative (Type II error)** misses a relevant query, potentially preventing timely nursing attention. Therefore recall, Medical misses and referral counts matter alongside accuracy.

The table shows measured results on the same 903 labeled queries. Local results use held-out outer predictions; the LLM uses its supplied, fixed predictions. The proposed two-stage router has not been evaluated.

| Pipeline | Accuracy | Precision | Recall | Relevant missed | Medical missed / 334 | Referrals |
|---|---:|---:|---:|---:|---:|---:|
| LR, default | 87.26% | 88.10% | 91.74% | 46 | 14 | 580 |
| RF, default | 87.04% | 87.93% | 91.56% | 47 | 7 | 580 |
| MLP, default | 87.26% | 88.91% | 90.66% | 52 | 11 | 568 |
| LR, recall-first | 83.94% | 80.29% | 98.03% | 11 | 1 | 680 |
| RF, recall-first | 84.83% | 81.53% | 97.49% | 14 | 0 | 666 |
| MLP, recall-first | 86.05% | 83.20% | 96.95% | 17 | 1 | 649 |
| Supplied LLM | 97.90% | 97.86% | 98.74% | 7 | 0 | 562 |

LR's recall-first policy prevents 35 additional relevant misses compared with default, at the cost of 100 additional referrals. RF has zero observed Medical misses under recall-first, but that does not guarantee zero future misses. The MLP does not establish a clear advantage for high-recall routing.

**Average precision**, a summary of precision across recall levels, is 0.958 for LR, 0.965 for RF and 0.946 for MLP. The notebook also includes **ROC AUC**, which measures ranking quality across thresholds. The LLM has no supplied probability scores, so neither score is reported for it. Its seven relevant misses comprise five Other and two Feedback queries; its 12 unnecessary referrals comprise eight Acknowledgment and four OSS queries. This shows why overall accuracy alone is insufficient. Small intent groups and the limited dataset make these estimates uncertain.

### Monthly cost at one million queries

- Input: 1,000,000 × 250 tokens × $2 / 1,000,000 = **$500**.
- Output: 1,000,000 × 5 tokens × $8 / 1,000,000 = **$40**.
- Sending every query to the LLM costs **$540/month in tokens**.
- Sending a fraction `r` costs **$540 × r**. For example, 20% would cost $108; 20% is an illustration, not a measured escalation rate.

The assignment assumes negligible classification-head compute. Embedding generation, infrastructure, retries and nurse workload are additional costs. A/B require local encoder capacity; C requires hosted encoder pricing. Encoder identities, throughput and end-to-end latency are unavailable, so equal total cost or speed across embedding alternatives cannot be assumed.

## 4. Recommend a production design

**Recommend evaluating a two-stage LR → LLM router in a monitored pilot.** LR is a simple first stage; its final selected embedding is B. The aim is to handle clear queries locally and reserve the LLM for uncertain cases, reducing LLM calls while protecting recall. These hybrid benefits remain hypotheses until measured.

1. LR produces a calibrated relevance probability.
2. Above an upper threshold, refer locally. Below a conservative lower threshold, classify as irrelevant. Send the middle uncertainty range to the LLM.
3. Use the LLM's routing label for escalated queries. Send failed, invalid or unresolved responses to human review rather than silently dropping the query.

Choose both thresholds using development data, prioritizing missed relevant and Medical queries while accounting for referrals and LLM calls. Do not simply choose a band around 0.50: models can be confidently wrong, including at low scores. Before deployment, evaluate the entire hybrid on held-out data and prospective human-labeled queries. Measure its recall, precision, subgroup misses, escalation rate and workload together. Comparing local models and the LLM separately does not establish hybrid quality.

Locally handled queries should avoid an LLM round trip; escalated queries incur both stages. This is an expected latency benefit, not a benchmark. Monitor actual latency, costs, failed calls, referral capacity, delayed-label errors and changing score distributions. Audit a sample of low-score rejections, and revalidate after new intents, language changes or encoder updates. **Distribution change** means future queries differ from the development data; it can invalidate both probabilities and thresholds.

Until the hybrid demonstrates an acceptable trade-off, the supplied LLM remains the strongest measured quality reference. The submitted local prediction file uses **RF recall-first**, selected by the notebook's rule of minimizing Medical misses first; it is not an LR–LLM hybrid prediction file. Choosing a model family from these results adds selection uncertainty and requires fresh validation.

**Reproducibility and evidence:** [SUBMISSION.ipynb](SUBMISSION.ipynb), the saved analysis in `src/analysis_outputs/`, model results in `src/experiments/`, and [test predictions](src/results/recommended_test_predictions.csv). Measured results, proposed design choices and cost assumptions are distinguished above. AI assistance was used for implementation and report preparation.
