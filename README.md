# Intent Classification Submission

[SUBMISSION.ipynb](SUBMISSION.ipynb) is the main notebook. It combines the data analysis, model comparisons, and production recommendation in one place, with explanations and charts.

## What the notebook covers

- The supplied queries, labels, and three embedding spaces.
- Logistic regression (LR), random forest (RF), and a small neural network (MLP).
- Validation, precision, recall, ROC AUC, calibration, and threshold trade-offs.
- Missed relevant queries, nurse referral workload, and the supplied LLM baseline.
- Production costs, limitations, and export of recommended local-classifier predictions.

The data contains 903 labeled training queries and 904 blind-test queries. Test ground truth is not supplied, so reported quality comes from held-out development predictions—not test accuracy.

## Setup and run

Use a Python environment with the packages listed in `src/requirements.txt`. From this submission folder, install them with:

```powershell
python -m pip install -r src/requirements.txt
```

Open `SUBMISSION.ipynb` in VS Code or Jupyter, select that Python environment as the notebook kernel, and run the cells from top to bottom.

The notebook defaults to:

```python
RETRAIN_ALL = False
```

This reads the included experiment results and creates the analysis tables and charts without retraining. Set it to `True` only to rerun all three training experiments. Retraining takes longer, uses CPU, and replaces the experiment outputs inside `src/experiments/`.

## Supporting files and outputs

All files used by the combined notebook are inside `src/`:

- `submission_tools.py`: data loading, plotting, and optional training support.
- `intent-classification-assignment/`: supplied data artifacts.
- `analysis_outputs/`: saved quantitative data-analysis results.
- `experiments/`: training implementations, saved models, and experiment results.
- `requirements.txt`: Python dependencies.

Running the final notebook cell writes:

```text
src/results/recommended_test_predictions.csv
src/results/recommendation.json
```

The CSV contains query IDs, probabilities, predicted labels, and the selected model, policy, and threshold. The local classifier is selected by prioritizing fewer Medical misses, then fewer relevant misses, then fewer false positives, and finally higher macro-F1. These priorities are stated assumptions, not clinical guarantees. The broader production recommendation may favor the LLM subject to operational constraints.

The separate `LR`, `RF`, and `MLP` folders contain earlier model-specific materials; the combined notebook does not depend on them.

The notebook distinguishes saved measurements from assumptions and unknown production costs or latency. It was assembled from the existing experiments and has not been executed end to end in this combined form. AI assistance was used for implementation, explanations, and organization.
