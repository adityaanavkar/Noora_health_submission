"""Create the canonical beginner-readable Version 3 notebook."""

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Version 3: a small neural network for intent routing

In this section, we will read the saved evidence for a small, regularized multi-layer perceptron (MLP). The MLP receives one of the supplied embedding vectors and estimates whether a query is relevant enough to refer to the nursing workflow.

This notebook is intentionally a reader for beginners. The model-fitting code stays in the core worker and the reusable reporting API. Set the single RUN_CORE switch in the first code cell to True when you want this notebook to train or reuse the Version 3 core. Leave it False when the core has already been run and you only want to render the evidence and reports.

The blind test has no labels. We can validate its shape, order, IDs, and probabilities, but we cannot calculate test quality.
"""
    ),
    code(
        """
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, display, Markdown

def discover_root(start):
    for candidate in [start, *start.parents]:
        if (candidate / "AGENTS.md").is_file() and (candidate / "intent-classification-assignment" / "intent-classification-assignment" / "manifest.json").is_file():
            return candidate
    return start

ROOT = discover_root(Path.cwd())
VERSION_DIR = ROOT / "version 3"
sys.path.insert(0, str(VERSION_DIR))

from src.reporting import (
    load_artifacts, final_recipe, plot_training_splits, plot_tuning_rank,
    plot_roc_pr, plot_threshold_development, plot_confusion_matrices,
    plot_calibration, calibration_bin_table, comparison_aggregate,
    plot_comparison, fold_variation_table, best_local_policy,
    recommended_predictions, write_reports,
)

# The one switch: True trains when outputs are missing and reuses them otherwise.
RUN_CORE = True

print("Repository root:", ROOT)
print("Canonical Version 3 directory:", VERSION_DIR)
print("RUN_CORE:", RUN_CORE)
"""
    ),
    code(
        """
if RUN_CORE:
    from src.experiment import run_experiment
    existing = load_artifacts(ROOT, strict=False)
    if existing["pending"]:
        print("Calling the core API to create the new V3 outputs.")
        run_experiment(VERSION_DIR, pilot=False)
    else:
        print("Reusing complete new V3 outputs; the core is not called.")
else:
    print("Core call skipped. Existing Version 3 outputs will be reused if present.")

data = load_artifacts(ROOT, strict=False)
READY = not data["pending"]
if READY:
    print("New V3 outputs are ready.")
else:
    print("New V3 is pending. Missing:", ", ".join(data["missing"]))
    print("The remaining evidence cells stay readable and do not invent results.")
"""
    ),
    markdown(
        """
## 1. Read the data and split evidence

The labeled development rows have a known answer: relevant or irrelevant. A grouped outer fold is a held-out slice used to estimate how the complete selection procedure behaves on unseen labeled rows. Inner folds sit inside the outer training slice and are used to choose the model, probability method, and thresholds.

Exact duplicate groups stay together. This prevents an exact duplicate from appearing on both sides of a split and making the estimate look better than it is.
"""
    ),
    code(
        """
if READY:
    outer = data["outer_predictions"]
    print("Development rows:", len(outer))
    print("Outer folds:", outer["outer_fold"].nunique())
    print("Labels:")
    display(outer["true_label"].value_counts().rename_axis("label").reset_index(name="rows"))
    print("Outer fold sizes:")
    display(outer.groupby("outer_fold").size().rename("held-out rows").reset_index())
    print("Fold choices:")
    display(data["fold_choices"].head())
else:
    print("Run the core first to see saved labels and fold choices.")
"""
    ),
    code(
        """
if READY:
    outer = data["outer_predictions"]
    test = data["test_predictions"]
    print("Final prediction artifact shape:", test.shape)
    print("Unique test IDs:", test["query_id"].nunique())
    print("Source-order IDs are unique:", test["query_id"].is_unique)
    print("Predicted labels:", sorted(test["predicted_label"].astype(str).unique()))
    display(test.head())
else:
    print("The final prediction artifact will appear after the core run.")
"""
    ),
    markdown(
        """
## 2. What an MLP learns

An MLP is a sequence of simple numeric transformations. The input layer receives the embedding coordinates. A hidden layer combines those coordinates, applies an activation function such as ReLU, and passes the result onward. A second hidden layer, when present, can combine the first layer's patterns into more flexible patterns. The output layer produces a score that can be read as a probability-like estimate for the relevant class.

Regularization is a restraint on flexibility. In this experiment, alpha penalizes overly large weights. Standardization puts input coordinates on a comparable scale, and it is fitted inside each training partition. A small dataset can be memorized by a large network, so the search is deliberately bounded. The saved configuration is the authority for the actual architecture, alpha, learning rate, batch size, iteration limit, tolerance, solver, and early-stopping setting; this notebook does not assume those values.

Training means fitting weights on the fit rows. Validation means using separate rows to choose settings or thresholds. The outer holdout is used only after those choices are frozen. Convergence diagnostics and warning counts are read from the saved configuration and summary when the core provides them.
"""
    ),
    code(
        """
if READY:
    recipe = final_recipe(data)
    settings = data["summary"].get("config", {}).get("settings", {})
    print("Actual final recipe from summary.json:")
    display(pd.Series(recipe))
    print("Actual core settings:")
    display(pd.Series(settings))
    print("Candidate rows by selection scope:")
    candidate = data["candidate_results"]
    group_columns = ["selection_scope"]
    if "outer_fold" in candidate.columns:
        group_columns.append("outer_fold")
    display(candidate.groupby(group_columns, dropna=False).size().rename("candidate rows").reset_index())
    diagnostic_columns = [column for column in candidate.columns if any(token in column.lower() for token in ["n_iter", "loss", "hit_max_iter", "convergence"])]
    if diagnostic_columns:
        print("Saved convergence diagnostics:")
        display(candidate[diagnostic_columns].head())
    display(Image(filename=str(plot_training_splits(data))))
    display(Image(filename=str(plot_tuning_rank(data))))
else:
    print("The actual MLP settings and tuning rank are available after the core run.")
"""
    ),
    markdown(
        """
## 3. Ranking quality: ROC and average precision

Precision asks: of the queries we refer, how many are truly relevant? Recall asks: of all truly relevant queries, how many did we catch? These answer different questions.

The ROC curve shows the catch rate against the false-alarm rate as the threshold moves. ROC AUC is a ranking summary: 0.5 is random ordering and 1.0 is perfect ordering. Average precision, or AP, summarizes the precision-recall curve. AP is also a ranking measure. Neither metric by itself says how many referrals the team will receive at a chosen threshold.
"""
    ),
    code(
        """
if READY:
    display(Image(filename=str(plot_roc_pr(data))))
    metrics = data["metrics"]
    row = metrics[(metrics["system"] == "tuned_mlp") & (metrics["policy"] == "default_0_5") & (metrics["evaluation"] == "outer_oof_aggregate")].iloc[0]
    print(f"Held-out development ROC AUC: {float(row['roc_auc']):.4f}")
    print(f"Held-out development average precision: {float(row['average_precision']):.4f}")
else:
    print("ROC and precision-recall charts require saved outer predictions.")
"""
    ),
    markdown(
        """
## 4. Thresholds, precision, recall, and workload

A threshold turns a score into an action. At 0.50, a score of at least one half is referred. The balanced macro-F1 policy chooses a threshold that balances the two class F1 scores on inner development data. The recall-first policy uses the declared illustrative targets of 98% relevant recall and 99% Medical recall when feasible, then minimizes false positives under its tie rules.

Lowering a threshold often catches more relevant queries, but it can also create more unnecessary referrals. We show both rates and counts because a team must understand the workload as well as the percentage.
"""
    ),
    code(
        """
if READY:
    display(Image(filename=str(plot_threshold_development(data))))
    policy = data["metrics"][(data["metrics"]["system"] == "tuned_mlp") & (data["metrics"]["evaluation"] == "outer_oof_aggregate")].copy()
    view = policy[policy["policy"].isin(["default_0_5", "balanced_macro_f1", "recall_first"])][["policy", "precision", "recall", "fn", "fp", "medical_fn", "medical_n", "medical_recall", "referrals"]].copy()
    view.columns = ["Policy", "Precision", "Relevant recall", "Relevant missed", "Unnecessary referrals", "Medical missed", "Medical total", "Medical recall", "Total referrals"]
    for column in ["Precision", "Relevant recall", "Medical recall"]:
        view[column] = view[column].map(lambda value: f"{value:.2%}")
    display(view)
else:
    print("Threshold trade-offs will be calculated from inner-development sweeps after the core run.")
"""
    ),
    markdown(
        """
## 5. Confusion matrices from saved fold-specific decisions

A false positive is an irrelevant query referred for review. A false negative is a relevant query missed by the router. Each matrix below aggregates saved decisions from the outer holdouts for one policy. The decisions are fold-specific: the threshold used for a row was selected without that row.
"""
    ),
    code(
        """
if READY:
    display(Image(filename=str(plot_confusion_matrices(data))))
    print("The three matrices aggregate the saved outer decisions. They are not the output of one model fitted on all 903 rows.")
else:
    print("Confusion matrices require saved outer-fold decisions.")
"""
    ),
    markdown(
        """
## 6. Calibration and probability bins

Calibration asks whether a score behaves like a probability. If many similar rows receive 0.80, roughly eight in ten should be relevant for a well-calibrated group. Calibration can be checked with a reliability curve, a Brier score, and the number of observations in each probability bin. A calibration plot is a diagnostic; it is not a safety certificate.
"""
    ),
    code(
        """
if READY:
    display(Image(filename=str(plot_calibration(data))))
    from sklearn.metrics import brier_score_loss
    outer = data["outer_predictions"]
    truth = pd.to_numeric(outer["truth"])
    chosen = pd.to_numeric(outer["p_chosen"])
    print(f"Held-out development Brier score: {brier_score_loss(truth, chosen):.4f}")
    display(calibration_bin_table(data))
else:
    print("Calibration requires saved selected probabilities.")
"""
    ),
    markdown(
        """
## 7. Compare model families and inspect fold variation

The compact table compares aggregate labeled-development results. The LLM has a frozen label but no probability, so ROC AUC and AP are undefined for it. Fold variation is shown separately because a pooled outer estimate does not prove stability.
"""
    ),
    code(
        """
if READY:
    comparison = comparison_aggregate(data)
    display(comparison[["display_label", "policy", "protocol", "accuracy", "macro_f1", "recall", "medical_fn", "medical_n", "fn", "fp", "total_workload"]])
    display(Image(filename=str(plot_comparison(data))))
    print("V3 fold variation:")
    display(fold_variation_table(data))
else:
    print("V1 Part2/V2 predecessor comparisons can render after their saved artifacts are present; new V3 rows are pending.")
"""
    ),
    markdown(
        """
## 8. Recommended local handoff and report generation

The assignment handoff uses a declared local rule: minimize Medical false negatives, then relevant false negatives, then false positives, then maximize macro-F1. This is a measured development selection rule, not an automatic production recommendation. Production may prefer the LLM if its advantage survives prospective validation and its cost, latency, privacy, and fallback requirements are acceptable.
"""
    ),
    code(
        """
data = load_artifacts(ROOT, strict=False)
if READY:
    choice = best_local_policy(data)
    print("Recommended local assignment policy:")
    display(choice[["display_label", "policy", "medical_fn", "fn", "fp", "macro_f1", "total_workload"]])
    print("Existing source-order test predictions will be exported after this cell.")
else:
    print("Recommended local predictions remain pending until new V3 outputs exist.")

# This cell is intentionally last: it regenerates substantive reports from saved artifacts.
write_reports(data)
print("Reports regenerated:", VERSION_DIR / "REPORT.md", VERSION_DIR / "COMPARISON_REPORT.md", VERSION_DIR / "ASSIGNMENT_RESPONSE.md")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
output = Path(__file__).resolve().parents[1] / "NEURAL_NETWORK.ipynb"
nbf.write(notebook, output)
print(f"Wrote {output}")
