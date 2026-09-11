"""Create the compact Version 2 evidence notebook."""
from pathlib import Path
import nbformat as nbf

def md(value):
    return nbf.v4.new_markdown_cell(value.strip())

def code(value):
    return nbf.v4.new_code_cell(value.strip())

cells = [
md("""
# Version 2: a careful Random Forest experiment

In this section, we will test a Random Forest router on the same grouped nested-validation design used for logistic regression. A forest is a committee of decision trees: each tree makes a sequence of simple splits, and the committee combines the trees into a score for whether a query is relevant and should reach the nursing team.

The complete bounded search has 24 forest settings across embedding spaces A, B, and C, or 72 candidates per search. The result is the best configuration within that bounded search, not a global maximum. The blind test has no labels, so it can receive predictions but cannot produce test quality metrics.
"""),
code("""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import Image, display, HTML

ROOT = Path.cwd()
VERSION_DIR = ROOT / "version 2"
sys.path.insert(0, str(VERSION_DIR / "src"))

from reporting import (
    load_artifacts, final_recipe, plot_tuning_rank, plot_roc_pr,
    plot_threshold_development, plot_confusion_matrices, plot_calibration,
    comparison_table, fold_variation_table, build_executive_summary_html,
    rf_aggregate_metrics, calibration_bin_table,
)
import json

data = load_artifacts(VERSION_DIR)
metrics = data["metrics"]
outer_predictions = data["outer_predictions"]
candidate_results = data["candidate_results"]
threshold_sweep = data["threshold_sweep"]
print("Validated Version 2 RF artifacts.")
print(f"Development rows: {len(outer_predictions):,}; outer folds: {outer_predictions['outer_fold'].nunique()}; test rows: {len(data['test_predictions']):,}.")
"""),
md("""
## 1. Read the saved evidence

Here, we will load the core tables and check the scope before interpreting anything. The core owns model fitting, split construction, calibration, threshold selection, and the final refit. This notebook reads its saved evidence and makes the results understandable.
"""),
code("""
label_counts = outer_predictions["true_label"].value_counts().rename_axis("label").reset_index(name="rows")
fold_counts = outer_predictions.groupby("outer_fold", as_index=False).size().rename(columns={"size": "held-out rows"})
print("Label counts:")
display(label_counts)
print("Outer-fold sizes:")
display(fold_counts)
outer_candidates = candidate_results[candidate_results["selection_scope"].eq("outer_fold")]
print("Candidate rows in saved outer searches:", len(outer_candidates))
print("Required count: 5 outer folds x 72 candidates = 360 rows, before final all-training rows.")
"""),
md("""
The labeled development set contains the examples used for grouped model selection and held-out evaluation. The outer fold is kept out while its model, probability method, and thresholds are chosen. Exact duplicate groups stay together. The blind test is structurally checked but has no labels.
"""),
code("""
recipe = final_recipe(data)
print("Final all-training recipe from summary.json:")
for key in ["final_embedding", "final_config_id", "final_forest_params", "final_probability_method", "final_mean_inner_average_precision"]:
    if key in recipe:
        print(f"  {key}: {recipe[key]}")
print("This recipe is read from the core final selection; it is not reconstructed from pooled outer candidate scores.")
"""),
md("""
## 2. Understand the forest controls

In this section, we will see the complete tuning grid and why each control matters. A tree asks questions about the embedding coordinates one at a time. `max_depth` is the maximum number of questions along one path: deeper paths can describe more detailed patterns. `min_samples_leaf` is the smallest group allowed at the end of a path: a smaller leaf can fit finer detail, while a larger leaf forces more smoothing. `max_features` is the fraction or rule for how many coordinates the tree may consider at each question; changing it makes the committee's trees less alike. `class_weight=balanced` gives more influence to the smaller routing class. The number of trees is fixed at 200. These settings control the kind of evidence the forest can fit; they do not make a clinical guarantee.
"""),
code("""
ranked = candidate_results[candidate_results["selection_scope"].eq("final_all_training")].copy()
if ranked.empty:
    ranked = candidate_results[candidate_results["selection_scope"].eq("outer_fold")].copy()
params = ranked["forest_params"].map(json.loads)
ranked["n_estimators"] = params.map(lambda value: value["n_estimators"])
ranked["max_depth"] = ranked["max_depth"].where(ranked["max_depth"].notna(), params.map(lambda value: value["max_depth"]))
ranked["min_samples_leaf"] = ranked["min_samples_leaf"].where(ranked["min_samples_leaf"].notna(), params.map(lambda value: value["min_samples_leaf"]))
ranked["max_features"] = ranked["max_features"].where(ranked["max_features"].notna(), params.map(lambda value: value["max_features"]))
ranked["class_weight"] = ranked["class_weight"].where(ranked["class_weight"].notna(), params.map(lambda value: value["class_weight"]))
ranked["mean_inner_average_precision"] = pd.to_numeric(ranked["mean_inner_average_precision"], errors="raise")
ranked_view = ranked.sort_values("mean_inner_average_precision", ascending=False).head(10).copy()
display(ranked_view[["embedding", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight", "mean_inner_average_precision"]])
print("The ranking is an inner-development tuning view. The final recipe above remains the authority for the final all-training fit.")
"""),
code("""
tuning_path = plot_tuning_rank(data)
display(Image(filename=str(tuning_path)))
print("The plot shows the top 12 rows from the complete 24-setting forest grid across A, B, and C; the saved table still contains the full candidate evidence.")
"""),
md("""
## 3. Measure ranking quality before choosing a threshold

Precision asks: of the rows we refer, how many are truly relevant? Recall asks: of all truly relevant rows, how many do we catch? A false positive is an unnecessary referral; a false negative is a missed relevant row. ROC plots false-alarm rate on the horizontal axis against catch rate, or relevant recall, on the vertical axis. Its AUC is 0.5 for random ranking and 1.0 for perfect ranking. Average precision summarizes the precision-recall curve as the cutoff moves; it is a ranking summary, not an accuracy score. These curves use saved selected probabilities from outer-held-out rows, and the supplied LLM has no probability curve.
"""),
code("""
roc_pr_path = plot_roc_pr(data)
display(Image(filename=str(roc_pr_path)))
from sklearn.metrics import roc_auc_score, average_precision_score
y_outer = pd.to_numeric(outer_predictions["truth"], errors="raise")
p_outer = pd.to_numeric(outer_predictions["p_chosen"], errors="raise")
print(f"Held-out RF ROC AUC: {roc_auc_score(y_outer, p_outer):.3f}; average precision: {average_precision_score(y_outer, p_outer):.3f}.")
"""),
md("""
## 4. Make the threshold trade-off visible

Here, we will inspect the inner-development sweep used to choose the three operating policies. Lowering the threshold usually sends more queries onward and can reduce misses; the observed direction and size must be read from the saved counts. The vertical markers belong to the development sweep only.
"""),
code("""
threshold_path = plot_threshold_development(data)
display(Image(filename=str(threshold_path)))
print("Default is 0.50. Balanced macro-F1 and recall-first use thresholds selected on inner-development data under the documented tie rules.")
"""),
md("""
## 5. Compare the three operating policies

The table below keeps rates and counts together because a threshold is an operating decision. Total referrals are TP + FP; unnecessary referrals are false positives. Recall-first is selected under assumed 98% relevant-recall and 99% Medical-recall targets when feasible. Those targets are development assumptions, not guarantees.
"""),
code("""
rf_rows = rf_aggregate_metrics(data)
policy_view = rf_rows[["policy_label", "precision", "recall", "fn", "fp", "total_workload", "medical_fn", "medical_n", "medical_recall", "macro_f1"]].copy()
policy_view.columns = ["Policy", "Precision", "Relevant recall", "Relevant missed", "Unnecessary referrals", "Total referrals", "Medical missed", "Medical total", "Medical recall", "Macro-F1"]
policy_view["Medical missed"] = policy_view["Medical missed"].astype(str) + "/" + policy_view["Medical total"].astype(str)
for column in ["Precision", "Relevant recall", "Medical recall", "Macro-F1"]:
    policy_view[column] = policy_view[column].map(lambda value: f"{value:.1%}")
display(policy_view[["Policy", "Precision", "Relevant recall", "Relevant missed", "Unnecessary referrals", "Total referrals", "Medical missed", "Medical recall", "Macro-F1"]])
default = rf_rows.set_index("policy").loc["default_0_5"]
recall_first = rf_rows.set_index("policy").loc["recall_first"]
print(f"Default -> recall-first: relevant misses {int(default['fn'])} -> {int(recall_first['fn'])}; Medical misses {int(default['medical_fn'])}/{int(default['medical_n'])} -> {int(recall_first['medical_fn'])}/{int(recall_first['medical_n'])}; total referrals {int(default['total_workload'])} -> {int(recall_first['total_workload'])}.")
"""),
md("""
The saved outer-fold decisions are used for this comparison. The change from default to recall-first can improve recall while worsening precision or workload; the actual sign is shown by the counts above. No policy is called a winner without considering the local cost of misses and review capacity.
"""),
md("""
## 6. Inspect held-out confusion matrices

Each matrix below aggregates all 903 outer-held-out rows for one policy. Every label came from the saved fold-specific decision made without that row. Fold-to-fold variation appears in a separate table, so the matrices show the requested policy comparison without pretending that one pooled row is a single fitted model.
"""),
code("""
confusion_path = plot_confusion_matrices(data)
display(Image(filename=str(confusion_path)))
print("These three matrices aggregate 903 saved outer-held-out decisions: default, balanced macro-F1, and recall-first.")
"""),
md("""
## 7. Check probability behavior

A predicted probability of 0.80 means that roughly eight of ten similar rows would be relevant if the probabilities were well calibrated. We compare raw and selected scores with a reliability plot, Brier scores, and counts in each selected-probability bin. Calibration is a diagnostic, not a safety certificate.
"""),
code("""
calibration_path = plot_calibration(data)
display(Image(filename=str(calibration_path)))
from sklearn.metrics import brier_score_loss
raw = pd.to_numeric(outer_predictions["p_raw"], errors="raise")
chosen = pd.to_numeric(outer_predictions["p_chosen"], errors="raise")
print(f"Outer Brier score, raw: {brier_score_loss(y_outer, raw):.3f}; selected: {brier_score_loss(y_outer, chosen):.3f}.")
print("Calibration bins (interval, count, predicted rate, observed relevant rate):")
display(calibration_bin_table(data))
print("The selected score is the probability method frozen by inner Brier selection.")
"""),
md("""
## 8. Compare RF, LR, and the supplied LLM

This is an aggregate comparison table only. RF and LR use the same outer development scope. The LLM has labels only, so its ROC AUC and average precision are undefined. Fold variation is shown separately so a single pooled row is not mistaken for stability.
"""),
code("""
comparison = comparison_table(data)
comparison_view = comparison[["display_label", "accuracy", "macro_f1", "precision", "recall", "medical_recall", "average_precision", "fn", "fp"]].copy()
comparison_view.columns = ["System", "Accuracy", "Macro-F1", "Precision", "Relevant recall", "Medical recall", "Average precision", "FN", "FP"]
for column in ["Accuracy", "Macro-F1", "Precision", "Relevant recall", "Medical recall"]:
    comparison_view[column] = comparison_view[column].map(lambda value: "—" if pd.isna(value) else f"{value:.1%}")
comparison_view["Average precision"] = comparison_view["Average precision"].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
display(comparison_view)
print("RF outer-fold variation:")
display(fold_variation_table(data))
"""),
md("""
The comparison describes measured development behavior. It does not establish that RF improves LR, and it does not turn the LLM label-only baseline into a score model. The next cell assembles the final white-paper-style summary from the same validated artifacts.
"""),
code("""
summary_html = build_executive_summary_html(data)
display(HTML(summary_html))
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
)
output = Path(__file__).resolve().parents[1] / "RANDOM_FOREST.ipynb"
nbf.write(notebook, output)
print(f"Wrote {output}")
