"""Read-only Version 3 reporting, charts, and report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

POLICIES = ("default_0_5", "balanced_macro_f1", "recall_first")
POLICY_LABELS = {
    "default_0_5": "Default 0.50",
    "balanced_macro_f1": "Balanced macro-F1",
    "recall_first": "Recall-first",
}
COLORS = {"teal": "#176b87", "orange": "#c46a2d", "plum": "#8e3b68", "grid": "#dbe4e8"}


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return pd.read_csv(path)


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def load_artifacts(root: Path | str, *, strict: bool = True) -> dict[str, Any]:
    """Read version 3/outputs; strict=False supports awaiting-run drafts."""
    root = Path(root).resolve()
    version_dir = root / "version 3"
    outputs = version_dir / "outputs"
    names = (
        "summary.json", "metrics.csv", "comparative_metrics.csv",
        "outer_predictions.csv", "candidate_results.csv", "threshold_sweep.csv",
        "fold_choices.csv", "splits.csv", "test_predictions.csv",
    )
    missing = [name for name in names if not (outputs / name).is_file()]
    if strict and missing:
        raise FileNotFoundError("New V3 core outputs are not ready: " + ", ".join(missing))
    data: dict[str, Any] = {
        "root": root, "version_dir": version_dir, "outputs_dir": outputs,
        "charts_dir": version_dir / "charts", "pending": bool(missing), "missing": missing,
    }
    if missing:
        return data
    data.update({
        "summary": json.loads((outputs / "summary.json").read_text(encoding="utf-8")),
        "metrics": _csv(outputs / "metrics.csv"),
        "comparative_metrics": _csv(outputs / "comparative_metrics.csv"),
        "outer_predictions": _csv(outputs / "outer_predictions.csv"),
        "candidate_results": _csv(outputs / "candidate_results.csv"),
        "threshold_sweep": _csv(outputs / "threshold_sweep.csv"),
        "fold_choices": _csv(outputs / "fold_choices.csv"),
        "splits": _csv(outputs / "splits.csv"),
        "test_predictions": _csv(outputs / "test_predictions.csv"),
    })
    _require(data["metrics"], ["system", "policy", "evaluation", "accuracy", "macro_f1", "precision", "recall", "fn", "fp", "medical_fn", "medical_n", "referrals"], "metrics.csv")
    _require(data["outer_predictions"], ["outer_fold", "truth", "p_chosen", "label_default_0_5", "label_balanced_macro_f1", "label_recall_first"], "outer_predictions.csv")
    _require(data["candidate_results"], ["selection_scope", "embedding", "config_id", "mean_inner_average_precision"], "candidate_results.csv")
    _require(data["threshold_sweep"], ["threshold", "relevant_recall", "medical_recall", "referrals", "fp", "fn"], "threshold_sweep.csv")
    _require(data["test_predictions"], ["query_id", "predicted_label", "p_relevant"], "test_predictions.csv")
    return data


def final_recipe(data: dict[str, Any]) -> dict[str, Any]:
    return data["summary"]["selection"]


def recommended_predictions(data: dict[str, Any]) -> pd.DataFrame:
    """Return the core-exported predictions for its recorded recommendation."""
    predictions = data["test_predictions"].copy()
    selection = final_recipe(data)
    policy = str(selection["recommended_policy"])
    if "policy" in predictions.columns and not predictions["policy"].astype(str).eq(policy).all():
        raise ValueError("test_predictions.csv policy does not match summary.json recommendation")
    if "threshold" in predictions.columns:
        expected = float(selection["recommended_threshold"])
        if not np.allclose(pd.to_numeric(predictions["threshold"]), expected):
            raise ValueError("test_predictions.csv threshold does not match summary.json recommendation")
    return predictions


def _save(fig: plt.Figure, data: dict[str, Any], filename: str) -> Path:
    data["charts_dir"].mkdir(parents=True, exist_ok=True)
    path = data["charts_dir"] / filename
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_splits(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    labels = outer["true_label"].value_counts().reindex(["irrelevant", "relevant"]).fillna(0)
    folds = outer.groupby("outer_fold").size()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels.plot.bar(ax=axes[0], color=[COLORS["orange"], COLORS["teal"]])
    axes[0].set(title="Labeled development rows", ylabel="Rows")
    axes[0].tick_params(axis="x", rotation=0)
    folds.plot.bar(ax=axes[1], color=COLORS["plum"])
    axes[1].set(title="Outer held-out rows by fold", ylabel="Rows", xlabel="Outer fold")
    axes[1].tick_params(axis="x", rotation=0)
    fig.suptitle("Training labels and grouped outer splits")
    fig.tight_layout()
    return _save(fig, data, "training_splits.png")


def plot_tuning_rank(data: dict[str, Any]) -> Path:
    frame = data["candidate_results"]
    frame = frame[frame["selection_scope"].eq("final_all_training")].copy()
    frame["mean_inner_average_precision"] = pd.to_numeric(frame["mean_inner_average_precision"])
    frame = frame.sort_values("mean_inner_average_precision")
    frame["display"] = frame["embedding"].str.upper() + " / " + frame["config_id"]
    best = frame["mean_inner_average_precision"].max()
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.barh(frame["display"], frame["mean_inner_average_precision"], color=[COLORS["teal"] if value == best else "#9cc6d1" for value in frame["mean_inner_average_precision"]])
    axis.set(title="Final all-training MLP tuning rank", xlabel="Mean inner average precision", xlim=(0, 1))
    axis.grid(axis="x", color=COLORS["grid"])
    fig.tight_layout()
    return _save(fig, data, "tuning_rank.png")


def plot_roc_pr(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    truth = pd.to_numeric(outer["truth"]).to_numpy()
    probability = pd.to_numeric(outer["p_chosen"]).to_numpy()
    fpr, tpr, _ = roc_curve(truth, probability)
    precision, recall, _ = precision_recall_curve(truth, probability)
    auc = roc_auc_score(truth, probability)
    ap = average_precision_score(truth, probability)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color=COLORS["teal"], label=f"MLP (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af")
    axes[0].set(title="ROC: held-out development", xlabel="False-positive rate", ylabel="Relevant recall")
    axes[1].plot(recall, precision, color=COLORS["orange"], label=f"MLP (AP {ap:.3f})")
    axes[1].set(title="Precision-recall: held-out development", xlabel="Relevant recall", ylabel="Precision")
    for axis in axes:
        axis.grid(color=COLORS["grid"])
        axis.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, data, "roc_pr.png")


def selected_thresholds(data: dict[str, Any]) -> dict[str, float]:
    sweep = data["threshold_sweep"]
    result = {"default_0_5": 0.5}
    for policy in ("balanced_macro_f1", "recall_first"):
        marker = "selected_" + policy
        rows = sweep[sweep[marker].astype(str).str.lower().isin(("true", "1"))]
        result[policy] = float(rows.iloc[0]["threshold"]) if not rows.empty else float("nan")
    return result


def plot_threshold_development(data: dict[str, Any]) -> Path:
    sweep = data["threshold_sweep"].sort_values("threshold")
    x = pd.to_numeric(sweep["threshold"])
    selected = selected_thresholds(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(x, sweep["relevant_recall"], label="Relevant recall", color=COLORS["teal"])
    axes[0].plot(x, sweep["medical_recall"], label="Medical recall", color=COLORS["plum"])
    axes[0].set(title="Development threshold rates", xlabel="Threshold", ylabel="Recall", ylim=(0, 1.05))
    axes[1].plot(x, sweep["referrals"], label="Total referrals", color=COLORS["orange"])
    axes[1].plot(x, sweep["fp"], label="Unnecessary referrals", color="#64748b")
    axes[1].set(title="Development referral counts", xlabel="Threshold", ylabel="Rows")
    for axis in axes:
        for policy, color in (("default_0_5", "#374151"), ("balanced_macro_f1", COLORS["teal"]), ("recall_first", COLORS["orange"])):
            axis.axvline(selected[policy], ls="--", lw=1, color=color, label=POLICY_LABELS[policy])
        axis.grid(color=COLORS["grid"])
        axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, data, "development_threshold_rate_count.png")


def plot_confusion_matrices(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    truth = pd.to_numeric(outer["truth"]).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, policy in zip(axes, POLICIES):
        predicted = pd.to_numeric(outer["label_" + policy]).astype(int)
        matrix = confusion_matrix(truth, predicted, labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["irrelevant", "relevant"], yticklabels=["irrelevant", "relevant"], xlabel="Predicted", ylabel="Actual", title=POLICY_LABELS[policy])
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, int(value), ha="center", va="center", fontsize=13)
    fig.suptitle("Aggregate matrices from saved fold-specific decisions")
    fig.tight_layout()
    return _save(fig, data, "confusion_matrices_three_policies.png")


def calibration_bin_table(data: dict[str, Any]) -> pd.DataFrame:
    outer = data["outer_predictions"]
    probability = pd.to_numeric(outer["p_chosen"])
    truth = pd.to_numeric(outer["truth"])
    bins = np.linspace(0, 1, 9)
    return pd.DataFrame({"bin": pd.cut(probability, bins=bins, include_lowest=True), "predicted_probability": probability, "observed_relevant": truth}).groupby("bin", observed=False).agg(rows=("observed_relevant", "size"), mean_predicted_probability=("predicted_probability", "mean"), observed_share_relevant=("observed_relevant", "mean")).reset_index()


def plot_calibration(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    truth = pd.to_numeric(outer["truth"]).to_numpy()
    probability = pd.to_numeric(outer["p_chosen"]).to_numpy()
    observed, predicted = calibration_curve(truth, probability, n_bins=8, strategy="uniform")
    bins = calibration_bin_table(data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0, 1], [0, 1], "--", color="#9ca3af")
    axes[0].plot(predicted, observed, "o-", color=COLORS["teal"], label="MLP selected score")
    axes[0].set(title="Reliability on held-out development", xlabel="Mean predicted probability", ylabel="Observed relevant share")
    axes[1].bar(range(len(bins)), bins["rows"], color=COLORS["orange"])
    axes[1].set(title="Counts per probability bin", xlabel="Selected-probability bin", ylabel="Rows", xticks=range(len(bins)), xticklabels=[str(value) for value in bins["bin"]])
    axes[1].tick_params(axis="x", rotation=45)
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(color=COLORS["grid"])
    fig.tight_layout()
    return _save(fig, data, "calibration_with_bin_counts.png")


def _aggregate(path: Path, system: str, policy: str) -> dict[str, Any]:
    frame = _csv(path)
    rows = frame[(frame["system"] == system) & (frame["policy"] == policy) & (frame["evaluation"] == "outer_oof_aggregate")]
    if rows.empty:
        raise ValueError(f"Missing aggregate {system}/{policy} in {path}")
    return rows.iloc[0].to_dict()


def comparison_aggregate(data: dict[str, Any]) -> pd.DataFrame:
    root = data["root"]
    sources = []
    sources.extend(("V1 Part2 tuned LR", root / "version 1" / "part2" / "outputs" / "metrics.csv", "tuned_lr", policy, "V1 Part2 nested grouped") for policy in POLICIES)
    sources.append(("V1 Part2 fixed B LR", root / "version 1" / "part2" / "outputs" / "metrics.csv", "baseline_lr_b", "default_0_5", "V1 Part2 nested grouped"))
    sources.extend(("V2 tuned RF", root / "version 2" / "outputs" / "metrics.csv", "tuned_rf", policy, "V2 nested grouped") for policy in POLICIES)
    sources.append(("Supplied LLM", root / "version 2" / "outputs" / "metrics.csv", "llm_train", "frozen_label", "V2 shared-row label-only reference"))
    if not data.get("pending"):
        sources.extend(("V3 tuned MLP", data["outputs_dir"] / "metrics.csv", "tuned_mlp", policy, "V3 nested grouped") for policy in POLICIES)
    rows = []
    for label, path, system, policy, protocol in sources:
        row = _aggregate(path, system, policy)
        threshold = float("nan")
        summary_path = path.parent / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            threshold = summary.get("selection", {}).get("final_thresholds", {}).get(policy, float("nan"))
        row.update({"display_label": label, "protocol": protocol, "threshold": threshold, "total_workload": int(row.get("referrals", row["tp"] + row["fp"]))})
        rows.append(row)
    return pd.DataFrame(rows)


def original_v1_table(data: dict[str, Any]) -> pd.DataFrame:
    """Read the original 181-row holdout summary, separate from nested results."""
    path = data["root"] / "version 1" / "outputs" / "summary.json"
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["metrics"])


def old_nn_audit(data: dict[str, Any]) -> dict[str, Any]:
    """Read the preserved no-space attempt only for a clearly exploratory audit."""
    old = data["root"] / "version3" / "outputs"
    result: dict[str, Any] = {"available": (old / "summary.json").is_file()}
    if result["available"]:
        result["summary"] = json.loads((old / "summary.json").read_text(encoding="utf-8"))
        result["metrics"] = _csv(old / "metrics.csv")
        result["candidate_rows"] = len(_csv(old / "candidate_results.csv"))
        result["split_rows"] = len(_csv(old / "splits.csv"))
    return result


def plot_comparison(data: dict[str, Any]) -> Path:
    frame = comparison_aggregate(data)
    frame = frame[frame["policy"].isin(["default_0_5", "frozen_label"])]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(len(frame))
    width = 0.36
    axis.bar(positions - width / 2, frame["accuracy"], width, label="Accuracy", color=COLORS["teal"])
    axis.bar(positions + width / 2, frame["macro_f1"], width, label="Macro-F1", color=COLORS["orange"])
    axis.set(xticks=positions, xticklabels=frame["display_label"], ylabel="Score", title="Compact aggregate comparison on labeled development rows", ylim=(0, 1.05))
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", color=COLORS["grid"])
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, data, "comparison_aggregate.png")


def _selection_candidates(data: dict[str, Any]) -> pd.DataFrame:
    frame = comparison_aggregate(data)
    return frame[frame["display_label"].str.contains("tuned LR|tuned RF|tuned MLP", regex=True)].copy()


def best_local_policy(data: dict[str, Any]) -> pd.Series | None:
    """Apply the declared miss-priority rule to measured local policies."""
    candidates = _selection_candidates(data)
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["medical_fn", "fn", "fp", "macro_f1", "display_label", "policy"],
        ascending=[True, True, True, False, True, True],
    ).iloc[0]


def export_recommended_predictions(data: dict[str, Any]) -> Path | None:
    """Copy an existing local-model test prediction into the V3 handoff file.

    This is deliberately a post-run export: it never fits a model.  A pending
    V3 run leaves the file absent rather than inventing a prediction artifact.
    """
    choice = best_local_policy(data)
    if choice is None:
        return None
    label = str(choice["display_label"])
    policy = str(choice["policy"])
    if "tuned LR" in label:
        source = data["root"] / "version 1" / "part2" / "outputs" / "test_predictions.csv"
        system = "tuned_lr"
    elif "tuned RF" in label:
        source = data["root"] / "version 2" / "outputs" / "test_predictions.csv"
        system = "tuned_rf"
    else:
        source = data["outputs_dir"] / "test_predictions.csv"
        system = "tuned_mlp"
    selected = predictions = _csv(source)
    summary_path = source.parent / "summary.json"
    threshold = float("nan")
    if summary_path.is_file():
        source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        threshold = source_summary.get("selection", {}).get("final_thresholds", {}).get(policy, float("nan"))
    if pd.isna(threshold) and "policy" in selected.columns:
        matching = selected[selected["policy"].astype(str).eq(policy)]
        if not matching.empty and "threshold" in matching.columns:
            threshold = float(matching.iloc[0]["threshold"])
    probability_column = "p_chosen" if "p_chosen" in selected.columns else "p_relevant"
    selected["recommended_probability"] = pd.to_numeric(selected[probability_column])
    if not pd.isna(threshold):
        selected["predicted_label"] = np.where(selected["recommended_probability"] >= threshold, "relevant", "irrelevant")
        selected["threshold"] = threshold
    selected.insert(0, "recommended_system", system)
    selected.insert(1, "recommended_policy", policy)
    selected.insert(2, "selection_rule", "minimum Medical FN, then relevant FN, then FP, then maximum macro-F1")
    output = data["version_dir"] / "recommended_test_predictions.csv"
    selected.to_csv(output, index=False)
    metadata = {
        "source_file": str(source),
        "system": system,
        "policy": policy,
        "selection_rule": "minimum Medical FN, then relevant FN, then FP, then maximum macro-F1",
        "production_recommendation": "separate decision; LLM may be preferred if its measured advantage and cost/latency/privacy constraints are acceptable",
    }
    (data["version_dir"] / "recommended_test_predictions_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def fold_variation_table(data: dict[str, Any]) -> pd.DataFrame:
    frame = data["metrics"]
    frame = frame[frame["system"].isin(["tuned_mlp", "tuned_nn"]) & frame["evaluation"].astype(str).str.startswith("outer_fold_")]
    return frame.groupby("policy", as_index=False).agg(recall_min=("recall", "min"), recall_max=("recall", "max"), macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"), fn_min=("fn", "min"), fn_max=("fn", "max"), medical_fn_min=("medical_fn", "min"), medical_fn_max=("medical_fn", "max"))


def _pct(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2%}"


def build_report(data: dict[str, Any]) -> str:
    if data.get("pending"):
        comparison = comparison_aggregate(data)
        lines = [
            "# Version 3 report: awaiting the core run", "",
            "New V3 outputs are not present yet. This is an awaiting-run draft and contains no fabricated MLP measurements.", "",
            "The new run is planned for 903 labeled rows, five grouped outer folds, three grouped inner folds, and a declared 54-candidate MLP grid. Model, calibration, threshold, and recommended-policy values will be filled from version 3/outputs/*.csv and summary.json after the user runs the core.", "",
            "## Existing measured predecessor evidence", "",
            "| System | Policy | Accuracy | Relevant recall | Medical FN / total | FN | FP | Referrals |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in comparison.iterrows():
            lines.append(f"| {row['display_label']} | {row['policy']} | {_pct(row['accuracy'])} | {_pct(row['recall'])} | {int(row['medical_fn'])} / {int(row['medical_n'])} | {int(row['fn'])} | {int(row['fp'])} | {int(row['total_workload'])} |")
        lines += [
            "", "V1 Part 2 and V2 use the shared nested grouped development design. The original V1 single holdout and old no-space NN are kept separate in COMPARISON_REPORT.md because their fairness protocols differ.", "",
            "Run the core and notebook commands in README.md. The final report will add the measured MLP policy table, ROC/AP, calibration, fold variation, recommendation, and recommended local prediction export.",
        ]
        return "\n".join(lines) + "\n"
    summary = data["summary"]
    selection = summary["selection"]
    settings = summary.get("config", {}).get("settings", {})
    counts = summary["counts"]
    duplicate_groups = counts.get("duplicate_groups", counts.get("group_count", "recorded"))
    frame = data["metrics"]
    rows = [frame[frame["system"].isin(["tuned_mlp", "tuned_nn"]) & (frame["policy"] == policy) & (frame["evaluation"] == "outer_oof_aggregate")].iloc[0] for policy in POLICIES]
    thresholds = selection["final_thresholds"]
    truth = pd.to_numeric(data["outer_predictions"]["truth"])
    probability = pd.to_numeric(data["outer_predictions"]["p_chosen"])
    fold = fold_variation_table(data)
    test = data["test_predictions"]
    lines = [
        "# Version 3 report: a regularized MLP router", "",
        "This report is generated from the executed notebook and the new V3 core artifacts. Values below are measured on held-out labeled development rows unless stated otherwise.", "",
        "## Scope and procedure", "",
        f"The data has {counts['train_rows']:,} labeled rows, {duplicate_groups:,} exact-duplicate groups, five grouped outer folds, and three grouped inner folds. The bounded search tests {counts['candidate_models_per_search']} MLP candidates per inner search across embeddings A, B, and C. The outer holdout is excluded from model, calibration, and threshold selection.", "",
        f"The final all-training recipe selected embedding {selection['final_embedding'].upper()}, configuration {selection['final_config_id']}, hidden layers {selection['final_hidden_layer_sizes']}, alpha {selection['final_alpha']}, learning rate {selection.get('final_learning_rate_init', 'recorded in the saved recipe')}, and mean inner average precision {selection['final_mean_inner_average_precision']:.4f}. The saved configuration records the actual solver, activation, batch size, max iterations, tolerance, and early-stopping setting: {settings}. The model uses fit-partition standardization.", "",
        "## Held-out development results", "",
        "| Policy | Accuracy | Macro-F1 | Precision | Relevant recall | Relevant missed | Medical missed / total | Medical recall | Total referrals | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, row in zip(POLICIES, rows):
        lines.append(f"| {POLICY_LABELS[policy]} | {_pct(row['accuracy'])} | {_pct(row['macro_f1'])} | {_pct(row['precision'])} | {_pct(row['recall'])} | {int(row['fn'])} | {int(row['medical_fn'])} / {int(row['medical_n'])} | {_pct(row['medical_recall'])} | {int(row['referrals'])} | {float(thresholds[policy]):.6f} |")
    lines += [
        "",
        f"Selected-score ranking on held-out development rows: ROC AUC {roc_auc_score(truth, probability):.4f}, average precision {average_precision_score(truth, probability):.4f}, and Brier score {brier_score_loss(truth, probability):.4f}. These are ranking and probability diagnostics, not blind-test quality or safety guarantees.", "",
        "## Fold variation", "",
        "| Policy | Recall range | Macro-F1 range | Relevant FN range | Medical FN range |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in fold.iterrows():
        lines.append(f"| {POLICY_LABELS[row['policy']]} | {_pct(row['recall_min'])}–{_pct(row['recall_max'])} | {_pct(row['macro_f1_min'])}–{_pct(row['macro_f1_max'])} | {int(row['fn_min'])}–{int(row['fn_max'])} | {int(row['medical_fn_min'])}–{int(row['medical_fn_max'])} |")
    lines += [
        "", "## Blind prediction artifact", "",
        f"The final prediction file contains {len(test):,} rows and {test['query_id'].nunique():,} unique IDs in source order. Blind labels are unavailable, so no blind-test quality metric is calculated.", "",
        "## Recommendation and limitations", "",
        f"The recorded recommended local policy is {selection['recommended_policy']} at threshold {float(selection['recommended_threshold']):.6f}. It is a conditional operating choice when the illustrative 98% relevant and 99% Medical recall targets are prioritized. It changes referral workload and misses as shown above; it does not create a future guarantee. Keep the MLP only when its measured local-head trade-off fits the operating policy. Exact duplicate grouping does not test patient, conversation, time, language, or future distribution shift, and Medical is a measurement subgroup rather than an input feature.", "",
        "Astra handled planning, coordination, and review. Luna implemented the core-facing API, notebook, reports, charts, and checks. All reported values are read from saved artifacts.",
    ]
    return "\n".join(lines) + "\n"


def build_comparison_report(data: dict[str, Any]) -> str:
    comparison = comparison_aggregate(data)
    original = original_v1_table(data)
    old = old_nn_audit(data)
    lines = [
        "# Comparison report: V1 LR, V2 RF, V3 MLP, and LLM", "",
        "This report reads actual predecessor CSV/JSON artifacts. New V3 rows are added only after the new V3 core outputs exist; the old no-space version3 attempt is never substituted for the new run.", "",
        "| System | Policy | Protocol | Accuracy | Macro-F1 | Precision | Relevant recall | Medical recall | AP | FN | FP | Referrals | Threshold |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        ap = "undefined" if pd.isna(row.get("average_precision")) else f"{float(row['average_precision']):.4f}"
        threshold = "—" if pd.isna(row.get("threshold")) else f"{float(row['threshold']):.6f}"
        lines.append(f"| {row['display_label']} | {row['policy']} | {row['protocol']} | {_pct(row['accuracy'])} | {_pct(row['macro_f1'])} | {_pct(row['precision'])} | {_pct(row['recall'])} | {_pct(row['medical_recall'])} | {ap} | {int(row['fn'])} | {int(row['fp'])} | {int(row['total_workload'])} | {threshold} |")
    lines += [
        "", "V1 Part 2, V2, and the planned new V3 use the shared five-fold grouped outer / three-fold grouped inner design when their coordinated artifacts are complete. The original V1 single approximately 80/20 holdout is a different protocol and is kept in the separate table below. Different protocols are descriptive evidence, not a causal ranking.", "",
        "Type I error is an unnecessary referral: a human spends time reviewing an irrelevant query. Type II error is a missed relevant query: the nursing workflow may fail to receive a query that should have been routed. The second error can carry greater service or safety consequence, so the threshold rule must be agreed with operations rather than chosen from accuracy alone.", "",
        "The LLM has labels only and no probability score, so ROC AUC and average precision are undefined for it. Local heads and an LLM also differ in encoder, latency, privacy, and cost assumptions.", "",
        "## Original V1 single-holdout audit (181 labeled rows)", "",
        "| Method | Accuracy | Macro-F1 | Relevant recall | Relevant FN | Medical FN / total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in original.iterrows():
        lines.append(f"| {row['method']} | {_pct(row['accuracy'])} | {_pct(row['macro_f1'])} | {_pct(row['relevant_recall'])} | {int(row['relevant_fn'])} | {int(row['medical_fn'])} / {int(row['medical_n'])} |")
    lines += ["", "This 181-row table is not merged into the nested comparison because its split, selection, and evaluation scope differ."]
    if old["available"]:
        old_summary = old["summary"]
        old_selection = old_summary.get("selection", {})
        lines += [
            "", "## Preserved old version3 NN audit (exploratory only)", "",
            f"The old no-space attempt selected {old_selection.get('final_embedding', 'unknown')} and {old_selection.get('final_config_id', 'unknown')}. It has {old['candidate_rows']} candidate rows and {old['split_rows']} saved split rows. Its fairness is unverified against the current contract because it used a six-candidate search, did not persist the full inner split table, and lacked the required calibration/Brier and provenance evidence. Its measured rows are excluded from the main comparison.",
        ]
    if data.get("pending"):
        lines += ["", "## New V3 status", "", "New V3 is pending. Run the commands in README.md to append all three measured MLP policy rows and the recommended local prediction export. Existing V1 Part 2, V2, LLM, EDA, and original V1 audit evidence above remain usable."]
    else:
        local = comparison[comparison["display_label"].str.contains("LR|RF|MLP") & comparison["policy"].eq("default_0_5")]
        best = local.sort_values("macro_f1", ascending=False).iloc[0]
        declared = best_local_policy(data)
        lines += ["", f"Among default-policy rows, {best['display_label']} has the highest measured macro-F1 ({_pct(best['macro_f1'])}). The declared local handoff rule is minimum Medical FN, then relevant FN, then FP, then maximum macro-F1; it selects {declared['display_label']} / {POLICY_LABELS[declared['policy']]} for the assignment prediction artifact. This is a measured development choice, not a production guarantee."]
    return "\n".join(lines) + "\n"


def subgroup_table(data: dict[str, Any]) -> pd.DataFrame:
    """Compute subgroup counts from saved outer predictions."""
    root = data["root"]
    specs = [
        ("V1 Part2 tuned LR", root / "version 1" / "part2" / "outputs" / "outer_predictions.csv", "label_default_0_5"),
        ("V2 tuned RF", root / "version 2" / "outputs" / "outer_predictions.csv", "label_default_0_5"),
        ("Supplied LLM", root / "version 1" / "part2" / "outputs" / "outer_predictions.csv", "llm_predicted_label"),
    ]
    if not data.get("pending"):
        specs.append(("V3 tuned MLP", data["outputs_dir"] / "outer_predictions.csv", "label_default_0_5"))
    intents = ["Medical", "Other", "Feedback", "Acknowledgment"]
    rows = []
    for name, path, prediction_column in specs:
        frame = _csv(path)
        truth = pd.to_numeric(frame["truth"]).astype(int)
        values = frame[prediction_column]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            predicted = numeric.astype(int)
        else:
            predicted = values.astype(str).str.lower().eq("relevant").astype(int)
        for intent in intents:
            mask = frame["intent"].astype(str).eq(intent)
            rows.append({
                "System": name,
                "Intent": intent,
                "Rows": int(mask.sum()),
                "Relevant": int(((truth == 1) & mask).sum()),
                "Type II FN": int(((truth == 1) & (predicted == 0) & mask).sum()),
                "Type I FP": int(((truth == 0) & (predicted == 1) & mask).sum()),
            })
    return pd.DataFrame(rows)


def eda_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    root = data["root"]
    duplicates = _csv(root / "analysis_outputs" / "exact_duplicates.csv")
    errors = _csv(root / "analysis_outputs" / "llm_routing_errors.csv")
    unknown = _csv(root / "analysis_outputs" / "llm_unknown_intents_train.csv")
    diagnostics = _csv(root / "analysis_outputs" / "training_neighborhood_diagnostics.csv")
    low_agreement = diagnostics[pd.to_numeric(diagnostics["label_agreement_k15"]) <= 0.20]
    return {
        "duplicate_rows_across_spaces": len(duplicates),
        "duplicate_query_ids": int(duplicates["query_id"].nunique()),
        "llm_errors": len(errors),
        "llm_fn": int(errors["error_type"].astype(str).str.contains("missed relevant").sum()),
        "llm_fp": int(errors["error_type"].astype(str).str.contains("unnecessary referral").sum()),
        "unknown_intent_errors": len(unknown),
        "low_agreement_rows": int(low_agreement["query_id"].nunique()),
    }


def build_assignment_response(data: dict[str, Any]) -> str:
    root = data["root"]
    pca = _csv(root / "analysis_outputs" / "pca_summary.csv").set_index("space")
    silhouette = _csv(root / "analysis_outputs" / "silhouette_scores.csv").set_index("space")
    neighbors = _csv(root / "analysis_outputs" / "neighborhood_agreement.csv")
    neighbors = neighbors[neighbors["k"] == 5].set_index("space")
    costs = _csv(root / "analysis_outputs" / "llm_cost_scenarios.csv")
    million = costs[costs["monthly_LLM_queries"] == 1000000].iloc[0]
    status = "pending; no new V3 result is inserted" if data.get("pending") else "complete; see the measured V3 report"
    quality = comparison_aggregate(data)
    subgroup = subgroup_table(data)
    eda = eda_snapshot(data)
    train_outer = _csv(root / "version 1" / "part2" / "outputs" / "outer_predictions.csv")
    label_counts = train_outer["true_label"].value_counts().to_dict()
    llm = quality[quality["display_label"].eq("Supplied LLM")].iloc[0]
    local = quality[quality["display_label"].str.contains("tuned LR|tuned RF|tuned MLP")].copy()
    local_choice = best_local_policy(data)
    local_choice_text = (
        f"The measured local handoff is {local_choice['display_label']} with {POLICY_LABELS[local_choice['policy']]} under the declared rule (Medical FN {int(local_choice['medical_fn'])}, relevant FN {int(local_choice['fn'])}, FP {int(local_choice['fp'])}, macro-F1 {_pct(local_choice['macro_f1'])}); its existing test probabilities are exported to recommended_test_predictions.csv."
        if local_choice is not None and not data.get("pending")
        else "The measured local handoff and recommended_test_predictions.csv remain pending until new V3 outputs exist."
    )
    production = (
        f"The supplied LLM currently has the highest measured aggregate accuracy ({_pct(llm['accuracy'])}) among the available labeled-development references. If it still dominates after the new V3 run, recommend the LLM behind a human fallback and a monitored confidence or escalation path; if a local head becomes preferable on the agreed miss rule or constraints, use that local artifact instead. LLM latency, encoder cost, privacy fit, and drift are unknown and require measurement."
        if data.get("pending")
        else f"The supplied LLM has the measured aggregate accuracy {_pct(llm['accuracy'])}; compare it with the measured local choice in COMPARISON_REPORT.md. Recommend the LLM behind a human fallback if its advantage survives prospective validation and its latency, privacy, encoder cost, and drift controls are acceptable. Otherwise deploy the exported local artifact under the declared miss-priority rule."
    )
    quality_lines = ["| System | Policy | Accuracy | Recall | Medical FN / total | FN | FP | Referrals |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in quality.iterrows():
        quality_lines.append(f"| {row['display_label']} | {row['policy']} | {_pct(row['accuracy'])} | {_pct(row['recall'])} | {int(row['medical_fn'])} / {int(row['medical_n'])} | {int(row['fn'])} | {int(row['fp'])} | {int(row['total_workload'])} |")
    subgroup_lines = ["| System | Intent | Rows | Relevant | Type II FN | Type I FP |", "|---|---|---:|---:|---:|---:|"]
    for _, row in subgroup.iterrows():
        subgroup_lines.append(f"| {row['System']} | {row['Intent']} | {row['Rows']} | {row['Relevant']} | {row['Type II FN']} | {row['Type I FP']} |")
    return f"""# Assignment response

## 1. Embedding structure

The first two PCA components explain {pca.loc['A', 'variance_first_2']:.2%} of A, {pca.loc['B', 'variance_first_2']:.2%} of B, and {pca.loc['C', 'variance_first_2']:.2%} of C. Reaching 95% variance requires {int(pca.loc['A', 'components_for_95pct'])}, {int(pca.loc['B', 'components_for_95pct'])}, and {int(pca.loc['C', 'components_for_95pct'])} components. Binary silhouette is {silhouette.loc['A', 'binary_silhouette']:.4f}, {silhouette.loc['B', 'binary_silhouette']:.4f}, and {silhouette.loc['C', 'binary_silhouette']:.4f} for A, B, and C. At k=5, neighborhood label agreement is {neighbors.loc['A', 'agreement']:.2%}, {neighbors.loc['B', 'agreement']:.2%}, and {neighbors.loc['C', 'agreement']:.2%}. These checks show local label structure but weak global separation.

PCA variance is not predictive signal. Silhouette depends on the distance and labels, and neighborhood agreement can reflect duplicates or local concentration. Grouped nested evaluation is the model-selection evidence. New V3 status: {status}.

The label count is {int(label_counts.get('relevant', 0))} relevant and {int(label_counts.get('irrelevant', 0))} irrelevant rows. The EDA found {eda['duplicate_query_ids']} duplicated query IDs represented in {eda['duplicate_rows_across_spaces']} embedding-space rows, {eda['llm_errors']} LLM routing errors ({eda['llm_fn']} missed-relevant and {eda['llm_fp']} unnecessary-referral errors), and {eda['unknown_intent_errors']} errors involving intents outside the frozen routing vocabulary. A simple diagnostic flag of k=15 neighborhood label agreement at or below 0.20 identifies {eda['low_agreement_rows']} query IDs; this is an investigation queue, not a ground-truth outlier label.

The 2D PCA and t-SNE figures are useful for orientation, but a 2D projection can hide separation, distort distances, and change apparent clusters. Use the saved high-dimensional geometry and grouped held-out results for decisions.

### Measured quality snapshot

{chr(10).join(quality_lines)}

For an individual subgroup, the saved outer predictions show:

{chr(10).join(subgroup_lines)}

Type I error means an irrelevant query is referred unnecessarily, consuming human capacity and possibly delaying higher-priority work. Type II error means a relevant query is missed, so the intended nursing workflow may never see it. Type II consequences can be more serious, so the threshold and fallback policy must be agreed with operations.

## 2. Pipeline

Validate IDs, labels, shapes, finiteness, and source order; align A/B/C embeddings; build exact duplicate groups from the union of embedding rows; and keep intent for subgroup measurement only. Use five stratified grouped outer folds and three grouped inner folds. Select the MLP by inner average precision, fit scaling inside each fit partition, compare raw and sigmoid-selected probabilities by inner Brier score, and choose default, balanced macro-F1, and recall-first thresholds inside development data. Evaluate frozen outer decisions, then refit the frozen recipe on all labeled rows and emit source-order blind predictions.

## 3. Cost and recommendation

The supplied scenario is 500 USD base plus 40 USD usage, or **540 USD/month**, for 1,000,000 monthly LLM-routed queries at its stated token assumptions. This is the requested hybrid 540 USD/month scenario and excludes embedding generation. Encoder cost and latency are unknown because embeddings are precomputed and no encoder benchmark was supplied.

{production}

The declared local handoff rule is minimum Medical FN, then relevant FN, then FP, then maximum macro-F1. It is applied only to measured local rows. {local_choice_text} New V3 is {status}.

## 4. Failure modes, monitoring, and deliverables

Monitor embedding validity and norms, score and referral-rate drift, class prevalence, duplicate rates, per-intent and Medical outcomes, false-negative samples, queue capacity, latency, and model/version fingerprints. Watch for new intents, language or channel shift, encoder changes, mislabeled queries, threshold drift, and overloaded human review. Recompute quality only when delayed human labels arrive.

Deliverables are the beginner MLP notebook and exported HTML, split/tuning/ROC-PR/threshold/confusion/calibration/comparison charts, REPORT.md, COMPARISON_REPORT.md, this response, src/reporting.py, execution scripts, and the core-produced final recommended-classifier prediction file. Blind labels are absent, so no blind-test quality is claimed.
"""


def write_reports(data: dict[str, Any]) -> None:
    version_dir = data["version_dir"]
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "REPORT.md").write_text(build_report(data), encoding="utf-8")
    (version_dir / "COMPARISON_REPORT.md").write_text(build_comparison_report(data), encoding="utf-8")
    (version_dir / "ASSIGNMENT_RESPONSE.md").write_text(build_assignment_response(data), encoding="utf-8")
    if not data.get("pending"):
        export_recommended_predictions(data)
