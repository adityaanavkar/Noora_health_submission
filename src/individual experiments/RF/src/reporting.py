"""Strict artifact readers, charts, and report text for the Random Forest notebook.

The experiment worker owns model fitting and all files under ``outputs``.  This
module only reads those files and writes figures/report documents under the
Version 2 directory.  It deliberately does not infer a final recipe from a
pooled candidate ranking: the final recipe is read from ``summary.json``.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


INK = "#202020"
RED = "#9f1239"
TEAL = "#176b87"
ORANGE = "#c46a2d"
SLATE = "#475569"
PALE = "#f7f4f1"
POLICIES = ("default_0_5", "balanced_macro_f1", "recall_first")
POLICY_LABELS = {
    "default_0_5": "Default 0.50",
    "balanced_macro_f1": "Balanced macro-F1",
    "recall_first": "Recall-first",
}

FOREST_CANDIDATE_COLUMNS = (
    "selection_scope",
    "outer_fold",
    "embedding",
    "config_id",
    "forest_params",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "mean_inner_average_precision",
)
BASE_METRIC_COLUMNS = (
    "system", "policy", "evaluation", "n", "accuracy", "macro_f1",
    "precision", "recall", "specificity", "tp", "tn", "fp", "fn",
    "medical_fn", "medical_n", "medical_recall", "referral_fraction",
    "roc_auc", "average_precision", "score_available",
    "referrals", "note",
)
OUTER_COLUMNS = (
    "row_index", "query_id", "outer_fold", "group_id", "truth", "true_label",
    "intent", "probability_method", "p_raw", "p_chosen",
    "selected_embedding", "selected_config_id", "selected_forest_params",
    "threshold_default_0_5", "threshold_balanced_macro_f1",
    "threshold_recall_first", "label_default_0_5", "label_balanced_macro_f1",
    "label_recall_first",
)
FOLD_CHOICE_COLUMNS = (
    "outer_fold", "selected_embedding", "selected_config_id",
    "selected_forest_params", "probability_method", "threshold_default_0_5",
    "threshold_balanced_macro_f1", "threshold_recall_first",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return pd.read_csv(path)


def load_artifacts(version_dir: Path | str) -> dict[str, Any]:
    """Read and validate the complete Version 2 reporting contract."""
    version_dir = Path(version_dir).resolve()
    outputs = version_dir / "outputs"
    required = (
        "metrics.csv", "comparative_metrics.csv", "outer_predictions.csv",
        "candidate_results.csv", "inner_search_records.csv", "actual_grid.csv",
        "threshold_sweep.csv", "fold_choices.csv", "splits.csv",
        "summary.json", "configuration.json", "protocol.json",
        "artifact_schema.json", "environment.json", "test_predictions.csv",
    )
    for filename in required:
        if not (outputs / filename).exists():
            raise FileNotFoundError(f"Required RF artifact is missing: {outputs / filename}")

    data: dict[str, Any] = {
        "version_dir": version_dir,
        "outputs_dir": outputs,
        "metrics": _read_csv(outputs / "metrics.csv"),
        "comparative_metrics": _read_csv(outputs / "comparative_metrics.csv"),
        "outer_predictions": _read_csv(outputs / "outer_predictions.csv"),
        "candidate_results": _read_csv(outputs / "candidate_results.csv"),
        "actual_grid": _read_csv(outputs / "actual_grid.csv"),
        "threshold_sweep": _read_csv(outputs / "threshold_sweep.csv"),
        "fold_choices": _read_csv(outputs / "fold_choices.csv"),
        "splits": _read_csv(outputs / "splits.csv"),
        "test_predictions": _read_csv(outputs / "test_predictions.csv"),
        "summary": json.loads((outputs / "summary.json").read_text(encoding="utf-8")),
        "configuration": json.loads((outputs / "configuration.json").read_text(encoding="utf-8")),
    }
    validate_artifacts(data)
    return data


def validate_artifacts(data: dict[str, Any]) -> None:
    metrics = data["metrics"]
    outer = data["outer_predictions"]
    candidates = data["candidate_results"]
    folds = data["fold_choices"]
    threshold = data["threshold_sweep"]
    splits = data["splits"]
    test = data["test_predictions"]
    actual_grid = data["actual_grid"]

    _require_columns(metrics, BASE_METRIC_COLUMNS, "metrics.csv")
    _require_columns(outer, OUTER_COLUMNS, "outer_predictions.csv")
    _require_columns(candidates, FOREST_CANDIDATE_COLUMNS, "candidate_results.csv")
    _require_columns(folds, FOLD_CHOICE_COLUMNS, "fold_choices.csv")
    _require_columns(threshold, ("threshold", "precision", "relevant_recall", "tp", "fp", "fn"), "threshold_sweep.csv")
    _require_columns(splits, ("split_level", "outer_fold", "role", "row_index", "query_id", "group_id", "truth", "label", "intent"), "splits.csv")
    _require_columns(test, ("query_id", "predicted_label", "p_relevant"), "test_predictions.csv")
    _require_columns(actual_grid, ("config_id", "max_depth", "min_samples_leaf", "max_features", "class_weight", "forest_params"), "actual_grid.csv")

    if len(outer) == 0 or outer["row_index"].duplicated().any():
        raise ValueError("outer_predictions.csv must contain exactly one unique row per development row.")
    for column in ("p_raw", "p_chosen"):
        values = pd.to_numeric(outer[column], errors="raise")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"outer_predictions.csv column {column} must be finite probabilities in [0, 1].")
    if outer["outer_fold"].nunique() != 5:
        raise ValueError("The RF report expects five outer folds.")
    if candidates["embedding"].astype(str).str.lower().nunique() != 3:
        raise ValueError("candidate_results.csv must cover embeddings A, B, and C.")
    if len(actual_grid) != 24 or actual_grid["config_id"].duplicated().any():
        raise ValueError("actual_grid.csv must contain exactly 24 unique forest configurations.")
    expected_levels = {"max_depth": {"8", "None"}, "min_samples_leaf": {"1", "4", "12"}, "max_features": {"sqrt", "0.1"}, "class_weight": {"None", "balanced"}}
    for column, expected in expected_levels.items():
        observed = set("None" if pd.isna(value) else str(value).replace(".0", "") for value in actual_grid[column])
        if observed != expected:
            raise ValueError(f"actual_grid.csv does not contain the complete declared level set for {column}: {sorted(observed)}")
    if len(candidates[candidates["selection_scope"].astype(str).eq("outer_fold")]) < 5 * 72:
        raise ValueError("candidate_results.csv must contain 72 frozen candidates for each outer fold (24 forest settings across A/B/C).")

    aggregate = metrics[metrics["evaluation"].astype(str).eq("outer_oof_aggregate")]
    rf = aggregate[aggregate["system"].astype(str).str.contains("rf|forest", case=False, regex=True)]
    if set(rf["policy"].astype(str)) & set(POLICIES) != set(POLICIES):
        raise ValueError("metrics.csv must contain all three aggregate RF policy rows.")
    comparison_systems = aggregate[aggregate["system"].astype(str).str.contains("rf|forest|lr|logistic|llm", case=False, regex=True)]
    if not any(comparison_systems["system"].astype(str).str.contains("lr|logistic", case=False, regex=True)):
        raise ValueError("metrics.csv is missing the aggregate LR comparison rows.")
    if not any(comparison_systems["system"].astype(str).str.contains("llm", case=False, regex=True)):
        raise ValueError("metrics.csv is missing the aggregate LLM comparison row.")

    if test["query_id"].duplicated().any() or len(test) != 904:
        raise ValueError("test_predictions.csv must contain 904 unique blind-test IDs.")


def rf_aggregate_metrics(data: dict[str, Any]) -> pd.DataFrame:
    metrics = data["metrics"]
    mask = metrics["evaluation"].astype(str).eq("outer_oof_aggregate")
    mask &= metrics["system"].astype(str).str.contains("rf|forest", case=False, regex=True)
    rows = metrics.loc[mask].copy()
    rows["policy_label"] = rows["policy"].map(POLICY_LABELS).fillna(rows["policy"])
    rows["total_workload"] = pd.to_numeric(rows["tp"], errors="raise") + pd.to_numeric(rows["fp"], errors="raise")
    return rows.set_index("policy").loc[list(POLICIES)].reset_index()


def final_recipe(data: dict[str, Any]) -> dict[str, Any]:
    """Return only the recipe saved by the core's final all-training selection."""
    summary = data["summary"]
    if not isinstance(summary.get("selection"), dict):
        raise ValueError("summary.json must contain the canonical final all-training recipe under selection.")
    selection = summary["selection"]
    required = ("final_embedding", "final_config_id", "final_forest_params", "final_probability_method", "final_mean_inner_average_precision", "final_thresholds")
    missing = [key for key in required if key not in selection]
    if missing:
        raise ValueError(f"summary.json selection is missing canonical final recipe fields: {', '.join(missing)}")
    return selection


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_roc_pr(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    y = pd.to_numeric(outer["truth"], errors="raise").to_numpy()
    p = pd.to_numeric(outer["p_chosen"], errors="raise").to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(fpr, tpr, color=TEAL, lw=2.5, label=f"RF selected score (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#b8b8b8", ls="--", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="Relevant recall", title="Held-out ROC curve")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(recall, precision, color=RED, lw=2.5, label=f"RF selected score (AP {ap:.3f})")
    axes[1].set(xlabel="Relevant recall", ylabel="Precision", title="Held-out precision-recall curve")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
    fig.suptitle("Out-of-fold ranking quality", fontsize=14, color=INK)
    fig.tight_layout()
    return _save(fig, data["version_dir"] / "charts" / "roc_pr_oof.png")


def _normalise_candidate_params(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    parsed = candidates["forest_params"].map(lambda value: json.loads(value) if isinstance(value, str) else value)
    candidates["n_estimators"] = parsed.map(lambda value: value.get("n_estimators", 200))
    candidates["max_depth"] = candidates["max_depth"].where(candidates["max_depth"].notna(), parsed.map(lambda value: value.get("max_depth")))
    candidates["min_samples_leaf"] = candidates["min_samples_leaf"].where(candidates["min_samples_leaf"].notna(), parsed.map(lambda value: value.get("min_samples_leaf")))
    candidates["max_features"] = candidates["max_features"].where(candidates["max_features"].notna(), parsed.map(lambda value: value.get("max_features")))
    candidates["class_weight"] = candidates["class_weight"].where(candidates["class_weight"].notna(), parsed.map(lambda value: value.get("class_weight")))
    return candidates


def _forest_label(row: pd.Series) -> str:
    depth = "unlimited" if str(row["max_depth"]).lower() in {"none", "nan"} else str(row["max_depth"])
    weight = "balanced" if str(row["class_weight"]).lower() == "balanced" else "plain"
    return f"trees={row['n_estimators']}, depth={depth}, leaf={row['min_samples_leaf']}, features={row['max_features']}, weight={weight}"


def plot_tuning_rank(data: dict[str, Any]) -> Path:
    candidates = _normalise_candidate_params(data["candidate_results"])
    candidates = candidates[candidates["selection_scope"].astype(str).eq("final_all_training")]
    if candidates.empty:
        candidates = data["candidate_results"].copy()
        candidates = candidates[candidates["selection_scope"].astype(str).eq("outer_fold")]
    candidates["mean_inner_average_precision"] = pd.to_numeric(candidates["mean_inner_average_precision"], errors="raise")
    ranked = candidates.groupby(["embedding", "config_id", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"], dropna=False, as_index=False)["mean_inner_average_precision"].mean()
    ranked = ranked.sort_values("mean_inner_average_precision", ascending=False).head(12).copy()
    ranked["label"] = ranked.apply(_forest_label, axis=1)
    fig, axis = plt.subplots(figsize=(11, 6))
    colors = [TEAL if str(e).lower() == "b" else ORANGE if str(e).lower() == "a" else RED for e in ranked["embedding"]]
    axis.barh(ranked["label"].iloc[::-1], ranked["mean_inner_average_precision"].iloc[::-1], color=colors[::-1])
    axis.set(xlabel="Mean inner average precision", title="Frozen forest candidates ranked by inner average precision")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _save(fig, data["version_dir"] / "charts" / "tuning_rank.png")


def _selected_thresholds(data: dict[str, Any]) -> dict[str, float]:
    sweep = data["threshold_sweep"]
    result: dict[str, float] = {"default_0_5": 0.5}
    for policy in ("balanced_macro_f1", "recall_first"):
        flag = f"selected_{policy}"
        if flag in sweep.columns:
            rows = sweep[sweep[flag].astype(bool)]
            if len(rows) == 1:
                result[policy] = float(rows.iloc[0]["threshold"])
                continue
        summary = data["summary"]
        selection = summary.get("selection", {})
        thresholds = selection.get("final_thresholds", summary.get("thresholds", {}))
        if policy in thresholds:
            result[policy] = float(thresholds[policy])
        else:
            raise ValueError(f"Cannot find the selected {policy} threshold in threshold_sweep.csv or summary.json.")
    return result


def plot_threshold_development(data: dict[str, Any]) -> Path:
    sweep = data["threshold_sweep"].copy()
    numeric = ["threshold", "precision", "relevant_recall", "tp", "fp", "fn"]
    for column in numeric:
        sweep[column] = pd.to_numeric(sweep[column], errors="raise")
    if "medical_recall" in sweep:
        sweep["medical_recall"] = pd.to_numeric(sweep["medical_recall"], errors="raise")
    thresholds = _selected_thresholds(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(sweep["threshold"], sweep["precision"], color=TEAL, lw=2, label="Precision")
    axes[0].plot(sweep["threshold"], sweep["relevant_recall"], color=ORANGE, lw=2, label="Relevant recall")
    if "medical_recall" in sweep:
        axes[0].plot(sweep["threshold"], sweep["medical_recall"], color=RED, lw=2, label="Medical recall")
    sweep["total_referrals"] = sweep["tp"] + sweep["fp"]
    axes[1].plot(sweep["threshold"], sweep["total_referrals"], color=TEAL, lw=2, label="Total referrals: TP + FP")
    axes[1].plot(sweep["threshold"], sweep["fp"], color=ORANGE, lw=2, label="FP: unnecessary referral")
    axes[1].plot(sweep["threshold"], sweep["fn"], color=RED, lw=2, label="FN: relevant missed")
    if "medical_fn" in sweep:
        axes[1].plot(sweep["threshold"], sweep["medical_fn"], color=SLATE, lw=2, label="Medical missed")
    for axis in axes:
        for policy, threshold in thresholds.items():
            axis.axvline(threshold, color=RED if policy == "recall_first" else SLATE, ls="--", lw=1, alpha=0.7)
            axis.text(threshold, 1.01, POLICY_LABELS[policy], rotation=90, transform=axis.get_xaxis_transform(), fontsize=8, ha="right")
        axis.grid(alpha=0.2)
        axis.set_xlabel("Decision threshold")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Development rates")
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Development counts")
    fig.suptitle("Threshold choices on inner-development data", fontsize=14, color=INK)
    fig.tight_layout()
    return _save(fig, data["version_dir"] / "charts" / "development_threshold_tradeoff.png")


def plot_confusion_matrices(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    y = pd.to_numeric(outer["truth"], errors="raise")
    fig, axes = plt.subplots(1, len(POLICIES), figsize=(10.5, 3.6))
    axes = np.atleast_1d(axes)
    for axis, policy in zip(axes, POLICIES):
        pred = pd.to_numeric(outer[f"label_{policy}"], errors="raise")
        matrix = confusion_matrix(y, pred, labels=[0, 1])
        axis.imshow(matrix, cmap="Reds", vmin=0)
        for (r, c), value in np.ndenumerate(matrix):
            text_color = "white" if value > matrix.max() / 2 else INK
            axis.text(c, r, int(value), ha="center", va="center", color=text_color, fontsize=12)
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Withhold", "Refer"], yticklabels=["Irrelevant", "Relevant"], xlabel="Predicted", ylabel="Actual", title=POLICY_LABELS[policy])
    fig.suptitle("903 outer-held-out rows: saved fold-specific policy decisions", fontsize=14, color=INK)
    fig.tight_layout()
    return _save(fig, data["version_dir"] / "charts" / "confusion_matrices_three_policies.png")


def plot_calibration(data: dict[str, Any]) -> Path:
    outer = data["outer_predictions"]
    y = pd.to_numeric(outer["truth"], errors="raise").to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1.4]})
    for column, label, color in (("p_raw", "Raw forest score", ORANGE), ("p_chosen", "Selected score", TEAL)):
        p = pd.to_numeric(outer[column], errors="raise").to_numpy()
        observed, predicted = calibration_curve(y, p, n_bins=8, strategy="uniform")
        axes[0].plot(predicted, observed, marker="o", lw=2, color=color, label=label)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#aaa", lw=1, label="Perfect agreement")
    axes[0].set(xlabel="Mean predicted probability of relevant", ylabel="Observed share relevant", title="Held-out score reliability")
    p = pd.to_numeric(outer["p_chosen"], errors="raise")
    axes[1].hist(p, bins=np.linspace(0, 1, 9), color=TEAL, edgecolor="white")
    axes[1].set(xlabel="Selected probability of relevant", ylabel="Rows", title="Rows in each probability bin")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8) if axis is axes[0] else None
    fig.tight_layout()
    return _save(fig, data["version_dir"] / "charts" / "calibration_and_probability_bins.png")


def calibration_bin_table(data: dict[str, Any]) -> pd.DataFrame:
    outer = data["outer_predictions"]
    y = pd.to_numeric(outer["truth"], errors="raise")
    p = pd.to_numeric(outer["p_chosen"], errors="raise")
    edges = np.linspace(0, 1, 9)
    bins = pd.cut(p, bins=edges, include_lowest=True, right=True)
    table = pd.DataFrame({"probability_bin": bins, "predicted_probability": p, "observed_relevant": y}).groupby("probability_bin", observed=False).agg(
        rows=("observed_relevant", "size"),
        mean_predicted_probability=("predicted_probability", "mean"),
        observed_relevant_rate=("observed_relevant", "mean"),
    ).reset_index()
    table["probability_bin"] = table["probability_bin"].astype(str)
    return table


def comparison_table(data: dict[str, Any]) -> pd.DataFrame:
    metrics = data["metrics"]
    rows = metrics[metrics["evaluation"].astype(str).eq("outer_oof_aggregate")].copy()
    selected = []
    for _, row in rows.iterrows():
        system = str(row["system"]).lower()
        policy = str(row["policy"])
        if ("rf" in system or "forest" in system) and policy == "default_0_5":
            selected.append(row)
        elif ("lr" in system or "logistic" in system) and policy == "default_0_5":
            selected.append(row)
        elif "llm" in system:
            selected.append(row)
    if len(selected) < 3:
        raise ValueError("Aggregate RF/LR/LLM comparison requires RF, LR, and LLM rows in metrics.csv.")
    result = pd.DataFrame(selected).copy()
    result["display_label"] = result["system"].map(lambda value: "RF" if ("rf" in str(value).lower() or "forest" in str(value).lower()) else "LR" if ("lr" in str(value).lower() or "logistic" in str(value).lower()) else "LLM")
    return result.drop_duplicates("display_label").set_index("display_label").loc[["RF", "LR", "LLM"]].reset_index()


def plot_comparison(data: dict[str, Any]) -> Path:
    table = comparison_table(data)
    x = np.arange(len(table))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(x - width / 2, table["precision"], width, label="Precision", color=TEAL)
    axis.bar(x + width / 2, table["recall"], width, label="Relevant recall", color=ORANGE)
    axis.set(xticks=x, xticklabels=table["display_label"], ylabel="Rate", ylim=(0, 1.05), title="Aggregate outer comparison")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, data["version_dir"] / "charts" / "comparison_rf_lr_llm.png")


def fold_variation_table(data: dict[str, Any]) -> pd.DataFrame:
    metrics = data["metrics"]
    system = metrics["system"].astype(str)
    rows = metrics[metrics["evaluation"].astype(str).str.startswith("outer_fold_") & system.str.contains("rf|forest", case=False, regex=True)].copy()
    if rows.empty:
        raise ValueError("metrics.csv has no RF per-outer-fold rows for the fold-variation table.")
    return rows.groupby("policy", as_index=False).agg(
        recall_min=("recall", "min"), recall_max=("recall", "max"),
        macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"),
        fn_min=("fn", "min"), fn_max=("fn", "max"),
        medical_fn_min=("medical_fn", "min"), medical_fn_max=("medical_fn", "max"),
    )


def _pct(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def _num(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def _policy_rows_html(data: dict[str, Any]) -> str:
    rows = []
    for _, row in rf_aggregate_metrics(data).iterrows():
        rows.append(f"<tr><td>{html.escape(str(row['policy_label']))}</td><td>{int(row['fn'])}</td><td>{int(row['fp'])}</td><td>{int(row['total_workload'])}</td><td>{_pct(row['precision'])}</td><td>{_pct(row['recall'])}</td><td>{int(row['medical_fn'])}/{int(row['medical_n'])}</td><td>{_pct(row['medical_recall'])}</td><td>{_pct(row['macro_f1'])}</td></tr>")
    return "".join(rows)


def build_executive_summary_html(data: dict[str, Any]) -> str:
    rows = rf_aggregate_metrics(data).set_index("policy")
    default = rows.loc["default_0_5"]
    recall = rows.loc["recall_first"]
    recipe = final_recipe(data)
    workload_delta = int(recall["total_workload"] - default["total_workload"])
    miss_delta = int(default["fn"] - recall["fn"])
    medical_delta = int(default["medical_fn"] - recall["medical_fn"])
    auc = roc_auc_score(pd.to_numeric(data["outer_predictions"]["truth"]), pd.to_numeric(data["outer_predictions"]["p_chosen"]))
    ap = average_precision_score(pd.to_numeric(data["outer_predictions"]["truth"]), pd.to_numeric(data["outer_predictions"]["p_chosen"]))
    comparison = comparison_table(data)
    llm = comparison.set_index("display_label").loc["LLM"]
    lr = comparison.set_index("display_label").loc["LR"]
    rf = comparison.set_index("display_label").loc["RF"]
    recipe_keys = ("final_embedding", "final_config_id", "final_forest_params", "final_probability_method", "final_mean_inner_average_precision")
    recipe_text = ", ".join(f"{html.escape(str(k).replace('final_', '').replace('_', ' '))}={html.escape(str(recipe[k]))}" for k in recipe_keys if k in recipe)
    return f"""
<article class="exec-paper">
<div class="top-rule"></div>
<div class="document-kicker">Random Forest experiment note</div>
<h1>Intent classification router</h1>
<div class="author">Version 2 · measured development evidence</div>
<div class="abstract"><div class="abstract-label">Executive summary</div><p>The RF selected score has held-out ROC AUC <strong>{auc:.3f}</strong> and average precision <strong>{ap:.3f}</strong> across {len(data['outer_predictions']):,} labeled development rows. The operational choice is the threshold: recall-first changes relevant misses from <strong>{int(default['fn'])}</strong> to <strong>{int(recall['fn'])}</strong>, Medical misses from <strong>{int(default['medical_fn'])}/{int(default['medical_n'])}</strong> to <strong>{int(recall['medical_fn'])}/{int(recall['medical_n'])}</strong>, and total referrals from <strong>{int(default['total_workload'])}</strong> to <strong>{int(recall['total_workload'])}</strong>. These are observed development counts.</p></div>
<div class="metric-strip"><div><strong>{len(data['outer_predictions']):,}</strong><span>labeled rows</span></div><div><strong>{auc:.3f}</strong><span>outer ROC AUC</span></div><div><strong>{ap:.3f}</strong><span>outer average precision</span></div><div><strong>{data['outer_predictions']['outer_fold'].nunique()}</strong><span>outer folds</span></div></div>
<h2><span>I</span> What was tested</h2><p>A forest is a committee of decision trees. Each tree sees a slightly different view of the training examples; the committee combines their votes into a score. The search fixed 200 trees and evaluated the complete 24-setting forest grid across embedding spaces A, B, and C: 72 candidates per search. Selection used mean inner average precision inside five outer folds with three inner folds.</p><p>The final recipe below is read from the core's all-training selection summary. It is the best result within this bounded search, not a global maximum, and it is not inferred from a pooled average of outer-fold candidates.</p><div class="recipe"><strong>Final RF recipe</strong><br>{recipe_text}</div>
<h2><span>II</span> Threshold decision</h2><p>The RF policies use the saved fold-specific decisions. Default uses 0.50. Balanced macro-F1 chooses the inner-development threshold with the best macro-F1. Recall-first chooses the highest feasible threshold under the assumed 98% relevant-recall and 99% Medical-recall targets.</p><table><thead><tr><th>Policy</th><th>Relevant missed</th><th>Unnecessary referrals</th><th>Total referrals</th><th>Precision</th><th>Relevant recall</th><th>Medical missed</th><th>Medical recall</th><th>Macro-F1</th></tr></thead><tbody>{_policy_rows_html(data)}</tbody></table><div class="finding"><strong>Observed workload change</strong><br>Moving from default to recall-first changes relevant misses from {int(default['fn'])} to {int(recall['fn'])}, Medical misses from {int(default['medical_fn'])}/{int(default['medical_n'])} to {int(recall['medical_fn'])}/{int(recall['medical_n'])}, and total referrals from {int(default['total_workload'])} to {int(recall['total_workload'])}. The change is a measured trade-off; precision, recall, or workload may worsen.</div>
<h2><span>III</span> Comparison</h2><p>RF and LR are compared on the same aggregate outer development rows. The supplied LLM contributes labels only, so its ROC AUC and average precision are undefined and no LLM probability curve is drawn. RF is not called better or safer from one table alone.</p><table><thead><tr><th>System</th><th>Accuracy</th><th>Macro-F1</th><th>Precision</th><th>Relevant recall</th><th>Medical recall</th><th>AP</th><th>FN</th><th>FP</th></tr></thead><tbody>{''.join(f"<tr><td>{r['display_label']}</td><td>{_pct(r['accuracy'])}</td><td>{_pct(r['macro_f1'])}</td><td>{_pct(r['precision'])}</td><td>{_pct(r['recall'])}</td><td>{_pct(r['medical_recall'])}</td><td>{_num(r['average_precision']) if str(r['display_label']) != 'LLM' else '—'}</td><td>{int(r['fn'])}</td><td>{int(r['fp'])}</td></tr>" for _, r in comparison.iterrows())}</tbody></table><div class="conditional"><strong>Conditional recommendation</strong><br>If avoiding relevant or Medical misses is worth additional referral work, recall-first is the candidate to validate prospectively. If review capacity is tighter, default is the lower-workload choice. The observed uncertainty across outer folds and the unlabeled blind test mean neither policy is a guarantee.</div>
<h2><span>IV</span> Limitations</h2><p>All metrics are development estimates from grouped outer out-of-fold predictions. The test file has no labels, so it cannot support test accuracy, recall, calibration, or safety claims. Medical is a source-intent subgroup used for measurement, not an inference-time feature. Exact duplicate grouping does not test unobserved patient, time, language, or workflow shift. The threshold targets are assumptions, not guarantees.</p>
</article>
"""


def build_markdown_report(data: dict[str, Any]) -> str:
    rows = rf_aggregate_metrics(data).set_index("policy")
    default, recall = rows.loc["default_0_5"], rows.loc["recall_first"]
    comparison = comparison_table(data)
    outer = data["outer_predictions"]
    auc = roc_auc_score(pd.to_numeric(outer["truth"]), pd.to_numeric(outer["p_chosen"]))
    ap = average_precision_score(pd.to_numeric(outer["truth"]), pd.to_numeric(outer["p_chosen"]))
    recipe = final_recipe(data)
    lines = [
        "# Version 2 report: nested Random Forest evaluation", "",
        "## Executive summary", "",
        f"The selected RF score has held-out ROC AUC **{auc:.3f}** and average precision **{ap:.3f}** on **{len(outer):,}** labeled development rows across **{outer['outer_fold'].nunique()}** outer folds. The final recipe is read from the core all-training summary: embedding **{recipe['final_embedding']}**, config **{recipe['final_config_id']}**, probability method **{recipe['final_probability_method']}**, forest parameters **{recipe['final_forest_params']}**.", "",
        "Recall-first changes the default operating point from "
        f"**{int(default['fn'])}** to **{int(recall['fn'])}** relevant misses, "
        f"**{int(default['medical_fn'])}/{int(default['medical_n'])}** to **{int(recall['medical_fn'])}/{int(recall['medical_n'])}** Medical misses, and "
        f"**{int(default['total_workload'])}** to **{int(recall['total_workload'])}** total referrals. "
        "The direction and size of this workload trade-off are observations, not guarantees.", "",
        "## Policy table", "", "| Policy | Relevant missed | Unnecessary referrals | Total referrals | Precision | Relevant recall | Medical missed | Medical recall | Macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(f"| {row['policy_label']} | {int(row['fn'])} | {int(row['fp'])} | {int(row['total_workload'])} | {_pct(row['precision'])} | {_pct(row['recall'])} | {int(row['medical_fn'])}/{int(row['medical_n'])} | {_pct(row['medical_recall'])} | {_pct(row['macro_f1'])} |")
    lines += ["", "## RF versus LR and the supplied LLM", "", "The comparison below uses aggregate outer development rows. The LLM has labels only; ROC AUC and average precision are undefined for it.", "", "| System | Accuracy | Macro-F1 | Precision | Relevant recall | Medical recall | Average precision | FN | FP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in comparison.iterrows():
        lines.append(f"| {row['display_label']} | {_pct(row['accuracy'])} | {_pct(row['macro_f1'])} | {_pct(row['precision'])} | {_pct(row['recall'])} | {_pct(row['medical_recall'])} | {_num(row['average_precision']) if row['display_label'] != 'LLM' else '—'} | {int(row['fn'])} | {int(row['fp'])} |")
    lines += ["", "## Recommendation", "", "If avoiding relevant or Medical misses is worth additional referral work, recall-first is the candidate to validate prospectively. If review capacity is tighter, default is the lower-workload choice. This is conditional on a local capacity and error-cost decision; it is not a safety or clinical claim.", "", "## Limitations", "", "Metrics are development estimates from grouped outer out-of-fold predictions. The blind test has no labels. Medical is a source-intent subgroup for measurement, not an inference-time feature. Exact duplicate grouping does not test unobserved patient, time, language, or workflow shift. The 98% relevant-recall and 99% Medical-recall targets are assumptions for development analysis.", "", "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe \"version 2\\scripts\\execute_rf.py\"", "# Recompute the RF core cache only when intentionally needed", ".\\.venv\\Scripts\\python.exe \"version 2\\scripts\\execute_rf.py\" --force-recompute", "```"]
    return "\n".join(lines) + "\n"


def write_reports(data: dict[str, Any]) -> None:
    version_dir = data["version_dir"]
    (version_dir / "REPORT.md").write_text(build_markdown_report(data), encoding="utf-8")
    standalone = "<!doctype html><html><head><meta charset='utf-8'><title>Random Forest report</title><style>body{margin:0;background:#ece8e4}.exec-paper{max-width:980px;margin:28px auto;padding:58px 72px 70px;background:#fff;color:#202020;font:15px/1.68 Georgia,serif}.top-rule{height:5px;background:#9f1239;margin-bottom:42px}.document-kicker,.abstract-label{font:700 11px Arial,sans-serif;text-transform:uppercase;letter-spacing:1.3px;color:#9f1239}.exec-paper h1{font:400 36px Georgia,serif;color:#9f1239}.exec-paper h2{font:700 19px Arial,sans-serif;border-bottom:2px solid #ddd;padding-bottom:7px;margin-top:42px}.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #bbb;border-bottom:1px solid #bbb;margin:24px 0}.metric-strip div{padding:15px;border-right:1px solid #ddd}.metric-strip strong{display:block;font:700 25px Arial,sans-serif;color:#176b87}.metric-strip span{display:block;color:#64748b;font:12px Arial,sans-serif}.abstract{border-top:1px solid #bbb;border-bottom:1px solid #bbb;padding:18px 0;margin-bottom:28px}table{width:100%;border-collapse:collapse;font:12px Arial,sans-serif;margin:16px 0}th{background:#123447;color:#fff;padding:9px}td{padding:9px;border-bottom:1px solid #e5e7eb;text-align:right}th:first-child,td:first-child{text-align:left}.recipe,.finding,.conditional{padding:16px 20px;margin:20px 0;background:#f7f4f1;border-left:5px solid #9f1239}</style></head><body>${build_executive_summary_html(data)}</body></html>"
    (version_dir / "RANDOM_FOREST_REPORT.html").write_text(standalone, encoding="utf-8")


__all__ = [
    "load_artifacts", "validate_artifacts", "final_recipe", "plot_roc_pr",
    "plot_tuning_rank", "plot_threshold_development", "plot_confusion_matrices",
    "plot_calibration", "plot_comparison", "comparison_table",
    "calibration_bin_table", "fold_variation_table", "build_executive_summary_html", "build_markdown_report",
    "write_reports",
]
