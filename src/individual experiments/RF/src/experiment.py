"""Nested grouped Random Forest evaluation for Version 2.

The core owns the complete reproducible experiment.  It shares the Version 1
training rows, duplicate groups and fold assignments, imports the saved LR
predictions for a fair comparison, and writes every generated artifact below
the supplied Version 2 directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


EMBEDDINGS = ("a", "b", "c")
EXPECTED_DIMS = {"a": 384, "b": 768, "c": 1536}
OUTER_SPLITS = 5
INNER_SPLITS = 3
OUTER_SEED = 42
INNER_SEED = 43
N_ESTIMATORS = 200
N_JOBS = 2
RELEVANT_RECALL_TARGET = 0.98
MEDICAL_RECALL_TARGET = 0.99
IMPLEMENTATION_VERSION = "rf-part2-nested-v2"


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if np.isscalar(value):
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(paths: Iterable[Path], *, content: bool = True) -> str:
    entries: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        entry: dict[str, Any] = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if content:
            entry["sha256"] = _sha256_file(path)
        entries.append(entry)
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _data_dir(version_dir: Path) -> Path:
    for parent in (version_dir.resolve(), *version_dir.resolve().parents):
        candidate = parent / "intent-classification-assignment" / "intent-classification-assignment"
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate intent-classification-assignment data")


def _lr_dir(version_dir: Path) -> Path:
    candidate = version_dir.parent / "version 1" / "part2" / "outputs"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Existing LR outputs are missing: {candidate}")
    return candidate


def _source_paths(data_dir: Path) -> list[Path]:
    return [
        data_dir / "train_queries.csv",
        *(data_dir / f"train_embeddings_{name}.npy" for name in EMBEDDINGS),
        data_dir / "llm_baseline_train_predictions.json",
        data_dir / "label_mapping.json",
        data_dir / "manifest.json",
    ]


def _test_paths(data_dir: Path) -> list[Path]:
    return [data_dir / "test_queries.csv", *(data_dir / f"test_embeddings_{name}.npy" for name in EMBEDDINGS)]


def _lr_paths(lr_dir: Path) -> list[Path]:
    return [lr_dir / name for name in ("outer_predictions.csv", "metrics.csv", "splits.csv", "configuration.json", "protocol.json")]


def _protocol(grid: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "model_family": "random_forest",
        "embeddings": {name: {"dimensions": EXPECTED_DIMS[name], "inputs": "raw normalized vectors"} for name in EMBEDDINGS},
        "duplicate_groups": "connected components of exact row matches in the union of training embeddings A/B/C",
        "outer": {"splitter": "StratifiedGroupKFold", "n_splits": OUTER_SPLITS, "shuffle": True, "random_state": OUTER_SEED, "stratification": "binary label", "shared_with": "version 1 part2"},
        "inner": {"splitter": "StratifiedGroupKFold", "n_splits": INNER_SPLITS, "shuffle": True, "random_state": INNER_SEED, "scope": "each outer training partition"},
        "search": {
            "n_estimators": N_ESTIMATORS,
            "random_state": OUTER_SEED,
            "n_jobs": N_JOBS,
            "universe": {"max_depth": [8, None], "min_samples_leaf": [1, 4, 12], "max_features": ["sqrt", 0.1], "class_weight": [None, "balanced"]},
            "frozen_configuration_count": len(grid),
            "configurations_per_search": len(grid) * len(EMBEDDINGS),
            "grid_selection": "exhaustive universe; frozen before outer evaluation",
            "selection_score": "mean inner average_precision",
            "tie_break": ["larger min_samples_leaf", "max_depth=8 before None", "max_features=sqrt before 0.1", "class_weight None before balanced", "embedding alphabetical"],
            "grid": grid,
        },
        "calibration": {"methods": ["raw", "sigmoid"], "selection_score": "lower pooled inner-heldout Brier score; raw on exact tie", "ensemble": False, "exclusion": "each inner-heldout row is excluded from both base-model fitting and calibration fitting"},
        "thresholds": {"default": 0.5, "balanced_macro_f1": "maximum macro-F1", "recall_first": {"minimum_relevant_recall": RELEVANT_RECALL_TARGET, "minimum_medical_recall": MEDICAL_RECALL_TARGET, "objective": "minimum false positives among feasible thresholds", "assumption": True}, "candidate_values": "0, 1, 0.5, and unique chosen inner-heldout probabilities"},
        "baselines": {"imported_logistic_regression": "saved Version 1 outer predictions and metrics, recomputed here", "llm": "frozen supplied training predictions joined by query_id", "always_relevant": "constant relevant label; no score"},
        "blind_test": "loaded only after final all-training inner development selection and threshold freeze; no test LLM file is read",
        "assumptions": "98% relevant recall and 99% Medical recall are illustrative development targets, not assignment requirements or safety guarantees",
    }


def _load_training(data_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    train = pd.read_csv(data_dir / "train_queries.csv", dtype="string")
    if list(train.columns) != ["query_id", "intent", "label"] or len(train) != 903 or train["query_id"].nunique() != 903:
        raise ValueError("Training query schema/count is not the declared 903-row source")
    if train["label"].isna().any() or set(train["label"].dropna()) != {"relevant", "irrelevant"}:
        raise ValueError("Training labels must be exactly relevant and irrelevant")
    embeddings: dict[str, np.ndarray] = {}
    for name in EMBEDDINGS:
        matrix = np.load(data_dir / f"train_embeddings_{name}.npy", allow_pickle=False, mmap_mode="r")
        if matrix.shape != (903, EXPECTED_DIMS[name]) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid training embedding {name} shape or finiteness")
        if not np.allclose(np.linalg.norm(np.asarray(matrix), axis=1), 1.0, atol=2e-3, rtol=2e-3):
            raise ValueError(f"Training embedding {name} is not normalized")
        embeddings[name] = matrix
    with (data_dir / "label_mapping.json").open(encoding="utf-8") as stream:
        policy = json.load(stream)
    with (data_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (data_dir / "llm_baseline_train_predictions.json").open(encoding="utf-8") as stream:
        llm = pd.DataFrame(json.load(stream))
    if set(llm.columns) != {"query_id", "predicted_intent", "predicted_label"} or len(llm) != 903 or llm["query_id"].nunique() != 903 or set(llm["query_id"]) != set(train["query_id"]):
        raise ValueError("Frozen training LLM baseline does not match training IDs/schema")
    llm_map = dict(zip(llm["query_id"].astype(str), llm["predicted_label"].astype(str)))
    if set(llm_map.values()) - {"relevant", "irrelevant"}:
        raise ValueError("Frozen LLM labels contain an invalid routing label")
    relevant = set(policy.get("relevant", []))
    irrelevant = set(policy.get("irrelevant", []))
    if not set(train["intent"]) <= relevant | irrelevant:
        raise ValueError("Training intents are not covered by label_mapping.json")
    expected = train["intent"].map(lambda value: "relevant" if value in relevant else "irrelevant")
    if not np.array_equal(expected.astype(str).to_numpy(), train["label"].astype(str).to_numpy()):
        raise ValueError("Intent-to-routing-label policy does not match labels")
    return train, embeddings, llm_map, {"policy": policy, "manifest": manifest}


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            nxt = int(self.parent[value])
            self.parent[value] = root
            value = nxt
        return root

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def _duplicate_groups(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    uf = _UnionFind(903)
    for name in EMBEDDINGS:
        matrix = np.asarray(embeddings[name])
        seen: dict[bytes, list[int]] = {}
        for row_index in range(903):
            row = np.ascontiguousarray(matrix[row_index])
            matches = seen.setdefault(hashlib.sha256(row.tobytes()).digest(), [])
            for previous in matches:
                if np.array_equal(row, matrix[previous]):
                    uf.union(row_index, previous)
            matches.append(row_index)
    roots = [uf.find(index) for index in range(903)]
    unique = sorted(set(roots), key=lambda root: min(i for i, item in enumerate(roots) if item == root))
    root_to_group = {root: group for group, root in enumerate(unique)}
    groups = np.asarray([root_to_group[root] for root in roots], dtype=np.int64)
    counts = np.bincount(groups)
    return groups, {"group_count": int(len(unique)), "duplicate_rows_absorbed": int(np.sum(counts - 1)), "largest_group": int(counts.max())}


def _group_splits(indices: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train_local, valid_local in splitter.split(np.zeros(len(indices)), y[indices], groups[indices]):
        fit_indices = np.sort(indices[train_local])
        heldout_indices = np.sort(indices[valid_local])
        if np.intersect1d(groups[fit_indices], groups[heldout_indices]).size:
            raise AssertionError("Group leakage detected")
        if set(y[fit_indices]) != {0, 1} or set(y[heldout_indices]) != {0, 1}:
            raise ValueError("Grouped stratified fold lacks one binary class")
        result.append((fit_indices, heldout_indices))
    return result


def _freeze_grid() -> list[dict[str, Any]]:
    universe = [{"max_depth": depth, "min_samples_leaf": leaf, "max_features": features, "class_weight": weight} for depth in (8, None) for leaf in (1, 4, 12) for features in ("sqrt", 0.1) for weight in (None, "balanced")]
    if len(universe) != 24:
        raise AssertionError("The declared RF universe must contain 24 configurations")
    # Stable IDs and ordering make the exact exhaustive grid portable and reviewable.
    ordered = sorted(universe, key=lambda item: (item["max_depth"] is None, item["max_depth"] or 0, -item["min_samples_leaf"], 0 if item["max_features"] == "sqrt" else 1, 0 if item["class_weight"] is None else 1))
    return [{"config_id": f"rf_{index:02d}", **item, "forest_params": {"n_estimators": N_ESTIMATORS, "max_depth": item["max_depth"], "min_samples_leaf": item["min_samples_leaf"], "max_features": item["max_features"], "class_weight": item["class_weight"], "random_state": OUTER_SEED, "n_jobs": N_JOBS}} for index, item in enumerate(ordered)]


def _params(config: dict[str, Any]) -> dict[str, Any]:
    return {"n_estimators": N_ESTIMATORS, "max_depth": config["max_depth"], "min_samples_leaf": int(config["min_samples_leaf"]), "max_features": config["max_features"], "class_weight": config["class_weight"], "random_state": OUTER_SEED, "n_jobs": N_JOBS}


def _make_rf(config: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(**_params(config))


def _fit_raw(X: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> RandomForestClassifier:
    model = _make_rf(config)
    model.fit(X, y)
    return model


def _relative_positions(full_indices: np.ndarray, selected: np.ndarray) -> np.ndarray:
    positions = {int(index): position for position, index in enumerate(full_indices)}
    return np.asarray([positions[int(index)] for index in selected], dtype=np.int64)


def _fit_sigmoid(X: np.ndarray, y: np.ndarray, fit_indices: np.ndarray, groups: np.ndarray, config: dict[str, Any], explicit_splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> CalibratedClassifierCV:
    fit_indices = np.sort(np.asarray(fit_indices, dtype=np.int64))
    calibration_splits = explicit_splits or _group_splits(fit_indices, y, groups, INNER_SPLITS, INNER_SEED)
    relative_splits = [(_relative_positions(fit_indices, left), _relative_positions(fit_indices, right)) for left, right in calibration_splits]
    calibrator = CalibratedClassifierCV(estimator=_make_rf(config), method="sigmoid", cv=relative_splits, ensemble=False, n_jobs=1)
    calibrator.fit(X[fit_indices], y[fit_indices])
    return calibrator


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    values = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(probabilities, dtype=float))))
    return np.sort(values[np.isfinite(values)])


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _threshold_row(y: np.ndarray, probabilities: np.ndarray, intents: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int8)
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    medical = intents == "Medical"
    medical_n = int(np.sum(medical))
    medical_fn = int(np.sum(medical & (predicted == 0)))
    relevant_recall = float(recall_score(y, predicted, zero_division=0))
    medical_recall = _safe_divide(medical_n - medical_fn, medical_n)
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)), "precision": float(precision_score(y, predicted, zero_division=0)), "relevant_recall": relevant_recall, "specificity": _safe_divide(tn, tn + fp), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "medical_fn": medical_fn, "medical_n": medical_n, "medical_recall": medical_recall, "referral_fraction": float(np.mean(predicted)), "referrals": int(np.sum(predicted)), "feasible_recall_first": bool(relevant_recall >= RELEVANT_RECALL_TARGET and (medical_n == 0 or (medical_recall or 0.0) >= MEDICAL_RECALL_TARGET))}


def _choose_thresholds(y: np.ndarray, probabilities: np.ndarray, intents: np.ndarray) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    rows = [_threshold_row(y, probabilities, intents, threshold) for threshold in _threshold_candidates(probabilities)]
    balanced = sorted(rows, key=lambda row: (-row["macro_f1"], -row["relevant_recall"], -row["precision"], -row["threshold"]))[0]
    feasible = [row for row in rows if row["feasible_recall_first"]]
    fallback = not feasible
    recall_first = sorted(feasible or rows, key=(lambda row: (row["fp"], -row["threshold"])) if feasible else (lambda row: (-row["relevant_recall"], -(row["medical_recall"] or 0.0), row["fp"], -row["threshold"])))[0]
    default = next(row for row in rows if row["threshold"] == 0.5)
    selected = {"default_0_5": dict(default), "balanced_macro_f1": dict(balanced), "recall_first": dict(recall_first)}
    selected["recall_first"]["fallback_used"] = fallback
    selected["recall_first"]["target_relevant_recall"] = RELEVANT_RECALL_TARGET
    selected["recall_first"]["target_medical_recall"] = MEDICAL_RECALL_TARGET
    sweep = pd.DataFrame(rows)
    for name in selected:
        sweep[f"selected_{name}"] = np.isclose(sweep["threshold"], selected[name]["threshold"], atol=0.0, rtol=0.0)
    return selected, sweep


def _crossfit_selected(X: np.ndarray, y: np.ndarray, intents: np.ndarray, groups: np.ndarray, fit_scope: np.ndarray, selected: dict[str, Any], inner_splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for inner_fold, (fit_indices, heldout_indices) in enumerate(inner_splits):
        raw_model = _fit_raw(X[fit_indices], y[fit_indices], selected)
        p_raw = raw_model.predict_proba(X[heldout_indices])[:, 1]
        sigmoid_model = _fit_sigmoid(X, y, fit_indices, groups, selected)
        p_sigmoid = sigmoid_model.predict_proba(X[heldout_indices])[:, 1]
        for row_index, raw, sigmoid in zip(heldout_indices, p_raw, p_sigmoid):
            records.append({"row_index": int(row_index), "inner_fold": int(inner_fold), "p_raw": float(raw), "p_sigmoid": float(sigmoid)})
    inner = pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)
    expected = np.sort(np.asarray(fit_scope, dtype=np.int64))
    if not np.array_equal(inner["row_index"].to_numpy(dtype=np.int64), expected):
        raise AssertionError("Inner cross-fitted probability coverage failure")
    if not np.isfinite(inner[["p_raw", "p_sigmoid"]].to_numpy()).all() or not ((inner[["p_raw", "p_sigmoid"]] >= 0) & (inner[["p_raw", "p_sigmoid"]] <= 1)).all().all():
        raise ValueError("Inner probability is non-finite or outside [0, 1]")
    y_inner = y[inner["row_index"].to_numpy(dtype=np.int64)]
    brier_raw = float(brier_score_loss(y_inner, inner["p_raw"]))
    brier_sigmoid = float(brier_score_loss(y_inner, inner["p_sigmoid"]))
    method = "sigmoid" if brier_sigmoid < brier_raw else "raw"
    selected_thresholds, sweep = _choose_thresholds(y_inner, inner[f"p_{method}"].to_numpy(dtype=float), intents[inner["row_index"].to_numpy(dtype=np.int64)])
    return inner, {"brier_raw": brier_raw, "brier_sigmoid": brier_sigmoid, "probability_method": method, "threshold_details": selected_thresholds, "inner_probability_coverage": True, "calibration_excludes_inner_heldout": True}, sweep


def _outer_metrics(y: np.ndarray, predicted: np.ndarray, score: np.ndarray | None, intents: np.ndarray, system: str, embedding: str, policy: str, evaluation: str, note: str) -> dict[str, Any]:
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    medical = intents == "Medical"
    medical_n = int(np.sum(medical))
    medical_fn = int(np.sum(medical & (predicted == 0)))
    row: dict[str, Any] = {"system": system, "embedding": embedding, "policy": policy, "evaluation": evaluation, "n": int(len(y)), "accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)), "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "specificity": _safe_divide(tn, tn + fp), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "medical_fn": medical_fn, "medical_n": medical_n, "medical_recall": _safe_divide(medical_n - medical_fn, medical_n), "referral_fraction": float(np.mean(predicted)), "referrals": int(np.sum(predicted)), "score_available": score is not None, "note": note}
    if score is None:
        row["roc_auc"] = None
        row["average_precision"] = None
    else:
        row["roc_auc"] = float(roc_auc_score(y, score))
        row["average_precision"] = float(average_precision_score(y, score))
    return row


def _append_split_rows(rows: list[dict[str, Any]], train: pd.DataFrame, groups: np.ndarray, y: np.ndarray, outer_fold: int, inner_fold: str | int, level: str, role: str, indices: np.ndarray) -> None:
    for index in indices:
        rows.append({"split_level": level, "outer_fold": outer_fold, "inner_fold": inner_fold, "role": role, "row_index": int(index), "query_id": str(train.iloc[index]["query_id"]), "group_id": f"g{int(groups[index]):04d}", "truth": int(y[index]), "label": str(train.iloc[index]["label"]), "intent": str(train.iloc[index]["intent"])})


def _load_test(data_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    test = pd.read_csv(data_dir / "test_queries.csv", dtype="string")
    if list(test.columns) != ["query_id"] or len(test) != 904 or test["query_id"].nunique() != 904:
        raise ValueError("Blind test schema/count is not the declared 904-row source")
    embeddings: dict[str, np.ndarray] = {}
    for name in EMBEDDINGS:
        matrix = np.load(data_dir / f"test_embeddings_{name}.npy", allow_pickle=False, mmap_mode="r")
        if matrix.shape != (904, EXPECTED_DIMS[name]) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid test embedding {name}")
        embeddings[name] = matrix
    return test, embeddings


def _artifact_paths(outputs: Path) -> list[Path]:
    return [outputs / name for name in ("summary.json", "configuration.json", "protocol.json", "environment.json", "requirements_versions.json", "artifact_schema.json", "actual_grid.csv", "candidate_results.csv", "inner_search_records.csv", "outer_predictions.csv", "metrics.csv", "comparative_metrics.csv", "fold_choices.csv", "splits.csv", "threshold_sweep.csv", "test_predictions.csv", "lr_validation.json", "final_model.joblib")]


def _artifact_schema() -> dict[str, Any]:
    return {"actual_grid.csv": {"key_columns": ["config_id", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight", "forest_params"]}, "candidate_results.csv": {"key_columns": ["selection_scope", "outer_fold", "embedding", "config_id", "n_estimators", "forest_params", "mean_inner_average_precision", "inner_average_precision_0", "inner_average_precision_1", "inner_average_precision_2"]}, "fold_choices.csv": {"key_columns": ["outer_fold", "selected_embedding", "selected_config_id", "n_estimators", "selected_forest_params", "probability_method", "threshold_default_0_5", "threshold_balanced_macro_f1", "threshold_recall_first"]}, "outer_predictions.csv": {"key_columns": ["row_index", "query_id", "outer_fold", "truth", "intent", "p_raw", "p_chosen", "threshold_default_0_5", "threshold_balanced_macro_f1", "threshold_recall_first", "label_default_0_5", "label_balanced_macro_f1", "label_recall_first"]}, "metrics.csv": {"key_columns": ["system", "policy", "evaluation", "accuracy", "macro_f1", "precision", "recall", "specificity", "medical_fn", "medical_n", "medical_recall", "referrals", "roc_auc", "average_precision"]}, "test_predictions.csv": {"key_columns": ["query_id", "p_relevant", "label_default_0_5", "label_balanced_macro_f1", "label_recall_first", "predicted_label", "policy", "threshold"]}}


def _validate_lr(version_dir: Path, train: pd.DataFrame, y: np.ndarray, intents: np.ndarray, groups: np.ndarray, outer_splits: list[tuple[np.ndarray, np.ndarray]], source_fingerprint: str, test_metadata_fingerprint: str, llm_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    lr_dir = _lr_dir(version_dir)
    paths = _lr_paths(lr_dir)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Existing LR outputs are incomplete")
    outer = pd.read_csv(lr_dir / "outer_predictions.csv")
    metrics = pd.read_csv(lr_dir / "metrics.csv")
    splits = pd.read_csv(lr_dir / "splits.csv")
    config = json.loads((lr_dir / "configuration.json").read_text(encoding="utf-8"))
    if config.get("source_data", {}).get("source_fingerprint") != source_fingerprint or config.get("source_data", {}).get("test_metadata_fingerprint") != test_metadata_fingerprint:
        raise AssertionError("Existing LR source fingerprints do not match current source inputs")
    if config.get("outer", {}).get("random_state") != OUTER_SEED or config.get("inner", {}).get("random_state") != INNER_SEED or config.get("outer", {}).get("n_splits") != OUTER_SPLITS or config.get("inner", {}).get("n_splits") != INNER_SPLITS:
        raise AssertionError("Existing LR protocol seeds/fold counts do not match RF protocol")
    required = {"row_index", "query_id", "outer_fold", "group_id", "truth", "intent", "p_chosen", "label_default_0_5", "label_balanced_macro_f1", "label_recall_first", "baseline_a_p_relevant", "baseline_b_p_relevant", "baseline_c_p_relevant", "baseline_a_label", "baseline_b_label", "baseline_c_label", "llm_predicted_label"}
    if not required <= set(outer.columns) or len(outer) != 903:
        raise AssertionError("Existing LR outer predictions have an incompatible schema")
    outer = outer.sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(outer["row_index"].to_numpy(dtype=int), np.arange(903)) or not np.array_equal(outer["query_id"].astype(str).to_numpy(), train["query_id"].astype(str).to_numpy()):
        raise AssertionError("Existing LR IDs/order do not match current training source")
    expected_folds = np.full(903, -1, dtype=int)
    for fold, (_, heldout) in enumerate(outer_splits):
        expected_folds[heldout] = fold
    if not np.array_equal(outer["outer_fold"].to_numpy(dtype=int), expected_folds):
        raise AssertionError("Existing LR outer fold IDs do not match shared fold assignments")
    outer_rows = splits[(splits["split_level"] == "outer") & (splits["role"] == "test")].sort_values("row_index")
    if len(outer_rows) != 903 or not np.array_equal(outer_rows["row_index"].to_numpy(dtype=int), np.arange(903)) or not np.array_equal(outer_rows["outer_fold"].to_numpy(dtype=int), expected_folds):
        raise AssertionError("Existing LR split file does not match shared outer fold signature")
    expected_inner_rows: list[dict[str, Any]] = []
    for fold, (outer_train, _outer_test) in enumerate(outer_splits):
        for inner_fold, (inner_train, inner_test) in enumerate(_group_splits(outer_train, y, groups, INNER_SPLITS, INNER_SEED)):
            for role, indices in (("train", inner_train), ("test", inner_test)):
                for index in indices:
                    expected_inner_rows.append({"outer_fold": fold, "inner_fold": inner_fold, "role": role, "row_index": int(index), "query_id": str(train.iloc[index]["query_id"]), "group_id": f"g{int(groups[index]):04d}"})
    expected_inner = pd.DataFrame(expected_inner_rows).sort_values(["outer_fold", "inner_fold", "role", "row_index"]).reset_index(drop=True)
    actual_inner = splits[splits["split_level"].astype(str).eq("inner")].copy()
    actual_inner["outer_fold"] = pd.to_numeric(actual_inner["outer_fold"], errors="raise").astype(int)
    actual_inner["inner_fold"] = pd.to_numeric(actual_inner["inner_fold"], errors="raise").astype(int)
    actual_inner = actual_inner[["outer_fold", "inner_fold", "role", "row_index", "query_id", "group_id"]].sort_values(["outer_fold", "inner_fold", "role", "row_index"]).reset_index(drop=True)
    actual_inner["row_index"] = pd.to_numeric(actual_inner["row_index"], errors="raise").astype(int)
    for column in ("outer_fold", "inner_fold", "role", "row_index", "query_id", "group_id"):
        actual_inner[column] = actual_inner[column].astype(str)
        expected_inner[column] = expected_inner[column].astype(str)
    if not actual_inner.equals(expected_inner):
        raise AssertionError("Existing LR inner split assignments do not match source-ID grouped recomputation")
    if not np.array_equal(outer["group_id"].astype(str).to_numpy(), np.asarray([f"g{int(group):04d}" for group in groups])):
        raise AssertionError("Existing LR group IDs do not match union-ABC duplicate groups")
    if not np.array_equal(outer["truth"].to_numpy(dtype=int), y) or not np.array_equal(outer["intent"].astype(str).to_numpy(), intents):
        raise AssertionError("Existing LR truth/source metadata do not match current source")
    if set(outer["llm_predicted_label"]) != set(llm_map.values()) or not np.array_equal(outer["llm_predicted_label"].astype(str).to_numpy(), np.asarray([llm_map[str(q)] for q in train["query_id"]])):
        raise AssertionError("Existing LR LLM joins do not match current frozen training baseline")
    specs = [("tuned_lr", "tuned", "p_chosen", "label_default_0_5", "default_0_5"), ("tuned_lr", "tuned", "p_chosen", "label_balanced_macro_f1", "balanced_macro_f1"), ("tuned_lr", "tuned", "p_chosen", "label_recall_first", "recall_first"), ("baseline_lr_a", "a", "baseline_a_p_relevant", "baseline_a_label", "default_0_5"), ("baseline_lr_b", "b", "baseline_b_p_relevant", "baseline_b_label", "default_0_5"), ("baseline_lr_c", "c", "baseline_c_p_relevant", "baseline_c_label", "default_0_5"), ("llm_train", "", "", "llm_predicted_label", "frozen_label"), ("always_relevant", "", "", "", "always_relevant")]
    recomputed: list[dict[str, Any]] = []
    for system, embedding, score_col, label_col, policy in specs:
        if system == "llm_train":
            predicted = (outer[label_col].astype(str).to_numpy() == "relevant").astype(np.int8)
            score = None
        elif system == "always_relevant":
            predicted = np.ones(903, dtype=np.int8)
            score = None
        else:
            predicted = outer[label_col].to_numpy(dtype=np.int8)
            score = outer[score_col].to_numpy(dtype=float)
        recomputed.append(_outer_metrics(y, predicted, score, intents, system, embedding, policy, "outer_oof_aggregate", "Recomputed from saved Version 1 outer predictions."))
        reported = metrics[(metrics["system"] == system) & (metrics["policy"] == policy) & (metrics["evaluation"] == "outer_oof_aggregate")]
        if len(reported) != 1:
            raise AssertionError(f"Existing LR metrics missing aggregate row for {system}/{policy}")
        for key in ("accuracy", "macro_f1", "precision", "recall", "medical_recall", "roc_auc", "average_precision"):
            actual = recomputed[-1].get(key)
            saved = reported.iloc[0].get(key)
            if pd.isna(saved):
                saved = None
            if actual is None and saved is None:
                continue
            if actual is None or saved is None or not np.isclose(float(actual), float(saved), rtol=1e-10, atol=1e-12):
                raise AssertionError(f"Existing LR metric mismatch for {system}/{policy}/{key}")
    return outer, {"source_fingerprints_match": True, "protocol_match": True, "id_order_match": True, "shared_outer_fold_signature_match": True, "shared_inner_split_signature_match": True, "duplicate_group_signature_match": True, "source_metadata_match": True, "saved_metrics_recomputed_match": True, "lr_output_fingerprint": _fingerprint(paths, content=True)}, recomputed


def run_experiment(version_dir: Path, force_recompute: bool = False) -> dict[str, Any]:
    """Run or reuse the Version 2 nested grouped Random Forest experiment."""
    version_dir = Path(version_dir).resolve()
    outputs = version_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    data_dir = _data_dir(version_dir)
    grid = _freeze_grid()
    protocol = _protocol(grid)
    source_fingerprint = _fingerprint(_source_paths(data_dir), content=True)
    test_metadata_fingerprint = _fingerprint(_test_paths(data_dir), content=False)
    lr_dir = _lr_dir(version_dir)
    lr_output_fingerprint = _fingerprint(_lr_paths(lr_dir), content=True)
    source_code_fingerprint = _sha256_file(Path(__file__))
    config_fingerprint = hashlib.sha256(json.dumps({"protocol": protocol, "source_code": source_code_fingerprint}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache_fingerprint = hashlib.sha256(f"{source_fingerprint}:{test_metadata_fingerprint}:{lr_output_fingerprint}:{config_fingerprint}".encode()).hexdigest()
    required = _artifact_paths(outputs)
    summary_path = outputs / "summary.json"
    if not force_recompute and summary_path.is_file() and all(path.is_file() for path in required):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("cache_fingerprint") == cache_fingerprint:
            cached["cache_hit"] = True
            print("[RF] cache hit: fingerprints match; no model fitting required")
            return cached

    train, embeddings, llm_map, _metadata = _load_training(data_dir)
    y = (train["label"].astype(str).to_numpy() == "relevant").astype(np.int8)
    intents = train["intent"].astype(str).to_numpy()
    groups, group_info = _duplicate_groups(embeddings)
    outer_splits = _group_splits(np.arange(903), y, groups, OUTER_SPLITS, OUTER_SEED)
    lr_outer, lr_validation, lr_metrics = _validate_lr(version_dir, train, y, intents, groups, outer_splits, source_fingerprint, test_metadata_fingerprint, llm_map)
    print(f"[RF] frozen exhaustive grid: 24 configs, all parameter levels covered; {len(grid) * len(EMBEDDINGS)} candidates/search; n_jobs={N_JOBS}")
    print(f"[RF] LR comparability validated: IDs, source fingerprints, duplicate groups, split signature, and saved metrics")

    actual_grid = pd.DataFrame([{**item, "n_estimators": N_ESTIMATORS, "forest_params": json.dumps(item["forest_params"], sort_keys=True, separators=(",", ":"))} for item in grid])
    split_rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    outer_prediction_rows: list[dict[str, Any]] = []
    fold_choices: list[dict[str, Any]] = []
    fold_times: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits):
        started = time.perf_counter()
        print(f"[RF] outer fold {outer_fold + 1}/{OUTER_SPLITS} started: train={len(outer_train)}, holdout={len(outer_test)}")
        _append_split_rows(split_rows, train, groups, y, outer_fold, "", "outer", "train", outer_train)
        _append_split_rows(split_rows, train, groups, y, outer_fold, "", "outer", "test", outer_test)
        inner_splits = _group_splits(outer_train, y, groups, INNER_SPLITS, INNER_SEED)
        for inner_fold, (inner_train, inner_test) in enumerate(inner_splits):
            _append_split_rows(split_rows, train, groups, y, outer_fold, inner_fold, "inner", "train", inner_train)
            _append_split_rows(split_rows, train, groups, y, outer_fold, inner_fold, "inner", "test", inner_test)
        fold_records: list[dict[str, Any]] = []
        for embedding in EMBEDDINGS:
            embedding_started = time.perf_counter()
            for config in grid:
                aps: list[float] = []
                for inner_train, inner_test in inner_splits:
                    model = _fit_raw(embeddings[embedding][inner_train], y[inner_train], config)
                    aps.append(float(average_precision_score(y[inner_test], model.predict_proba(embeddings[embedding][inner_test])[:, 1])))
                record = {"selection_scope": "outer_fold", "outer_fold": outer_fold, "embedding": embedding, "config_id": config["config_id"], "n_estimators": N_ESTIMATORS, "forest_params": json.dumps(config["forest_params"], sort_keys=True, separators=(",", ":")), "max_depth": config["max_depth"], "min_samples_leaf": config["min_samples_leaf"], "max_features": config["max_features"], "class_weight": "None" if config["class_weight"] is None else config["class_weight"], "mean_inner_average_precision": float(np.mean(aps)), "inner_average_precision_0": aps[0], "inner_average_precision_1": aps[1], "inner_average_precision_2": aps[2], "inner_rows": int(len(outer_train))}
                fold_records.append(record)
                candidate_records.append(record)
            print(f"[RF] outer fold {outer_fold + 1}/{OUTER_SPLITS} embedding {embedding} complete: 12 configs, {time.perf_counter() - embedding_started:.1f}s")
        selected_record = sorted(fold_records, key=lambda record: (-record["mean_inner_average_precision"], -int(record["min_samples_leaf"]), 0 if record["max_depth"] == 8 else 1, 0 if record["max_features"] == "sqrt" else 1, 0 if record["class_weight"] == "None" else 1, record["embedding"]))[0]
        selected = next(config for config in grid if config["config_id"] == selected_record["config_id"])
        selected["embedding"] = selected_record["embedding"]
        _inner_oof, choice, _inner_sweep = _crossfit_selected(embeddings[selected["embedding"]], y, intents, groups, outer_train, selected, inner_splits)
        thresholds = choice["threshold_details"]
        raw_model = _fit_raw(embeddings[selected["embedding"]][outer_train], y[outer_train], selected)
        p_raw = raw_model.predict_proba(embeddings[selected["embedding"]][outer_test])[:, 1]
        if choice["probability_method"] == "sigmoid":
            chosen_model = _fit_sigmoid(embeddings[selected["embedding"]], y, outer_train, groups, selected)
            p_chosen = chosen_model.predict_proba(embeddings[selected["embedding"]][outer_test])[:, 1]
        else:
            chosen_model = raw_model
            p_chosen = p_raw.copy()
        for position, row_index in enumerate(outer_test):
            row: dict[str, Any] = {"row_index": int(row_index), "query_id": str(train.iloc[row_index]["query_id"]), "outer_fold": outer_fold, "group_id": f"g{int(groups[row_index]):04d}", "truth": int(y[row_index]), "true_label": str(train.iloc[row_index]["label"]), "intent": str(train.iloc[row_index]["intent"]), "selected_embedding": selected["embedding"], "selected_config_id": selected["config_id"], "selected_forest_params": json.dumps(selected["forest_params"], sort_keys=True, separators=(",", ":")), "probability_method": choice["probability_method"], "p_raw": float(p_raw[position]), "p_chosen": float(p_chosen[position]), "threshold_default_0_5": thresholds["default_0_5"]["threshold"], "threshold_balanced_macro_f1": thresholds["balanced_macro_f1"]["threshold"], "threshold_recall_first": thresholds["recall_first"]["threshold"], "label_default_0_5": int(p_chosen[position] >= thresholds["default_0_5"]["threshold"]), "label_balanced_macro_f1": int(p_chosen[position] >= thresholds["balanced_macro_f1"]["threshold"]), "label_recall_first": int(p_chosen[position] >= thresholds["recall_first"]["threshold"]), "llm_predicted_label": llm_map[str(train.iloc[row_index]["query_id"])]}
            for column in ("p_chosen", "label_default_0_5", "label_balanced_macro_f1", "label_recall_first"):
                row[f"lr_{column}"] = lr_outer.iloc[row_index][column]
            outer_prediction_rows.append(row)
        fold_runtime = time.perf_counter() - started
        fold_times.append({"outer_fold": outer_fold, "runtime_seconds": fold_runtime, "n_jobs": N_JOBS, "n_estimators": N_ESTIMATORS})
        fold_choices.append({"outer_fold": outer_fold, "outer_train_rows": len(outer_train), "outer_test_rows": len(outer_test), "selected_embedding": selected["embedding"], "selected_config_id": selected["config_id"], "n_estimators": N_ESTIMATORS, "selected_forest_params": json.dumps(selected["forest_params"], sort_keys=True, separators=(",", ":")), "selected_mean_inner_average_precision": selected_record["mean_inner_average_precision"], "brier_raw": choice["brier_raw"], "brier_sigmoid": choice["brier_sigmoid"], "probability_method": choice["probability_method"], "threshold_default_0_5": thresholds["default_0_5"]["threshold"], "threshold_balanced_macro_f1": thresholds["balanced_macro_f1"]["threshold"], "threshold_recall_first": thresholds["recall_first"]["threshold"], "recall_first_feasible": thresholds["recall_first"]["feasible_recall_first"], "recall_first_fallback_used": thresholds["recall_first"].get("fallback_used", False), "inner_probability_coverage": choice["inner_probability_coverage"], "calibration_excludes_inner_heldout": choice["calibration_excludes_inner_heldout"], "outer_holdout_excluded_selection": True, "runtime_seconds": fold_runtime, "n_jobs": N_JOBS, "n_estimators": N_ESTIMATORS})
        print(f"[RF] outer fold {outer_fold + 1}/{OUTER_SPLITS} complete: selected {selected['embedding']}/{selected['config_id']}, method={choice['probability_method']}, runtime={fold_runtime:.1f}s")

    outer_predictions = pd.DataFrame(outer_prediction_rows).sort_values("row_index").reset_index(drop=True)
    if len(outer_predictions) != 903 or not np.array_equal(outer_predictions["row_index"].to_numpy(dtype=int), np.arange(903)) or outer_predictions["query_id"].nunique() != 903:
        raise AssertionError("RF outer predictions do not cover every training row exactly once")
    if not np.isfinite(outer_predictions[["p_raw", "p_chosen"]].to_numpy()).all() or not ((outer_predictions[["p_raw", "p_chosen"]] >= 0) & (outer_predictions[["p_raw", "p_chosen"]] <= 1)).all().all():
        raise ValueError("RF outer probabilities are non-finite or outside [0, 1]")

    metrics: list[dict[str, Any]] = []
    rf_specs = [("default_0_5", "label_default_0_5"), ("balanced_macro_f1", "label_balanced_macro_f1"), ("recall_first", "label_recall_first")]
    for policy, label_col in rf_specs:
        metrics.append(_outer_metrics(y, outer_predictions[label_col].to_numpy(dtype=np.int8), outer_predictions["p_chosen"].to_numpy(dtype=float), intents, "tuned_rf", "tuned", policy, "outer_oof_aggregate", "Nested selected RF; pooled OOF scores are descriptive."))
        for fold in range(OUTER_SPLITS):
            mask = outer_predictions["outer_fold"].to_numpy(dtype=int) == fold
            metrics.append(_outer_metrics(y[mask], outer_predictions.loc[mask, label_col].to_numpy(dtype=np.int8), outer_predictions.loc[mask, "p_chosen"].to_numpy(dtype=float), intents[mask], "tuned_rf", "tuned", policy, f"outer_fold_{fold}", "Single outer-fold held-out result."))
    metrics.extend(lr_metrics)

    final_inner_splits = _group_splits(np.arange(903), y, groups, INNER_SPLITS, INNER_SEED)
    final_records: list[dict[str, Any]] = []
    for embedding in EMBEDDINGS:
        for config in grid:
            aps: list[float] = []
            for inner_train, inner_test in final_inner_splits:
                model = _fit_raw(embeddings[embedding][inner_train], y[inner_train], config)
                aps.append(float(average_precision_score(y[inner_test], model.predict_proba(embeddings[embedding][inner_test])[:, 1])))
            final_records.append({"selection_scope": "final_all_training", "outer_fold": "final", "embedding": embedding, "config_id": config["config_id"], "n_estimators": N_ESTIMATORS, "forest_params": json.dumps(config["forest_params"], sort_keys=True, separators=(",", ":")), "max_depth": config["max_depth"], "min_samples_leaf": config["min_samples_leaf"], "max_features": config["max_features"], "class_weight": "None" if config["class_weight"] is None else config["class_weight"], "mean_inner_average_precision": float(np.mean(aps)), "inner_average_precision_0": aps[0], "inner_average_precision_1": aps[1], "inner_average_precision_2": aps[2], "inner_rows": 903})
    candidate_records.extend(final_records)
    final_selected_record = sorted(final_records, key=lambda record: (-record["mean_inner_average_precision"], -int(record["min_samples_leaf"]), 0 if record["max_depth"] == 8 else 1, 0 if record["max_features"] == "sqrt" else 1, 0 if record["class_weight"] == "None" else 1, record["embedding"]))[0]
    final_selected = next(config.copy() for config in grid if config["config_id"] == final_selected_record["config_id"])
    final_selected["embedding"] = final_selected_record["embedding"]
    _final_inner_oof, final_choice, final_sweep = _crossfit_selected(embeddings[final_selected["embedding"]], y, intents, groups, np.arange(903), final_selected, final_inner_splits)
    final_thresholds = final_choice["threshold_details"]
    print(f"[RF] final all-training selection: {final_selected['embedding']}/{final_selected['config_id']}, method={final_choice['probability_method']}; loading test vectors now")
    test, test_embeddings = _load_test(data_dir)
    final_raw_model = _fit_raw(embeddings[final_selected["embedding"]], y, final_selected)
    final_p_raw = final_raw_model.predict_proba(test_embeddings[final_selected["embedding"]])[:, 1]
    if final_choice["probability_method"] == "sigmoid":
        final_model = _fit_sigmoid(embeddings[final_selected["embedding"]], y, np.arange(903), groups, final_selected, explicit_splits=final_inner_splits)
        final_p_chosen = final_model.predict_proba(test_embeddings[final_selected["embedding"]])[:, 1]
    else:
        final_model = final_raw_model
        final_p_chosen = final_p_raw.copy()
    test_predictions = pd.DataFrame({"query_id": test["query_id"].astype(str), "p_relevant": final_p_chosen, "p_raw": final_p_raw, "p_chosen": final_p_chosen, "label_default_0_5": (final_p_chosen >= final_thresholds["default_0_5"]["threshold"]).astype(np.int8), "label_balanced_macro_f1": (final_p_chosen >= final_thresholds["balanced_macro_f1"]["threshold"]).astype(np.int8), "label_recall_first": (final_p_chosen >= final_thresholds["recall_first"]["threshold"]).astype(np.int8), "predicted_label": np.where(final_p_chosen >= final_thresholds["recall_first"]["threshold"], "relevant", "irrelevant"), "policy": "recall_first", "threshold": final_thresholds["recall_first"]["threshold"], "threshold_default_0_5": final_thresholds["default_0_5"]["threshold"], "threshold_balanced_macro_f1": final_thresholds["balanced_macro_f1"]["threshold"], "threshold_recall_first": final_thresholds["recall_first"]["threshold"], "probability_method": final_choice["probability_method"], "selected_embedding": final_selected["embedding"], "selected_config_id": final_selected["config_id"], "selected_forest_params": json.dumps(final_selected["forest_params"], sort_keys=True, separators=(",", ":"))})
    if len(test_predictions) != 904 or not test_predictions["query_id"].equals(test["query_id"].astype(str)):
        raise AssertionError("Test prediction count/order does not match source")
    if not np.isfinite(test_predictions[["p_relevant", "p_raw", "p_chosen"]].to_numpy()).all() or not ((test_predictions[["p_relevant", "p_raw", "p_chosen"]] >= 0) & (test_predictions[["p_relevant", "p_raw", "p_chosen"]] <= 1)).all().all():
        raise ValueError("Test probabilities are non-finite or outside [0, 1]")

    final_model_path = outputs / "final_model.joblib"
    joblib_model_saved = False
    if joblib is not None:
        joblib.dump({"model": final_model, "embedding": final_selected["embedding"], "config_id": final_selected["config_id"], "forest_params": final_selected["forest_params"], "probability_method": final_choice["probability_method"], "thresholds": {name: values["threshold"] for name, values in final_thresholds.items()}, "protocol": protocol}, final_model_path)
        loaded = joblib.load(final_model_path)["model"]
        loaded_p = loaded.predict_proba(test_embeddings[final_selected["embedding"]])[:, 1]
        if not np.allclose(loaded_p, final_p_chosen, rtol=1e-12, atol=1e-12):
            raise AssertionError("Saved joblib model does not reproduce final test probabilities")
        joblib_model_saved = True

    candidates = pd.DataFrame(candidate_records)
    candidates["selection_rank"] = candidates.groupby(["selection_scope", "outer_fold"], dropna=False)["mean_inner_average_precision"].rank(method="first", ascending=False).astype(int)
    final_sweep.insert(0, "selection_scope", "final_all_training")
    splits = pd.DataFrame(split_rows)
    splits["_split_order"] = splits["split_level"].map({"outer": 0, "inner": 1})
    splits["_role_order"] = splits["role"].map({"train": 0, "test": 1})
    splits["_outer_order"] = pd.to_numeric(splits["outer_fold"], errors="coerce")
    splits["_inner_order"] = pd.to_numeric(splits["inner_fold"], errors="coerce").fillna(-1)
    splits = splits.sort_values(["_split_order", "_outer_order", "_inner_order", "_role_order", "row_index"]).drop(columns=["_split_order", "_role_order", "_outer_order", "_inner_order"]).reset_index(drop=True)
    comparative = pd.DataFrame([row for row in metrics if row["evaluation"] == "outer_oof_aggregate"])
    protocol_payload = {**protocol, "source_data": {"relative_data_dir": str(data_dir), "source_fingerprint": source_fingerprint, "test_metadata_fingerprint": test_metadata_fingerprint, "imported_lr_output_fingerprint": lr_output_fingerprint}, "config_fingerprint": config_fingerprint, "cache_fingerprint": cache_fingerprint}
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__, "joblib": getattr(joblib, "__version__", None), "cpu_count": os.cpu_count(), "n_jobs": N_JOBS, "seeds": {"outer": OUTER_SEED, "inner": INNER_SEED}}
    checks = {"train_rows_903": len(train) == 903, "train_ids_unique": train["query_id"].nunique() == 903, "labels_valid": set(train["label"]) == {"relevant", "irrelevant"}, "embedding_shapes_and_finiteness": True, "normalized_training_embeddings": True, "duplicate_groups_union_ABC": group_info["group_count"] == 902, "frozen_grid_24": len(grid) == 24, "grid_all_parameter_levels": all(len({item[key] for item in grid}) == len(levels) for key, levels in {"max_depth": (8, None), "min_samples_leaf": (1, 4, 12), "max_features": ("sqrt", 0.1), "class_weight": (None, "balanced")}.items()), "outer_folds_5": len(outer_splits) == 5, "outer_group_disjoint": True, "inner_group_disjoint": True, "shared_outer_split_for_ABC": True, "shared_outer_split_with_lr": lr_validation["shared_outer_fold_signature_match"], "lr_source_and_metrics_validated": all(lr_validation.values()), "outer_oof_coverage_903": len(outer_predictions) == 903 and outer_predictions["query_id"].nunique() == 903, "inner_crossfit_coverage": all(choice["inner_probability_coverage"] for choice in fold_choices) and final_choice["inner_probability_coverage"], "calibration_complete_exclusion": all(choice["calibration_excludes_inner_heldout"] for choice in fold_choices) and final_choice["calibration_excludes_inner_heldout"], "threshold_endpoints_and_default_included": True, "outer_holdout_excluded_all_selection": all(choice["outer_holdout_excluded_selection"] for choice in fold_choices), "auc_ap_probabilities_only": True, "llm_auc_ap_null": all(row["roc_auc"] is None and row["average_precision"] is None for row in metrics if row["system"] in {"llm_train", "always_relevant"}), "test_rows_904": len(test_predictions) == 904, "test_order_preserved": test_predictions["query_id"].equals(test["query_id"].astype(str)), "test_probabilities_finite_and_bounded": True, "test_loaded_after_final_selection": True, "joblib_model_saved": joblib_model_saved, "saved_model_matches_test_export": joblib_model_saved}
    for frame, name in [(actual_grid, "actual_grid.csv"), (candidates, "candidate_results.csv"), (candidates, "inner_search_records.csv"), (outer_predictions, "outer_predictions.csv"), (pd.DataFrame(metrics), "metrics.csv"), (comparative, "comparative_metrics.csv"), (pd.DataFrame(fold_choices), "fold_choices.csv"), (splits, "splits.csv"), (final_sweep, "threshold_sweep.csv"), (test_predictions, "test_predictions.csv")]:
        frame.to_csv(outputs / name, index=False)
    _write_json(outputs / "lr_validation.json", lr_validation)
    _write_json(outputs / "configuration.json", protocol_payload)
    _write_json(outputs / "protocol.json", protocol_payload)
    _write_json(outputs / "environment.json", environment)
    _write_json(outputs / "requirements_versions.json", {"runtime": environment, "declared": ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "joblib>=1.4"]})
    _write_json(outputs / "artifact_schema.json", _artifact_schema())
    aggregate_metrics = [row for row in metrics if row["evaluation"] == "outer_oof_aggregate"]
    artifact_map = {path.name: str(path.relative_to(version_dir)) for path in _artifact_paths(outputs) if path.exists()}
    summary = {"stage": "version_2_random_forest", "cache_hit": False, "source_fingerprint": source_fingerprint, "test_metadata_fingerprint": test_metadata_fingerprint, "imported_lr_output_fingerprint": lr_output_fingerprint, "config_fingerprint": config_fingerprint, "cache_fingerprint": cache_fingerprint, "selection": {"outer_fold_choices": fold_choices, "final_embedding": final_selected["embedding"], "final_config_id": final_selected["config_id"], "final_forest_params": final_selected["forest_params"], "final_mean_inner_average_precision": final_selected_record["mean_inner_average_precision"], "final_probability_method": final_choice["probability_method"], "final_brier_raw": final_choice["brier_raw"], "final_brier_sigmoid": final_choice["brier_sigmoid"], "final_thresholds": {name: values["threshold"] for name, values in final_thresholds.items()}, "recommended_policy": "recall_first", "recommended_threshold": final_thresholds["recall_first"]["threshold"]}, "config": protocol_payload, "counts": {"train_rows": 903, "test_rows": 904, "outer_folds": 5, "inner_folds": 3, "candidate_configurations_per_search": 72, **group_info, "medical_train_rows": int(np.sum(intents == "Medical"))}, "timing": {"outer_folds": fold_times, "total_seconds": float(sum(item["runtime_seconds"] for item in fold_times))}, "metrics": aggregate_metrics, "comparative_metrics": aggregate_metrics, "checks": checks, "artifacts": artifact_map, "artifact_schema": _artifact_schema(), "limitations": ["Blind test has no supplied truth labels.", "Recall-first Medical and relevant targets are assumptions, not safety guarantees.", "Pooled outer OOF AUC/AP combine scores from fold-specific fitted models and are descriptive.", "RF is not promised to improve accuracy or any other metric.", "The exhaustive 24-configuration grid is bounded by 200 trees and n_jobs=2; it is not a global optimization guarantee.", "The frozen LLM provides labels only, so ROC-AUC/AP are null."], "environment": environment}
    _write_json(summary_path, summary)
    print(f"[RF] complete: total outer runtime={summary['timing']['total_seconds']:.1f}s; final export has {len(test_predictions)} ordered IDs")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_dir", type=Path)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.version_dir, force_recompute=args.force_recompute)
    print(json.dumps({"cache_hit": result.get("cache_hit"), "artifacts": result.get("artifacts")}, indent=2))
