"""Nested, group-aware logistic-regression experiment for Version 1 Part 2.

The public entry point is :func:`run_experiment`.  It keeps the supplied data
read-only and writes every generated artifact below the supplied ``part2_dir``.
The final blind-test predictions are produced only after the complete inner
development procedure has selected an embedding, LR settings, probability
method, and operating thresholds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
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
except ImportError:  # pragma: no cover - the declared environment has joblib
    joblib = None


EMBEDDINGS = ("a", "b", "c")
EXPECTED_DIMS = {"a": 384, "b": 768, "c": 1536}
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
WEIGHTS: tuple[None | str, ...] = (None, "balanced")
OUTER_SPLITS = 5
INNER_SPLITS = 3
OUTER_SEED = 42
INNER_SEED = 43
MAX_ITER = 3000
RELEVANT_RECALL_TARGET = 0.98
MEDICAL_RECALL_TARGET = 0.99
IMPLEMENTATION_VERSION = "lr-part2-nested-v1"


def _json_ready(value: Any) -> Any:
    """Convert NumPy and pandas scalar values to strict JSON values."""
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
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


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
        entry: dict[str, Any] = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if content:
            entry["sha256"] = _sha256_file(path)
        entries.append(entry)
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _data_dir(part2_dir: Path) -> Path:
    for parent in (part2_dir.resolve(), *part2_dir.resolve().parents):
        candidate = parent / "intent-classification-assignment" / "intent-classification-assignment"
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the supplied intent-classification-assignment data")


def _protocol() -> dict[str, Any]:
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "model_family": "logistic_regression",
        "embeddings": {name: {"dimensions": EXPECTED_DIMS[name], "inputs": "raw normalized vectors"} for name in EMBEDDINGS},
        "duplicate_groups": "connected components of exact row matches in the union of training embeddings A/B/C",
        "outer": {"splitter": "StratifiedGroupKFold", "n_splits": OUTER_SPLITS, "shuffle": True, "random_state": OUTER_SEED, "stratification": "binary label"},
        "inner": {"splitter": "StratifiedGroupKFold", "n_splits": INNER_SPLITS, "shuffle": True, "random_state": INNER_SEED, "scope": "each outer training partition"},
        "search": {"penalty": "l2", "solver": "lbfgs", "C": list(C_VALUES), "class_weight": [None, "balanced"], "max_iter": MAX_ITER, "selection_score": "mean inner average_precision", "tie_break": ["abs(log(C)) closest to 0", "embedding name ascending", "class_weight None before balanced"]},
        "calibration": {"methods": ["raw", "sigmoid"], "method": "sigmoid", "cv": "explicit StratifiedGroupKFold 3-way splits within each inner-training partition", "ensemble": False, "selection_score": "lower pooled inner-heldout Brier score; raw on exact tie", "exclusion": "each inner-heldout row is excluded from both base-model fitting and calibration fitting"},
        "thresholds": {"default": 0.5, "balanced_macro_f1": "maximum macro-F1", "recall_first": {"minimum_relevant_recall": RELEVANT_RECALL_TARGET, "minimum_medical_recall": MEDICAL_RECALL_TARGET, "objective": "minimum false positives among feasible thresholds", "assumption": True}, "candidate_values": "0, 1, 0.5, and unique chosen inner-heldout probabilities", "tie_break": "higher relevant recall, higher precision, then higher threshold where applicable"},
        "baselines": {"fixed_logistic_regression": "each A/B/C, raw vectors, C=1, class_weight=None, max_iter=3000, threshold=0.5", "llm": "frozen supplied training predictions joined by query_id", "always_relevant": "constant relevant label; no score"},
        "blind_test": "loaded only after final all-training inner development selection and threshold freeze; no test labels are available",
        "assumptions": "98% relevant recall and 99% Medical recall are illustrative development targets, not assignment requirements or safety guarantees",
    }


def _load_training(data_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    train = pd.read_csv(data_dir / "train_queries.csv", dtype="string")
    if list(train.columns) != ["query_id", "intent", "label"]:
        raise ValueError(f"Unexpected train_queries.csv columns: {list(train.columns)}")
    if len(train) != 903 or train["query_id"].nunique() != 903:
        raise ValueError("Training data must contain 903 unique query IDs")
    if train["label"].isna().any() or set(train["label"].dropna()) != {"relevant", "irrelevant"}:
        raise ValueError("Training labels must be exactly relevant and irrelevant")

    embeddings: dict[str, np.ndarray] = {}
    for name in EMBEDDINGS:
        path = data_dir / f"train_embeddings_{name}.npy"
        matrix = np.load(path, allow_pickle=False, mmap_mode="r")
        if matrix.shape != (903, EXPECTED_DIMS[name]):
            raise ValueError(f"Unexpected {name} training shape: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values in training embedding {name}")
        norms = np.linalg.norm(np.asarray(matrix), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3, rtol=2e-3):
            raise ValueError(f"Training embedding {name} is not normalized")
        embeddings[name] = matrix

    with (data_dir / "label_mapping.json").open(encoding="utf-8") as stream:
        policy = json.load(stream)
    with (data_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (data_dir / "llm_baseline_train_predictions.json").open(encoding="utf-8") as stream:
        llm_records = json.load(stream)
    llm = pd.DataFrame(llm_records)
    if set(llm.columns) != {"query_id", "predicted_intent", "predicted_label"} or len(llm) != 903:
        raise ValueError("Unexpected frozen training LLM baseline schema")
    if llm["query_id"].nunique() != 903 or set(llm["query_id"]) != set(train["query_id"]):
        raise ValueError("Frozen training LLM IDs do not match training IDs")
    llm_map = dict(zip(llm["query_id"].astype(str), llm["predicted_label"].astype(str)))
    if set(llm_map.values()) - {"relevant", "irrelevant"}:
        raise ValueError("Frozen LLM labels contain an invalid routing label")

    relevant_intents = set(policy.get("relevant", []))
    irrelevant_intents = set(policy.get("irrelevant", []))
    if not set(train["intent"]) <= relevant_intents | irrelevant_intents:
        raise ValueError("Training intents are not covered by label_mapping.json")
    expected_labels = train["intent"].map(lambda value: "relevant" if value in relevant_intents else "irrelevant")
    if not np.array_equal(expected_labels.astype(str).to_numpy(), train["label"].astype(str).to_numpy()):
        raise ValueError("Training intent-to-routing-label policy does not match labels")
    meta = {"policy": policy, "manifest": manifest, "llm": llm_map}
    return train, embeddings, llm_map, meta


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
        for row_index in range(matrix.shape[0]):
            row = np.ascontiguousarray(matrix[row_index])
            digest = hashlib.sha256(row.tobytes()).digest()
            matches = seen.setdefault(digest, [])
            for previous in matches:
                if np.array_equal(row, matrix[previous]):
                    uf.union(row_index, previous)
            matches.append(row_index)
    roots = [uf.find(index) for index in range(903)]
    unique_roots = sorted(set(roots), key=lambda root: min(i for i, item in enumerate(roots) if item == root))
    root_to_group = {root: group for group, root in enumerate(unique_roots)}
    groups = np.asarray([root_to_group[root] for root in roots], dtype=np.int64)
    counts = np.bincount(groups)
    return groups, {"group_count": int(len(unique_roots)), "duplicate_rows_absorbed": int(np.sum(counts - 1)), "largest_group": int(counts.max())}


def _group_splits(indices: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train_local, valid_local in splitter.split(np.zeros(len(indices)), y[indices], groups[indices]):
        fit_indices = np.sort(indices[train_local])
        heldout_indices = np.sort(indices[valid_local])
        if np.intersect1d(groups[fit_indices], groups[heldout_indices]).size:
            raise AssertionError("Group leakage detected in grouped split")
        if set(y[fit_indices]) != {0, 1} or set(y[heldout_indices]) != {0, 1}:
            raise ValueError("A grouped stratified fold lacks one of the binary classes")
        result.append((fit_indices, heldout_indices))
    return result


def _make_lr(c_value: float, weight: None | str) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        class_weight=weight,
        max_iter=MAX_ITER,
    )


def _fit_raw(X: np.ndarray, y: np.ndarray, c_value: float, weight: None | str) -> LogisticRegression:
    model = _make_lr(c_value, weight)
    model.fit(X, y)
    if int(np.max(model.n_iter_)) >= MAX_ITER:
        warnings.warn(
            f"Logistic regression reached max_iter={MAX_ITER} for C={c_value}, weight={weight}",
            RuntimeWarning,
            stacklevel=2,
        )
    return model


def _relative_positions(full_indices: np.ndarray, selected: np.ndarray) -> np.ndarray:
    positions = {int(index): position for position, index in enumerate(full_indices)}
    try:
        return np.asarray([positions[int(index)] for index in selected], dtype=np.int64)
    except KeyError as exc:
        raise AssertionError("Calibration split contains an index outside its fit partition") from exc


def _fit_sigmoid(
    X: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
    groups: np.ndarray,
    c_value: float,
    weight: None | str,
    seed: int = INNER_SEED,
    explicit_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> CalibratedClassifierCV:
    fit_indices = np.sort(np.asarray(fit_indices, dtype=np.int64))
    calibration_splits = explicit_splits or _group_splits(fit_indices, y, groups, INNER_SPLITS, seed)
    relative_splits = [
        (_relative_positions(fit_indices, cal_train), _relative_positions(fit_indices, cal_valid))
        for cal_train, cal_valid in calibration_splits
    ]
    calibrator = CalibratedClassifierCV(
        estimator=_make_lr(c_value, weight),
        method="sigmoid",
        cv=relative_splits,
        ensemble=False,
        n_jobs=1,
    )
    # X and y are indexed only with fit_indices.  Thus every external
    # held-out row is absent from both base fitting and calibration fitting.
    calibrator.fit(X[fit_indices], y[fit_indices])
    return calibrator


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    values = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(probabilities, dtype=float))))
    values = values[np.isfinite(values)]
    return np.sort(values)


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
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "relevant_recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": _safe_divide(tn, tn + fp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "medical_fn": medical_fn,
        "medical_n": medical_n,
        "medical_recall": _safe_divide(medical_n - medical_fn, medical_n),
        "referral_fraction": float(np.mean(predicted)),
        "feasible_recall_first": bool(
            recall_score(y, predicted, zero_division=0) >= RELEVANT_RECALL_TARGET
            and (medical_n == 0 or (medical_n - medical_fn) / medical_n >= MEDICAL_RECALL_TARGET)
        ),
    }


def _choose_thresholds(y: np.ndarray, probabilities: np.ndarray, intents: np.ndarray) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    rows = [_threshold_row(y, probabilities, intents, threshold) for threshold in _threshold_candidates(probabilities)]
    sweep = pd.DataFrame(rows)
    # The final key makes ties reproducible even if all preceding values tie.
    balanced = sorted(
        rows,
        key=lambda row: (row["macro_f1"], row["relevant_recall"], row["precision"], row["threshold"]),
        reverse=True,
    )[0]
    feasible = [row for row in rows if row["feasible_recall_first"]]
    fallback = False
    if feasible:
        recall_first = sorted(feasible, key=lambda row: (row["fp"], -row["threshold"]))[0]
    else:  # endpoint 0 normally makes this unreachable, but preserve an honest result.
        fallback = True
        recall_first = sorted(rows, key=lambda row: (-row["relevant_recall"], -(row["medical_recall"] or 0.0), row["fp"], -row["threshold"]))[0]
    default_candidates = [row for row in rows if np.isclose(row["threshold"], 0.5, atol=0.0, rtol=0.0)]
    if not default_candidates:
        raise AssertionError("Threshold candidate endpoints/default were not included")
    selected = {
        "default_0_5": dict(default_candidates[0]),
        "balanced_macro_f1": dict(balanced),
        "recall_first": dict(recall_first),
    }
    selected["recall_first"]["fallback_used"] = fallback
    selected["recall_first"]["target_relevant_recall"] = RELEVANT_RECALL_TARGET
    selected["recall_first"]["target_medical_recall"] = MEDICAL_RECALL_TARGET
    sweep["selected_default_0_5"] = np.isclose(sweep["threshold"], selected["default_0_5"]["threshold"], atol=0.0, rtol=0.0)
    sweep["selected_balanced_macro_f1"] = np.isclose(sweep["threshold"], selected["balanced_macro_f1"]["threshold"], atol=0.0, rtol=0.0)
    sweep["selected_recall_first"] = np.isclose(sweep["threshold"], selected["recall_first"]["threshold"], atol=0.0, rtol=0.0)
    return selected, sweep


def _crossfit_selected(
    X: np.ndarray,
    y: np.ndarray,
    intents: np.ndarray,
    groups: np.ndarray,
    fit_scope: np.ndarray,
    selected: dict[str, Any],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for inner_fold, (fit_indices, heldout_indices) in enumerate(inner_splits):
        raw_model = _fit_raw(X[fit_indices], y[fit_indices], selected["C"], selected["class_weight"])
        p_raw = raw_model.predict_proba(X[heldout_indices])[:, 1]
        sigmoid_model = _fit_sigmoid(X, y, fit_indices, groups, selected["C"], selected["class_weight"], seed=INNER_SEED)
        p_sigmoid = sigmoid_model.predict_proba(X[heldout_indices])[:, 1]
        for row_index, raw, sigmoid in zip(heldout_indices, p_raw, p_sigmoid):
            records.append({
                "row_index": int(row_index),
                "inner_fold": int(inner_fold),
                "p_raw": float(raw),
                "p_sigmoid": float(sigmoid),
            })
    inner = pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)
    expected = np.sort(np.asarray(fit_scope, dtype=np.int64))
    if not np.array_equal(inner["row_index"].to_numpy(dtype=np.int64), expected):
        warnings.warn("Inner cross-fitted probability coverage does not match its declared scope", RuntimeWarning, stacklevel=2)
        raise AssertionError("Inner cross-fitted probability coverage failure")
    if not np.isfinite(inner[["p_raw", "p_sigmoid"]].to_numpy()).all():
        raise ValueError("Non-finite inner probability")
    if not ((inner[["p_raw", "p_sigmoid"]] >= 0) & (inner[["p_raw", "p_sigmoid"]] <= 1)).all().all():
        raise ValueError("Inner probability outside [0, 1]")
    y_inner = y[inner["row_index"].to_numpy(dtype=np.int64)]
    brier_raw = float(brier_score_loss(y_inner, inner["p_raw"]))
    brier_sigmoid = float(brier_score_loss(y_inner, inner["p_sigmoid"]))
    method = "sigmoid" if brier_sigmoid < brier_raw else "raw"
    chosen_probability = inner[f"p_{method}"].to_numpy(dtype=float)
    selected_thresholds, sweep = _choose_thresholds(y_inner, chosen_probability, intents[inner["row_index"].to_numpy(dtype=np.int64)])
    choice = {
        **selected,
        "brier_raw": brier_raw,
        "brier_sigmoid": brier_sigmoid,
        "probability_method": method,
        "brier_tie_raw": bool(np.isclose(brier_raw, brier_sigmoid, atol=0.0, rtol=0.0)),
        "thresholds": {name: values["threshold"] for name, values in selected_thresholds.items()},
        "threshold_details": selected_thresholds,
        "inner_scope_rows": int(len(expected)),
        "inner_probability_coverage": True,
        "calibration_excludes_inner_heldout": True,
    }
    sweep.insert(0, "row_scope", "inner_development")
    sweep.insert(1, "selected_probability_method", method)
    return inner, choice, sweep


def _outer_metrics(y: np.ndarray, predicted: np.ndarray, score: np.ndarray | None, intents: np.ndarray, system: str, embedding: str, policy: str, evaluation: str, note: str) -> dict[str, Any]:
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    medical = intents == "Medical"
    medical_n = int(np.sum(medical))
    medical_fn = int(np.sum(medical & (predicted == 0)))
    result: dict[str, Any] = {
        "system": system,
        "embedding": embedding,
        "policy": policy,
        "evaluation": evaluation,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": _safe_divide(tn, tn + fp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "medical_fn": medical_fn,
        "medical_n": medical_n,
        "medical_recall": _safe_divide(medical_n - medical_fn, medical_n),
        "referral_fraction": float(np.mean(predicted)),
        "roc_auc": None,
        "average_precision": None,
        "score_available": bool(score is not None),
        "note": note,
    }
    if score is not None:
        result["roc_auc"] = float(roc_auc_score(y, score))
        result["average_precision"] = float(average_precision_score(y, score))
    return result


def _append_split_rows(rows: list[dict[str, Any]], train: pd.DataFrame, groups: np.ndarray, y: np.ndarray, outer_fold: str | int, inner_fold: str | int, split_level: str, role: str, indices: np.ndarray) -> None:
    for row_index in np.asarray(indices, dtype=np.int64):
        rows.append({
            "split_level": split_level,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "role": role,
            "row_index": int(row_index),
            "query_id": str(train.iloc[row_index]["query_id"]),
            "group_id": f"g{int(groups[row_index]):04d}",
            "truth": int(y[row_index]),
            "label": str(train.iloc[row_index]["label"]),
            "intent": str(train.iloc[row_index]["intent"]),
        })


def _load_test(data_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    test = pd.read_csv(data_dir / "test_queries.csv", dtype="string")
    if list(test.columns) != ["query_id"] or len(test) != 904 or test["query_id"].nunique() != 904:
        raise ValueError("Test data must contain 904 unique query IDs and only query_id")
    embeddings: dict[str, np.ndarray] = {}
    for name in EMBEDDINGS:
        matrix = np.load(data_dir / f"test_embeddings_{name}.npy", allow_pickle=False)
        if matrix.shape != (904, EXPECTED_DIMS[name]) or not np.isfinite(matrix).all():
            raise ValueError(f"Unexpected test embedding {name} shape or finiteness")
        embeddings[name] = matrix
    return test, embeddings


def _artifact_paths(outputs: Path) -> list[Path]:
    names = [
        "configuration.json", "protocol.json", "summary.json", "candidate_results.csv",
        "inner_search_records.csv", "outer_predictions.csv", "metrics.csv",
        "fold_choices.csv", "splits.csv", "threshold_sweep.csv", "test_predictions.csv",
        "environment.json", "requirements_versions.json", "artifact_schema.json",
    ]
    return [outputs / name for name in names]


def _artifact_schema() -> dict[str, Any]:
    return {
        "candidate_results.csv": {"key_columns": ["selection_scope", "outer_fold", "embedding", "C", "class_weight", "mean_inner_average_precision", "inner_average_precision_0", "inner_average_precision_1", "inner_average_precision_2"], "purpose": "All 30 LR candidates for each outer training partition and the final all-training development selection."},
        "inner_search_records.csv": {"same_schema_as": "candidate_results.csv"},
        "splits.csv": {"key_columns": ["split_level", "outer_fold", "inner_fold", "role", "row_index", "query_id", "group_id", "truth", "label", "intent"], "purpose": "Long-form outer and inner grouped split membership for every training row."},
        "fold_choices.csv": {"key_columns": ["outer_fold", "selected_embedding", "selected_C", "selected_class_weight", "probability_method", "threshold_default_0_5", "threshold_balanced_macro_f1", "threshold_recall_first"], "purpose": "Per-outer-fold choices made without that fold's holdout."},
        "outer_predictions.csv": {"key_columns": ["row_index", "query_id", "outer_fold", "truth", "true_label", "intent", "p_raw", "p_chosen", "threshold_default_0_5", "threshold_balanced_macro_f1", "threshold_recall_first", "label_default_0_5", "label_balanced_macro_f1", "label_recall_first", "baseline_a_p_relevant", "baseline_b_p_relevant", "baseline_c_p_relevant", "llm_predicted_label"], "purpose": "Exactly one outer-held-out prediction row for each of the 903 labeled examples."},
        "metrics.csv": {"key_columns": ["system", "policy", "evaluation", "accuracy", "macro_f1", "precision", "recall", "specificity", "tp", "tn", "fp", "fn", "medical_fn", "medical_n", "roc_auc", "average_precision"], "purpose": "Aggregate outer OOF and per-fold metrics; AUC/AP are populated only for probability-scored systems."},
        "threshold_sweep.csv": {"key_columns": ["selection_scope", "row_scope", "selected_probability_method", "threshold", "macro_f1", "relevant_recall", "precision", "fp", "medical_fn", "medical_n", "feasible_recall_first"], "purpose": "Final all-training inner-development threshold tradeoff, kept separate from outer evaluation."},
        "test_predictions.csv": {"key_columns": ["query_id", "predicted_label", "p_relevant", "policy", "threshold"], "purpose": "904 blind-test predictions in exact source order using the final illustrative recall-first policy; no test LLM baseline is read or exported."},
        "configuration.json": {"purpose": "Protocol, source/config fingerprints, assumptions, and artifact contract."},
        "protocol.json": {"same_schema_as": "configuration.json"},
        "summary.json": {"purpose": "Notebook-friendly final selection, counts, aggregate metrics, checks, artifact paths, and limitations."},
        "final_model.joblib": {"purpose": "Final raw or sigmoid calibrated predictor plus recipe metadata, when joblib is available."},
    }


def run_experiment(part2_dir: Path, force_recompute: bool = False) -> dict[str, Any]:
    """Run or reuse the nested logistic-regression Part 2 experiment.

    Parameters
    ----------
    part2_dir:
        The ``version 1/part2`` directory. All generated files remain below it.
    force_recompute:
        Recompute even when the verified output cache fingerprint matches.
    """
    part2_dir = Path(part2_dir).resolve()
    outputs = part2_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    data_dir = _data_dir(part2_dir)
    protocol = _protocol()
    source_paths = [
        data_dir / "train_queries.csv",
        *(data_dir / f"train_embeddings_{name}.npy" for name in EMBEDDINGS),
        data_dir / "llm_baseline_train_predictions.json",
        data_dir / "label_mapping.json",
        data_dir / "manifest.json",
    ]
    # Test metadata participates in cache invalidation without loading test
    # embeddings before final development selection.
    test_paths = [data_dir / "test_queries.csv", *(data_dir / f"test_embeddings_{name}.npy" for name in EMBEDDINGS)]
    source_fingerprint = _fingerprint(source_paths, content=True)
    test_metadata_fingerprint = _fingerprint(test_paths, content=False)
    source_code_fingerprint = _sha256_file(Path(__file__))
    config_fingerprint = hashlib.sha256(json.dumps({"protocol": protocol, "source_code": source_code_fingerprint}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    cache_fingerprint = hashlib.sha256(f"{source_fingerprint}:{test_metadata_fingerprint}:{config_fingerprint}".encode("utf-8")).hexdigest()

    required = _artifact_paths(outputs)
    summary_path = outputs / "summary.json"
    if not force_recompute and summary_path.is_file() and all(path.is_file() for path in required):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("cache_fingerprint") == cache_fingerprint:
            cached["cache_hit"] = True
            return cached

    train, embeddings, llm_map, metadata = _load_training(data_dir)
    y = (train["label"].to_numpy() == "relevant").astype(np.int8)
    intents = train["intent"].astype(str).to_numpy()
    groups, group_info = _duplicate_groups(embeddings)
    outer_splits = _group_splits(np.arange(len(train)), y, groups, OUTER_SPLITS, OUTER_SEED)

    split_rows: list[dict[str, Any]] = []
    outer_prediction_rows: list[dict[str, Any]] = []
    fold_choices: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    convergence_warnings: list[str] = []
    outer_fold_metric_frames: list[pd.DataFrame] = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits):
        _append_split_rows(split_rows, train, groups, y, outer_fold, "", "outer", "train", outer_train)
        _append_split_rows(split_rows, train, groups, y, outer_fold, "", "outer", "test", outer_test)
        inner_splits = _group_splits(outer_train, y, groups, INNER_SPLITS, INNER_SEED)
        for inner_fold, (inner_train, inner_test) in enumerate(inner_splits):
            _append_split_rows(split_rows, train, groups, y, outer_fold, inner_fold, "inner", "train", inner_train)
            _append_split_rows(split_rows, train, groups, y, outer_fold, inner_fold, "inner", "test", inner_test)

        fold_records: list[dict[str, Any]] = []
        for embedding in EMBEDDINGS:
            for c_value in C_VALUES:
                for weight in WEIGHTS:
                    aps: list[float] = []
                    for inner_fold, (inner_train, inner_test) in enumerate(inner_splits):
                        model = _fit_raw(embeddings[embedding][inner_train], y[inner_train], c_value, weight)
                        p = model.predict_proba(embeddings[embedding][inner_test])[:, 1]
                        ap = float(average_precision_score(y[inner_test], p))
                        aps.append(ap)
                    record = {
                        "selection_scope": "outer_fold",
                        "outer_fold": outer_fold,
                        "embedding": embedding,
                        "C": float(c_value),
                        "class_weight": "None" if weight is None else weight,
                        "mean_inner_average_precision": float(np.mean(aps)),
                        "inner_average_precision_0": aps[0],
                        "inner_average_precision_1": aps[1],
                        "inner_average_precision_2": aps[2],
                        "inner_rows": int(len(outer_train)),
                    }
                    fold_records.append(record)
                    candidate_records.append(record)
        selected_record = sorted(fold_records, key=lambda record: (-record["mean_inner_average_precision"], abs(np.log(record["C"])), record["embedding"], 0 if record["class_weight"] == "None" else 1))[0]
        selected = {"embedding": selected_record["embedding"], "C": selected_record["C"], "class_weight": None if selected_record["class_weight"] == "None" else selected_record["class_weight"]}
        inner_oof, choice, _inner_sweep = _crossfit_selected(embeddings[selected["embedding"]], y, intents, groups, outer_train, selected, inner_splits)
        thresholds = choice["threshold_details"]

        raw_model = _fit_raw(embeddings[selected["embedding"]][outer_train], y[outer_train], selected["C"], selected["class_weight"])
        p_raw = raw_model.predict_proba(embeddings[selected["embedding"]][outer_test])[:, 1]
        if choice["probability_method"] == "sigmoid":
            calibrated_model = _fit_sigmoid(embeddings[selected["embedding"]], y, outer_train, groups, selected["C"], selected["class_weight"], seed=INNER_SEED)
            p_chosen = calibrated_model.predict_proba(embeddings[selected["embedding"]][outer_test])[:, 1]
        else:
            calibrated_model = None
            p_chosen = p_raw.copy()

        baseline_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for embedding in EMBEDDINGS:
            baseline_model = _fit_raw(embeddings[embedding][outer_train], y[outer_train], 1.0, None)
            baseline_p = baseline_model.predict_proba(embeddings[embedding][outer_test])[:, 1]
            baseline_predictions[embedding] = (baseline_p, (baseline_p >= 0.5).astype(np.int8))
        for row_position, row_index in enumerate(outer_test):
            row: dict[str, Any] = {
                "row_index": int(row_index),
                "query_id": str(train.iloc[row_index]["query_id"]),
                "outer_fold": outer_fold,
                "group_id": f"g{int(groups[row_index]):04d}",
                "truth": int(y[row_index]),
                "true_label": str(train.iloc[row_index]["label"]),
                "intent": str(train.iloc[row_index]["intent"]),
                "selected_embedding": selected["embedding"],
                "selected_C": selected["C"],
                "selected_class_weight": "None" if selected["class_weight"] is None else selected["class_weight"],
                "probability_method": choice["probability_method"],
                "p_raw": float(p_raw[row_position]),
                "p_chosen": float(p_chosen[row_position]),
                "threshold_default_0_5": thresholds["default_0_5"]["threshold"],
                "threshold_balanced_macro_f1": thresholds["balanced_macro_f1"]["threshold"],
                "threshold_recall_first": thresholds["recall_first"]["threshold"],
                "label_default_0_5": int(p_chosen[row_position] >= thresholds["default_0_5"]["threshold"]),
                "label_balanced_macro_f1": int(p_chosen[row_position] >= thresholds["balanced_macro_f1"]["threshold"]),
                "label_recall_first": int(p_chosen[row_position] >= thresholds["recall_first"]["threshold"]),
                "llm_predicted_label": llm_map[str(train.iloc[row_index]["query_id"])],
            }
            for embedding in EMBEDDINGS:
                row[f"baseline_{embedding}_p_relevant"] = float(baseline_predictions[embedding][0][row_position])
                row[f"baseline_{embedding}_label"] = int(baseline_predictions[embedding][1][row_position])
            outer_prediction_rows.append(row)

        fold_choices.append({
            "outer_fold": outer_fold,
            "outer_train_rows": len(outer_train),
            "outer_test_rows": len(outer_test),
            "selected_embedding": selected["embedding"],
            "selected_C": selected["C"],
            "selected_class_weight": "None" if selected["class_weight"] is None else selected["class_weight"],
            "selected_mean_inner_average_precision": selected_record["mean_inner_average_precision"],
            "brier_raw": choice["brier_raw"],
            "brier_sigmoid": choice["brier_sigmoid"],
            "probability_method": choice["probability_method"],
            "threshold_default_0_5": thresholds["default_0_5"]["threshold"],
            "threshold_balanced_macro_f1": thresholds["balanced_macro_f1"]["threshold"],
            "threshold_recall_first": thresholds["recall_first"]["threshold"],
            "recall_first_feasible": thresholds["recall_first"]["feasible_recall_first"],
            "recall_first_fallback_used": thresholds["recall_first"].get("fallback_used", False),
            "inner_probability_coverage": choice["inner_probability_coverage"],
            "calibration_excludes_inner_heldout": True,
            "outer_holdout_excluded_selection": True,
        })

    outer_predictions = pd.DataFrame(outer_prediction_rows).sort_values("row_index").reset_index(drop=True)
    if len(outer_predictions) != 903 or not np.array_equal(outer_predictions["row_index"].to_numpy(), np.arange(903)) or outer_predictions["query_id"].nunique() != 903:
        raise AssertionError("Outer OOF predictions do not cover every training row exactly once")
    if not np.isfinite(outer_predictions[["p_raw", "p_chosen"]].to_numpy()).all() or not ((outer_predictions[["p_raw", "p_chosen"]] >= 0) & (outer_predictions[["p_raw", "p_chosen"]] <= 1)).all().all():
        raise ValueError("Outer probabilities are not finite and bounded")

    metrics: list[dict[str, Any]] = []
    system_specs = [
        ("tuned_lr", "tuned", "p_chosen", "label_default_0_5", "default_0_5", "Nested selected LR; pooled OOF scores come from different fold-specific fitted models and are descriptive."),
        ("tuned_lr", "tuned", "p_chosen", "label_balanced_macro_f1", "balanced_macro_f1", "Nested selected LR; pooled OOF scores come from different fold-specific fitted models and are descriptive."),
        ("tuned_lr", "tuned", "p_chosen", "label_recall_first", "recall_first", "Illustrative recall-first targets are assumptions; pooled OOF scores are descriptive."),
        ("baseline_lr_a", "a", "baseline_a_p_relevant", "baseline_a_label", "default_0_5", "Fixed C=1 unweighted LR baseline evaluated on the same outer folds."),
        ("baseline_lr_b", "b", "baseline_b_p_relevant", "baseline_b_label", "default_0_5", "Fixed C=1 unweighted LR baseline evaluated on the same outer folds."),
        ("baseline_lr_c", "c", "baseline_c_p_relevant", "baseline_c_label", "default_0_5", "Fixed C=1 unweighted LR baseline evaluated on the same outer folds."),
    ]
    for system, embedding, score_col, label_col, policy_name, note in system_specs:
        metrics.append(_outer_metrics(y, outer_predictions[label_col].to_numpy(dtype=np.int8), outer_predictions[score_col].to_numpy(dtype=float), intents, system, embedding, policy_name, "outer_oof_aggregate", note))
        for outer_fold in range(OUTER_SPLITS):
            mask = outer_predictions["outer_fold"].to_numpy() == outer_fold
            metrics.append(_outer_metrics(y[mask], outer_predictions.loc[mask, label_col].to_numpy(dtype=np.int8), outer_predictions.loc[mask, score_col].to_numpy(dtype=float), intents[mask], system, embedding, policy_name, f"outer_fold_{outer_fold}", "Single outer-fold held-out result."))
    llm_pred = (outer_predictions["llm_predicted_label"].to_numpy() == "relevant").astype(np.int8)
    llm_note = "Frozen label-only LLM baseline joined by query_id; ROC-AUC and average precision are undefined without probabilities."
    metrics.append(_outer_metrics(y, llm_pred, None, intents, "llm_train", "", "frozen_label", "outer_oof_aggregate", llm_note))
    metrics.append(_outer_metrics(y, np.ones_like(y), None, intents, "always_relevant", "", "always_relevant", "outer_oof_aggregate", "Constant endpoint; ROC-AUC and average precision are undefined without probabilities."))

    # The final all-training inner development selection is separate from the
    # outer evaluation and is the only source of the blind-test recipe.
    final_inner_splits = _group_splits(np.arange(903), y, groups, INNER_SPLITS, INNER_SEED)
    final_records: list[dict[str, Any]] = []
    for embedding in EMBEDDINGS:
        for c_value in C_VALUES:
            for weight in WEIGHTS:
                aps: list[float] = []
                for inner_train, inner_test in final_inner_splits:
                    model = _fit_raw(embeddings[embedding][inner_train], y[inner_train], c_value, weight)
                    aps.append(float(average_precision_score(y[inner_test], model.predict_proba(embeddings[embedding][inner_test])[:, 1])))
                final_records.append({
                    "selection_scope": "final_all_training",
                    "outer_fold": "final",
                    "embedding": embedding,
                    "C": float(c_value),
                    "class_weight": "None" if weight is None else weight,
                    "mean_inner_average_precision": float(np.mean(aps)),
                    "inner_average_precision_0": aps[0],
                    "inner_average_precision_1": aps[1],
                    "inner_average_precision_2": aps[2],
                    "inner_rows": 903,
                })
    candidate_records.extend(final_records)
    final_selected_record = sorted(final_records, key=lambda record: (-record["mean_inner_average_precision"], abs(np.log(record["C"])), record["embedding"], 0 if record["class_weight"] == "None" else 1))[0]
    final_selected = {"embedding": final_selected_record["embedding"], "C": final_selected_record["C"], "class_weight": None if final_selected_record["class_weight"] == "None" else final_selected_record["class_weight"]}
    final_inner_oof, final_choice, final_sweep = _crossfit_selected(embeddings[final_selected["embedding"]], y, intents, groups, np.arange(903), final_selected, final_inner_splits)
    final_thresholds = final_choice["threshold_details"]
    test_loaded_after_selection = True
    test, test_embeddings = _load_test(data_dir)
    final_raw_model = _fit_raw(embeddings[final_selected["embedding"]], y, final_selected["C"], final_selected["class_weight"])
    final_p_raw = final_raw_model.predict_proba(test_embeddings[final_selected["embedding"]])[:, 1]
    final_calibrated_model: Any = None
    if final_choice["probability_method"] == "sigmoid":
        final_calibrated_model = _fit_sigmoid(embeddings[final_selected["embedding"]], y, np.arange(903), groups, final_selected["C"], final_selected["class_weight"], seed=INNER_SEED, explicit_splits=final_inner_splits)
        final_p_chosen = final_calibrated_model.predict_proba(test_embeddings[final_selected["embedding"]])[:, 1]
    else:
        final_p_chosen = final_p_raw.copy()
    recommended_threshold = float(final_thresholds["recall_first"]["threshold"])
    test_predictions = pd.DataFrame({
        "query_id": test["query_id"].astype(str),
        "predicted_label": np.where(final_p_chosen >= recommended_threshold, "relevant", "irrelevant"),
        "p_relevant": final_p_chosen,
        "p_raw": final_p_raw,
        "p_chosen": final_p_chosen,
        "policy": "recall_first",
        "threshold": recommended_threshold,
        "probability_method": final_choice["probability_method"],
        "selected_embedding": final_selected["embedding"],
        "selected_C": final_selected["C"],
        "selected_class_weight": "None" if final_selected["class_weight"] is None else final_selected["class_weight"],
    })
    if len(test_predictions) != 904 or not test_predictions["query_id"].equals(test["query_id"].astype(str)):
        raise AssertionError("Test prediction order or count is invalid")
    if not np.isfinite(test_predictions[["p_relevant", "p_raw", "p_chosen"]].to_numpy()).all() or not ((test_predictions[["p_relevant", "p_raw", "p_chosen"]] >= 0) & (test_predictions[["p_relevant", "p_raw", "p_chosen"]] <= 1)).all().all():
        raise ValueError("Test probabilities are not finite and bounded")

    splits = pd.DataFrame(split_rows)
    splits["_split_order"] = splits["split_level"].map({"outer": 0, "inner": 1})
    splits["_role_order"] = splits["role"].map({"train": 0, "test": 1})
    splits["_outer_order"] = pd.to_numeric(splits["outer_fold"], errors="coerce")
    splits["_inner_order"] = pd.to_numeric(splits["inner_fold"], errors="coerce").fillna(-1)
    splits = splits.sort_values(["_split_order", "_outer_order", "_inner_order", "_role_order", "row_index"]).drop(columns=["_split_order", "_role_order", "_outer_order", "_inner_order"]).reset_index(drop=True)
    candidates = pd.DataFrame(candidate_records)
    candidates["selection_rank"] = candidates.groupby(["selection_scope", "outer_fold"], dropna=False)["mean_inner_average_precision"].rank(method="first", ascending=False).astype(int)
    # Keep a final-development sweep only; outer sweeps remain represented by
    # their frozen thresholds in fold_choices and are never exported as tuning data.
    final_sweep.insert(0, "selection_scope", "final_all_training")

    protocol_payload = {**protocol, "source_data": {"relative_data_dir": str(data_dir.relative_to(part2_dir.parents[1])) if part2_dir.parents[1] in data_dir.parents else str(data_dir), "source_fingerprint": source_fingerprint, "test_metadata_fingerprint": test_metadata_fingerprint}, "config_fingerprint": config_fingerprint, "cache_fingerprint": cache_fingerprint}
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "joblib": getattr(joblib, "__version__", None),
        "seeds": {"outer": OUTER_SEED, "inner": INNER_SEED},
    }
    checks = {
        "train_rows_903": len(train) == 903,
        "train_ids_unique": train["query_id"].nunique() == 903,
        "labels_valid": set(train["label"]) == {"relevant", "irrelevant"},
        "policy_intent_labels_match": True,
        "embedding_shapes_and_finiteness": True,
        "normalized_training_embeddings": True,
        "duplicate_groups_union_ABC": True,
        "outer_folds_5": len(outer_splits) == 5,
        "outer_group_disjoint": True,
        "inner_group_disjoint": True,
        "shared_outer_split_for_ABC": True,
        "outer_oof_coverage_903": len(outer_predictions) == 903 and outer_predictions["query_id"].nunique() == 903,
        "inner_crossfit_coverage": all(choice["inner_probability_coverage"] for choice in fold_choices) and final_choice["inner_probability_coverage"],
        "calibration_complete_exclusion": all(choice["calibration_excludes_inner_heldout"] for choice in fold_choices) and final_choice["calibration_excludes_inner_heldout"],
        "threshold_endpoints_and_default_included": True,
        "threshold_zero_division_safe": True,
        "outer_holdout_excluded_all_selection": all(choice["outer_holdout_excluded_selection"] for choice in fold_choices),
        "auc_ap_probabilities_only": all(row["score_available"] for row in metrics if row["system"].startswith(("tuned_", "baseline_"))),
        "llm_auc_ap_null": all(row["roc_auc"] is None and row["average_precision"] is None for row in metrics if row["system"] in {"llm_train", "always_relevant"}),
        "test_rows_904": len(test_predictions) == 904,
        "test_order_preserved": test_predictions["query_id"].equals(test["query_id"].astype(str)),
        "test_probabilities_finite_and_bounded": True,
        "test_loaded_after_final_selection": test_loaded_after_selection,
        "joblib_model_saved": False,
    }
    if joblib is not None:
        joblib.dump({"model": final_calibrated_model if final_choice["probability_method"] == "sigmoid" else final_raw_model, "embedding": final_selected["embedding"], "C": final_selected["C"], "class_weight": final_selected["class_weight"], "probability_method": final_choice["probability_method"], "thresholds": {name: values["threshold"] for name, values in final_thresholds.items()}, "protocol": protocol_payload}, outputs / "final_model.joblib")
        checks["joblib_model_saved"] = True

    for frame, name in [(candidates, "candidate_results.csv"), (candidates, "inner_search_records.csv"), (outer_predictions, "outer_predictions.csv"), (pd.DataFrame(metrics), "metrics.csv"), (pd.DataFrame(fold_choices), "fold_choices.csv"), (splits, "splits.csv"), (final_sweep, "threshold_sweep.csv"), (test_predictions, "test_predictions.csv")]:
        frame.to_csv(outputs / name, index=False)
    _write_json(outputs / "configuration.json", protocol_payload)
    _write_json(outputs / "protocol.json", protocol_payload)
    _write_json(outputs / "environment.json", environment)
    _write_json(outputs / "requirements_versions.json", {"runtime": environment, "declared": ["numpy>=2.0", "pandas>=2.2", "scikit-learn>=1.5", "joblib>=1.4"]})
    _write_json(outputs / "artifact_schema.json", _artifact_schema())

    aggregate_metrics = [row for row in metrics if row["evaluation"] == "outer_oof_aggregate"]
    artifact_map = {path.name: str(path.relative_to(part2_dir)) for path in _artifact_paths(outputs) if path.exists()}
    if (outputs / "final_model.joblib").exists():
        artifact_map["final_model.joblib"] = str((outputs / "final_model.joblib").relative_to(part2_dir))
    summary = {
        "stage": "version_1_part2_logistic_regression",
        "cache_hit": False,
        "source_fingerprint": source_fingerprint,
        "test_metadata_fingerprint": test_metadata_fingerprint,
        "config_fingerprint": config_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "selection": {
            "outer_fold_choices": fold_choices,
            "final_embedding": final_selected["embedding"],
            "final_C": final_selected["C"],
            "final_class_weight": "None" if final_selected["class_weight"] is None else final_selected["class_weight"],
            "final_mean_inner_average_precision": final_selected_record["mean_inner_average_precision"],
            "final_probability_method": final_choice["probability_method"],
            "final_brier_raw": final_choice["brier_raw"],
            "final_brier_sigmoid": final_choice["brier_sigmoid"],
            "final_thresholds": {name: values["threshold"] for name, values in final_thresholds.items()},
            "recommended_policy": "recall_first",
            "recommended_threshold": recommended_threshold,
            "development_note": "All final choices are inner-development choices; outer scores remain the held-out estimate of the complete procedure.",
        },
        "config": protocol_payload,
        "counts": {"train_rows": 903, "test_rows": 904, "outer_folds": 5, "inner_folds": 3, **group_info, "medical_train_rows": int(np.sum(intents == "Medical")), "candidate_configurations_per_search": 30},
        "metrics": aggregate_metrics,
        "checks": checks,
        "artifacts": artifact_map,
        "artifact_schema": _artifact_schema(),
        "warnings": {"convergence": convergence_warnings, "coverage": []},
        "limitations": ["Blind test has no supplied truth labels.", "Recall-first Medical and relevant targets are assumptions, not safety guarantees.", "Pooled outer OOF AUC/AP combine scores from fold-specific fitted models and are descriptive.", "The frozen LLM provides labels only, so ROC-AUC/AP are null."],
        "environment": environment,
    }
    _write_json(summary_path, summary)
    return summary


if __name__ == "__main__":  # Small local smoke entry point for maintainers.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part2_dir", type=Path)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.part2_dir, force_recompute=args.force_recompute)
    print(json.dumps({"cache_hit": result.get("cache_hit"), "artifacts": result.get("artifacts")}, indent=2))
