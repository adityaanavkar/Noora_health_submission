"""Leakage-aware, grouped nested-validation MLP experiment.

This module writes all run artifacts below the caller-selected Version 3
directory.  Importing it is side-effect free; fitting happens only through
``run_experiment`` or the command-line wrapper.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


EMBEDDINGS = ("a", "b", "c")
EXPECTED_DIMS = {"a": 384, "b": 768, "c": 1536}
N_TRAIN = 903
N_TEST = 904
OUTER_FOLDS = 5
INNER_FOLDS = 3
OUTER_SEED = 42
INNER_SEED = 43
RELEVANT_TARGET = 0.98
MEDICAL_TARGET = 0.99
IMPLEMENTATION_VERSION = "mlp-part3-nested-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    """Predeclared settings.  The 54-row grid is generated from these values."""

    architectures: tuple[tuple[int, ...], ...] = ((32,), (64,), (32, 16))
    alphas: tuple[float, ...] = (1e-2, 1.0, 100.0)
    learning_rates: tuple[float, ...] = (5e-4, 1e-3)
    max_iter: int = 300
    early_stopping: bool = False
    batch_size: int = 64
    tol: float = 1e-4
    solver: str = "adam"
    activation: str = "relu"
    n_jobs: int = 1
    pilot_max_fits: int = 1


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
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if np.isscalar(value):
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(paths: Iterable[Path]) -> str:
    rows = [{"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)} for path in paths]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _data_dir(version_dir: Path) -> Path:
    for parent in (version_dir.resolve(), *version_dir.resolve().parents):
        candidate = parent / "intent-classification-assignment" / "intent-classification-assignment"
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate intent-classification-assignment data")


def _prior_dirs(version_dir: Path) -> tuple[Path, Path]:
    root = version_dir.parent
    v1 = root / "version 1" / "part2" / "outputs"
    v2 = root / "version 2" / "outputs"
    for path in (v1, v2):
        if not path.is_dir():
            raise FileNotFoundError(f"Required predecessor outputs are missing: {path}")
    return v1, v2


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


def _load_training(data_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    train = pd.read_csv(data_dir / "train_queries.csv", dtype="string")
    if list(train.columns) != ["query_id", "intent", "label"] or len(train) != N_TRAIN or train["query_id"].nunique() != N_TRAIN:
        raise ValueError("Training CSV must contain 903 unique query IDs and the declared columns")
    if set(train["label"].dropna()) != {"relevant", "irrelevant"}:
        raise ValueError("Training labels must be exactly relevant and irrelevant")
    embeddings: dict[str, np.ndarray] = {}
    for name in EMBEDDINGS:
        matrix = np.load(data_dir / f"train_embeddings_{name}.npy", allow_pickle=False, mmap_mode="r")
        if matrix.shape != (N_TRAIN, EXPECTED_DIMS[name]) or not np.isfinite(matrix).all():
            raise ValueError(f"Training embedding {name} has the wrong shape or non-finite values")
        if not np.allclose(np.linalg.norm(np.asarray(matrix), axis=1), 1.0, atol=2e-3, rtol=2e-3):
            raise ValueError(f"Training embedding {name} is not normalized")
        embeddings[name] = matrix
    policy = json.loads((data_dir / "label_mapping.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    llm_rows = pd.DataFrame(json.loads((data_dir / "llm_baseline_train_predictions.json").read_text(encoding="utf-8")))
    expected_llm_columns = {"query_id", "predicted_intent", "predicted_label"}
    if set(llm_rows.columns) != expected_llm_columns or len(llm_rows) != N_TRAIN or llm_rows["query_id"].nunique() != N_TRAIN or set(llm_rows["query_id"]) != set(train["query_id"]):
        raise ValueError("Frozen training LLM baseline does not exactly cover training IDs")
    llm = dict(zip(llm_rows["query_id"].astype(str), llm_rows["predicted_label"].astype(str)))
    relevant = set(policy.get("relevant", []))
    irrelevant = set(policy.get("irrelevant", []))
    expected = train["intent"].map(lambda item: "relevant" if item in relevant else "irrelevant")
    if not set(train["intent"]) <= relevant | irrelevant or not np.array_equal(expected.astype(str), train["label"].astype(str)):
        raise ValueError("Training intent labels do not match label_mapping.json")
    return train, embeddings, llm, {"policy": policy, "manifest": manifest}


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


def duplicate_groups(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return connected components of exact duplicate rows across union ABC."""
    uf = _UnionFind(N_TRAIN)
    for name in EMBEDDINGS:
        matrix = np.asarray(embeddings[name])
        seen: dict[bytes, list[int]] = {}
        for row_index in range(N_TRAIN):
            row = np.ascontiguousarray(matrix[row_index])
            matches = seen.setdefault(hashlib.sha256(row.tobytes()).digest(), [])
            for previous in matches:
                if np.array_equal(row, matrix[previous]):
                    uf.union(row_index, previous)
            matches.append(row_index)
    roots = [uf.find(index) for index in range(N_TRAIN)]
    ordered = sorted(set(roots), key=lambda root: min(i for i, item in enumerate(roots) if item == root))
    group_id = {root: index for index, root in enumerate(ordered)}
    groups = np.asarray([group_id[root] for root in roots], dtype=np.int64)
    counts = np.bincount(groups)
    return groups, {"group_count": int(len(ordered)), "duplicate_rows_absorbed": int(np.sum(counts - 1)), "largest_group": int(counts.max())}


def _relative_splits(full_indices: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = {int(value): index for index, value in enumerate(full_indices)}
    return [(np.asarray([positions[int(v)] for v in left]), np.asarray([positions[int(v)] for v in right])) for left, right in splits]


def _fit_with_diagnostics(call: Any) -> tuple[Any, list[warnings.WarningMessage]]:
    """Capture convergence warnings while preserving unrelated warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        value = call()
    for item in caught:
        if not isinstance(item.message, ConvergenceWarning):
            warnings.warn(item.message, item.category, stacklevel=2)
    return value, caught


def grouped_splits(indices: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = []
    for train_local, test_local in splitter.split(np.zeros(len(indices)), y[indices], groups[indices]):
        left, right = np.sort(indices[train_local]), np.sort(indices[test_local])
        if np.intersect1d(groups[left], groups[right]).size:
            raise AssertionError("Grouped fold leakage detected")
        if set(y[left]) != {0, 1} or set(y[right]) != {0, 1}:
            raise ValueError("A grouped fold does not contain both binary classes")
        result.append((left, right))
    return result


def _first_inner_fit_indices(inner_by_outer_fold: dict[int, list[tuple[np.ndarray, np.ndarray]]]) -> np.ndarray:
    """Normalize the first pilot fit partition to a one-dimensional index array."""
    fold_splits = inner_by_outer_fold[0]
    if not isinstance(fold_splits, list) or not fold_splits or not isinstance(fold_splits[0], tuple) or len(fold_splits[0]) != 2:
        raise TypeError("Expected inner splits shaped as {outer_fold: [(fit_indices, holdout_indices), ...]}")
    fit_indices = np.asarray(fold_splits[0][0], dtype=np.int64)
    if fit_indices.ndim != 1:
        raise AssertionError("Pilot fit indices must be one-dimensional")
    return fit_indices


def _load_shared_splits(version_dir: Path, train: pd.DataFrame, groups: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, list[tuple[np.ndarray, np.ndarray]]], pd.DataFrame, dict[str, Any]]:
    _, v2 = _prior_dirs(version_dir)
    split_path = v2 / "splits.csv"
    split = pd.read_csv(split_path)
    required = {"split_level", "outer_fold", "inner_fold", "role", "row_index", "query_id", "group_id", "truth", "label", "intent"}
    if set(split.columns) != required:
        raise ValueError("Version 2 split schema changed")
    v1_split = pd.read_csv(version_dir.parent / "version 1" / "part2" / "outputs" / "splits.csv")
    split_keys = ["split_level", "outer_fold", "inner_fold", "role", "row_index", "query_id", "group_id", "truth", "label", "intent"]
    if not split.sort_values(split_keys, kind="mergesort").reset_index(drop=True).equals(v1_split.sort_values(split_keys, kind="mergesort").reset_index(drop=True)):
        raise AssertionError("Version 1 Part 2 and Version 2 split tables are not identical")
    expected_ids = train["query_id"].astype(str).to_numpy()
    row_indices = split["row_index"].to_numpy(dtype=np.int64)
    if set(row_indices) != set(range(N_TRAIN)):
        raise AssertionError("Split row_index values do not cover training rows 0..902")
    if not np.array_equal(split["query_id"].astype(str).to_numpy(), expected_ids[row_indices]):
        raise AssertionError("Split query IDs do not map to training CSV IDs by row_index")
    computed_group_ids = np.asarray([f"g{item:04d}" for item in groups])
    y = train["label"].eq("relevant").astype(np.int8).to_numpy()
    if not np.array_equal(split["truth"].to_numpy(dtype=np.int8), y[row_indices]):
        raise AssertionError("Split truth values do not map to training labels by row_index")
    if not np.array_equal(split["label"].astype(str).to_numpy(), train["label"].astype(str).to_numpy()[row_indices]):
        raise AssertionError("Split labels do not map to training labels by row_index")
    if not np.array_equal(split["intent"].astype(str).to_numpy(), train["intent"].astype(str).to_numpy()[row_indices]):
        raise AssertionError("Split intents do not map to training intents by row_index")
    if not np.array_equal(split["group_id"].astype(str).to_numpy(), computed_group_ids[row_indices]):
        raise AssertionError("Split group IDs do not map to exact union-ABC groups by row_index")
    outer: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    inner: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for fold in range(OUTER_FOLDS):
        rows = split[(split.split_level == "outer") & (split.outer_fold == fold)]
        outer[fold] = (np.sort(rows.loc[rows.role == "train", "row_index"].to_numpy(dtype=np.int64)), np.sort(rows.loc[rows.role == "test", "row_index"].to_numpy(dtype=np.int64)))
        inner[fold] = []
        for inner_fold in range(INNER_FOLDS):
            rows = split[(split.split_level == "inner") & (split.outer_fold == fold) & (split.inner_fold == inner_fold)]
            inner[fold].append((np.sort(rows.loc[rows.role == "train", "row_index"].to_numpy(dtype=np.int64)), np.sort(rows.loc[rows.role == "test", "row_index"].to_numpy(dtype=np.int64))))
        if np.unique(np.concatenate(outer[fold])).size != N_TRAIN or np.intersect1d(groups[outer[fold][0]], groups[outer[fold][1]]).size:
            raise AssertionError("Outer split coverage or group disjointness failed")
        for left, right in inner[fold]:
            if np.unique(np.concatenate((left, right))).size != len(outer[fold][0]) or np.intersect1d(groups[left], groups[right]).size:
                raise AssertionError("Inner split coverage or group disjointness failed")
    checks = {"shared_v1_v2_split_bytes": _sha256(split_path) == _sha256(version_dir.parent / "version 1" / "part2" / "outputs" / "splits.csv"), "outer_folds_5": True, "inner_folds_3_per_outer": True, "outer_group_disjoint": True, "inner_group_disjoint": True, "outer_coverage_903": True}
    return outer, inner, split, checks


def make_grid(config: ExperimentConfig) -> list[dict[str, Any]]:
    rows = []
    for embedding in EMBEDDINGS:
        for architecture in config.architectures:
            for alpha in config.alphas:
                for learning_rate in config.learning_rates:
                    rows.append({"config_id": f"mlp_{embedding}_{len([r for r in rows if r['embedding'] == embedding]):02d}", "embedding": embedding, "hidden_layer_sizes": json.dumps(list(architecture), separators=(",", ":")), "alpha": alpha, "learning_rate_init": learning_rate, "max_iter": config.max_iter, "early_stopping": config.early_stopping, "batch_size": config.batch_size, "tol": config.tol, "solver": config.solver, "activation": config.activation})
    if len(rows) != 54:
        raise AssertionError("The predeclared MLP universe must contain exactly 54 candidates")
    return rows


def _model(config_row: dict[str, Any], settings: ExperimentConfig) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=tuple(json.loads(config_row["hidden_layer_sizes"])), alpha=float(config_row["alpha"]), learning_rate_init=float(config_row["learning_rate_init"]), max_iter=settings.max_iter, early_stopping=settings.early_stopping, batch_size=settings.batch_size, tol=settings.tol, solver=settings.solver, activation=settings.activation, random_state=OUTER_SEED, shuffle=True, n_iter_no_change=10, validation_fraction=0.1))])


def _fit_sigmoid(X: np.ndarray, y: np.ndarray, fit_indices: np.ndarray, groups: np.ndarray, config_row: dict[str, Any], settings: ExperimentConfig, calibration_splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    splits = calibration_splits or grouped_splits(fit_indices, y, groups, INNER_FOLDS, INNER_SEED)
    calibrated = CalibratedClassifierCV(estimator=_model(config_row, settings), method="sigmoid", cv=_relative_splits(np.sort(fit_indices), splits), ensemble=False, n_jobs=1)
    calibrated, caught = _fit_with_diagnostics(lambda: calibrated.fit(X[np.sort(fit_indices)], y[np.sort(fit_indices)]))
    return calibrated, {
        "convergence_warning_count": int(sum(isinstance(item.message, ConvergenceWarning) for item in caught)),
        "convergence_warning_messages": [str(item.message) for item in caught if isinstance(item.message, ConvergenceWarning)],
    }


def _fit_raw(X: np.ndarray, y: np.ndarray, indices: np.ndarray, config_row: dict[str, Any], settings: ExperimentConfig) -> Pipeline:
    model = _model(config_row, settings)
    model, caught = _fit_with_diagnostics(lambda: model.fit(X[indices], y[indices]))
    mlp = model.named_steps["mlp"]
    return model, {
        "n_iter": int(mlp.n_iter_),
        "loss": float(mlp.loss_),
        "hit_max_iter": bool(mlp.n_iter_ >= settings.max_iter),
        "convergence_warning": bool(any(isinstance(item.message, ConvergenceWarning) for item in caught)),
    }


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    values = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(probabilities, dtype=float))))
    return np.sort(values[np.isfinite(values)])


def _safe(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _threshold_row(y: np.ndarray, probabilities: np.ndarray, intents: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int8)
    tn = int(np.sum((y == 0) & (predicted == 0))); fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0))); tp = int(np.sum((y == 1) & (predicted == 1)))
    medical = intents == "Medical"; medical_n = int(medical.sum()); medical_fn = int(np.sum(medical & (predicted == 0)))
    relevant_recall = float(recall_score(y, predicted, zero_division=0)); medical_recall = _safe(medical_n - medical_fn, medical_n)
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)), "precision": float(precision_score(y, predicted, zero_division=0)), "relevant_recall": relevant_recall, "specificity": _safe(tn, tn + fp), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "medical_fn": medical_fn, "medical_n": medical_n, "medical_recall": medical_recall, "referral_fraction": float(predicted.mean()), "referrals": int(predicted.sum()), "feasible_recall_first": bool(relevant_recall >= RELEVANT_TARGET and (medical_n == 0 or (medical_recall or 0.0) >= MEDICAL_TARGET))}


def choose_thresholds(y: np.ndarray, probabilities: np.ndarray, intents: np.ndarray) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    rows = [_threshold_row(y, probabilities, intents, threshold) for threshold in _threshold_candidates(probabilities)]
    balanced = sorted(rows, key=lambda row: (-row["macro_f1"], -row["relevant_recall"], -row["precision"], -row["threshold"]))[0]
    feasible = [row for row in rows if row["feasible_recall_first"]]
    fallback = not feasible
    recall_first = sorted(feasible or rows, key=(lambda row: (row["fp"], -row["threshold"])) if feasible else (lambda row: (-row["relevant_recall"], -(row["medical_recall"] or 0.0), row["fp"], -row["threshold"])))[0]
    default = next(row for row in rows if row["threshold"] == 0.5)
    selected = {"default_0_5": dict(default), "balanced_macro_f1": dict(balanced), "recall_first": dict(recall_first)}
    selected["recall_first"].update({"fallback_used": fallback, "target_relevant_recall": RELEVANT_TARGET, "target_medical_recall": MEDICAL_TARGET})
    sweep = pd.DataFrame(rows)
    for name, value in selected.items():
        sweep[f"selected_{name}"] = np.isclose(sweep.threshold, value["threshold"], atol=0.0, rtol=0.0)
    return selected, sweep


def _search(Xs: dict[str, np.ndarray], y: np.ndarray, inner: list[tuple[np.ndarray, np.ndarray]], grid: list[dict[str, Any]], settings: ExperimentConfig, scope: str, checkpoint: Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grid_row in grid:
        X = Xs[grid_row["embedding"]]; scores = []; start = time.perf_counter()
        for inner_fold, (fit_idx, hold_idx) in enumerate(inner):
            model, fit_diag = _fit_raw(X, y, fit_idx, grid_row, settings)
            p = model.predict_proba(X[hold_idx])[:, 1]
            scores.append(float(average_precision_score(y[hold_idx], p)))
            if inner_fold == 0:
                diagnostics = []
            diagnostics.append(fit_diag)
        row = dict(grid_row); row.update({"selection_scope": scope, "inner_average_precision_0": scores[0], "inner_average_precision_1": scores[1], "inner_average_precision_2": scores[2], "mean_inner_average_precision": float(np.mean(scores)), "inner_rows": int(sum(len(v[1]) for v in inner)), "runtime_seconds": float(time.perf_counter() - start)})
        for inner_fold, diag in enumerate(diagnostics):
            row[f"inner_n_iter_{inner_fold}"] = diag["n_iter"]
            row[f"inner_loss_{inner_fold}"] = diag["loss"]
            row[f"inner_hit_max_iter_{inner_fold}"] = diag["hit_max_iter"]
            row[f"inner_convergence_warning_{inner_fold}"] = diag["convergence_warning"]
        rows.append(row)
        if checkpoint is not None:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    return pd.DataFrame(rows)


def _select(candidate: pd.DataFrame) -> dict[str, Any]:
    work = candidate.copy()
    work["parameter_count"] = work.hidden_layer_sizes.map(lambda value: sum(json.loads(value)))
    ordered = work.sort_values(["mean_inner_average_precision", "parameter_count", "alpha", "learning_rate_init", "embedding", "config_id"], ascending=[False, True, False, True, True, True], kind="mergesort")
    return ordered.iloc[0].drop(labels=["parameter_count"]).to_dict()


def _crossfit(X: np.ndarray, y: np.ndarray, intents: np.ndarray, groups: np.ndarray, inner: list[tuple[np.ndarray, np.ndarray]], selected: dict[str, Any], settings: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    records = []
    for inner_fold, (fit_idx, hold_idx) in enumerate(inner):
        raw, raw_diag = _fit_raw(X, y, fit_idx, selected, settings); p_raw = raw.predict_proba(X[hold_idx])[:, 1]
        sigmoid, sigmoid_diag = _fit_sigmoid(X, y, fit_idx, groups, selected, settings); p_sigmoid = sigmoid.predict_proba(X[hold_idx])[:, 1]
        records.extend({"row_index": int(row), "inner_fold": inner_fold, "p_raw": float(pr), "p_sigmoid": float(ps), "raw_n_iter": raw_diag["n_iter"], "raw_loss": raw_diag["loss"], "raw_hit_max_iter": raw_diag["hit_max_iter"], "raw_convergence_warning": raw_diag["convergence_warning"], "sigmoid_convergence_warning_count": sigmoid_diag["convergence_warning_count"]} for row, pr, ps in zip(hold_idx, p_raw, p_sigmoid))
    inner_frame = pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)
    expected = np.sort(np.concatenate([part[1] for part in inner]))
    if not np.array_equal(inner_frame.row_index.to_numpy(), expected) or not np.isfinite(inner_frame[["p_raw", "p_sigmoid"]]).all().all():
        raise AssertionError("Inner cross-fit probability coverage or finiteness failed")
    indices = inner_frame.row_index.to_numpy(dtype=np.int64); y_inner = y[indices]
    brier_raw = float(brier_score_loss(y_inner, inner_frame.p_raw)); brier_sigmoid = float(brier_score_loss(y_inner, inner_frame.p_sigmoid)); method = "sigmoid" if brier_sigmoid < brier_raw else "raw"
    thresholds, sweep = choose_thresholds(y_inner, inner_frame[f"p_{method}"].to_numpy(), intents[indices])
    return inner_frame, {"brier_raw": brier_raw, "brier_sigmoid": brier_sigmoid, "probability_method": method, "thresholds": thresholds}, sweep


def _metrics(system: str, embedding: str, policy: str, evaluation: str, y: np.ndarray, p: np.ndarray | None, predicted: np.ndarray, intents: np.ndarray, note: str) -> dict[str, Any]:
    tn = int(np.sum((y == 0) & (predicted == 0))); fp = int(np.sum((y == 0) & (predicted == 1))); fn = int(np.sum((y == 1) & (predicted == 0))); tp = int(np.sum((y == 1) & (predicted == 1)))
    medical = intents == "Medical"; medical_n = int(medical.sum()); medical_fn = int(np.sum(medical & (predicted == 0)))
    result = {"system": system, "embedding": embedding, "policy": policy, "evaluation": evaluation, "n": int(len(y)), "accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)), "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "specificity": _safe(tn, tn + fp), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "medical_fn": medical_fn, "medical_n": medical_n, "medical_recall": _safe(medical_n - medical_fn, medical_n), "referral_fraction": float(predicted.mean()), "referrals": int(predicted.sum()), "score_available": p is not None, "note": note, "roc_auc": None if p is None else float(roc_auc_score(y, p)), "average_precision": None if p is None else float(average_precision_score(y, p))}
    return result


def _hardware() -> dict[str, Any]:
    nvidia = shutil.which("nvidia-smi"); details = None
    if nvidia:
        try:
            details = subprocess.check_output([nvidia, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            details = None
    return {"backend": "sklearn MLPClassifier CPU", "torch_available": importlib.util.find_spec("torch") is not None, "nvidia_smi_available": nvidia is not None, "nvidia_smi_result": details, "gpu_hardware_detected": bool(details)}


def _artifact_schema() -> dict[str, Any]:
    return {"actual_grid.csv": {"purpose": "The complete 54-candidate ABC MLP grid frozen before outer evaluation."}, "candidate_results.csv": {"purpose": "All 54 candidates per outer training scope and final all-training scope with three inner AP values."}, "fold_choices.csv": {"purpose": "Outer-fold choices, calibration Brier scores, thresholds, and exclusion checks."}, "splits.csv": {"purpose": "Exact Version 2 grouped outer/inner split table plus final_inner rows."}, "outer_predictions.csv": {"purpose": "Exactly one held-out prediction row per labeled training ID, with imported LR/RF/LLM columns joined by exact ID."}, "metrics.csv": {"purpose": "MLP outer aggregate/per-fold metrics plus exact-ID imported predecessor and LLM comparisons; LLM score metrics are null."}, "comparative_metrics.csv": {"purpose": "Version 1, Version 2, and Version 3 development metrics with source_version."}, "threshold_sweep.csv": {"purpose": "Final all-training inner-development threshold sweep only."}, "test_predictions.csv": {"purpose": "Exact 904-row source-order predictions produced only after all choices are frozen."}, "final_model.joblib": {"purpose": "Final selected sklearn raw or sigmoid model, scaler, recipe, and validation metadata."}, "summary.json": {"purpose": "Notebook-ready counts, selections, actual metrics, checks, paths, and limitations."}}


def _call_hubble_reporting(root: Path) -> dict[str, Any]:
    """Call the shared reporting owner only after all core outputs exist."""
    try:
        from src.reporting import load_artifacts, write_reports
    except ImportError:
        return {"status": "pending", "message": "src.reporting is not installed yet; core reports were not written."}
    data = load_artifacts(root)
    result = write_reports(data)
    return {"status": "complete", "result": result}


def run_experiment(version_dir: str | Path, *, pilot: bool = False) -> dict[str, Any]:
    """Run the complete experiment, or a one-fit pilot when ``pilot=True``."""
    version_dir = Path(version_dir).resolve(); out = version_dir / "outputs"; out.mkdir(parents=True, exist_ok=True)
    settings = ExperimentConfig(); data_dir = _data_dir(version_dir); train, embeddings, llm, metadata = _load_training(data_dir); y = train.label.eq("relevant").astype(np.int8).to_numpy(); intents = train.intent.astype(str).to_numpy(); groups, group_info = duplicate_groups(embeddings)
    outer, inner, shared_splits, split_checks = _load_shared_splits(version_dir, train, groups)
    if pilot:
        pilot_fit_indices = _first_inner_fit_indices(inner); grid = make_grid(settings); start = time.perf_counter(); _, fit_diag = _fit_raw(embeddings[grid[0]["embedding"]], y, pilot_fit_indices, grid[0], settings); elapsed = time.perf_counter() - start
        payload = {"purpose": "one-fit runtime pilot; no full experiment executed", "backend": _hardware(), "config": asdict(settings), "candidate": grid[0], "fit_rows": len(pilot_fit_indices), "runtime_seconds": elapsed, "fit_diagnostics": fit_diag, "estimated_972_fit_seconds": elapsed * 972}
        _write_json(out / "pilot_runtime.json", payload); return payload
    grid = make_grid(settings); pd.DataFrame(grid).to_csv(out / "actual_grid.csv", index=False)
    Xs = {key: np.asarray(value) for key, value in embeddings.items()}; candidate_frames = []; fold_choices = []; outer_records = []; scope_dir = out / ".checkpoints"; scope_dir.mkdir(exist_ok=True)
    for fold in range(OUTER_FOLDS):
        candidate_path = scope_dir / f"candidate_outer_{fold}.csv"; candidates = _search(Xs, y, inner[fold], grid, settings, f"outer_fold_{fold}", candidate_path); candidate_frames.append(candidates); selected = _select(candidates); embedding = selected["embedding"]; inner_crossfit, calibration, sweep = _crossfit(Xs[embedding], y, intents, groups, inner[fold], selected, settings); outer_train, outer_test = outer[fold]; raw, raw_diag = _fit_raw(Xs[embedding], y, outer_train, selected, settings); sigmoid, sigmoid_diag = _fit_sigmoid(Xs[embedding], y, outer_train, groups, selected, settings, inner[fold]); p_raw = raw.predict_proba(Xs[embedding][outer_test])[:, 1]; p_sigmoid = sigmoid.predict_proba(Xs[embedding][outer_test])[:, 1]; method = calibration["probability_method"]; chosen = p_sigmoid if method == "sigmoid" else p_raw; thresholds = calibration["thresholds"]
        fold_choices.append({"outer_fold": fold, "outer_train_rows": len(outer_train), "outer_test_rows": len(outer_test), "selected_embedding": embedding, "selected_config_id": selected["config_id"], "selected_hidden_layer_sizes": selected["hidden_layer_sizes"], "selected_alpha": selected["alpha"], "selected_learning_rate_init": selected["learning_rate_init"], "selected_mean_inner_average_precision": selected["mean_inner_average_precision"], "brier_raw": calibration["brier_raw"], "brier_sigmoid": calibration["brier_sigmoid"], "probability_method": method, "threshold_default_0_5": thresholds["default_0_5"]["threshold"], "threshold_balanced_macro_f1": thresholds["balanced_macro_f1"]["threshold"], "threshold_recall_first": thresholds["recall_first"]["threshold"], "recall_first_feasible": not thresholds["recall_first"].get("fallback_used", False), "recall_first_fallback_used": thresholds["recall_first"].get("fallback_used", False), "inner_probability_coverage": True, "calibration_excludes_inner_heldout": True, "outer_holdout_excluded_selection": True})
        for row, pr, pc in zip(outer_test, p_raw, chosen):
            rec = {"row_index": int(row), "query_id": str(train.iloc[row].query_id), "outer_fold": fold, "group_id": f"g{groups[row]:04d}", "truth": int(y[row]), "true_label": str(train.iloc[row].label), "intent": str(train.iloc[row].intent), "selected_embedding": embedding, "selected_config_id": selected["config_id"], "probability_method": method, "p_raw": float(pr), "p_chosen": float(pc)}
            for policy in ("default_0_5", "balanced_macro_f1", "recall_first"): rec[f"threshold_{policy}"] = thresholds[policy]["threshold"]; rec[f"label_{policy}"] = int(pc >= thresholds[policy]["threshold"])
            outer_records.append(rec)
        fold_choices[-1].update({"outer_raw_n_iter": raw_diag["n_iter"], "outer_raw_loss": raw_diag["loss"], "outer_raw_hit_max_iter": raw_diag["hit_max_iter"], "outer_raw_convergence_warning": raw_diag["convergence_warning"], "outer_sigmoid_convergence_warning_count": sigmoid_diag["convergence_warning_count"]})
        pd.DataFrame(outer_records).sort_values("row_index").to_csv(scope_dir / "outer_predictions_partial.csv", index=False); pd.DataFrame(fold_choices).to_csv(out / "fold_choices_partial.csv", index=False)
    candidate_results = pd.concat(candidate_frames, ignore_index=True); final_inner = grouped_splits(np.arange(N_TRAIN), y, groups, INNER_FOLDS, INNER_SEED); final_candidates = _search(Xs, y, final_inner, grid, settings, "final_all_training", out / ".checkpoints" / "candidate_final.csv"); candidate_results = pd.concat([candidate_results, final_candidates], ignore_index=True); final = _select(final_candidates); final_embedding = final["embedding"]; final_inner_probs, final_calibration, final_sweep = _crossfit(Xs[final_embedding], y, intents, groups, final_inner, final, settings); final_thresholds = final_calibration["thresholds"]; final_raw, final_raw_diag = _fit_raw(Xs[final_embedding], y, np.arange(N_TRAIN), final, settings); final_sigmoid, final_sigmoid_diag = _fit_sigmoid(Xs[final_embedding], y, np.arange(N_TRAIN), groups, final, settings, final_inner); chosen_model = final_sigmoid if final_calibration["probability_method"] == "sigmoid" else final_raw
    outer_frame = pd.DataFrame(outer_records).sort_values("row_index").reset_index(drop=True); outer_frame["baseline_lr_a_p_relevant"] = pd.NA; outer_frame["baseline_lr_b_p_relevant"] = pd.NA; outer_frame["baseline_lr_c_p_relevant"] = pd.NA; outer_frame["rf_p_chosen_v2"] = pd.NA; outer_frame["llm_predicted_label"] = outer_frame.query_id.map(llm)
    v1_dir, v2_dir = _prior_dirs(version_dir); v1p = pd.read_csv(v1_dir / "outer_predictions.csv"); v2p = pd.read_csv(v2_dir / "outer_predictions.csv");
    for prior, source in ((v1p, "v1"), (v2p, "v2")):
        if not np.array_equal(prior.sort_values("row_index").query_id.astype(str).to_numpy(), train.query_id.astype(str).to_numpy()): raise AssertionError(f"{source} outer IDs do not exactly match training order")
    by_id = v1p.set_index("query_id"); outer_frame["baseline_lr_a_p_relevant"] = outer_frame.query_id.map(by_id.baseline_a_p_relevant); outer_frame["baseline_lr_b_p_relevant"] = outer_frame.query_id.map(by_id.baseline_b_p_relevant); outer_frame["baseline_lr_c_p_relevant"] = outer_frame.query_id.map(by_id.baseline_c_p_relevant); outer_frame["rf_p_chosen_v2"] = outer_frame.query_id.map(v2p.set_index("query_id").p_chosen)
    metrics = []
    for policy in ("default_0_5", "balanced_macro_f1", "recall_first"):
        metrics.append(_metrics("tuned_mlp", "tuned", policy, "outer_oof_aggregate", y, outer_frame.p_chosen.to_numpy(), outer_frame[f"label_{policy}"].to_numpy(), intents, "Nested grouped MLP; pooled outer OOF scores are descriptive."))
        for fold in range(OUTER_FOLDS):
            mask = outer_frame.outer_fold.eq(fold).to_numpy(); metrics.append(_metrics("tuned_mlp", "tuned", policy, f"outer_fold_{fold}", y[mask], outer_frame.loc[mask, "p_chosen"].to_numpy(), outer_frame.loc[mask, f"label_{policy}"].to_numpy(), intents[mask], "Outer held-out fold."))
    llm_pred = outer_frame.llm_predicted_label.eq("relevant").astype(np.int8).to_numpy(); metrics.append(_metrics("llm_train", "", "frozen_label", "outer_oof_aggregate", y, None, llm_pred, intents, "Frozen training LLM labels joined by exact query_id; score metrics are undefined and null."))
    metrics.append(_metrics("always_relevant", "", "always_relevant", "outer_oof_aggregate", y, None, np.ones_like(y), intents, "Constant reference."))
    prior_metrics = pd.read_csv(v2_dir / "comparative_metrics.csv"); prior_metrics["source_version"] = "version2_imported_exact_id_contract"; metrics_frame = pd.DataFrame(metrics); metrics_frame["source_version"] = "version3"; metrics_frame.to_csv(out / "metrics.csv", index=False); pd.concat([prior_metrics, metrics_frame], ignore_index=True, sort=False).to_csv(out / "comparative_metrics.csv", index=False)
    candidate_results.to_csv(out / "candidate_results.csv", index=False); candidate_results.to_csv(out / "inner_search_records.csv", index=False); pd.DataFrame(fold_choices).to_csv(out / "fold_choices.csv", index=False); outer_frame.to_csv(out / "outer_predictions.csv", index=False); final_sweep.to_csv(out / "threshold_sweep.csv", index=False); shared_splits = pd.concat([shared_splits, pd.DataFrame([{"split_level": "final_inner", "outer_fold": "final", "inner_fold": fold, "role": role, "row_index": int(row), "query_id": str(train.iloc[row].query_id), "group_id": f"g{groups[row]:04d}", "truth": int(y[row]), "label": str(train.iloc[row].label), "intent": str(train.iloc[row].intent)} for fold, (left, right) in enumerate(final_inner) for role, values in (("train", left), ("test", right)) for row in values])], ignore_index=True); shared_splits.to_csv(out / "splits.csv", index=False)
    # Test arrays are loaded only after all development selections and threshold choices are frozen.
    test_queries = pd.read_csv(data_dir / "test_queries.csv", dtype="string"); test_embeddings = {name: np.load(data_dir / f"test_embeddings_{name}.npy", allow_pickle=False) for name in EMBEDDINGS};
    if list(test_queries.columns) != ["query_id"] or len(test_queries) != N_TEST or test_queries.query_id.nunique() != N_TEST: raise ValueError("Test metadata must contain 904 unique IDs")
    test_p = chosen_model.predict_proba(test_embeddings[final_embedding])[:, 1]; test_threshold = final_thresholds["recall_first"]["threshold"]; test_predictions = pd.DataFrame({"query_id": test_queries.query_id.astype(str), "p_relevant": test_p, "predicted_label": np.where(test_p >= test_threshold, "relevant", "irrelevant"), "policy": "recall_first", "threshold": test_threshold, "probability_method": final_calibration["probability_method"], "selected_embedding": final_embedding, "selected_config_id": final["config_id"]}); test_predictions.to_csv(out / "test_predictions.csv", index=False)
    if joblib is None: raise ImportError("joblib is required for final_model.joblib export")
    joblib.dump(
        {
            "model": chosen_model,
            "recipe": {
                "embedding": final_embedding,
                "config": final,
                "probability_method": final_calibration["probability_method"],
                "thresholds": {key: value["threshold"] for key, value in final_thresholds.items()},
                "backend": _hardware(),
            },
        },
        out / "final_model.joblib",
    )
    checks = {**group_info, **split_checks, "train_rows_903": len(train) == N_TRAIN, "test_rows_904": len(test_queries) == N_TEST, "outer_coverage_903": len(outer_frame) == N_TRAIN and outer_frame.row_index.is_unique, "finite_outer_probabilities": bool(np.isfinite(outer_frame[["p_raw", "p_chosen"]].to_numpy()).all()), "probabilities_bounded": bool(((outer_frame[["p_raw", "p_chosen"]] >= 0) & (outer_frame[["p_raw", "p_chosen"]] <= 1)).all().all() and ((test_p >= 0) & (test_p <= 1)).all()), "test_order_preserved": bool(np.array_equal(test_predictions.query_id.to_numpy(), test_queries.query_id.astype(str).to_numpy())), "no_test_selection": True, "joblib_model_saved": (out / "final_model.joblib").is_file(), "candidate_count_per_search_54": bool(len(final_candidates) == 54)}
    summary = {"stage": "version_3_mlp", "model_family": "MLPClassifier", "counts": {"train_rows": N_TRAIN, "test_rows": N_TEST, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, **group_info, "candidate_models_per_search": 54}, "selection": {"final_embedding": final_embedding, "final_config_id": final["config_id"], "final_hidden_layer_sizes": json.loads(final["hidden_layer_sizes"]), "final_alpha": final["alpha"], "final_learning_rate_init": final["learning_rate_init"], "final_mean_inner_average_precision": final["mean_inner_average_precision"], "final_probability_method": final_calibration["probability_method"], "final_brier_raw": final_calibration["brier_raw"], "final_brier_sigmoid": final_calibration["brier_sigmoid"], "final_thresholds": {key: value["threshold"] for key, value in final_thresholds.items()}, "recommended_policy": "recall_first", "recommended_threshold": final_thresholds["recall_first"]["threshold"]}, "metrics": metrics, "checks": checks, "hardware": _hardware(), "source_fingerprint": _fingerprint(_source_paths(data_dir)), "test_source_fingerprint": _fingerprint(_test_paths(data_dir)), "predecessor_split_fingerprint": _sha256(v2_dir / "splits.csv"), "config": {"implementation_version": IMPLEMENTATION_VERSION, "settings": asdict(settings), "grid_count": 54, "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED, "duplicate_groups": "exact connected components in union ABC", "selection_score": "mean inner average precision", "threshold_targets": {"relevant": RELEVANT_TARGET, "medical": MEDICAL_TARGET}}, "artifacts": {name: f"outputs/{name}" for name in ("configuration.json", "protocol.json", "summary.json", "actual_grid.csv", "candidate_results.csv", "inner_search_records.csv", "fold_choices.csv", "outer_predictions.csv", "metrics.csv", "comparative_metrics.csv", "threshold_sweep.csv", "splits.csv", "test_predictions.csv", "final_model.joblib", "artifact_schema.json")}, "limitations": ["Blind test has no supplied truth labels.", "Recall-first targets are illustrative development assumptions.", "Pooled outer scores are descriptive estimates of the complete selection procedure.", "Frozen LLM labels have no probabilities, so ROC-AUC/AP are null."]}
    _write_json(out / "summary.json", summary); configuration = {"summary": summary["config"], "source_fingerprint": summary["source_fingerprint"], "test_source_fingerprint": summary["test_source_fingerprint"], "protocol": summary["config"], "backend": summary["hardware"]}; _write_json(out / "configuration.json", configuration); _write_json(out / "protocol.json", configuration); _write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__, "joblib": getattr(joblib, "__version__", "unknown"), "hardware": summary["hardware"]}); _write_json(out / "requirements_versions.json", {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__, "joblib": getattr(joblib, "__version__", "unknown")}); _write_json(out / "artifact_schema.json", _artifact_schema()); summary["reporting"] = _call_hubble_reporting(version_dir); _write_json(out / "summary.json", summary); return summary
