"""
Pre-filter detection metrics: Precision, Recall, F1, AUC-ROC with bootstrap CI.
Per signal and aggregated, broken down by injection_type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset.schema import PrefilterScore


def _bootstrap_metric(y_true, y_score, metric_fn, n_iter=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            stats.append(metric_fn(np.array(y_true)[idx], np.array(y_score)[idx]))
        except Exception:
            pass
    if not stats:
        return 0.0, 0.0, 0.0
    alpha = (1 - ci) / 2
    return (
        float(np.mean(stats)),
        float(np.percentile(stats, alpha * 100)),
        float(np.percentile(stats, (1 - alpha) * 100)),
    )


def youden_threshold(y_true: list, scores: list) -> float:
    """Find threshold maximising Youden's J (sensitivity + specificity - 1)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.5


def compute_detection_metrics(
    scores: list[PrefilterScore],
    signal: str = "aggregated",
) -> dict:
    """Compute detection metrics for one signal on a flat list of PrefilterScores."""
    signal_map = {
        "embedding": lambda s: s.embedding_score,
        "entropy": lambda s: s.entropy_score,
        "classifier": lambda s: s.classifier_score,
        "aggregated": lambda s: s.aggregated_score,
    }
    get_score = signal_map[signal]

    y_true = [int(s.ground_truth_poisoned) for s in scores]
    y_score = [get_score(s) for s in scores]

    if not any(y_true) or all(y_true):
        return {"signal": signal, "error": "no_variance_in_labels"}

    opt_threshold = youden_threshold(y_true, y_score)
    y_pred = [int(s > opt_threshold) for s in y_score]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5

    _, f1_lo, f1_hi = _bootstrap_metric(
        y_true, y_pred, lambda yt, yp: f1_score(yt, yp, zero_division=0)
    )
    _, auc_lo, auc_hi = _bootstrap_metric(
        y_true, y_score, lambda yt, ys: roc_auc_score(yt, ys) if len(set(yt)) > 1 else 0.5
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "signal": signal,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_ci_lower": f1_lo,
        "f1_ci_upper": f1_hi,
        "auc_roc": float(auc),
        "auc_ci_lower": auc_lo,
        "auc_ci_upper": auc_hi,
        "optimal_threshold": float(opt_threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def save_detection_tables(
    all_scores: list[PrefilterScore],
    config: dict,
) -> None:
    out_dir = Path(config["experiment"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = ["embedding", "entropy", "classifier", "aggregated"]
    rows_per_signal = []
    rows_aggregated = []

    injection_types = {
        getattr(s, "injection_type", None) or _infer_injection(s.triplet_id) for s in all_scores
    }

    for inj_type in injection_types:
        subset = [s for s in all_scores if _match_injection(s, inj_type)]
        for signal in signals:
            m = compute_detection_metrics(subset, signal)
            m["injection_type"] = inj_type
            if signal == "aggregated":
                rows_aggregated.append(m)
            else:
                rows_per_signal.append(m)

    if rows_per_signal:
        df = pd.DataFrame(rows_per_signal)
        df.to_csv(out_dir / "table_4_5_signal_performance.csv", index=False)
        print("Saved table_4_5_signal_performance.csv")

    if rows_aggregated:
        df = pd.DataFrame(rows_aggregated)
        df.to_csv(out_dir / "table_4_6_aggregated_performance.csv", index=False)
        print("Saved table_4_6_aggregated_performance.csv")


def _infer_injection(triplet_id: str) -> str:
    if "_rn_" in triplet_id:
        return "random_noise"
    if "_af_" in triplet_id:
        return "adversarial_fact"
    if "_pr_" in triplet_id:
        return "poisonedrag_style"
    return "unknown"


def _match_injection(s: PrefilterScore, inj_type: str) -> bool:
    return _infer_injection(s.triplet_id) == inj_type
