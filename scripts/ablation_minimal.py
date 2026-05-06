"""
Minimal ablation: per-signal contribution and DeBERTa vs RoBERTa comparison.
Reads existing score files - no model inference.

Outputs:
  exports/table_ablation_per_signal.csv
  exports/table_ablation_classifier.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve

ROOT    = Path(__file__).parent.parent
EXPORTS = ROOT / "exports"
ARCHIVE = EXPORTS / "_archived_tables"

# Weights from config (same order as signal columns below)
# [embedding, entropy, classifier, crossencoder, answer_span]
WEIGHTS = np.array([0.15, 0.15, 0.35, 0.15, 0.20])

SIGNAL_COLS = {
    "embedding":    "embedding_score",
    "entropy":      "perplexity_score",
    "classifier":   "deberta_score",
    "crossencoder": "crossencoder_score",
    "answer_span":  "answer_recall_score",
}
NEUTRAL_VALS = {
    "embedding":    0.0,
    "entropy":      0.0,
    "classifier":   0.5,
    "crossencoder": 0.5,
    "answer_span":  0.5,
}


def best_f1_threshold(y_true, y_score) -> tuple[float, float, float, float]:
    """Return (threshold, f1, precision, recall) at the Youden-optimal point."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = np.argmax(j)
    thresh   = thresholds[best_idx]
    y_pred   = (y_score >= thresh).astype(int)
    return thresh, f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)


def weighted_vote(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    cols = list(SIGNAL_COLS.values())
    X = df[cols].values
    return X @ weights


def load_scores() -> pd.DataFrame:
    for p in [ARCHIVE / "prefilter_passage_scores.csv",
              EXPORTS  / "prefilter_passage_scores.csv"]:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("prefilter_passage_scores.csv not found in exports/ or _archived_tables/")


# ablation 1: per-signal contribution

def run_signal_ablation(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy().reset_index(drop=True)
    y_true = test["is_poisoned_passage"].astype(int).values

    rows = []
    signal_names = list(SIGNAL_COLS.keys()) + ["all_signals"]

    for removed in signal_names:
        scores = test.copy()
        if removed != "all_signals":
            scores[SIGNAL_COLS[removed]] = NEUTRAL_VALS[removed]

        agg = weighted_vote(scores, WEIGHTS)
        thresh, f1, prec, rec = best_f1_threshold(y_true, agg)
        rows.append({
            "variant":        "all_signals" if removed == "all_signals" else f"no_{removed}",
            "removed_signal": "none" if removed == "all_signals" else removed,
            "threshold":      round(thresh, 4),
            "f1":             round(f1,     4),
            "precision":      round(prec,   4),
            "recall":         round(rec,    4),
            "f1_delta":       0.0,          # filled below
        })

    result = pd.DataFrame(rows)
    baseline_f1 = result.loc[result["variant"] == "all_signals", "f1"].iloc[0]
    result["f1_delta"] = result["f1"] - baseline_f1
    return result


# ablation 2: classifier backbone comparison

def run_classifier_comparison() -> pd.DataFrame:
    path = ROOT / "results" / "v2" / "models" / "classifier_comparison.json"
    if not path.exists():
        print(f"  WARNING: {path} not found - skipping classifier comparison")
        return pd.DataFrame()

    data = json.loads(path.read_text())
    rows = [
        {"classifier": "DeBERTa-v3-small", "val_f1": round(data.get("deberta_val_f1", float("nan")), 4)},
        {"classifier": "RoBERTa-base",      "val_f1": round(data.get("roberta_val_f1", float("nan")), 4)},
    ]
    return pd.DataFrame(rows)


# main

def main():
    EXPORTS.mkdir(exist_ok=True)

    print("Loading prefilter passage scores...")
    df = load_scores()
    print(f"  {len(df)} passages | test split: {(df['split']=='test').sum()}")

    print("Running per-signal ablation...")
    sig_tbl = run_signal_ablation(df)
    sig_tbl.to_csv(EXPORTS / "table_ablation_per_signal.csv", index=False)
    print(sig_tbl.to_string(index=False))
    print(f"  -> exports/table_ablation_per_signal.csv ({len(sig_tbl)} rows)")

    print("\nRunning classifier backbone comparison...")
    clf_tbl = run_classifier_comparison()
    if not clf_tbl.empty:
        clf_tbl.to_csv(EXPORTS / "table_ablation_classifier.csv", index=False)
        print(clf_tbl.to_string(index=False))
        print(f"  -> exports/table_ablation_classifier.csv ({len(clf_tbl)} rows)")


if __name__ == "__main__":
    main()
