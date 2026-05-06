"""
Phase 10 — Export thesis data to analysis-ready CSVs.

Reads raw v2 experiment outputs and writes 6 canonical CSV files to exports/.
Safe to rerun: overwrites existing files.

Usage:
    python scripts/10_export_data.py [--config config/v2.yaml] [--out exports]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def _load_splits(splits_dir: Path) -> tuple[list, dict, dict]:
    """Return (all_triplets, triplet_by_id, split_label_by_id)."""
    triplet_by_id: dict[str, dict] = {}
    split_label: dict[str, str] = {}
    for split in ("train", "val", "test"):
        p = splits_dir / f"{split}.json"
        if not p.exists():
            continue
        for t in _load_json(p):
            triplet_by_id[t["id"]] = t
            split_label[t["id"]] = split
    return list(triplet_by_id.values()), triplet_by_id, split_label


def _scores_from_record(r: dict) -> dict:
    """Extract faithfulness/answer_relevance/context_relevance safely.

    Must use explicit None checks — scores can be 0.0 which is falsy.
    """
    scores = r.get("scores") or {}

    def _get(top_key: str) -> float | None:
        v = r.get(top_key)
        if v is not None:
            return v
        return scores.get(top_key)

    return {
        "faithfulness_score":     _get("faithfulness"),
        "answer_relevance_score": _get("answer_relevance"),
        "context_relevance_score":_get("context_relevance"),
    }


# ── File 1 — judge_scores_v2.csv ─────────────────────────────────────────────

def build_judge_scores(v2_raw: Path, triplet_by_id: dict, split_label: dict) -> pd.DataFrame:
    """One row per actual scored record from v2 raw JSON files.

    The universe of triplets is derived from the raw JSON files themselves,
    NOT from data/splits/ (which may contain v1 triplets at different noise levels).
    Gap-filling is intentionally omitted: only records with actual scores are included.
    """

    JUDGE_FILES = {
        ("gpt-5.4-nano",   "baseline"):             "baseline_gpt.json",
        ("gemma-4-26b",    "baseline"):              "baseline_gemini.json",
        ("deepseek-v3.2",  "baseline"):              "baseline_deepseek.json",
        ("gpt-5.4-nano",   "postfilter_stat"):       "postfilter_gpt.json",
        ("gemma-4-26b",    "postfilter_stat"):       "postfilter_gemini.json",
        ("deepseek-v3.2",  "postfilter_stat"):       "postfilter_deepseek.json",
        ("gpt-5.4-nano",   "postfilter_llm"):        "postfilter_llm_gpt.json",
        ("gemma-4-26b",    "postfilter_llm"):        "postfilter_llm_gemini.json",
        ("deepseek-v3.2",  "postfilter_llm"):        "postfilter_llm_deepseek.json",
    }

    rows = []

    for (judge, condition), fname in JUDGE_FILES.items():
        fpath = v2_raw / fname
        if not fpath.exists():
            print(f"  WARN: {fname} not found — skipping")
            continue
        records = _load_json(fpath)
        for r in records:
            tid = r.get("triplet_id") or r.get("id")
            if not tid:
                continue
            sc = _scores_from_record(r)
            # Prefer is_poisoned/injection_type/noise_level from the raw record itself;
            # fall back to triplet_by_id only if absent.
            t = triplet_by_id.get(tid, {})
            rows.append({
                "triplet_id":            tid,
                "judge_model":           judge,   # canonical name from JUDGE_FILES
                "condition":             condition,
                "faithfulness_score":    sc["faithfulness_score"],
                "answer_relevance_score":sc["answer_relevance_score"],
                "context_relevance_score":sc["context_relevance_score"],
                "is_poisoned":           r.get("is_poisoned") if r.get("is_poisoned") is not None else t.get("is_poisoned"),
                "injection_type":        r.get("injection_type") or t.get("injection_type"),
                "noise_level":           r.get("noise_level") if r.get("noise_level") is not None else t.get("noise_level"),
                "latency_ms":            r.get("latency_ms", 0.0),
                "split":                 split_label.get(tid),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["judge_model", "condition", "triplet_id"]).reset_index(drop=True)
    return df


# ── File 2 — prefilter_passage_scores.csv ────────────────────────────────────

def build_prefilter_scores(v2_scores: Path, triplet_by_id: dict, split_label: dict) -> pd.DataFrame:
    agg_path = v2_scores / "aggregated_scores.json"
    if not agg_path.exists():
        print(f"  WARN: {agg_path} not found — skipping prefilter export")
        return pd.DataFrame()

    records = _load_json(agg_path)
    rows = []
    for r in records:
        tid = r.get("triplet_id")
        t = triplet_by_id.get(tid, {})
        pidxs = set(t.get("poisoned_passage_indices") or [])
        pidx = r.get("passage_index", 0)
        rows.append({
            "triplet_id":          tid,
            "passage_index":       pidx,
            "embedding_score":     r.get("embedding_score"),
            "perplexity_score":    r.get("entropy_score"),       # renamed
            "deberta_score":       r.get("classifier_score"),    # renamed
            "answer_recall_score": r.get("answer_span_score"),   # renamed
            "crossencoder_score":  r.get("crossencoder_score", 0.0),
            "aggregated_score":    r.get("aggregated_score"),
            "flagged":             r.get("flagged"),
            "is_poisoned_passage": pidx in pidxs,
            "injection_type":      t.get("injection_type"),
            "split":               split_label.get(tid),
        })

    return pd.DataFrame(rows).sort_values(["triplet_id", "passage_index"]).reset_index(drop=True)


# ── File 3 — mistral_triplet_predictions.csv ─────────────────────────────────

def build_mistral_predictions(v2_scores: Path, triplet_by_id: dict) -> pd.DataFrame:
    """Build triplet-level Mistral predictions from passage-level flagged.json.

    llm_prefilter_flagged.json is a dict keyed by "{triplet_id}__p{passage_index}"
    with bool values indicating whether that passage was flagged as suspicious.
    Aggregation: if any passage in a triplet was flagged → triplet predicted_poisoned=True.
    """
    flagged_path = v2_scores / "llm_prefilter_flagged.json"
    if not flagged_path.exists():
        print(f"  WARN: {flagged_path} not found — skipping mistral export")
        return pd.DataFrame()

    data = _load_json(flagged_path)

    # Build triplet-level aggregation
    flagged_by_tid: dict[str, bool] = defaultdict(bool)

    if isinstance(data, dict):
        # Keys are "{triplet_id}__p{index}" — strip the passage suffix
        for key, val in data.items():
            # Split on last "__p" occurrence to get triplet_id
            if "__" in key:
                # Format: "{tid}__p{N}" — take everything before the last "__"
                tid = "__".join(key.split("__")[:-1])
            else:
                tid = key
            flagged_by_tid[tid] = flagged_by_tid[tid] or bool(val)
    elif isinstance(data, list):
        for r in data:
            tid = r.get("triplet_id")
            if tid:
                flagged_by_tid[tid] = flagged_by_tid[tid] or bool(r.get("flagged", False))

    # Build rows only for triplets that actually have scores (triplet_by_id universe)
    rows = []
    for tid, t in triplet_by_id.items():
        rows.append({
            "triplet_id":        tid,
            "is_poisoned":       t.get("is_poisoned"),
            "predicted_poisoned":flagged_by_tid.get(tid, False),
            "mistral_reasoning": None,   # not stored
        })

    df = pd.DataFrame(rows).sort_values("triplet_id").reset_index(drop=True)
    n_flagged = df["predicted_poisoned"].sum()
    print(f"  Mistral: {n_flagged} triplets flagged out of {len(df)} "
          f"({df[df['is_poisoned']]['predicted_poisoned'].sum()} poisoned, "
          f"{df[~df['is_poisoned']]['predicted_poisoned'].sum()} clean)")
    return df


# ── File 4 — justification_analysis.csv ──────────────────────────────────────

def build_justification_analysis(v2_just: Path, triplet_by_id: dict) -> pd.DataFrame:
    FILES = {
        "standard":           "justification_samples.json",
        "standard_postfilter":"justification_samples_postfilter.json",
        "poison_aware":       "justification_samples_poison_aware.json",
    }
    rows = []
    for condition, fname in FILES.items():
        fpath = v2_just / fname
        if not fpath.exists():
            print(f"  WARN: {fpath} not found — skipping condition '{condition}'")
            continue
        for r in _load_json(fpath):
            tid = r.get("triplet_id") or r.get("id")
            t = triplet_by_id.get(tid, {})
            # parse scores — structure varies by prompt type
            sc = r.get("scores") or {}
            def _s(key):
                v = sc.get(key)
                if isinstance(v, dict):
                    return v.get("score")
                return v

            rows.append({
                "triplet_id":            tid,
                "prompt_condition":      condition,
                "judge_model":           r.get("judge_model", "gpt-5.4-nano"),
                "is_poisoned":           t.get("is_poisoned"),
                "injection_type":        t.get("injection_type"),
                "noise_level":           t.get("noise_level"),
                "faithfulness_score":    _s("faithfulness"),
                "answer_relevance_score":_s("answer_relevance"),
                "context_relevance_score":_s("context_relevance"),
                "faithfulness_reason":   sc.get("faithfulness", {}).get("reason") if isinstance(sc.get("faithfulness"), dict) else None,
                "poisoning_detected":    sc.get("poisoning_detected"),
                "suspicious_patterns":   sc.get("suspicious_patterns"),
            })

    return pd.DataFrame(rows).sort_values(["prompt_condition", "triplet_id"]).reset_index(drop=True)


# ── File 5 — dataset_metadata.csv ────────────────────────────────────────────

def build_dataset_metadata(all_triplets: list) -> pd.DataFrame:
    rows = []
    for t in all_triplets:
        twa = t.get("target_wrong_answer") or ""
        rows.append({
            "triplet_id":          t["id"],
            "question":            t.get("question"),
            "answer":              t.get("answer"),
            "injection_type":      t.get("injection_type"),
            "noise_level":         t.get("noise_level"),
            "is_poisoned":         t.get("is_poisoned"),
            "n_passages_original": len((t.get("original_context") or "").split("\n\n")),
            "n_passages_poisoned": len((t.get("poisoned_context") or "").split("\n\n")),
            "poisoned_passage_indices": json.dumps(t.get("poisoned_passage_indices") or []),
            "target_wrong_answer": twa if (t.get("injection_type") == "poisonedrag_style" and t.get("is_poisoned") and twa) else None,
        })

    return pd.DataFrame(rows).sort_values("triplet_id").reset_index(drop=True)


# ── File 6 — prefilter_training_log.csv ──────────────────────────────────────

def build_training_log(results_dir: Path) -> pd.DataFrame:
    rows = []

    # test_eval.json (XGBoost test-set evaluation)
    p = results_dir / "models" / "test_eval.json"
    if p.exists():
        ev = _load_json(p)
        for thresh_row in (ev if isinstance(ev, list) else [ev]):
            rows.append({
                "phase":       "phase3b_xgboost",
                "split":       "test",
                "metric":      thresh_row.get("metric", "xgboost"),
                "value":       thresh_row.get("value") or thresh_row.get("f1"),
                "signal":      "xgboost_aggregator",
                "clean_fpr":   thresh_row.get("clean_fpr"),
            })

    # table_ablation_aggregation.csv
    agg_csv = results_dir / "tables" / "table_ablation_aggregation.csv"
    if agg_csv.exists():
        agg = pd.read_csv(agg_csv)
        for _, r in agg.iterrows():
            for col in ["f1", "precision", "recall", "auc"]:
                if col in r and pd.notna(r[col]):
                    rows.append({
                        "phase":     "phase3c_aggregation",
                        "split":     r.get("split", "val"),
                        "metric":    col,
                        "value":     r[col],
                        "signal":    r.get("method") or r.get("aggregation_method"),
                        "clean_fpr": r.get("clean_fpr") if "clean_fpr" in r else None,
                    })

    # table_4_5_signal_performance.csv (signal ablation)
    sig_csv = results_dir / "tables" / "table_4_5_signal_performance.csv"
    if sig_csv.exists():
        sig = pd.read_csv(sig_csv)
        for _, r in sig.iterrows():
            for col in ["f1", "precision", "recall", "auc"]:
                if col in r and pd.notna(r[col]):
                    rows.append({
                        "phase":     "phase7_signal_ablation",
                        "split":     r.get("split", "test"),
                        "metric":    col,
                        "value":     r[col],
                        "signal":    r.get("signal") or r.get("ablation_signal"),
                        "clean_fpr": None,
                    })

    # classifier_comparison.json (DeBERTa vs RoBERTa)
    clf_p = results_dir / "models" / "classifier_comparison.json"
    if clf_p.exists():
        cmp = _load_json(clf_p)
        if isinstance(cmp, dict):
            cmp = [cmp]
        for entry in cmp:
            rows.append({
                "phase":     "phase3a_classifier",
                "split":     "val",
                "metric":    "f1",
                "value":     entry.get("f1") or entry.get("val_f1"),
                "signal":    entry.get("model") or entry.get("backbone"),
                "clean_fpr": None,
            })

    if not rows:
        print("  WARN: no training log files found — returning empty DataFrame")
        return pd.DataFrame(columns=["phase","split","metric","value","signal","clean_fpr"])

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis data to analysis-ready CSVs.")
    parser.add_argument("--config", default="config/v2.yaml")
    parser.add_argument("--out", default="exports", help="Output directory for CSVs")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--v2-raw", default="results/v2/raw_scores",
                        help="Directory containing baseline/postfilter JSON files")
    parser.add_argument("--v2-scores", default="results/v2/prefilter_scores",
                        help="Directory containing passage-level score JSONs")
    parser.add_argument("--v2-just", default="results/v2/justifications",
                        help="Directory containing justification JSON files")
    parser.add_argument("--results", default="results/v2",
                        help="Root results directory (for tables/models sub-dirs)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir  = ROOT / args.splits
    v2_raw      = ROOT / args.v2_raw
    v2_scores   = ROOT / args.v2_scores
    v2_just     = ROOT / args.v2_just
    results_dir = ROOT / args.results

    print(f"Loading splits from {splits_dir} ...")
    all_triplets, triplet_by_id, split_label = _load_splits(splits_dir)
    print(f"  {len(triplet_by_id)} triplets loaded")

    # File 1
    print("\n[1/6] Building judge_scores_v2.csv ...")
    df1 = build_judge_scores(v2_raw, triplet_by_id, split_label)
    p1 = out_dir / "judge_scores_v2.csv"
    df1.to_csv(p1, index=False)
    print(f"  → {len(df1):,} rows  →  {p1}")

    # File 2
    print("\n[2/6] Building prefilter_passage_scores.csv ...")
    df2 = build_prefilter_scores(v2_scores, triplet_by_id, split_label)
    if not df2.empty:
        p2 = out_dir / "prefilter_passage_scores.csv"
        df2.to_csv(p2, index=False)
        print(f"  → {len(df2):,} rows  →  {p2}")

    # File 3
    print("\n[3/6] Building mistral_triplet_predictions.csv ...")
    df3 = build_mistral_predictions(v2_scores, triplet_by_id)
    if not df3.empty:
        p3 = out_dir / "mistral_triplet_predictions.csv"
        df3.to_csv(p3, index=False)
        print(f"  → {len(df3):,} rows  →  {p3}")

    # File 4
    print("\n[4/6] Building justification_analysis.csv ...")
    df4 = build_justification_analysis(v2_just, triplet_by_id)
    if not df4.empty:
        p4 = out_dir / "justification_analysis.csv"
        df4.to_csv(p4, index=False)
        print(f"  → {len(df4):,} rows  →  {p4}")

    # File 5
    print("\n[5/6] Building dataset_metadata.csv ...")
    df5 = build_dataset_metadata(all_triplets)
    p5 = out_dir / "dataset_metadata.csv"
    df5.to_csv(p5, index=False)
    print(f"  → {len(df5):,} rows  →  {p5}")

    # File 6
    print("\n[6/6] Building prefilter_training_log.csv ...")
    df6 = build_training_log(results_dir)
    p6 = out_dir / "prefilter_training_log.csv"
    df6.to_csv(p6, index=False)
    print(f"  → {len(df6):,} rows  →  {p6}")

    print(f"\nDone — 6 CSV files written to {out_dir}/")


if __name__ == "__main__":
    main()
