"""
Phase 5: Generate ALL thesis figures, tables, and experiment log.

Usage:
    python scripts/09_generate_report.py --config config/base.yaml [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.analysis.ablation import run_ablation, run_aggregation_comparison
from src.analysis.correlation import save_correlation_table
from src.analysis.plots import generate_all
from src.dataset.schema import PrefilterScore
from src.metrics.detection_metrics import save_detection_tables
from src.metrics.judge_metrics import save_judge_metrics_tables
from src.metrics.system_metrics import save_system_tables
from src.utils.config import load_config

import pandas as pd


def save_context_relevance_fpr_table(all_scores: list[dict], config: dict, threshold: float = 0.5) -> None:
    """
    Aggregate context_relevance FPR across all judges × injection types × noise levels.
    Saves table_cr_fpr_summary.csv — a supplementary table surfacing the context detection
    signal without any new experiments.
    """
    out_dir = Path(config["experiment"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    judges = sorted({s.get("judge_model") for s in all_scores if s.get("judge_model")})
    injection_types = sorted({s.get("injection_type") for s in all_scores if s.get("injection_type")})
    noise_levels = sorted({s.get("noise_level") for s in all_scores if s.get("noise_level") is not None})

    for judge in judges:
        for inj_type in injection_types:
            for nl in noise_levels:
                subset = [
                    s for s in all_scores
                    if s.get("judge_model") == judge
                    and s.get("injection_type") == inj_type
                    and s.get("noise_level") == nl
                ]
                poisoned = [s for s in subset if s.get("is_poisoned")]
                clean = [s for s in subset if not s.get("is_poisoned")]
                if not poisoned:
                    continue
                cr_fpr = sum(1 for s in poisoned if (s.get("context_relevance") or 0) >= threshold) / len(poisoned)
                faith_fpr = sum(1 for s in poisoned if (s.get("faithfulness") or 0) >= threshold) / len(poisoned)
                cr_clean_mean = sum(s.get("context_relevance") or 0 for s in clean) / len(clean) if clean else None
                cr_poisoned_mean = sum(s.get("context_relevance") or 0 for s in poisoned) / len(poisoned)
                rows.append({
                    "judge": judge,
                    "injection_type": inj_type,
                    "noise_level": nl,
                    "cr_fpr": round(cr_fpr, 4),
                    "faithfulness_fpr": round(faith_fpr, 4),
                    "cr_fpr_vs_faith_fpr": round(cr_fpr - faith_fpr, 4),
                    "cr_poisoned_mean": round(cr_poisoned_mean, 4),
                    "cr_clean_mean": round(cr_clean_mean, 4) if cr_clean_mean is not None else None,
                    "n_poisoned": len(poisoned),
                })

    if rows:
        df = pd.DataFrame(rows).sort_values(["judge", "injection_type", "noise_level"])
        path = out_dir / "table_cr_fpr_summary.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}")

        # Print a quick cross-judge summary to stdout
        summary = df.groupby(["judge", "injection_type"])[["cr_fpr", "faithfulness_fpr"]].mean().round(3)
        print("\nContext Relevance FPR vs Faithfulness FPR (mean over noise levels):")
        print(summary.to_string())


def _load_scores(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config["experiment"]["results_dir"])
    raw_scores_dir = results_dir / "raw_scores"

    baseline_gpt = _load_scores(raw_scores_dir / "baseline_gpt.json")
    baseline_gemini = _load_scores(raw_scores_dir / "baseline_gemini.json")
    baseline_deepseek = _load_scores(raw_scores_dir / "baseline_deepseek.json")
    postfilter_gpt = _load_scores(raw_scores_dir / "postfilter_gpt.json")
    postfilter_gemini = _load_scores(raw_scores_dir / "postfilter_gemini.json")
    postfilter_deepseek = _load_scores(raw_scores_dir / "postfilter_deepseek.json")
    postfilter_llm_gpt = _load_scores(raw_scores_dir / "postfilter_llm_gpt.json")
    postfilter_llm_gemini = _load_scores(raw_scores_dir / "postfilter_llm_gemini.json")
    postfilter_llm_deepseek = _load_scores(raw_scores_dir / "postfilter_llm_deepseek.json")

    baseline_scores = baseline_gpt + baseline_gemini + baseline_deepseek
    postfilter_scores = postfilter_gpt + postfilter_gemini + postfilter_deepseek
    all_judge_scores = baseline_scores

    prefilter_raw_path = results_dir / "prefilter_scores" / "aggregated_scores.json"
    prefilter_scores: list[PrefilterScore] = []
    if prefilter_raw_path.exists():
        raw = json.loads(prefilter_raw_path.read_text())
        prefilter_scores = [PrefilterScore(**r) for r in raw]

    bench_path = results_dir / "benchmarks" / "latency_profile.json"
    profiles = json.loads(bench_path.read_text()) if bench_path.exists() else {}

    print(
        f"\nLoaded: {len(all_judge_scores)} judge scores, {len(prefilter_scores)} prefilter scores"
    )
    print(f"Baseline: {len(baseline_scores)}, Post-filter: {len(postfilter_scores)}")

    if all_judge_scores:
        print("\nGenerating judge metrics tables...")
        threshold = config["judges"]["scoring"]["scoring_threshold"]
        save_judge_metrics_tables(all_judge_scores, config, threshold)
        save_correlation_table(all_judge_scores, config)
        print("\nGenerating context relevance FPR summary table...")
        save_context_relevance_fpr_table(all_judge_scores, config, threshold)

    if prefilter_scores:
        print("\nGenerating detection metrics tables...")
        save_detection_tables(prefilter_scores, config)

    if baseline_scores and postfilter_scores:
        print("\nGenerating system comparison tables...")
        save_system_tables(baseline_scores, postfilter_scores, config)

    injection_map = {
        s.get("triplet_id", s.get("id", "")): s.get("injection_type", "unknown")
        for s in baseline_scores
        if s.get("triplet_id") or s.get("id")
    }

    if prefilter_scores:
        ablation_results = run_ablation(prefilter_scores, config, injection_map)
        run_aggregation_comparison(prefilter_scores, config, injection_map)
    else:
        ablation_results = {}

    print("\nGenerating thesis figures...")
    generate_all(
        judge_scores=all_judge_scores,
        baseline_scores=baseline_scores,
        postfilter_scores=postfilter_scores,
        prefilter_scores=prefilter_scores,
        ablation_results=ablation_results,
        profiles=profiles,
        config=config,
    )

    experiment_log = {
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "data_summary": {
            "total_judge_scores": len(all_judge_scores),
            "baseline_gpt": len(baseline_gpt),
            "baseline_gemini": len(baseline_gemini),
            "baseline_deepseek": len(baseline_deepseek),
            "postfilter_gpt": len(postfilter_gpt),
            "postfilter_gemini": len(postfilter_gemini),
            "postfilter_deepseek": len(postfilter_deepseek),
            "postfilter_llm_gpt": len(postfilter_llm_gpt),
            "postfilter_llm_gemini": len(postfilter_llm_gemini),
            "postfilter_llm_deepseek": len(postfilter_llm_deepseek),
            "prefilter_scores": len(prefilter_scores),
        },
        "figures_generated": (
            sorted([p.name for p in (results_dir / "figures").glob("*.png")])
            if (results_dir / "figures").exists()
            else []
        ),
        "tables_generated": (
            sorted([p.name for p in (results_dir / "tables").glob("*.csv")])
            if (results_dir / "tables").exists()
            else []
        ),
    }

    log_path = results_dir / "experiment_log.json"
    log_path.write_text(json.dumps(experiment_log, indent=2))
    print(f"\nSaved experiment log to {log_path}")

    figures = (
        sorted((results_dir / "figures").glob("*.png"))
        if (results_dir / "figures").exists()
        else []
    )
    tables = (
        sorted((results_dir / "tables").glob("*.csv")) if (results_dir / "tables").exists() else []
    )
    print(f"\nFigures: {len(figures)} PNG files in results/figures/")
    print(f"Tables:  {len(tables)} CSV files in results/tables/")

    print("\nRunning residual-aware analysis...")
    run_residual_analysis(
        exports_dir=Path("exports"),
        out_dir=Path("exports")
    )


def run_residual_analysis(exports_dir: Path, out_dir: Path) -> None:
    """Residual-aware analysis: corrects the Prefilter Paradox labelling artifact."""
    import shutil
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    from sklearn.metrics import cohen_kappa_score

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    BLUE   = "#4a7ab8"
    RED    = "#c44e52"
    ORANGE = "#dd8452"
    PURPLE = "#8172b3"
    GREEN  = "#55a868"
    THRESHOLD = 0.5
    DPI = 200

    SOURCE_FILES = {
        "judge_scores_v2.csv",
        "prefilter_passage_scores.csv",
        "mistral_triplet_predictions.csv",
        "dataset_metadata.csv",
        "prefilter_training_log.csv",
        "justification_analysis.csv",
    }

    # ── Step 0 — Archive legacy outputs ──────────────────────────────────────
    print("\n── Step 0: Archiving legacy outputs ──")
    legacy_dir = exports_dir / "legacy_analysis"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    for item in list(exports_dir.iterdir()):
        if item.name in SOURCE_FILES:
            continue
        if item.name == "legacy_analysis":
            continue
        if item.is_file():
            dest = legacy_dir / item.name
            shutil.move(str(item), str(dest))
            print(f"  Archived file: {item.name}")
        elif item.is_dir() and item.name == "figures":
            dest = legacy_dir / "figures"
            if dest.exists():
                shutil.rmtree(str(dest))
            shutil.move(str(item), str(dest))
            print(f"  Archived directory: figures/")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load source files ────────────────────────────────────────────────────
    judge_path   = exports_dir / "judge_scores_v2.csv"
    passage_path = exports_dir / "prefilter_passage_scores.csv"
    mistral_path = exports_dir / "mistral_triplet_predictions.csv"

    missing = [p for p in [judge_path, passage_path, mistral_path] if not p.exists()]
    if missing:
        print(f"WARNING: Missing source files: {[p.name for p in missing]}. Skipping residual analysis.")
        return

    df_judges   = pd.read_csv(judge_path)
    df_passages = pd.read_csv(passage_path)
    df_mistral  = pd.read_csv(mistral_path)

    # Coerce bool columns that might be strings
    for col in ["is_poisoned"]:
        if df_judges[col].dtype == object:
            df_judges[col] = df_judges[col].map({"True": True, "False": False, True: True, False: False})
        df_judges[col] = df_judges[col].astype(bool)

    for col in ["flagged", "is_poisoned_passage"]:
        if df_passages[col].dtype == object:
            df_passages[col] = df_passages[col].map({"True": True, "False": False, True: True, False: False})
        df_passages[col] = df_passages[col].astype(bool)

    for col in ["is_poisoned", "predicted_poisoned"]:
        if df_mistral[col].dtype == object:
            df_mistral[col] = df_mistral[col].map({"True": True, "False": False, True: True, False: False})
        df_mistral[col] = df_mistral[col].astype(bool)

    # Drop rows with null faithfulness
    before = len(df_judges)
    df_judges = df_judges[df_judges["faithfulness_score"].notna()].copy()
    print(f"  Dropped {before - len(df_judges)} rows with null faithfulness_score. Remaining: {len(df_judges)}")

    # ── Step 1 — Build joined frame ──────────────────────────────────────────
    print("\n── Step 1: Building joined frame ──")

    trip_agg = (
        df_passages
        .groupby("triplet_id")
        .apply(lambda g: pd.Series({
            "n_orig_poisoned":      g["is_poisoned_passage"].sum(),
            "n_flagged_poisoned":   (g["is_poisoned_passage"] & g["flagged"]).sum(),
        }))
        .reset_index()
    )
    trip_agg["n_residual_poisoned_stat"] = trip_agg["n_orig_poisoned"] - trip_agg["n_flagged_poisoned"]

    print(f"  Passage-level triplet aggregates: {len(trip_agg)} triplets")

    df = df_judges.merge(trip_agg, on="triplet_id", how="left")
    # Triplets not in passage scores get 0
    for col in ["n_orig_poisoned", "n_flagged_poisoned", "n_residual_poisoned_stat"]:
        df[col] = df[col].fillna(0).astype(int)

    print(f"  After merging passage aggregates: {len(df)} rows")

    # Merge Mistral predictions
    mistral_sub = df_mistral[["triplet_id", "predicted_poisoned"]].drop_duplicates("triplet_id")
    df = df.merge(mistral_sub, on="triplet_id", how="left")

    missing_mistral = df["predicted_poisoned"].isna().sum()
    if missing_mistral > 0:
        print(f"  WARNING: {missing_mistral} rows missing Mistral prediction — filling with False")
        df["predicted_poisoned"] = df["predicted_poisoned"].fillna(False).astype(bool)
    else:
        df["predicted_poisoned"] = df["predicted_poisoned"].astype(bool)

    # Derived boolean columns
    df["effectively_poisoned_stat"] = df["is_poisoned"] & (df["n_residual_poisoned_stat"] > 0)
    df["filter_cleaned_stat"]       = df["is_poisoned"] & (df["n_residual_poisoned_stat"] == 0)
    df["originally_clean"]          = ~df["is_poisoned"]

    df["mistral_correct"]       = df["is_poisoned"] & df["predicted_poisoned"]
    df["mistral_missed"]        = df["is_poisoned"] & ~df["predicted_poisoned"]
    df["mistral_false_alarm"]   = ~df["is_poisoned"] & df["predicted_poisoned"]
    df["mistral_true_negative"] = ~df["is_poisoned"] & ~df["predicted_poisoned"]

    # Verify partitions
    partition_stat  = (df["effectively_poisoned_stat"].astype(int) +
                       df["filter_cleaned_stat"].astype(int) +
                       df["originally_clean"].astype(int))
    partition_mistr = (df["mistral_correct"].astype(int) +
                       df["mistral_missed"].astype(int) +
                       df["mistral_false_alarm"].astype(int) +
                       df["mistral_true_negative"].astype(int))

    if not (partition_stat == 1).all():
        print("  FAIL: stat partition does not sum to 1 for all rows")
    else:
        print("  PASS: stat partition sums to 1 for all rows")

    if not (partition_mistr == 1).all():
        print("  FAIL: mistral partition does not sum to 1 for all rows")
    else:
        print("  PASS: mistral partition sums to 1 for all rows")

    print("\nHead(3):")
    print(df[["triplet_id","judge_model","condition","is_poisoned","effectively_poisoned_stat",
              "filter_cleaned_stat","originally_clean","mistral_correct","mistral_missed"]].head(3).to_string())

    bool_cols = ["effectively_poisoned_stat","filter_cleaned_stat","originally_clean",
                 "mistral_correct","mistral_missed","mistral_false_alarm","mistral_true_negative"]
    for c in bool_cols:
        print(f"  {c}: True={df[c].sum()}, False={(~df[c]).sum()}")
    print(f"  Total rows: {len(df)}")

    out_path = exports_dir / "joined_with_residual_labels.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # ── Step 2 — Master headline table ───────────────────────────────────────
    print("\n── Step 2: Master headline table ──")

    METRICS = {
        "faithfulness":       "faithfulness_score",
        "context_relevance":  "context_relevance_score",
        "answer_relevance":   "answer_relevance_score",
    }

    def _fpr(series: pd.Series) -> float:
        valid = series.dropna()
        if len(valid) == 0:
            return float("nan")
        return (valid >= THRESHOLD).mean()

    headline_rows = []

    # Pre-compute baseline FPR lookup
    baseline_fpr_lookup: dict[tuple, float] = {}
    for judge in df["judge_model"].unique():
        for inj in df["injection_type"].unique():
            for metric_short, metric_col in METRICS.items():
                sub = df[(df["judge_model"] == judge) &
                         (df["condition"] == "baseline") &
                         (df["injection_type"] == inj) &
                         (df["is_poisoned"] == True)]
                baseline_fpr_lookup[(judge, inj, metric_short)] = _fpr(sub[metric_col])

    for judge in df["judge_model"].unique():
        for inj in df["injection_type"].unique():
            for metric_short, metric_col in METRICS.items():
                base_fpr = baseline_fpr_lookup.get((judge, inj, metric_short), float("nan"))

                # baseline
                sub_b = df[(df["judge_model"] == judge) & (df["condition"] == "baseline") &
                           (df["injection_type"] == inj) & (df["is_poisoned"] == True)]
                fpr_b = _fpr(sub_b[metric_col])
                headline_rows.append({
                    "judge_model": judge, "condition": "baseline",
                    "injection_type": inj, "metric": metric_short,
                    "n": len(sub_b[metric_col].dropna()),
                    "mean_score": sub_b[metric_col].mean(),
                    "fpr": fpr_b, "tnr": 1 - fpr_b,
                    "delta_fpr_vs_baseline": 0.0,
                    "n_total": None, "mean_score_total": None, "fpr_total": None, "tnr_total": None,
                    "n_mistral_correct": None, "mean_score_mistral_correct": None, "fpr_mistral_correct": None,
                    "n_mistral_missed": None, "mean_score_mistral_missed": None, "fpr_mistral_missed": None,
                    "delta_fpr_mistral_missed_vs_baseline": None,
                })

                # postfilter_stat
                sub_s = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                           (df["injection_type"] == inj) & (df["effectively_poisoned_stat"] == True)]
                fpr_s = _fpr(sub_s[metric_col])
                headline_rows.append({
                    "judge_model": judge, "condition": "postfilter_stat",
                    "injection_type": inj, "metric": metric_short,
                    "n": len(sub_s[metric_col].dropna()),
                    "mean_score": sub_s[metric_col].mean(),
                    "fpr": fpr_s, "tnr": 1 - fpr_s,
                    "delta_fpr_vs_baseline": fpr_s - base_fpr,
                    "n_total": None, "mean_score_total": None, "fpr_total": None, "tnr_total": None,
                    "n_mistral_correct": None, "mean_score_mistral_correct": None, "fpr_mistral_correct": None,
                    "n_mistral_missed": None, "mean_score_mistral_missed": None, "fpr_mistral_missed": None,
                    "delta_fpr_mistral_missed_vs_baseline": None,
                })

                # postfilter_llm
                sub_l_total  = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                                  (df["injection_type"] == inj) & (df["is_poisoned"] == True)]
                sub_l_mc     = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                                  (df["injection_type"] == inj) & (df["mistral_correct"] == True)]
                sub_l_mm     = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                                  (df["injection_type"] == inj) & (df["mistral_missed"] == True)]
                fpr_total = _fpr(sub_l_total[metric_col])
                fpr_mc    = _fpr(sub_l_mc[metric_col])
                fpr_mm    = _fpr(sub_l_mm[metric_col])
                headline_rows.append({
                    "judge_model": judge, "condition": "postfilter_llm",
                    "injection_type": inj, "metric": metric_short,
                    "n": len(sub_l_total[metric_col].dropna()),
                    "mean_score": sub_l_total[metric_col].mean(),
                    "fpr": fpr_total, "tnr": 1 - fpr_total,
                    "delta_fpr_vs_baseline": fpr_total - base_fpr,
                    "n_total": len(sub_l_total[metric_col].dropna()),
                    "mean_score_total": sub_l_total[metric_col].mean(),
                    "fpr_total": fpr_total, "tnr_total": 1 - fpr_total,
                    "n_mistral_correct": len(sub_l_mc[metric_col].dropna()),
                    "mean_score_mistral_correct": sub_l_mc[metric_col].mean(),
                    "fpr_mistral_correct": fpr_mc,
                    "n_mistral_missed": len(sub_l_mm[metric_col].dropna()),
                    "mean_score_mistral_missed": sub_l_mm[metric_col].mean(),
                    "fpr_mistral_missed": fpr_mm,
                    "delta_fpr_mistral_missed_vs_baseline": fpr_mm - base_fpr,
                })

    df_headline = pd.DataFrame(headline_rows)
    df_headline.to_csv(exports_dir / "table_master_headline.csv", index=False)
    print(f"  Rows: {len(df_headline)} (expected 81)")
    print(df_headline.head(5).to_string())

    # ── Step 3 — Per-noise breakdown ─────────────────────────────────────────
    print("\n── Step 3: Per-noise breakdown ──")

    noise_rows = []
    for judge in df["judge_model"].unique():
        for inj in df["injection_type"].unique():
            for nl in sorted(df["noise_level"].dropna().unique()):
                for metric_short, metric_col in METRICS.items():
                    base_fpr = _fpr(df[(df["judge_model"] == judge) & (df["condition"] == "baseline") &
                                       (df["injection_type"] == inj) & (df["noise_level"] == nl) &
                                       (df["is_poisoned"] == True)][metric_col])

                    sub_b = df[(df["judge_model"] == judge) & (df["condition"] == "baseline") &
                               (df["injection_type"] == inj) & (df["noise_level"] == nl) & (df["is_poisoned"] == True)]
                    fpr_b = _fpr(sub_b[metric_col])
                    noise_rows.append({
                        "judge_model": judge, "condition": "baseline",
                        "injection_type": inj, "noise_level": nl, "metric": metric_short,
                        "n": len(sub_b[metric_col].dropna()),
                        "mean_score": sub_b[metric_col].mean(),
                        "fpr": fpr_b, "tnr": 1 - fpr_b, "delta_fpr_vs_baseline": 0.0,
                        "fpr_mistral_missed": None, "delta_fpr_mistral_missed_vs_baseline": None,
                    })

                    sub_s = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                               (df["injection_type"] == inj) & (df["noise_level"] == nl) &
                               (df["effectively_poisoned_stat"] == True)]
                    fpr_s = _fpr(sub_s[metric_col])
                    noise_rows.append({
                        "judge_model": judge, "condition": "postfilter_stat",
                        "injection_type": inj, "noise_level": nl, "metric": metric_short,
                        "n": len(sub_s[metric_col].dropna()),
                        "mean_score": sub_s[metric_col].mean(),
                        "fpr": fpr_s, "tnr": 1 - fpr_s, "delta_fpr_vs_baseline": fpr_s - base_fpr,
                        "fpr_mistral_missed": None, "delta_fpr_mistral_missed_vs_baseline": None,
                    })

                    sub_l_total = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                                     (df["injection_type"] == inj) & (df["noise_level"] == nl) &
                                     (df["is_poisoned"] == True)]
                    sub_l_mm    = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                                     (df["injection_type"] == inj) & (df["noise_level"] == nl) &
                                     (df["mistral_missed"] == True)]
                    fpr_total = _fpr(sub_l_total[metric_col])
                    fpr_mm    = _fpr(sub_l_mm[metric_col])
                    noise_rows.append({
                        "judge_model": judge, "condition": "postfilter_llm",
                        "injection_type": inj, "noise_level": nl, "metric": metric_short,
                        "n": len(sub_l_total[metric_col].dropna()),
                        "mean_score": sub_l_total[metric_col].mean(),
                        "fpr": fpr_total, "tnr": 1 - fpr_total,
                        "delta_fpr_vs_baseline": fpr_total - base_fpr,
                        "fpr_mistral_missed": fpr_mm,
                        "delta_fpr_mistral_missed_vs_baseline": fpr_mm - base_fpr,
                    })

    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv(exports_dir / "table_per_noise_breakdown.csv", index=False)
    print(f"  Rows: {len(df_noise)} (expected ≤324)")

    # ── Step 4 — Filter behaviour comparison ─────────────────────────────────
    print("\n── Step 4: Filter behaviour comparison ──")

    # Use baseline condition to get one row per triplet per judge (avoid duplication)
    baseline_df = df[df["condition"] == "baseline"].drop_duplicates("triplet_id")

    filter_rows = []
    for inj in sorted(df["injection_type"].unique()):
        trip_inj = baseline_df[baseline_df["injection_type"] == inj]

        poisoned_trips = trip_inj[trip_inj["is_poisoned"] == True]
        clean_trips    = trip_inj[trip_inj["is_poisoned"] == False]

        n_poisoned_trips = len(poisoned_trips)

        # For stat filter passage-level metrics, use the trip_agg aligned data
        # average n_orig_poisoned and n_flagged_poisoned per triplet
        mean_orig_poisoned = poisoned_trips["n_orig_poisoned"].mean()
        mean_flagged       = poisoned_trips["n_flagged_poisoned"].mean()
        mean_residual      = poisoned_trips["n_residual_poisoned_stat"].mean()
        n_fully_cleaned    = poisoned_trips["filter_cleaned_stat"].sum()
        n_residual_left    = poisoned_trips["effectively_poisoned_stat"].sum()

        # Passage-level recall on test split
        pass_test_inj = df_passages[(df_passages["injection_type"] == inj) &
                                    (df_passages["split"] == "test") &
                                    (df_passages["is_poisoned_passage"] == True)]
        if len(pass_test_inj) > 0:
            stat_recall_test = pass_test_inj["flagged"].sum() / len(pass_test_inj)
        else:
            stat_recall_test = float("nan")

        # All passages per triplet (mean)
        pass_inj = df_passages[df_passages["injection_type"] == inj]
        mean_passages_per_trip = pass_inj.groupby("triplet_id").size().mean()

        # LLM filter metrics (from mistral, not baseline_df)
        mistr_inj = df_mistral.copy()
        # get injection_type from baseline_df
        trip_inj_map = baseline_df.set_index("triplet_id")["injection_type"].to_dict()
        is_poisoned_map = baseline_df.set_index("triplet_id")["is_poisoned"].to_dict()
        mistr_inj["injection_type"] = mistr_inj["triplet_id"].map(trip_inj_map)
        mistr_inj["is_poisoned_trip"] = mistr_inj["triplet_id"].map(is_poisoned_map)
        mistr_inj_sub = mistr_inj[mistr_inj["injection_type"] == inj]
        mistr_poisoned = mistr_inj_sub[mistr_inj_sub["is_poisoned_trip"] == True]
        mistr_clean    = mistr_inj_sub[mistr_inj_sub["is_poisoned_trip"] == False]
        n_caught   = (mistr_poisoned["predicted_poisoned"] == True).sum()
        n_missed_m = (mistr_poisoned["predicted_poisoned"] == False).sum()
        mistral_recall = n_caught / (n_caught + n_missed_m) if (n_caught + n_missed_m) > 0 else float("nan")
        n_clean_flagged = (mistr_clean["predicted_poisoned"] == True).sum()
        mistral_clean_fpr = n_clean_flagged / len(mistr_clean) if len(mistr_clean) > 0 else float("nan")

        filter_rows.append({
            "injection_type": inj,
            "filter_mechanism": "statistical (XGBoost, passage-level)",
            "n_triplets_originally_poisoned": n_poisoned_trips,
            "mean_passages_per_triplet": round(mean_passages_per_trip, 2),
            "mean_originally_poisoned_passages_per_triplet": round(mean_orig_poisoned, 2),
            "mean_passages_caught_by_stat_filter": round(mean_flagged, 2),
            "mean_passages_remaining_after_stat_filter": round(mean_residual, 2),
            "n_triplets_fully_cleaned_by_stat_filter": int(n_fully_cleaned),
            "n_triplets_with_residual_after_stat_filter": int(n_residual_left),
            "stat_filter_recall_on_test_split": round(stat_recall_test, 4) if not np.isnan(stat_recall_test) else None,
            "n_triplets_caught_by_mistral_llm_filter": None,
            "n_triplets_missed_by_mistral_llm_filter": None,
            "mistral_recall": None,
            "n_clean_triplets_flagged_by_mistral": None,
            "mistral_clean_fpr": None,
        })
        filter_rows.append({
            "injection_type": inj,
            "filter_mechanism": "LLM (Mistral, triplet-level)",
            "n_triplets_originally_poisoned": n_poisoned_trips,
            "mean_passages_per_triplet": None,
            "mean_originally_poisoned_passages_per_triplet": None,
            "mean_passages_caught_by_stat_filter": None,
            "mean_passages_remaining_after_stat_filter": None,
            "n_triplets_fully_cleaned_by_stat_filter": None,
            "n_triplets_with_residual_after_stat_filter": None,
            "stat_filter_recall_on_test_split": None,
            "n_triplets_caught_by_mistral_llm_filter": int(n_caught),
            "n_triplets_missed_by_mistral_llm_filter": int(n_missed_m),
            "mistral_recall": round(mistral_recall, 4) if not np.isnan(mistral_recall) else None,
            "n_clean_triplets_flagged_by_mistral": int(n_clean_flagged),
            "mistral_clean_fpr": round(mistral_clean_fpr, 4) if not np.isnan(mistral_clean_fpr) else None,
        })

    df_filter = pd.DataFrame(filter_rows)
    df_filter.to_csv(exports_dir / "table_filter_behaviour.csv", index=False)
    print(df_filter.to_string())

    # ── Step 5 — Cleaned-vs-clean calibration ───────────────────────────────
    print("\n── Step 5: Cleaned-vs-clean calibration ──")

    cal_rows = []
    for judge in df["judge_model"].unique():
        for inj in df["injection_type"].unique():
            for metric_short, metric_col in METRICS.items():
                sub_fc = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                            (df["injection_type"] == inj) & (df["filter_cleaned_stat"] == True)]
                sub_oc = df[(df["judge_model"] == judge) & (df["condition"] == "baseline") &
                            (df["injection_type"] == inj) & (df["originally_clean"] == True)]
                ms_fc = sub_fc[metric_col].mean()
                ms_oc = sub_oc[metric_col].mean()
                delta  = ms_fc - ms_oc if (not pd.isna(ms_fc) and not pd.isna(ms_oc)) else float("nan")
                cal_rows.append({
                    "judge_model": judge, "injection_type": inj, "metric": metric_short,
                    "mean_score_filter_cleaned": ms_fc,
                    "mean_score_originally_clean_baseline": ms_oc,
                    "delta": delta,
                    "n_filter_cleaned": len(sub_fc[metric_col].dropna()),
                    "n_originally_clean": len(sub_oc[metric_col].dropna()),
                    "note": "LLM filter not included: does not modify content",
                })

    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_csv(exports_dir / "table_cleaned_vs_clean_calibration.csv", index=False)
    print(f"  Rows: {len(df_cal)}")

    # ── Step 6 — Mistral-judge agreement ─────────────────────────────────────
    print("\n── Step 6: Mistral-judge agreement ──")

    agree_rows = []
    for judge in df["judge_model"].unique():
        for inj in df["injection_type"].unique():
            for metric_short, metric_col in METRICS.items():
                sub = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                         (df["injection_type"] == inj) & (df["is_poisoned"] == True) &
                         df[metric_col].notna()]
                if len(sub) == 0:
                    continue

                judge_faithful       = (sub[metric_col] >= THRESHOLD)
                mistral_flagged      = sub["predicted_poisoned"].astype(bool)
                judge_unfaithful     = ~judge_faithful
                mistral_missed_mask  = ~mistral_flagged

                a = (mistral_flagged & judge_faithful).sum()
                b = (mistral_flagged & judge_unfaithful).sum()
                c = (mistral_missed_mask & judge_faithful).sum()
                d = (mistral_missed_mask & judge_unfaithful).sum()

                kappa = float("nan")
                if len(sub) >= 2 and len(mistral_flagged.unique()) > 1 and len(judge_unfaithful.unique()) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            kappa = cohen_kappa_score(mistral_flagged, judge_unfaithful)
                        except Exception:
                            pass

                jrwmc  = b / (a + b) if (a + b) > 0 else None
                jrwmm  = d / (c + d) if (c + d) > 0 else None

                agree_rows.append({
                    "judge_model": judge, "injection_type": inj, "metric": metric_short,
                    "a_mistral_flagged_judge_faithful": int(a),
                    "b_mistral_flagged_judge_unfaithful": int(b),
                    "c_mistral_missed_judge_faithful": int(c),
                    "d_mistral_missed_judge_unfaithful": int(d),
                    "n_total": int(a + b + c + d),
                    "cohen_kappa": kappa,
                    "judge_recall_when_mistral_correct": jrwmc,
                    "judge_recall_when_mistral_missed": jrwmm,
                })

    df_agree = pd.DataFrame(agree_rows)
    df_agree.to_csv(exports_dir / "table_mistral_judge_agreement.csv", index=False)
    print(f"  Rows: {len(df_agree)}")

    # ── Step 7 — McNemar tests ────────────────────────────────────────────────
    print("\n── Step 7: McNemar tests ──")

    mcnemar_rows = []
    for judge in df["judge_model"].unique():
        for inj in df["injection_type"].unique():
            for metric_short, metric_col in METRICS.items():
                sub_b = df[(df["judge_model"] == judge) & (df["condition"] == "baseline") &
                           (df["injection_type"] == inj) & (df["effectively_poisoned_stat"] == True) &
                           df[metric_col].notna()][["triplet_id", metric_col]].rename(columns={metric_col: "score_baseline"})
                sub_s = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                           (df["injection_type"] == inj) & (df["effectively_poisoned_stat"] == True) &
                           df[metric_col].notna()][["triplet_id", metric_col]].rename(columns={metric_col: "score_stat"})

                paired = sub_b.merge(sub_s, on="triplet_id")
                n_paired = len(paired)

                n10 = ((paired["score_baseline"] >= THRESHOLD) & (paired["score_stat"] < THRESHOLD)).sum()
                n01 = ((paired["score_baseline"] < THRESHOLD) & (paired["score_stat"] >= THRESHOLD)).sum()

                if (n10 + n01) < 10:
                    chi2_val = float("nan")
                    p_val    = float("nan")
                else:
                    chi2_val = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
                    p_val    = scipy_stats.chi2.sf(chi2_val, df=1)

                mcnemar_rows.append({
                    "judge_model": judge, "injection_type": inj, "metric": metric_short,
                    "n_paired": n_paired, "n10": int(n10), "n01": int(n01),
                    "chi2": chi2_val, "p_value": p_val,
                })

    df_mcnemar = pd.DataFrame(mcnemar_rows)
    df_mcnemar.to_csv(exports_dir / "table_mcnemar_corrected.csv", index=False)
    print(f"  Rows: {len(df_mcnemar)}")

    # ── Step 8 — Sanity verification ─────────────────────────────────────────
    print("\n── Step 8: Sanity verification ──")

    baseline_only = df[df["condition"] == "baseline"]
    for inj in sorted(df["injection_type"].unique()):
        sub = baseline_only[baseline_only["injection_type"] == inj]
        poisoned_rows = sub[sub["is_poisoned"] == True]
        eff_frac = poisoned_rows["effectively_poisoned_stat"].mean() if len(poisoned_rows) > 0 else float("nan")
        print(f"  {inj}: effectively_poisoned_stat fraction = {eff_frac:.3f}")

    LEGACY_FPR = {
        ("gpt-5.4-nano", "faithfulness"):   0.555,
        ("gemma-4-26b",  "faithfulness"):   0.699,
        ("deepseek-v3.2","faithfulness"):   0.412,
    }
    print("\n  Baseline FPR verification (faithfulness, all injection types combined):")
    for (judge, metric_short), legacy_val in LEGACY_FPR.items():
        sub = baseline_only[(baseline_only["judge_model"] == judge) & (baseline_only["is_poisoned"] == True)]
        computed = _fpr(sub["faithfulness_score"])
        diff = abs(computed - legacy_val)
        status = "PASS" if diff <= 0.01 else "FAIL (diff={:.4f})".format(diff)
        print(f"  [{status}] {judge} faithfulness FPR: computed={computed:.3f}, legacy={legacy_val:.3f}")

    print("\n  Corrected stat-filter ΔFPR (faithfulness, all injections combined):")
    for judge in ["gpt-5.4-nano", "gemma-4-26b", "deepseek-v3.2"]:
        sub_b = baseline_only[(baseline_only["judge_model"] == judge) & (baseline_only["is_poisoned"] == True)]
        sub_s = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                   (df["effectively_poisoned_stat"] == True)]
        fpr_b = _fpr(sub_b["faithfulness_score"])
        fpr_s = _fpr(sub_s["faithfulness_score"])
        delta = fpr_s - fpr_b
        print(f"  {judge}: baseline_fpr={fpr_b:.3f}, stat_fpr={fpr_s:.3f}, delta={delta:+.3f}")

    print("\n  LLM filter aggregate FPR vs baseline (faithfulness, all injections):")
    for judge in ["gpt-5.4-nano", "gemma-4-26b", "deepseek-v3.2"]:
        sub_b = baseline_only[(baseline_only["judge_model"] == judge) & (baseline_only["is_poisoned"] == True)]
        sub_l = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") & (df["is_poisoned"] == True)]
        fpr_b = _fpr(sub_b["faithfulness_score"])
        fpr_l = _fpr(sub_l["faithfulness_score"])
        delta = fpr_l - fpr_b
        print(f"  {judge}: baseline={fpr_b:.3f}, llm_filter={fpr_l:.3f}, delta={delta:+.3f}")

    print("\n  Partition sums (all 1):", (partition_stat == 1).all(), "/", (partition_mistr == 1).all())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ──")

    JUDGES = sorted(df["judge_model"].unique())
    INJECTIONS = sorted(df["injection_type"].unique())

    # ── Plot 1 — Score distributions ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for i, judge in enumerate(JUDGES):
        ax = axes[i]
        sub_b = df[(df["judge_model"] == judge) & (df["condition"] == "baseline")]
        clean_scores   = sub_b[sub_b["is_poisoned"] == False]["faithfulness_score"].dropna()
        poisoned_scores = sub_b[sub_b["is_poisoned"] == True]["faithfulness_score"].dropna()
        ax.hist(clean_scores,   bins=20, range=(0,1), density=True, alpha=0.55, color=BLUE, label="clean")
        ax.hist(poisoned_scores, bins=20, range=(0,1), density=True, alpha=0.55, color=RED, label="poisoned")
        ax.axvline(x=THRESHOLD, color="black", linestyle="--", linewidth=1)
        ax.set_title(judge)
        ax.set_xlabel("Faithfulness score")
        if i == 0:
            ax.set_ylabel("Density")
            ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_score_distributions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig01_score_distributions.png")

    # ── Plot 2 — Filter behaviour ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: stat filter stacked bars
    inj_labels = INJECTIONS
    cleaned_counts = []
    residual_counts = []
    for inj in inj_labels:
        trip_inj_b = df[(df["condition"] == "baseline") & (df["injection_type"] == inj)].drop_duplicates("triplet_id")
        cleaned_counts.append(trip_inj_b["filter_cleaned_stat"].sum())
        residual_counts.append(trip_inj_b["effectively_poisoned_stat"].sum())

    y_pos = range(len(inj_labels))
    bars1 = ax1.barh(list(y_pos), residual_counts, color=RED, label="Residual poisoned")
    bars2 = ax1.barh(list(y_pos), cleaned_counts, left=residual_counts, color=PURPLE, label="Filter cleaned")
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(inj_labels)
    ax1.set_xlabel("# triplets")
    ax1.set_title("Statistical filter (passage-level)")
    ax1.legend()
    for bar, cnt in zip(bars1, residual_counts):
        ax1.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(cnt),
                 ha="center", va="center", fontsize=8, color="white")
    for bar, cnt, off in zip(bars2, cleaned_counts, residual_counts):
        ax1.text(off + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(cnt),
                 ha="center", va="center", fontsize=8, color="white")

    # Right: LLM filter stacked bars
    correct_counts = []
    missed_counts  = []
    for inj in inj_labels:
        trip_inj_b = df[(df["condition"] == "baseline") & (df["injection_type"] == inj)].drop_duplicates("triplet_id")
        correct_counts.append(trip_inj_b["mistral_correct"].sum())
        missed_counts.append(trip_inj_b["mistral_missed"].sum())

    bars3 = ax2.barh(list(y_pos), correct_counts, color=GREEN, label="Mistral correct")
    bars4 = ax2.barh(list(y_pos), missed_counts, left=correct_counts, color=RED, label="Mistral missed")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(inj_labels)
    ax2.set_xlabel("# triplets")
    ax2.set_title("LLM filter (Mistral, triplet-level)")
    ax2.legend()
    for bar, cnt in zip(bars3, correct_counts):
        ax2.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(cnt),
                 ha="center", va="center", fontsize=8, color="white")
    for bar, cnt, off in zip(bars4, missed_counts, correct_counts):
        ax2.text(off + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(cnt),
                 ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_filter_behaviour.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig02_filter_behaviour.png")

    # ── Plots 3a/b/c — Headline FPR bars ────────────────────────────────────
    for metric_short, metric_col in METRICS.items():
        suffix = {"faithfulness": "a", "context_relevance": "b", "answer_relevance": "c"}[metric_short]
        fname  = f"fig03{suffix}_headline_{metric_short}.png"

        # Aggregate over injection types
        fpr_baseline = []
        fpr_stat     = []
        fpr_llm_mm   = []
        delta_stat   = []
        delta_llm_mm = []
        n_llm_mm     = []

        for judge in JUDGES:
            sub_b  = baseline_only[(baseline_only["judge_model"] == judge) & (baseline_only["is_poisoned"] == True)]
            fpr_b  = _fpr(sub_b[metric_col])
            fpr_baseline.append(fpr_b)

            sub_s  = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                        (df["effectively_poisoned_stat"] == True)]
            fpr_s  = _fpr(sub_s[metric_col])
            fpr_stat.append(fpr_s)
            delta_stat.append(fpr_s - fpr_b)

            sub_mm = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                        (df["mistral_missed"] == True)]
            fpr_mm = _fpr(sub_mm[metric_col])
            n_mm   = len(sub_mm[metric_col].dropna())
            fpr_llm_mm.append(fpr_mm)
            delta_llm_mm.append(fpr_mm - fpr_b)
            n_llm_mm.append(n_mm)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        x = range(len(JUDGES))
        short_labels = [j.replace("gpt-5.4-nano", "GPT").replace("gemma-4-26b", "Gemma").replace("deepseek-v3.2", "DeepSeek") for j in JUDGES]

        # Panel 1: baseline
        ax = axes[0]
        bars = ax.bar(list(x), fpr_baseline, color=BLUE)
        ax.set_xticks(list(x)); ax.set_xticklabels(short_labels)
        ax.set_ylim(0, 1); ax.set_ylabel("FPR (fraction judge misses poison)")
        ax.axhline(y=THRESHOLD, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Baseline")

        # Panel 2: stat filter
        ax = axes[1]
        bars = ax.bar(list(x), fpr_stat, color=RED)
        for xi, delta in zip(x, delta_stat):
            ax.text(xi, fpr_stat[xi] + 0.02, f"{delta:+.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels(short_labels)
        ax.set_ylim(0, 1); ax.axhline(y=THRESHOLD, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Stat filter (corrected)")

        # Panel 3: LLM filter Mistral-missed
        ax = axes[2]
        bars = ax.bar(list(x), fpr_llm_mm, color=ORANGE)
        for xi, (delta, n) in enumerate(zip(delta_llm_mm, n_llm_mm)):
            fpr_val = fpr_llm_mm[xi]
            label = f"{delta:+.2f}\nn={n}"
            ax.text(xi, fpr_val + 0.02 if not np.isnan(fpr_val) else 0.5, label,
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(x)); ax.set_xticklabels(short_labels)
        ax.set_ylim(0, 1); ax.axhline(y=THRESHOLD, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("LLM filter (Mistral-missed)")

        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Plots 4a/b/c — Per-injection FPR ─────────────────────────────────────
    for metric_short, metric_col in METRICS.items():
        suffix = {"faithfulness": "a", "context_relevance": "b", "answer_relevance": "c"}[metric_short]
        fname  = f"fig04{suffix}_per_injection_{metric_short}.png"

        fig, axes = plt.subplots(2, 3, figsize=(13, 9))

        for col_idx, judge in enumerate(JUDGES):
            short = judge.replace("gpt-5.4-nano","GPT").replace("gemma-4-26b","Gemma").replace("deepseek-v3.2","DeepSeek")

            # Top row: stat filter comparison
            ax = axes[0][col_idx]
            fpr_b_vals = []
            fpr_s_vals = []
            for inj in INJECTIONS:
                sb = baseline_only[(baseline_only["judge_model"] == judge) &
                                   (baseline_only["injection_type"] == inj) &
                                   (baseline_only["is_poisoned"] == True)]
                ss = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                        (df["injection_type"] == inj) & (df["effectively_poisoned_stat"] == True)]
                fpr_b_vals.append(_fpr(sb[metric_col]))
                fpr_s_vals.append(_fpr(ss[metric_col]))

            xi = np.arange(len(INJECTIONS))
            width = 0.35
            ax.bar(xi - width/2, fpr_b_vals, width, color=BLUE, label="Baseline")
            ax.bar(xi + width/2, fpr_s_vals, width, color=RED, label="Stat filter")
            for i, (fb, fs) in enumerate(zip(fpr_b_vals, fpr_s_vals)):
                delta = fs - fb
                if not np.isnan(delta):
                    ax.text(i + width/2, fs + 0.02, f"{delta:+.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(xi); ax.set_xticklabels([inj.replace("_","\n") for inj in INJECTIONS], fontsize=7)
            ax.set_ylim(0, 1); ax.set_title(f"{short} — stat filter")
            if col_idx == 0:
                ax.set_ylabel("FPR"); ax.legend(fontsize=7)

            # Bottom row: LLM filter Mistral-missed
            ax = axes[1][col_idx]
            fpr_b_vals2 = []
            fpr_mm_vals = []
            n_mm_vals   = []
            for inj in INJECTIONS:
                sb = baseline_only[(baseline_only["judge_model"] == judge) &
                                   (baseline_only["injection_type"] == inj) &
                                   (baseline_only["is_poisoned"] == True)]
                sm = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                        (df["injection_type"] == inj) & (df["mistral_missed"] == True)]
                fpr_b_vals2.append(_fpr(sb[metric_col]))
                fpr_mm_vals.append(_fpr(sm[metric_col]))
                n_mm_vals.append(len(sm[metric_col].dropna()))

            ax.bar(xi - width/2, fpr_b_vals2, width, color=BLUE, label="Baseline")
            bars_llm = ax.bar(xi + width/2, fpr_mm_vals, width, color=ORANGE, label="LLM filter (missed)")
            for i, (fb, fm, n) in enumerate(zip(fpr_b_vals2, fpr_mm_vals, n_mm_vals)):
                delta = fm - fb
                alpha_val = 0.3 if n < 20 else 1.0
                bars_llm[i].set_alpha(alpha_val)
                if not np.isnan(delta):
                    ax.text(i + width/2, fm + 0.02, f"{delta:+.2f}", ha="center", va="bottom", fontsize=7)
                if n < 20:
                    ax.text(i + width/2, 0.05, f"n={n}", ha="center", va="bottom", fontsize=6, color="gray")
            ax.set_xticks(xi); ax.set_xticklabels([inj.replace("_","\n") for inj in INJECTIONS], fontsize=7)
            ax.set_ylim(0, 1); ax.set_title(f"{short} — LLM filter (missed)")
            if col_idx == 0:
                ax.set_ylabel("FPR"); ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Plots 5a/b/c — Cleaned-vs-clean calibration ──────────────────────────
    for metric_short, metric_col in METRICS.items():
        suffix = {"faithfulness": "a", "context_relevance": "b", "answer_relevance": "c"}[metric_short]
        fname  = f"fig05{suffix}_cleaned_vs_clean_{metric_short}.png"

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        for row_idx, judge in enumerate(JUDGES):
            for col_idx, inj in enumerate(INJECTIONS):
                ax = axes[row_idx][col_idx]
                sub_fc = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_stat") &
                            (df["injection_type"] == inj) & (df["filter_cleaned_stat"] == True)]
                sub_oc = df[(df["judge_model"] == judge) & (df["condition"] == "baseline") &
                            (df["injection_type"] == inj) & (df["originally_clean"] == True)]
                ms_fc  = sub_fc[metric_col].mean()
                ms_oc  = sub_oc[metric_col].mean()
                n_fc   = len(sub_fc[metric_col].dropna())
                n_oc   = len(sub_oc[metric_col].dropna())

                vals  = [ms_fc, ms_oc]
                cols  = [PURPLE, BLUE]
                xlabs = ["Filter-cleaned", "Originally clean"]
                bars  = ax.bar([0, 1], vals, color=cols)
                if n_fc < 20:
                    bars[0].set_alpha(0.3)
                    ax.text(0, (ms_fc or 0) + 0.02, f"n={n_fc}", ha="center", va="bottom", fontsize=7)
                if not (pd.isna(ms_fc) or pd.isna(ms_oc)):
                    delta = ms_fc - ms_oc
                    ax.text(0.5, max(ms_fc, ms_oc) + 0.05,
                            f"Δ={delta:+.2f}", ha="center", va="bottom", fontsize=7, transform=ax.transData)
                ax.set_xticks([0, 1]); ax.set_xticklabels(xlabs, fontsize=7)
                ax.set_ylim(0, 1.1)
                if col_idx == 0:
                    short = judge.replace("gpt-5.4-nano","GPT").replace("gemma-4-26b","Gemma").replace("deepseek-v3.2","DeepSeek")
                    ax.set_ylabel(short, fontsize=8)
                if row_idx == 0:
                    ax.set_title(inj.replace("_","\n"), fontsize=8)

        fig.suptitle(f"Cleaned-vs-clean calibration ({metric_short})", fontsize=10, y=1.01)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Plots 6a/b/c — Mistral-judge agreement heatmaps ──────────────────────
    for metric_short, metric_col in METRICS.items():
        suffix = {"faithfulness": "a", "context_relevance": "b", "answer_relevance": "c"}[metric_short]
        fname  = f"fig06{suffix}_mistral_judge_agreement_{metric_short}.png"

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for row_idx, judge in enumerate(JUDGES):
            for col_idx, inj in enumerate(INJECTIONS):
                ax = axes[row_idx][col_idx]
                sub = df[(df["judge_model"] == judge) & (df["condition"] == "postfilter_llm") &
                         (df["injection_type"] == inj) & (df["is_poisoned"] == True) &
                         df[metric_col].notna()]

                judge_faithful   = (sub[metric_col] >= THRESHOLD)
                mistral_flagged  = sub["predicted_poisoned"].astype(bool)

                a = (mistral_flagged  & judge_faithful).sum()
                b = (mistral_flagged  & ~judge_faithful).sum()
                c = (~mistral_flagged & judge_faithful).sum()
                d = (~mistral_flagged & ~judge_faithful).sum()
                n_total = a + b + c + d

                matrix = np.array([[b, a], [d, c]], dtype=float)  # rows=judge(unfaith,faith), cols=mistral(flagged,missed)
                pct    = matrix / n_total * 100 if n_total > 0 else matrix

                im = ax.imshow(matrix, cmap="Oranges", aspect="auto", vmin=0)
                for ri in range(2):
                    for ci in range(2):
                        val = int(matrix[ri, ci])
                        pv  = pct[ri, ci]
                        ax.text(ci, ri, f"{val}\n({pv:.0f}%)", ha="center", va="center", fontsize=7)

                ax.set_xticks([0, 1]); ax.set_xticklabels(["Flagged", "Missed"], fontsize=7)
                ax.set_yticks([0, 1]); ax.set_yticklabels(["Unfaithful", "Faithful"], fontsize=7)

                # kappa
                kappa_row = df_agree[(df_agree["judge_model"] == judge) &
                                     (df_agree["injection_type"] == inj) &
                                     (df_agree["metric"] == metric_short)]
                kappa_val = kappa_row["cohen_kappa"].values[0] if len(kappa_row) > 0 else float("nan")
                kappa_str = f"κ = {kappa_val:.2f}" if not np.isnan(kappa_val) else "κ = N/A"
                ax.set_title(kappa_str, fontsize=8)

                if col_idx == 0:
                    short = judge.replace("gpt-5.4-nano","GPT").replace("gemma-4-26b","Gemma").replace("deepseek-v3.2","DeepSeek")
                    ax.set_ylabel(short, fontsize=8)
                if row_idx == 0:
                    ax.set_xlabel(inj.replace("_","\n"), fontsize=8)
                    ax.xaxis.set_label_position("top")

        fig.suptitle(f"Mistral-judge agreement ({metric_short})", fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Plot 7 — Inter-judge disagreement ────────────────────────────────────
    # Pivot baseline faithfulness to get one row per triplet, one col per judge
    pivot = (baseline_only[baseline_only["faithfulness_score"].notna()]
             .pivot_table(index="triplet_id", columns="judge_model",
                          values="faithfulness_score", aggfunc="mean"))
    pivot = pivot.dropna()
    pivot_std = pivot.std(axis=1)

    clean_mask   = baseline_only.drop_duplicates("triplet_id").set_index("triplet_id")["is_poisoned"]
    clean_mask   = (~clean_mask).reindex(pivot_std.index)

    std_clean    = pivot_std[clean_mask == True]
    std_poisoned = pivot_std[clean_mask == False]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(std_clean,    bins=30, density=True, alpha=0.55, color=BLUE,  label="clean")
    ax.hist(std_poisoned, bins=30, density=True, alpha=0.55, color=RED,   label="poisoned")
    mean_clean_std    = std_clean.mean()
    mean_poisoned_std = std_poisoned.mean()
    ax.axvline(mean_clean_std,    color=BLUE,  linestyle="--", linewidth=1)
    ax.axvline(mean_poisoned_std, color=RED,   linestyle="--", linewidth=1)

    if len(std_clean) > 2 and len(std_poisoned) > 2:
        from scipy.stats import pearsonr
        # Compute pearson r between std and is_poisoned
        is_poisoned_aligned = (~clean_mask).astype(int).reindex(pivot_std.index).fillna(0)
        r_val, p_val_r = pearsonr(pivot_std.values, is_poisoned_aligned.values)
        n_val = len(pivot_std)
        p_str = "p < 0.001" if p_val_r < 0.001 else f"p = {p_val_r:.3f}"
        ax.text(0.98, 0.95, f"Pearson r = {r_val:+.3f}, {p_str}, n = {n_val:,}",
                ha="right", va="top", transform=ax.transAxes, fontsize=8)

    ax.set_xlabel("Std of faithfulness across 3 judges")
    ax.set_ylabel("Density")
    ax.set_title("Inter-judge disagreement (baseline faithfulness)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_interjudge_disagreement.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig07_interjudge_disagreement.png")

    # ── Plot 8 — McNemar corrected heatmap ───────────────────────────────────
    df_mc_faith = df_mcnemar[df_mcnemar["metric"] == "faithfulness"].copy()

    # Build pivot: judges × injections
    mc_pivot = df_mc_faith.pivot(index="judge_model", columns="injection_type", values="chi2")
    mc_pval  = df_mc_faith.pivot(index="judge_model", columns="injection_type", values="p_value")

    fig, ax = plt.subplots(figsize=(7, 3))
    data_vals = mc_pivot.values.astype(float)
    im = ax.imshow(data_vals, cmap="Reds", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="chi2")

    for ri, judge in enumerate(mc_pivot.index):
        for ci, inj in enumerate(mc_pivot.columns):
            chi2_v = data_vals[ri, ci]
            p_v    = mc_pval.loc[judge, inj]
            p_str  = ("p<0.001" if p_v < 0.001 else f"p={p_v:.2f}") if not np.isnan(p_v) else "n/a"
            ax.text(ci, ri, f"{chi2_v:.1f}\n{p_str}", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(len(mc_pivot.columns))); ax.set_xticklabels(mc_pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(mc_pivot.index)));   ax.set_yticklabels(mc_pivot.index)
    ax.set_title("McNemar test (stat filter, faithfulness)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig08_mcnemar_corrected.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig08_mcnemar_corrected.png")

    # ── Summary file ─────────────────────────────────────────────────────────
    print("\n── Writing summary.md ──")

    def _get_headline(judge, condition, metric, inj="all"):
        if inj == "all":
            rows = df_headline[(df_headline["judge_model"] == judge) &
                               (df_headline["condition"] == condition) &
                               (df_headline["metric"] == metric)]
        else:
            rows = df_headline[(df_headline["judge_model"] == judge) &
                               (df_headline["condition"] == condition) &
                               (df_headline["injection_type"] == inj) &
                               (df_headline["metric"] == metric)]
        return rows

    # Aggregate stat filter corrected ΔFPR across all injections per judge
    def _agg_delta(judge, condition, metric):
        sub_b = baseline_only[(baseline_only["judge_model"] == judge) & (baseline_only["is_poisoned"] == True)]
        fpr_b = _fpr(sub_b[METRICS[metric]])
        if condition == "postfilter_stat":
            sub_c = df[(df["judge_model"] == judge) & (df["condition"] == condition) &
                       (df["effectively_poisoned_stat"] == True)]
        else:
            sub_c = df[(df["judge_model"] == judge) & (df["condition"] == condition) &
                       (df["mistral_missed"] == True)]
        fpr_c = _fpr(sub_c[METRICS[metric]])
        return fpr_b, fpr_c, fpr_c - fpr_b

    lines = []
    lines.append("# Residual-Aware Analysis Summary")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("### Stat filter corrected ΔFPR (faithfulness, all injections)")
    for judge in JUDGES:
        fpr_b, fpr_s, delta = _agg_delta(judge, "postfilter_stat", "faithfulness")
        lines.append(f"- {judge}: baseline={fpr_b:.3f}, stat_filter={fpr_s:.3f}, ΔFPR={delta:+.3f}")
    lines.append("")
    lines.append("### LLM filter Mistral-missed FPR (faithfulness, all injections)")
    for judge in JUDGES:
        fpr_b, fpr_mm, delta = _agg_delta(judge, "postfilter_llm", "faithfulness")
        lines.append(f"- {judge}: baseline={fpr_b:.3f}, mistral_missed_fpr={fpr_mm:.3f}, ΔFPR={delta:+.3f}")
    lines.append("")
    lines.append("### Baseline FPR (faithfulness, all injections combined)")
    for judge in JUDGES:
        sub = baseline_only[(baseline_only["judge_model"] == judge) & (baseline_only["is_poisoned"] == True)]
        fpr = _fpr(sub["faithfulness_score"])
        lines.append(f"- {judge}: {fpr:.3f}")
    lines.append("")
    lines.append("### Filter behaviour")
    lines.append(df_filter[["injection_type","filter_mechanism","n_triplets_originally_poisoned",
                             "n_triplets_fully_cleaned_by_stat_filter","mistral_recall",
                             "mistral_clean_fpr"]].to_string(index=False))
    lines.append("")
    lines.append("### McNemar test (faithfulness) — chi2 values")
    lines.append(df_mc_faith[["judge_model","injection_type","chi2","p_value"]].to_string(index=False))
    lines.append("")
    lines.append("### Prefilter Paradox correction note")
    lines.append("The legacy analysis used `is_poisoned` as denominator for post-filter FPR.")
    lines.append("This counted filter-cleaned triplets as FPs when the judge scored them faithfully.")
    lines.append("Corrected analysis uses `effectively_poisoned_stat` (residual poisoned passages > 0).")
    lines.append("Triplets fully cleaned by the statistical filter are excluded from FPR denominator.")

    summary_text = "\n".join(lines)
    summary_path = exports_dir / "summary.md"
    summary_path.write_text(summary_text)
    print(f"\n{'='*60}")
    print(summary_text)
    print(f"{'='*60}")
    print(f"\nSaved summary to {summary_path}")

    # Final file listing
    files_created = list(exports_dir.glob("table_*.csv")) + list(exports_dir.glob("*.csv")) + \
                    list(fig_dir.glob("fig*.png")) + [summary_path]
    new_files = [f for f in (list(exports_dir.glob("table_*.csv")) +
                              list(exports_dir.glob("joined_*.csv")) +
                              list(exports_dir.glob("summary.md")) +
                              list(fig_dir.glob("fig*.png"))) if f.exists()]
    print(f"\nCreated {len(new_files)} output files:")
    for f in sorted(new_files):
        print(f"  {f}")


if __name__ == "__main__":
    main()
