"""Pearson/Spearman correlation between noise_level and FPR."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats


def compute_noise_fpr_correlation(
    judge_scores: list[dict],
    threshold: float = 0.5,
    metrics: list[str] = ("faithfulness", "answer_relevance", "context_relevance"),
) -> dict:
    """
    For each (judge, injection_type, metric) group, compute:
      - Pearson r (noise_level vs FPR)
      - Spearman rho (noise_level vs FPR)
    """
    from collections import defaultdict

    # Aggregate FPR per (judge, inj_type, noise_level, metric)
    groups: dict = defaultdict(lambda: {"poisoned": 0, "total_poisoned": 0})

    for s in judge_scores:
        if not s.get("is_poisoned", False):
            continue
        judge = s.get("judge_model", "unknown")
        inj = s.get("injection_type", "unknown")
        nl = s.get("noise_level", 0.0)
        for metric in metrics:
            val = s.get(metric, None)
            if val is None:
                continue
            key = (judge, inj, nl, metric)
            groups[key]["total_poisoned"] += 1
            if val > threshold:
                groups[key]["poisoned"] += 1

    # Build dataframe
    rows = []
    for (judge, inj, nl, metric), counts in groups.items():
        fpr = counts["poisoned"] / counts["total_poisoned"] if counts["total_poisoned"] > 0 else 0.0
        rows.append(
            {
                "judge": judge,
                "injection_type": inj,
                "noise_level": nl,
                "metric": metric,
                "fpr": fpr,
                "n": counts["total_poisoned"],
            }
        )

    df = pd.DataFrame(rows)
    results = {}

    for (judge, inj, metric), grp in df.groupby(["judge", "injection_type", "metric"]):
        grp = grp.sort_values("noise_level")
        if len(grp) < 3:
            continue
        x = grp["noise_level"].values
        y = grp["fpr"].values
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        results[f"{judge}_{inj}_{metric}"] = {
            "judge": judge,
            "injection_type": inj,
            "metric": metric,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n_noise_levels": len(grp),
        }

    return results


def save_correlation_table(judge_scores: list[dict], config: dict) -> None:
    threshold = config["judges"]["scoring"]["scoring_threshold"]
    results = compute_noise_fpr_correlation(judge_scores, threshold)

    out_dir = Path(config["experiment"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(results.values())
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "table_4_4_correlations.csv", index=False)
        print("Saved table_4_4_correlations.csv")
