"""
End-to-end system metrics: FPR before vs. after pre-filter.
Wilcoxon signed-rank test, per-metric improvement, clean sample false flagging rate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats


def compute_system_metrics(
    baseline_scores: list[dict],
    postfilter_scores: list[dict],
    threshold: float = 0.5,
    metrics: list[str] = ("faithfulness", "answer_relevance", "context_relevance"),
) -> dict:
    """
    Compare baseline and post-filter judge scores.
    Returns per-(judge, injection_type, metric) improvement stats.
    """

    def _fpr(scores, metric):
        poisoned = [s for s in scores if s.get("is_poisoned", False)]
        if not poisoned:
            return 0.0
        return sum(1 for s in poisoned if s.get(metric, 0) >= threshold) / len(poisoned)

    def _flag_rate_clean(scores, metric):
        clean = [s for s in scores if not s.get("is_poisoned", False)]
        if not clean:
            return 0.0
        return sum(1 for s in clean if s.get(metric, 0) <= threshold) / len(clean)

    # Index by triplet_id for pairing (some error records use "id" instead)
    def _tid(s):
        return s.get("triplet_id") or s.get("id", "")

    results = {}
    judges = {s.get("judge_model") for s in baseline_scores}
    injection_types = {s.get("injection_type") for s in baseline_scores}

    for judge in judges:
        # Build per-judge indices so Wilcoxon pairs are within the same judge
        base_idx = {
            _tid(s): s
            for s in baseline_scores
            if _tid(s) and s.get("judge_model") == judge and s.get("faithfulness") is not None
        }
        post_idx = {
            _tid(s): s
            for s in postfilter_scores
            if _tid(s) and s.get("judge_model") == judge and s.get("faithfulness") is not None
        }
        common_ids = set(base_idx) & set(post_idx)

        for inj_type in injection_types:
            base_sub = [
                s
                for s in baseline_scores
                if s.get("judge_model") == judge and s.get("injection_type") == inj_type
            ]
            post_sub = [
                s
                for s in postfilter_scores
                if s.get("judge_model") == judge and s.get("injection_type") == inj_type
            ]

            if not base_sub or not post_sub:
                continue

            key = f"{judge}_{inj_type}"
            results[key] = {"judge": judge, "injection_type": inj_type}

            for metric in metrics:
                fpr_before = _fpr(base_sub, metric)
                fpr_after = _fpr(post_sub, metric)
                abs_reduction = fpr_before - fpr_after
                rel_improvement = abs_reduction / fpr_before if fpr_before > 0 else 0.0

                # Paired Wilcoxon test on matched samples (within this judge)
                paired_ids = [
                    tid for tid in common_ids
                    if base_idx[tid].get("injection_type") == inj_type
                ]
                base_vals = [
                    base_idx[tid][metric]
                    for tid in paired_ids
                    if metric in base_idx[tid] and base_idx[tid][metric] is not None
                ]
                post_vals = [
                    post_idx[tid][metric]
                    for tid in paired_ids
                    if tid in post_idx and metric in post_idx[tid] and post_idx[tid][metric] is not None
                ]

                wilcoxon_p = None
                if len(base_vals) >= 10 and len(base_vals) == len(post_vals):
                    try:
                        _, wilcoxon_p = stats.wilcoxon(base_vals, post_vals)
                    except ValueError:
                        pass

                clean_flag_rate = _flag_rate_clean(post_sub, metric)

                results[key][metric] = {
                    "fpr_before": fpr_before,
                    "fpr_after": fpr_after,
                    "abs_reduction": abs_reduction,
                    "rel_improvement_pct": rel_improvement * 100,
                    "wilcoxon_p": float(wilcoxon_p) if wilcoxon_p is not None else None,
                    "clean_false_flag_rate": clean_flag_rate,
                }

    return results


def save_system_tables(
    baseline_scores: list[dict],
    postfilter_scores: list[dict],
    config: dict,
) -> None:
    threshold = config["judges"]["scoring"]["scoring_threshold"]
    metrics = ["faithfulness", "answer_relevance", "context_relevance"]
    out_dir = Path(config["experiment"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = compute_system_metrics(baseline_scores, postfilter_scores, threshold, metrics)

    rows_before_after, rows_clean, rows_per_metric = [], [], []

    for key, m in results.items():
        for metric in metrics:
            if metric not in m:
                continue
            d = m[metric]
            rows_before_after.append(
                {
                    "judge": m["judge"],
                    "injection_type": m["injection_type"],
                    "metric": metric,
                    "fpr_before": d["fpr_before"],
                    "fpr_after": d["fpr_after"],
                    "abs_reduction": d["abs_reduction"],
                    "rel_improvement_pct": d["rel_improvement_pct"],
                    "wilcoxon_p": d["wilcoxon_p"],
                }
            )
            rows_clean.append(
                {
                    "judge": m["judge"],
                    "injection_type": m["injection_type"],
                    "metric": metric,
                    "clean_false_flag_rate": d["clean_false_flag_rate"],
                }
            )
            rows_per_metric.append(
                {
                    "judge": m["judge"],
                    "injection_type": m["injection_type"],
                    "metric": metric,
                    "fpr_improvement": d["abs_reduction"],
                }
            )

    for fname, rows in [
        ("table_4_7_fpr_before_after.csv", rows_before_after),
        ("table_4_8_clean_sample_impact.csv", rows_clean),
        ("table_4_9_per_metric_improvement.csv", rows_per_metric),
    ]:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / fname, index=False)
            print(f"Saved {fname}")
