"""
Signal ablation: measure detection metrics with each signal removed one at a time.
Reuses existing PrefilterScores  no re-inference needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.schema import PrefilterScore
from src.metrics.detection_metrics import compute_detection_metrics
from src.prefilter.aggregator import aggregate


def run_ablation(
    all_scores: list[PrefilterScore],
    config: dict,
    injection_map: dict[str, str],
) -> dict:
    """
    For each ablation variant, zero out one signal and re-aggregate with weighted_vote.
    Returns dict: variant_name -> metrics (per injection type and overall).
    """
    variants = {
        "all_signals": None,
        "no_embedding": "embedding",
        "no_entropy": "entropy",
        "no_classifier": "classifier",
        "no_answer_span": "answer_span",
    }

    results = {}

    for variant_name, removed_signal in variants.items():
        modified = []
        for s in all_scores:
            m = s.model_copy()
            if removed_signal == "embedding":
                m.embedding_score = 0.0
            elif removed_signal == "entropy":
                m.entropy_score = 0.0
            elif removed_signal == "classifier":
                m.classifier_score = 0.5  # neutral, not 0
            elif removed_signal == "answer_span":
                m.answer_span_score = 0.5  # neutral (yes/no default)
            modified.append(m)

        aggregate(modified, config, method="weighted_vote")

        # Overall metrics
        metrics = compute_detection_metrics(modified, "aggregated")
        results[f"{variant_name}_overall"] = {
            "variant": variant_name,
            "removed_signal": removed_signal or "none",
            "injection_type": "overall",
            **{k: v for k, v in metrics.items() if k not in ("signal", "error")},
        }

        # Per injection type
        for inj_type in ("random_noise", "adversarial_fact", "poisonedrag_style"):
            subset = [s for s in modified if injection_map.get(s.triplet_id) == inj_type]
            if not subset:
                continue
            metrics = compute_detection_metrics(subset, "aggregated")
            results[f"{variant_name}_{inj_type}"] = {
                "variant": variant_name,
                "removed_signal": removed_signal or "none",
                "injection_type": inj_type,
                **{k: v for k, v in metrics.items() if k not in ("signal", "error")},
            }

    return results


def run_aggregation_comparison(
    all_scores: list[PrefilterScore],
    config: dict,
    injection_map: dict[str, str],
) -> dict:
    """Compare all three aggregation methods overall and per injection type."""
    methods = ["weighted_vote", "xgboost", "majority_vote"]
    results = {}

    for method in methods:
        modified = [s.model_copy() for s in all_scores]
        aggregate(modified, config, method=method)

        metrics = compute_detection_metrics(modified, "aggregated")
        note = "in-sample (val)" if method == "xgboost" else "out-of-sample"
        results[f"{method}_overall"] = {
            "method": method,
            "injection_type": "overall",
            "note": note,
            **{k: v for k, v in metrics.items() if k != "signal"},
        }

        for inj_type in ("random_noise", "adversarial_fact", "poisonedrag_style"):
            subset = [s for s in modified if injection_map.get(s.triplet_id) == inj_type]
            if not subset:
                continue
            metrics = compute_detection_metrics(subset, "aggregated")
            results[f"{method}_{inj_type}"] = {
                "method": method,
                "injection_type": inj_type,
                "note": note,
                **{k: v for k, v in metrics.items() if k != "signal"},
            }

    return results


def run_signal_only(
    all_scores: list[PrefilterScore],
    config: dict,
    injection_map: dict[str, str],
) -> dict:
    """Each signal alone as the sole classifier (others zeroed out)."""
    signal_configs = {
        "embedding_only": {
            "embedding_score": None,
            "entropy_score": 0.0,
            "classifier_score": 0.5,
            "answer_span_score": 0.5,
        },
        "entropy_only": {
            "embedding_score": 0.0,
            "entropy_score": None,
            "classifier_score": 0.5,
            "answer_span_score": 0.5,
        },
        "classifier_only": {
            "embedding_score": 0.0,
            "entropy_score": 0.0,
            "classifier_score": None,
            "answer_span_score": 0.5,
        },
        "answer_span_only": {
            "embedding_score": 0.0,
            "entropy_score": 0.0,
            "classifier_score": 0.5,
            "answer_span_score": None,
        },
    }

    results = {}
    for variant_name, overrides in signal_configs.items():
        modified = []
        for s in all_scores:
            m = s.model_copy()
            for field, val in overrides.items():
                if val is not None:
                    setattr(m, field, val)
            modified.append(m)

        aggregate(modified, config, method="weighted_vote")
        metrics = compute_detection_metrics(modified, "aggregated")
        results[variant_name] = {
            "variant": variant_name,
            "injection_type": "overall",
            **{k: v for k, v in metrics.items() if k not in ("signal", "error")},
        }

    return results


def run_prefilter_strategy_comparison(
    baseline_scores: list[dict],
    postfilter_scores: list[dict],
    postfilter_llm_scores: list[dict],
    config: dict,
) -> dict:
    """
    Compare three strategies  no filter, statistical prefilter, LLM prefilter 
    using judge scores. Computes FPR reduction per judge and injection type.
    """
    threshold = config["judges"]["scoring"]["scoring_threshold"]

    def _fpr(scores, judge=None, inj_type=None):
        subset = [s for s in scores if s.get("is_poisoned", False)]
        if judge:
            subset = [s for s in subset if s.get("judge_model") == judge]
        if inj_type:
            subset = [s for s in subset if s.get("injection_type") == inj_type]
        if not subset:
            return None
        return sum(1 for s in subset if s.get("faithfulness", 0) > threshold) / len(subset)

    judges = sorted({s.get("judge_model") for s in baseline_scores})
    injection_types = ["random_noise", "adversarial_fact", "poisonedrag_style", "overall"]

    results = {}
    for judge in judges:
        for inj_type in injection_types:
            inj = None if inj_type == "overall" else inj_type
            fpr_base = _fpr(baseline_scores, judge, inj)
            fpr_stat = _fpr(postfilter_scores, judge, inj)
            fpr_llm = _fpr(postfilter_llm_scores, judge, inj)
            if fpr_base is None:
                continue
            key = f"{judge}_{inj_type}"
            results[key] = {
                "judge": judge,
                "injection_type": inj_type,
                "fpr_baseline": round(fpr_base, 4),
                "fpr_statistical": round(fpr_stat, 4) if fpr_stat is not None else None,
                "fpr_llm": round(fpr_llm, 4) if fpr_llm is not None else None,
                "delta_statistical": (
                    round(fpr_stat - fpr_base, 4) if fpr_stat is not None else None
                ),
                "delta_llm": round(fpr_llm - fpr_base, 4) if fpr_llm is not None else None,
            }

    return results


def save_ablation_tables(
    all_scores: list[PrefilterScore],
    config: dict,
    injection_map: dict[str, str],
    baseline_scores: list[dict] | None = None,
    postfilter_scores: list[dict] | None = None,
    postfilter_llm_scores: list[dict] | None = None,
) -> None:
    out_dir = Path(config["experiment"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation = run_ablation(all_scores, config, injection_map)
    agg_comparison = run_aggregation_comparison(all_scores, config, injection_map)
    signal_only = run_signal_only(all_scores, config, injection_map)

    for filename, rows in [
        ("table_ablation_signal_removal.csv", list(ablation.values())),
        ("table_ablation_aggregation.csv", list(agg_comparison.values())),
        ("table_ablation_signal_only.csv", list(signal_only.values())),
    ]:
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / filename, index=False)
            print(f"Saved {filename}")

    # Prefilter strategy comparison (only if all three judge score sets are available)
    if baseline_scores and postfilter_scores and postfilter_llm_scores:
        strategy = run_prefilter_strategy_comparison(
            baseline_scores, postfilter_scores, postfilter_llm_scores, config
        )
        if strategy:
            pd.DataFrame(list(strategy.values())).to_csv(
                out_dir / "table_ablation_prefilter_strategy.csv", index=False
            )
            print("Saved table_ablation_prefilter_strategy.csv")
