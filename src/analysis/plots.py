"""
All thesis plot generation functions.
Saves to results/figures/ at 300 DPI PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.dataset.schema import PrefilterScore


def _setup(config: dict):
    style = config["reporting"].get("figure_style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    return Path(config["experiment"]["results_dir"]) / "figures"


def _save(fig, path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = config["reporting"].get("figure_dpi", 300)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# Figure 4.1: FPR vs Noise Level Lines (faithfulness, primary)

METRICS = ["faithfulness", "answer_relevance", "context_relevance"]
METRIC_LABELS = {"faithfulness": "Faithfulness", "answer_relevance": "Answer Relevance", "context_relevance": "Context Relevance"}


def _fpr_series(judge_scores, judge, inj_type, noise_levels, threshold, metric):
    fprs, cis_lo, cis_hi = [], [], []
    for nl in noise_levels:
        subset = [
            s for s in judge_scores
            if s.get("judge_model") == judge
            and s.get("injection_type") == inj_type
            and s.get("noise_level") == nl
            and s.get("is_poisoned", False)
            and s.get(metric) is not None
        ]
        if not subset:
            fprs.append(None); cis_lo.append(None); cis_hi.append(None)
            continue
        vals = [int(s.get(metric, 0) >= threshold) for s in subset]
        fpr = np.mean(vals)
        ci = 1.96 * np.sqrt(fpr * (1 - fpr) / max(len(vals), 1))
        fprs.append(fpr)
        cis_lo.append(max(0, fpr - ci))
        cis_hi.append(min(1, fpr + ci))
    return fprs, cis_lo, cis_hi


def fig_fpr_vs_noise_lines(judge_scores: list[dict], config: dict) -> None:
    out_dir = _setup(config)
    threshold = config["judges"]["scoring"]["scoring_threshold"]
    injection_types = ["random_noise", "adversarial_fact", "poisonedrag_style"]
    judges = sorted({s.get("judge_model") for s in judge_scores if s.get("judge_model")})
    noise_levels = sorted(config["dataset"]["noise_levels"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("FPR (Faithfulness) vs. Noise Level by Injection Type", fontsize=14)

    for ax, inj_type in zip(axes, injection_types):
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="FPR=0.5")
        for judge in judges:
            fprs, cis_lo, cis_hi = _fpr_series(judge_scores, judge, inj_type, noise_levels, threshold, "faithfulness")
            valid = [(nl, f, lo, hi) for nl, f, lo, hi in zip(noise_levels, fprs, cis_lo, cis_hi) if f is not None]
            if not valid:
                continue
            xs, ys, los, his = zip(*valid)
            line = ax.plot(xs, ys, marker="o", label=judge)[0]
            ax.fill_between(xs, los, his, alpha=0.2, color=line.get_color())

        ax.set_title(inj_type.replace("_", " ").title())
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("FPR (Faithfulness)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    _save(fig, out_dir / "fig_4_1_fpr_vs_noise_lines.png", config)


# Figure 4.1b: FPR vs Noise Level - All 3 Metrics


def fig_fpr_all_metrics(judge_scores: list[dict], config: dict) -> None:
    """3×3 grid: rows=metrics, cols=injection types. Shows FPR across all metrics."""
    out_dir = _setup(config)
    threshold = config["judges"]["scoring"]["scoring_threshold"]
    injection_types = ["random_noise", "adversarial_fact", "poisonedrag_style"]
    judges = sorted({s.get("judge_model") for s in judge_scores if s.get("judge_model")})
    noise_levels = sorted(config["dataset"]["noise_levels"])

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    fig.suptitle("FPR vs. Noise Level - All Metrics", fontsize=14)

    for row_i, metric in enumerate(METRICS):
        for col_i, inj_type in enumerate(injection_types):
            ax = axes[row_i][col_i]
            ax.axhline(0.5, linestyle="--", color="gray", alpha=0.4)
            for judge in judges:
                fprs, cis_lo, cis_hi = _fpr_series(judge_scores, judge, inj_type, noise_levels, threshold, metric)
                valid = [(nl, f, lo, hi) for nl, f, lo, hi in zip(noise_levels, fprs, cis_lo, cis_hi) if f is not None]
                if not valid:
                    continue
                xs, ys, los, his = zip(*valid)
                line = ax.plot(xs, ys, marker="o", label=judge)[0]
                ax.fill_between(xs, los, his, alpha=0.15, color=line.get_color())
            if row_i == 0:
                ax.set_title(inj_type.replace("_", " ").title(), fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"FPR\n({METRIC_LABELS[metric]})", fontsize=9)
            ax.set_xlabel("Noise Level" if row_i == 2 else "")
            ax.set_ylim(0, 1.05)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig_4_1b_fpr_all_metrics.png", config)


# Figure 4.2: FPR Heatmaps


def fig_fpr_heatmaps(judge_scores: list[dict], config: dict) -> None:
    """3 rows (one per metric) × len(judges) cols."""
    out_dir = _setup(config)
    threshold = config["judges"]["scoring"]["scoring_threshold"]
    judges = sorted({s.get("judge_model") for s in judge_scores if s.get("judge_model")})
    noise_levels = sorted(config["dataset"]["noise_levels"])
    injection_types = ["random_noise", "adversarial_fact", "poisonedrag_style"]

    fig, axes = plt.subplots(3, len(judges), figsize=(7 * len(judges), 12))
    if len(judges) == 1:
        axes = [[ax] for ax in axes]
    fig.suptitle("FPR Heatmaps by Judge and Metric", fontsize=14)

    for row_i, metric in enumerate(METRICS):
        for col_i, judge in enumerate(judges):
            ax = axes[row_i][col_i]
            data = []
            for inj_type in injection_types:
                row = []
                for nl in noise_levels:
                    subset = [
                        s for s in judge_scores
                        if s.get("judge_model") == judge
                        and s.get("injection_type") == inj_type
                        and s.get("noise_level") == nl
                        and s.get("is_poisoned", False)
                        and s.get(metric) is not None
                    ]
                    row.append(np.mean([int(s.get(metric, 0) >= threshold) for s in subset]) if subset else np.nan)
                data.append(row)

            df = pd.DataFrame(data, index=[t.replace("_", "\n") for t in injection_types], columns=noise_levels)
            sns.heatmap(df, ax=ax, vmin=0, vmax=1, center=0.5, cmap="RdYlGn_r",
                        annot=True, fmt=".2f", cbar_kws={"label": "FPR"})
            title = judge if row_i == 0 else ""
            if title:
                ax.set_title(title)
            ax.set_xlabel("Noise Level")
            ax.set_ylabel(METRIC_LABELS[metric] if col_i == 0 else "")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir / "fig_4_2_fpr_heatmaps.png", config)


# Figure 4.3: Score Distributions - all metrics


def fig_kde_score_distributions(judge_scores: list[dict], config: dict) -> None:
    """Separate figure per metric: judges × injection types grid."""
    out_dir = _setup(config)
    judges = sorted({s.get("judge_model") for s in judge_scores if s.get("judge_model")})
    injection_types = ["random_noise", "adversarial_fact", "poisonedrag_style"]

    for metric in METRICS:
        fig, axes = plt.subplots(len(judges), 3, figsize=(15, 5 * len(judges)))
        if len(judges) == 1:
            axes = [axes]
        fig.suptitle(f"Score Distributions ({METRIC_LABELS[metric]}): Clean vs. Poisoned", fontsize=14)

        for row_i, judge in enumerate(judges):
            for col_i, inj_type in enumerate(injection_types):
                ax = axes[row_i][col_i]
                clean_vals = [
                    s.get(metric, 0) for s in judge_scores
                    if s.get("judge_model") == judge and s.get("injection_type") == inj_type
                    and not s.get("is_poisoned", False) and s.get(metric) is not None
                ]
                poisoned_vals = [
                    s.get(metric, 0) for s in judge_scores
                    if s.get("judge_model") == judge and s.get("injection_type") == inj_type
                    and s.get("is_poisoned", False) and s.get(metric) is not None
                ]
                if clean_vals:
                    ax.hist(clean_vals, bins=20, alpha=0.5, density=True, label="Clean", color="steelblue")
                if poisoned_vals:
                    ax.hist(poisoned_vals, bins=20, alpha=0.5, density=True, label="Poisoned", color="salmon")
                ax.set_title(f"{judge} | {inj_type.replace('_', ' ')}", fontsize=9)
                ax.set_xlabel(f"{METRIC_LABELS[metric]} Score")
                ax.legend(fontsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        slug = metric.replace("_", "")
        _save(fig, out_dir / f"fig_4_3_score_distributions_{slug}.png", config)


# Figure 4.4: ROC Curves


def fig_roc_curves(all_scores: list[PrefilterScore], config: dict) -> None:
    out_dir = _setup(config)
    injection_types = sorted({_infer_injection(s.triplet_id) for s in all_scores} - {"unknown"})
    signals = [
        ("embedding", "Embedding"),
        ("entropy", "Entropy"),
        ("classifier", "Classifier"),
        ("aggregated", "Aggregated"),
    ]

    fig, axes = plt.subplots(1, len(injection_types), figsize=(6 * len(injection_types), 5))
    if len(injection_types) == 1:
        axes = [axes]
    fig.suptitle("ROC Curves by Signal and Injection Type", fontsize=14)

    for ax, inj_type in zip(axes, injection_types):
        subset = [s for s in all_scores if _infer_injection(s.triplet_id) == inj_type]
        y_true = [int(s.ground_truth_poisoned) for s in subset]

        if not any(y_true):
            continue

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        for attr, label in signals:
            y_score = [getattr(s, f"{attr}_score") for s in subset]
            try:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.2f})")
            except Exception:
                pass

        ax.set_title(inj_type.replace("_", " ").title())
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)

    _save(fig, out_dir / "fig_4_4_roc_curves_signals.png", config)


# Figure 4.5: Confusion Matrices


def fig_confusion_matrices(all_scores: list[PrefilterScore], config: dict) -> None:
    out_dir = _setup(config)
    threshold = config["prefilter"]["aggregation"]["threshold"]
    methods = config["prefilter"]["aggregation"]["methods"]

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices at Optimal Threshold", fontsize=14)

    for ax, method in zip(axes, methods):
        y_true = [int(s.ground_truth_poisoned) for s in all_scores]
        y_pred = [int(s.aggregated_score > threshold) for s in all_scores]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax,
            cmap="Blues",
            xticklabels=["Clean", "Poisoned"],
            yticklabels=["Clean", "Poisoned"],
        )
        ax.set_title(method.replace("_", " ").title())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    _save(fig, out_dir / "fig_4_5_confusion_matrices.png", config)


# Figure 4.5b: Metric Means Overview


def fig_metric_means_overview(judge_scores: list[dict], config: dict) -> None:
    """Bar chart: clean vs poisoned mean for all 3 metrics × judges. One subplot per judge."""
    out_dir = _setup(config)
    judges = sorted({s.get("judge_model") for s in judge_scores if s.get("judge_model")})

    fig, axes = plt.subplots(1, len(judges), figsize=(7 * len(judges), 5), sharey=True)
    if len(judges) == 1:
        axes = [axes]
    fig.suptitle("Mean Judge Scores: Clean vs. Poisoned (All Metrics)", fontsize=14)

    x = np.arange(len(METRICS))
    width = 0.35

    for ax, judge in zip(axes, judges):
        clean_means, poisoned_means = [], []
        for metric in METRICS:
            clean_vals = [s.get(metric) for s in judge_scores if s.get("judge_model") == judge and not s.get("is_poisoned") and s.get(metric) is not None]
            poisoned_vals = [s.get(metric) for s in judge_scores if s.get("judge_model") == judge and s.get("is_poisoned") and s.get(metric) is not None]
            clean_means.append(np.mean(clean_vals) if clean_vals else 0)
            poisoned_means.append(np.mean(poisoned_vals) if poisoned_vals else 0)

        bars_c = ax.bar(x - width / 2, clean_means, width, label="Clean", color="steelblue", alpha=0.85)
        bars_p = ax.bar(x + width / 2, poisoned_means, width, label="Poisoned", color="salmon", alpha=0.85)
        ax.bar_label(bars_c, fmt="%.2f", fontsize=7, padding=2)
        ax.bar_label(bars_p, fmt="%.2f", fontsize=7, padding=2)

        # Gap annotations
        for i, (c, p) in enumerate(zip(clean_means, poisoned_means)):
            ax.annotate(f"Delta{c-p:.2f}", xy=(i, max(c, p) + 0.05), ha="center", fontsize=7, color="darkred")

        ax.set_title(judge)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Mean Score")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "fig_4_5b_metric_means_overview.png", config)


# Figure 4.6: FPR Before/After Bars - all 3 metrics


def fig_fpr_before_after(
    baseline_scores: list[dict],
    postfilter_scores: list[dict],
    config: dict,
) -> None:
    """One subplot per metric showing FPR before/after per judge × injection type."""
    out_dir = _setup(config)
    threshold = config["judges"]["scoring"]["scoring_threshold"]
    injection_types = ["random_noise", "adversarial_fact", "poisonedrag_style"]
    judges = sorted({s.get("judge_model") for s in baseline_scores if s.get("judge_model")})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FPR Before vs. After Pre-filter (All Metrics)", fontsize=14)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric in zip(axes, METRICS):
        x = np.arange(len(injection_types))
        width = 0.8 / (2 * len(judges))

        for j_idx, judge in enumerate(judges):
            for phase_idx, (scores, label) in enumerate(
                [(baseline_scores, "Before"), (postfilter_scores, "After")]
            ):
                fprs = []
                for inj_type in injection_types:
                    subset = [
                        s for s in scores
                        if s.get("judge_model") == judge
                        and s.get("injection_type") == inj_type
                        and s.get("is_poisoned", False)
                        and s.get(metric) is not None
                    ]
                    fprs.append(np.mean([int(s.get(metric, 0) >= threshold) for s in subset]) if subset else 0.0)

                pos = x + (j_idx * 2 + phase_idx - len(judges) + 0.5) * width
                color = colors[j_idx % len(colors)]
                alpha = 1.0 if label == "Before" else 0.5
                ax.bar(pos, fprs, width, label=f"{judge} ({label})" if metric == METRICS[0] else "", color=color, alpha=alpha)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in injection_types], fontsize=8)
        ax.set_ylabel(f"FPR ({METRIC_LABELS[metric]})")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 1.1)
        if metric == METRICS[0]:
            ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "fig_4_6_fpr_before_after_bars.png", config)


# Figure 4.7: Latency Breakdown


def fig_latency_breakdown(profiles: dict, config: dict) -> None:
    out_dir = _setup(config)
    components = ["embedding", "entropy", "classifier"]
    latencies = [profiles.get(c, {}).get("latency_mean_ms", 0) for c in components]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(components, latencies, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.bar_label(bars, fmt="%.1f ms", padding=3)
    ax.set_ylabel("Mean Latency per Triplet (ms)")
    ax.set_title("Pre-filter Component Latency Breakdown")
    _save(fig, out_dir / "fig_4_7_latency_breakdown.png", config)


# Figure 4.8: Calibration Curves


def fig_calibration_curves(all_scores: list[PrefilterScore], config: dict) -> None:
    out_dir = _setup(config)
    n_bins = 10
    y_true = np.array([int(s.ground_truth_poisoned) for s in all_scores])
    y_score = np.array([s.aggregated_score for s in all_scores])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    bins = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_fracs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() > 0:
            bin_means.append(float(np.mean(y_score[mask])))
            bin_fracs.append(float(np.mean(y_true[mask])))

    ax.plot(bin_means, bin_fracs, "o-", label="Weighted Vote")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Aggregated)")
    ax.legend()
    _save(fig, out_dir / "fig_4_8_calibration_curves.png", config)


# Figure 4.9: Ablation Bars


def fig_ablation_bars(ablation_results: dict, config: dict) -> None:
    out_dir = _setup(config)
    injection_types = sorted(
        {r["injection_type"] for r in ablation_results.values() if "injection_type" in r}
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    variants = ["all_signals", "no_embedding", "no_entropy", "no_classifier"]
    x = np.arange(len(injection_types))
    width = 0.8 / len(variants)

    baseline_f1 = {
        inj: ablation_results.get(f"all_signals_{inj}", {}).get("f1", 0) for inj in injection_types
    }

    for v_idx, variant in enumerate(variants):
        f1s = [ablation_results.get(f"{variant}_{inj}", {}).get("f1", 0) for inj in injection_types]
        pos = x + (v_idx - len(variants) / 2 + 0.5) * width
        ax.bar(pos, f1s, width, label=variant.replace("_", " ").title())

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in injection_types])
    ax.set_ylabel("F1 Score")
    ax.set_title("Signal Ablation Study")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    _save(fig, out_dir / "fig_4_9_ablation_bars.png", config)


# Generate all figures


def generate_all(
    judge_scores: list[dict],
    baseline_scores: list[dict],
    postfilter_scores: list[dict],
    prefilter_scores: list[PrefilterScore],
    ablation_results: dict,
    profiles: dict,
    config: dict,
) -> None:
    print("Generating all thesis figures...")
    if judge_scores:
        fig_fpr_vs_noise_lines(judge_scores, config)
        fig_fpr_all_metrics(judge_scores, config)
        fig_fpr_heatmaps(judge_scores, config)
        fig_kde_score_distributions(judge_scores, config)
        fig_metric_means_overview(judge_scores, config)
    if prefilter_scores:
        fig_roc_curves(prefilter_scores, config)
        fig_confusion_matrices(prefilter_scores, config)
        fig_calibration_curves(prefilter_scores, config)
    if baseline_scores and postfilter_scores:
        fig_fpr_before_after(baseline_scores, postfilter_scores, config)
    if profiles:
        fig_latency_breakdown(profiles, config)
    if ablation_results:
        fig_ablation_bars(ablation_results, config)
    print("All figures saved to results/figures/")


def _infer_injection(triplet_id: str) -> str:
    if "_rn_" in triplet_id or "random_noise" in triplet_id:
        return "random_noise"
    if "_af_" in triplet_id or "adversarial_fact" in triplet_id:
        return "adversarial_fact"
    if "_pr_" in triplet_id or "poisonedrag_style" in triplet_id:
        return "poisonedrag_style"
    return "unknown"
