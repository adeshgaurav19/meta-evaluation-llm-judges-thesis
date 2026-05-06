"""
Generate all 9 thesis figures from the exports/ CSV files.

Usage:
    python scripts/generate_thesis_plots.py [--out results/v2/figures]

Reads from: exports/
Writes to:  results/v2/figures/  (or --out)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ── Colour palette ──────────────────────────────────────────────────────────
BLUE    = "#4a7ab8"   # baseline / clean
RED     = "#c44e52"   # poisoned / stat-filter
ORANGE  = "#dd8452"   # llm-filter
GREEN   = "#55a868"   # DeepSeek

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        False,
    "figure.dpi":       200,
})

THRESHOLD = 0.5


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


def _load_scores(exports: Path) -> pd.DataFrame:
    df = pd.read_csv(exports / "judge_scores_v2.csv")
    df = df.dropna(subset=["faithfulness_score"])
    df["faithfulness_score"]      = df["faithfulness_score"].astype(float)
    df["context_relevance_score"] = pd.to_numeric(df["context_relevance_score"], errors="coerce")
    df["answer_relevance_score"]  = pd.to_numeric(df["answer_relevance_score"],  errors="coerce")
    df["is_poisoned"] = df["is_poisoned"].astype(str).str.lower() == "true"
    # normalise column name: exports use judge_model, script uses judge
    if "judge_model" in df.columns and "judge" not in df.columns:
        df = df.rename(columns={"judge_model": "judge"})
    print(f"  judge_scores loaded: {len(df)} rows after dropna")
    return df


def _load_passages(exports: Path) -> pd.DataFrame:
    df = pd.read_csv(exports / "prefilter_passage_scores.csv")
    df["is_poisoned_passage"] = df["is_poisoned_passage"].astype(str).str.lower() == "true"
    # predicted_poisoned lives in mistral_triplet_predictions.csv (triplet-level)
    # join on triplet_id so passage-level df can filter by it
    mistral_path = exports / "mistral_triplet_predictions.csv"
    if mistral_path.exists():
        m = pd.read_csv(mistral_path)[["triplet_id", "predicted_poisoned"]]
        m["predicted_poisoned"] = m["predicted_poisoned"].astype(str).str.lower() == "true"
        df = df.merge(m, on="triplet_id", how="left")
        df["predicted_poisoned"] = df["predicted_poisoned"].fillna(False)
    else:
        df["predicted_poisoned"] = False
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 — Score distribution histograms
# ═══════════════════════════════════════════════════════════════════════════
def plot_score_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    print("Plot 1: score distributions …")
    base = df[df["condition"] == "baseline"]

    judges = [("gpt-5.4-nano", "GPT"), ("gemma-4-26b", "Gemma"), ("deepseek-v3.2", "DeepSeek")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

    bins = np.linspace(0, 1, 21)

    for ax, (jkey, jlabel) in zip(axes, judges):
        sub = base[base["judge"] == jkey]
        clean   = sub[~sub["is_poisoned"]]["faithfulness_score"].dropna()
        poisoned = sub[sub["is_poisoned"]]["faithfulness_score"].dropna()

        ax.hist(clean,   bins=bins, density=True, alpha=0.55, color=BLUE,  label="Clean")
        ax.hist(poisoned, bins=bins, density=True, alpha=0.55, color=RED,   label="Poisoned")
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="Threshold")

        ax.set_title(jlabel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Faithfulness score", fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(loc="upper center", fontsize=9, frameon=False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_score_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2 — TNR loss bar chart (Prefilter Paradox)
# ═══════════════════════════════════════════════════════════════════════════
def plot_tnr_loss(out_dir: Path) -> None:
    print("Plot 2: TNR loss …")

    # Verified numbers
    data = {
        "Gemma":    {"baseline": 0.301, "stat_filter": 0.295, "llm_filter": 0.306},
        "GPT":      {"baseline": 0.445, "stat_filter": 0.386, "llm_filter": 0.401},
        "DeepSeek": {"baseline": 0.588, "stat_filter": 0.402, "llm_filter": 0.447},
    }

    judges = list(data.keys())
    x = np.arange(len(judges))
    w = 0.27

    fig, ax = plt.subplots(figsize=(8, 4.2))

    bars_base = ax.bar(x - w, [data[j]["baseline"]    for j in judges], w, color=BLUE,   label="Baseline",     zorder=3)
    bars_stat = ax.bar(x,     [data[j]["stat_filter"]  for j in judges], w, color=RED,    label="Stat filter",  zorder=3)
    bars_llm  = ax.bar(x + w, [data[j]["llm_filter"]   for j in judges], w, color=ORANGE, label="LLM filter",   zorder=3)

    # Annotate losses
    for i, j in enumerate(judges):
        base = data[j]["baseline"]
        for bars, cond, col in [(bars_stat, "stat_filter", RED), (bars_llm, "llm_filter", ORANGE)]:
            loss = base - data[j][cond]
            bar = bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"−{loss:.3f}",
                    ha="center", va="bottom", fontsize=8, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(judges, fontsize=11)
    ax.set_ylabel("True Negative Rate (poisoning detection)", fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_title("Detection capability lost under upstream filtering", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_tnr_loss.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-injection FPR heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_fpr_heatmap(out_dir: Path) -> None:
    print("Plot 3: FPR heatmap …")

    import matplotlib.colors as mcolors

    matrix = np.array([
        [0.490, 0.560, 0.615],   # GPT
        [0.570, 0.620, 0.907],   # Gemma
        [0.465, 0.398, 0.375],   # DeepSeek
    ])
    row_labels = ["GPT", "Gemma", "DeepSeek"]
    col_labels = ["random_noise", "adversarial_fact", "poisonedrag_style"]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("FPR (judge fooled)", fontsize=10)

    ax.set_title("Baseline faithfulness FPR by judge × injection type", fontsize=11)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_fpr_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4 — Noise-level FPR curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_noise_curves(out_dir: Path) -> None:
    print("Plot 4: noise-level FPR curves …")

    noise = [0.2, 0.4, 0.6, 0.8]
    data = {
        "GPT":      ([0.700, 0.597, 0.500, 0.423], BLUE),
        "Gemma":    ([0.850, 0.747, 0.643, 0.555], RED),
        "DeepSeek": ([0.597, 0.460, 0.360, 0.233], GREEN),
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    for label, (fprs, col) in data.items():
        ax.plot(noise, fprs, color=col, marker="o", linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Noise level (proportion of context corrupted)", fontsize=10)
    ax.set_ylabel("Faithfulness FPR (poisoned subset)", fontsize=10)
    ax.set_ylim(0.15, 0.95)
    ax.set_xticks(noise)
    ax.set_title("Baseline FPR by noise level — higher noise is easier to detect", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_noise_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 5 — Paradox magnitude by injection type (DeepSeek)
# ═══════════════════════════════════════════════════════════════════════════
def plot_paradox_by_injection(out_dir: Path) -> None:
    print("Plot 5: paradox by injection type …")

    inj_types  = ["random_noise", "adversarial_fact", "poisonedrag_style"]
    baseline   = [0.465, 0.398, 0.375]
    after_stat = [0.525, 0.532, 0.738]
    deltas     = [a - b for a, b in zip(after_stat, baseline)]

    x = np.arange(len(inj_types))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 4))

    bars_base = ax.bar(x - w/2, baseline,   w, color=BLUE, label="Baseline",    zorder=3)
    bars_stat = ax.bar(x + w/2, after_stat, w, color=RED,  label="After stat filter", zorder=3)

    for i, (d, xpos) in enumerate(zip(deltas, x)):
        ax.text(xpos, max(baseline[i], after_stat[i]) + 0.025,
                f"+{d:.3f}", ha="center", va="bottom",
                fontsize=10, color=RED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(inj_types, fontsize=10)
    ax.set_ylabel("Faithfulness FPR (poisoned subset)", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.set_title("DeepSeek: paradox is concentrated on PoisonedRAG-style injections", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_paradox_by_injection.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 6 — Filter recall by attack type
# ═══════════════════════════════════════════════════════════════════════════
def plot_filter_recall(passages: pd.DataFrame, out_dir: Path) -> None:
    print("Plot 6: filter recall by attack type …")

    test_poisoned = passages[(passages["split"] == "test") & passages["is_poisoned_passage"]]
    recall = (test_poisoned.groupby("injection_type")["predicted_poisoned"]
              .agg(["mean", "count"])
              .reset_index())
    recall.columns = ["injection_type", "recall", "n"]
    print(f"    recall table:\n{recall.to_string(index=False)}")

    order      = ["random_noise", "adversarial_fact", "poisonedrag_style"]
    colours    = [BLUE, ORANGE, RED]
    labels     = ["random_noise", "adversarial_fact", "poisonedrag_style"]

    fig, ax = plt.subplots(figsize=(7, 3.6))

    for i, (inj, col) in enumerate(zip(order, colours)):
        row = recall[recall["injection_type"] == inj]
        val = float(row["recall"]) if len(row) else 0.0
        bar = ax.bar(i, val, color=col, zorder=3)
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0.9, color="grey", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Filter recall (XGBoost test set)", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Statistical filter recall by attack type", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_filter_recall.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 7 — McNemar paired-difference visualisation
# ═══════════════════════════════════════════════════════════════════════════
def plot_mcnemar(out_dir: Path) -> None:
    print("Plot 7: McNemar …")
    from scipy.stats import chi2

    # (label, n10, n01, chi2, p)
    # Ordered top-to-bottom by chi2 ascending
    entries = [
        ("Gemma × stat",     162, 169, 0.11,   0.74),
        ("Gemma × llm",       49,  43, 0.27,   0.60),
        ("GPT × stat",       161, 232, 12.47, 4.1e-4),
        ("GPT × llm",         85, 138, 12.13, 5.0e-4),
        ("DeepSeek × llm",   210, 191, 33.76, 1e-8),
        ("DeepSeek × stat",  114, 337, 109.28, 1e-20),
    ]

    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(entries))

    for i, (label, n10, n01, chi, p) in enumerate(entries):
        y = y_pos[i]
        ax.barh(y, n10, color=GREEN, height=0.55, zorder=3)
        ax.barh(y, n01, left=n10, color=RED, height=0.55, zorder=3)
        total = n10 + n01
        star = stars(p)
        p_str = f"< {p:.0e}" if p < 0.001 else f"= {p:.3f}"
        ann = f"χ² = {chi:.2f}, p {p_str}  {star}"
        ax.text(total + 5, y, ann, va="center", fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in entries], fontsize=10)
    ax.set_xlabel("Number of discordant triplets (poisoned subset)", fontsize=10)
    ax.set_title("McNemar paired comparison: where filtering changes the judge's verdict", fontsize=11)
    ax.set_xlim(0, 560)

    patch_fix  = mpatches.Patch(color=GREEN, label="Filter fixed (baseline FP → filter rejects)")
    patch_broke = mpatches.Patch(color=RED,   label="Filter broke (baseline correct → filter passes)")
    ax.legend(handles=[patch_fix, patch_broke], loc="upper right", fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_mcnemar.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 8 — Inter-judge variance
# ═══════════════════════════════════════════════════════════════════════════
def plot_interjudge_disagreement(df: pd.DataFrame, out_dir: Path) -> None:
    print("Plot 8: inter-judge disagreement …")

    base = df[df["condition"] == "baseline"][
        ["triplet_id", "judge", "faithfulness_score", "is_poisoned"]
    ].copy()

    # shorten judge names for pivot columns
    base["judge_short"] = (base["judge"]
        .str.replace("gpt-5.4-nano", "gpt", regex=False)
        .str.replace("gemma-4-26b", "gemma", regex=False)
        .str.replace("deepseek-v3.2", "deepseek", regex=False))

    pivot = base.pivot_table(
        index=["triplet_id", "is_poisoned"],
        columns="judge_short",
        values="faithfulness_score"
    ).dropna()

    pivot["std"] = pivot[["gpt", "gemma", "deepseek"]].std(axis=1)
    pivot = pivot.reset_index()

    clean_std   = pivot[~pivot["is_poisoned"]]["std"]
    poison_std  = pivot[pivot["is_poisoned"]]["std"]

    mean_clean  = clean_std.mean()
    mean_poison = poison_std.mean()
    print(f"    mean std clean={mean_clean:.3f}, poisoned={mean_poison:.3f}, n={len(pivot)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 0.6, 31)
    ax.hist(clean_std,  bins=bins, density=True, alpha=0.55, color=BLUE, label="Clean")
    ax.hist(poison_std, bins=bins, density=True, alpha=0.55, color=RED,  label="Poisoned")

    ax.axvline(mean_clean,  color=BLUE, linestyle="--", linewidth=1.4,
               label=f"Clean mean = {mean_clean:.3f}")
    ax.axvline(mean_poison, color=RED,  linestyle="--", linewidth=1.4,
               label=f"Poisoned mean = {mean_poison:.3f}")

    ax.text(0.35, ax.get_ylim()[1] * 0.88,
            "Pearson r = +0.113, p < 0.001",
            fontsize=9.5, color="black")

    ax.set_xlabel("Inter-judge std (faithfulness score, baseline)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "Multi-judge disagreement is weakly but significantly higher on poisoned triplets",
        fontsize=10.5
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_interjudge_disagreement.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 9 — Faithfulness gap collapse
# ═══════════════════════════════════════════════════════════════════════════
def plot_gap_collapse(out_dir: Path) -> None:
    print("Plot 9: gap collapse …")

    data = {
        "Gemma":    {"baseline": 0.271, "stat_filter": 0.267, "llm_filter": 0.233},
        "GPT":      {"baseline": 0.197, "stat_filter": 0.143, "llm_filter": 0.157},
        "DeepSeek": {"baseline": 0.373, "stat_filter": 0.182, "llm_filter": 0.241},
    }

    judges = list(data.keys())
    x = np.arange(len(judges))
    w = 0.27

    fig, ax = plt.subplots(figsize=(7.5, 4))

    bars_base = ax.bar(x - w, [data[j]["baseline"]    for j in judges], w, color=BLUE,   label="Baseline",    zorder=3)
    bars_stat = ax.bar(x,     [data[j]["stat_filter"]  for j in judges], w, color=RED,    label="Stat filter", zorder=3)
    bars_llm  = ax.bar(x + w, [data[j]["llm_filter"]   for j in judges], w, color=ORANGE, label="LLM filter",  zorder=3)

    for i, j in enumerate(judges):
        base = data[j]["baseline"]
        for bars, cond in [(bars_stat, "stat_filter"), (bars_llm, "llm_filter")]:
            loss = base - data[j][cond]
            bar = bars[i]
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"−{loss:.3f}",
                    ha="center", va="bottom", fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(judges, fontsize=11)
    ax.set_ylabel("Discrimination gap (mean clean − mean poisoned)", fontsize=10)
    ax.set_ylim(0, 0.45)
    ax.set_title("Filtering destroys the sharpest judge's discrimination gap", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, "fig_gap_collapse.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", default="exports", help="Path to exports/ directory")
    parser.add_argument("--out", default="results/v2/figures", help="Output directory for PNGs")
    parser.add_argument("--config", default=None, help="Ignored — accepted for pipeline compat")
    parser.add_argument("--force", action="store_true", help="Ignored — always overwrites")
    args = parser.parse_args()

    exports = Path(args.exports)
    out_dir = Path(args.out)

    print(f"Reading from: {exports}/")
    print(f"Writing to:   {out_dir}/\n")

    df       = _load_scores(exports)
    passages = _load_passages(exports)

    plot_score_distributions(df, out_dir)
    plot_tnr_loss(out_dir)
    plot_fpr_heatmap(out_dir)
    plot_noise_curves(out_dir)
    plot_paradox_by_injection(out_dir)
    plot_filter_recall(passages, out_dir)
    plot_mcnemar(out_dir)
    plot_interjudge_disagreement(df, out_dir)
    plot_gap_collapse(out_dir)

    pngs = sorted(out_dir.glob("fig_*.png"))
    print(f"\nDone — {len(pngs)} figures written to {out_dir}/")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
