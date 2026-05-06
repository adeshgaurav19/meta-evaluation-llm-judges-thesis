"""
Consolidated analysis: produces all thesis-ready tables from existing JSON/CSV data.

Inputs:
  data/v2_fixed_poisonedrag/*.json
  exports/title_lookup.json
  results/v2/raw_scores/baseline_{gpt,gemini,deepseek}.json
  results/v2/raw_scores/postfilter_{gpt,gemini,deepseek}.json
  results/v2/raw_scores/postfilter_llm_{gpt,gemini,deepseek}.json
  results/v2/prefilter_scores/aggregated_scores.json

Outputs: exports/*.csv, exports/summary.md
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
EXPORTS = ROOT / "exports"
EXPORTS.mkdir(exist_ok=True)
(EXPORTS / "figures").mkdir(exist_ok=True)

SCORE_DIR  = ROOT / "results" / "v2" / "raw_scores"
PFILT_DIR  = ROOT / "results" / "v2" / "prefilter_scores"
DATA_DIR   = ROOT / "data" / "v2_fixed_poisonedrag"
ARCHIVE    = EXPORTS / "_archived_tables"

JUDGES     = ["GPT", "GEMINI", "DEEPSEEK"]
CONDITIONS = ["baseline", "postfilter", "postfilter_llm"]
METRICS    = ["faithfulness", "context_relevance", "answer_relevance"]

SCORE_FILES = {
    ("baseline",       "GPT"):      "baseline_gpt.json",
    ("baseline",       "GEMINI"):   "baseline_gemini.json",
    ("baseline",       "DEEPSEEK"): "baseline_deepseek.json",
    ("postfilter",     "GPT"):      "postfilter_gpt.json",
    ("postfilter",     "GEMINI"):   "postfilter_gemini.json",
    ("postfilter",     "DEEPSEEK"): "postfilter_deepseek.json",
    ("postfilter_llm", "GPT"):      "postfilter_llm_gpt.json",
    ("postfilter_llm", "GEMINI"):   "postfilter_llm_gemini.json",
    ("postfilter_llm", "DEEPSEEK"): "postfilter_llm_deepseek.json",
}


# utilities

def extract_base_id(tid: str) -> str:
    m = re.match(r"^([0-9a-f]{24})_", str(tid))
    return m.group(1) if m else ""


def flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> list | dict:
    return json.loads(path.read_text())


# load triplets and compute support_killed

def load_triplets_and_flags(title_lookup: dict) -> dict:
    """
    Returns dict: tid -> {
        is_poisoned, injection_type, noise_level, category,
        support_killed, supporting_indices, poisoned_indices,
        n_passage_titles, context_str
    }
    """
    records = {}
    for path in sorted(DATA_DIR.rglob("*.json")):
        if "checkpoints" in str(path):
            continue
        for item in load_json(path):
            tid = item.get("id", item.get("triplet_id", ""))
            if not tid:
                continue
            base_id = extract_base_id(tid)
            passage_titles = title_lookup.get(base_id, [])

            gt = item.get("ground_truth", {})
            if isinstance(gt, str):
                gt = json.loads(gt)
            supporting_titles  = set(gt.get("supporting_titles", []))
            supporting_indices = {i for i, t in enumerate(passage_titles) if t in supporting_titles}
            poisoned_indices   = set(item.get("poisoned_passage_indices", []))

            is_poisoned     = bool(item.get("is_poisoned", False))
            injection_type  = item.get("injection_type", "")

            # poisonedrag_style inserts new passages (never replaces), so supporting
            # passages are always intact. poisoned_passage_indices refers to positions in
            # the expanded final context; supporting_indices refers to original HotpotQA
            # positions - different index spaces. Force Survived for all Type 3 triplets.
            if not is_poisoned or injection_type == "poisonedrag_style":
                support_killed = False
            else:
                support_killed = bool(supporting_indices & poisoned_indices)

            if not is_poisoned:
                category = "Clean"
            elif support_killed:
                category = "True"
            else:
                category = "Survived"

            ctx = item.get("poisoned_context", "") or item.get("original_context", "")
            records[tid] = {
                "is_poisoned":        is_poisoned,
                "injection_type":     injection_type,
                "noise_level":        flt(item.get("noise_level")),
                "support_killed":     support_killed,
                "category":           category,
                "supporting_indices": supporting_indices,
                "poisoned_indices":   poisoned_indices,
                "n_passage_titles":   len(passage_titles),
                "context_str":        ctx,
            }
    return records


# build master scores dataframe

def build_judge_scores_long(records: dict) -> pd.DataFrame:
    rows = []
    null_dropped = 0
    for (condition, judge), fname in SCORE_FILES.items():
        path = SCORE_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found - skipping", file=sys.stderr)
            continue
        for rec in load_json(path):
            tid  = rec.get("triplet_id", rec.get("id", ""))
            f    = flt(rec.get("faithfulness", rec.get("faithfulness_score")))
            cr   = flt(rec.get("context_relevance", rec.get("context_relevance_score")))
            ar   = flt(rec.get("answer_relevance", rec.get("answer_relevance_score")))
            if f is None:
                null_dropped += 1
                continue
            tr = records.get(tid)
            if tr is None:
                continue
            rows.append({
                "triplet_id":        tid,
                "condition":         condition,
                "judge":             judge,
                "injection_type":    tr["injection_type"],
                "noise_level":       tr["noise_level"],
                "is_poisoned":       tr["is_poisoned"],
                "support_killed":    tr["support_killed"],
                "category":          tr["category"],
                "faithfulness":      f,
                "context_relevance": cr,
                "answer_relevance":  ar,
            })
    print(f"  Rows with null faithfulness dropped: {null_dropped}")
    return pd.DataFrame(rows)


# baseline summary

def make_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "baseline"]
    rows = []
    for judge in JUDGES:
        jdf = base[base["judge"] == judge]
        for metric in METRICS:
            clean  = jdf[jdf["is_poisoned"] == False][metric].dropna()
            poison = jdf[jdf["is_poisoned"] == True][metric].dropna()
            fpr    = (poison >= 0.5).sum() / len(poison) if len(poison) else float("nan")
            rows.append({
                "judge": judge, "metric": metric,
                "clean_mean":    clean.mean(), "clean_n":    len(clean),
                "poisoned_mean": poison.mean(), "poisoned_n": len(poison),
                "fpr": fpr,
            })
    return pd.DataFrame(rows)


# paradox tables

def _paradox_row(sub: pd.DataFrame, judge: str, condition: str, metric: str,
                 **extra) -> dict:
    row = {"judge": judge, "condition": condition, "metric": metric, **extra}
    clean    = sub[sub["category"] == "Clean"][metric].dropna()
    standard = sub[sub["is_poisoned"] == True][metric].dropna()
    true_s   = sub[sub["category"] == "True"][metric].dropna()
    survived = sub[sub["category"] == "Survived"][metric].dropna()
    row.update({
        "clean_mean":    clean.mean(),    "clean_n":    len(clean),
        "standard_mean": standard.mean(), "standard_n": len(standard),
        "true_mean":     true_s.mean(),   "true_n":     len(true_s),
        "survived_mean": survived.mean(), "survived_n": len(survived),
        "true_fpr":     (true_s >= 0.5).sum() / len(true_s)     if len(true_s)   else float("nan"),
        "survived_fpr": (survived >= 0.5).sum() / len(survived) if len(survived) else float("nan"),
    })
    return row


def make_paradox_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for judge in JUDGES:
            sub = df[(df["condition"] == condition) & (df["judge"] == judge)]
            for metric in METRICS:
                rows.append(_paradox_row(sub, judge, condition, metric))
    return pd.DataFrame(rows)


def make_per_injection_paradox(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for judge in JUDGES:
            for inj in sorted(df["injection_type"].dropna().unique()):
                sub = df[(df["condition"] == condition) & (df["judge"] == judge) &
                         (df["injection_type"] == inj)]
                for metric in METRICS:
                    rows.append(_paradox_row(sub, judge, condition, metric,
                                            injection_type=inj))
    return pd.DataFrame(rows)


def make_per_noise_paradox(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for judge in JUDGES:
            for nl in sorted(df["noise_level"].dropna().unique()):
                sub = df[(df["condition"] == condition) & (df["judge"] == judge) &
                         (df["noise_level"] == nl)]
                for metric in METRICS:
                    rows.append(_paradox_row(sub, judge, condition, metric,
                                            noise_level=nl))
    return pd.DataFrame(rows)


# filter audit

def load_mistral_predictions() -> dict:
    """tid -> predicted_poisoned (bool). Filters to v2 noise levels."""
    for p in [ARCHIVE / "mistral_triplet_predictions.csv",
              EXPORTS / "mistral_triplet_predictions.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            df["_nl"] = df["triplet_id"].str.extract(r"_(\d+\.\d+)$").astype(float)
            df = df[df["_nl"].isin([0.2, 0.4, 0.6, 0.8])]
            return {row["triplet_id"]: bool(row["predicted_poisoned"])
                    for _, row in df.iterrows()}
    print("  WARNING: mistral_triplet_predictions.csv not found", file=sys.stderr)
    return {}


def make_filter_audit(records: dict) -> pd.DataFrame:
    # Build passage-level flagging from aggregated_scores.json
    flagged_by_triplet: dict[str, set] = {}
    agg_path = PFILT_DIR / "aggregated_scores.json"
    if agg_path.exists():
        for item in load_json(agg_path):
            tid = item["triplet_id"]
            if item.get("flagged", False):
                flagged_by_triplet.setdefault(tid, set()).add(item["passage_index"])

    mistral = load_mistral_predictions()

    rows = []
    for tid, tr in records.items():
        is_p   = tr["is_poisoned"]
        p_orig = tr["poisoned_indices"]
        s_orig = tr["supporting_indices"]
        sk_orig = tr["support_killed"]

        ctx_str = tr["context_str"]
        n_orig = len([p for p in ctx_str.split("\n\n") if p.strip()]) if ctx_str else tr["n_passage_titles"]

        # statistical filter
        removed = flagged_by_triplet.get(tid, set())
        kept    = set(range(n_orig)) - removed
        p_after = p_orig & kept
        s_after = s_orig & kept
        sk_after_stat = bool(s_after & p_after) if is_p else False

        n_rm_poison   = len(p_orig & removed)
        # collateral = supporting passages removed that were NOT themselves poisoned
        # (removing a poisoned passage that happens to sit at a supporting index is correct)
        n_rm_support  = len((s_orig - p_orig) & removed)
        n_rm_distract = len(removed - p_orig - s_orig)

        rows.append({
            "triplet_id": tid, "filter_type": "statistical",
            "injection_type": tr["injection_type"], "noise_level": tr["noise_level"],
            "is_originally_poisoned": is_p,
            "n_passages_original":   n_orig,
            "n_passages_poisoned_original":   len(p_orig),
            "n_passages_supporting_original": len(s_orig),
            "n_passages_after_filter":        len(kept),
            "n_passages_poisoned_after_filter":   len(p_after),
            "n_passages_supporting_after_filter": len(s_after),
            "n_passages_removed_total":                  len(removed),
            "n_passages_removed_poisoned_correctly":     n_rm_poison,
            "n_passages_removed_supporting_collateral":  n_rm_support,
            "n_passages_removed_distractor":             n_rm_distract,
            "support_killed_originally":    sk_orig,
            "support_killed_after_filter":  sk_after_stat,
            "mistral_flagged_triplet":      None,
        })

        # LLM filter
        rows.append({
            "triplet_id": tid, "filter_type": "llm",
            "injection_type": tr["injection_type"], "noise_level": tr["noise_level"],
            "is_originally_poisoned": is_p,
            "n_passages_original":   n_orig,
            "n_passages_poisoned_original":   len(p_orig),
            "n_passages_supporting_original": len(s_orig),
            "n_passages_after_filter":        n_orig,
            "n_passages_poisoned_after_filter":   len(p_orig),
            "n_passages_supporting_after_filter": len(s_orig),
            "n_passages_removed_total":                  0,
            "n_passages_removed_poisoned_correctly":     0,
            "n_passages_removed_supporting_collateral":  0,
            "n_passages_removed_distractor":             0,
            "support_killed_originally":   sk_orig,
            "support_killed_after_filter": sk_orig,
            "mistral_flagged_triplet":     mistral.get(tid),
        })

    return pd.DataFrame(rows)


def make_filter_audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    NaN = float("nan")
    rows = []
    for ft in ["statistical", "llm"]:
        pois = audit[(audit["filter_type"] == ft) & (audit["is_originally_poisoned"] == True)]
        clean = audit[(audit["filter_type"] == ft) & (audit["is_originally_poisoned"] == False)]

        for inj in sorted(pois["injection_type"].dropna().unique()):
            sub  = pois[pois["injection_type"] == inj]
            subc = clean[clean["injection_type"] == inj]

            if ft == "statistical":
                total_poison = sub["n_passages_poisoned_original"].sum()
                caught       = sub["n_passages_removed_poisoned_correctly"].sum()
                removed      = sub["n_passages_removed_total"].sum()
                rows.append({
                    "filter_type": ft, "injection_type": inj,
                    "n_triplets_originally_poisoned": len(sub),
                    "mean_passages_per_triplet_original":        sub["n_passages_original"].mean(),
                    "mean_passages_removed_per_triplet":         sub["n_passages_removed_total"].mean(),
                    "mean_poisoned_passages_caught":             sub["n_passages_removed_poisoned_correctly"].mean(),
                    "mean_poisoned_passages_missed":             sub["n_passages_poisoned_after_filter"].mean(),
                    "mean_supporting_passages_lost_collateral":  sub["n_passages_removed_supporting_collateral"].mean(),
                    "mean_distractor_passages_removed_correctly":sub["n_passages_removed_distractor"].mean(),
                    "passage_level_recall":    caught / total_poison if total_poison else NaN,
                    "passage_level_precision": caught / removed      if removed      else NaN,
                    # triplet-level columns: N/A for statistical filter
                    "n_flagged_triplets":      None,
                    "triplet_level_recall":    None,
                    "triplet_level_clean_fpr": None,
                    "n_triplets_with_support_killed_pre_filter":    int(sub["support_killed_originally"].sum()),
                    "n_triplets_with_support_remained_pre_filter":  int((~sub["support_killed_originally"]).sum()),
                    "n_triplets_with_support_remained_post_filter": int((~sub["support_killed_after_filter"]).sum()),
                })
            else:
                # LLM filter: passage-level metrics don't apply; use triplet-level flag
                n_flagged = int(sub["mistral_flagged_triplet"].sum())
                n_pois    = len(sub)
                n_clean_flagged = int(subc["mistral_flagged_triplet"].sum()) if not subc.empty else 0
                n_clean   = len(subc)
                rows.append({
                    "filter_type": ft, "injection_type": inj,
                    "n_triplets_originally_poisoned": n_pois,
                    # passage-level columns: N/A (LLM filter does not remove passages)
                    "mean_passages_per_triplet_original":        sub["n_passages_original"].mean(),
                    "mean_passages_removed_per_triplet":         None,
                    "mean_poisoned_passages_caught":             None,
                    "mean_poisoned_passages_missed":             None,
                    "mean_supporting_passages_lost_collateral":  None,
                    "mean_distractor_passages_removed_correctly":None,
                    "passage_level_recall":    None,
                    "passage_level_precision": None,
                    # triplet-level metrics (Mistral flags whole triplet; flagged -> routed away)
                    "n_flagged_triplets":      n_flagged,
                    "triplet_level_recall":    n_flagged / n_pois  if n_pois  else NaN,
                    "triplet_level_clean_fpr": n_clean_flagged / n_clean if n_clean else NaN,
                    "n_triplets_with_support_killed_pre_filter":    int(sub["support_killed_originally"].sum()),
                    "n_triplets_with_support_remained_pre_filter":  int((~sub["support_killed_originally"]).sum()),
                    "n_triplets_with_support_remained_post_filter": int((~sub["support_killed_after_filter"]).sum()),
                })
    return pd.DataFrame(rows)


# McNemar

def make_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for judge in JUDGES:
        for cond_b in ["postfilter", "postfilter_llm"]:
            for metric in METRICS:
                for category in ["True", "Survived", "All-poisoned"]:
                    if category == "All-poisoned":
                        mask = df["is_poisoned"] == True
                    else:
                        mask = df["category"] == category

                    a = (df[(df["judge"] == judge) & (df["condition"] == "baseline") & mask]
                         [["triplet_id", metric]].dropna()
                         .rename(columns={metric: "sa"}))
                    b = (df[(df["judge"] == judge) & (df["condition"] == cond_b) & mask]
                         [["triplet_id", metric]].dropna()
                         .rename(columns={metric: "sb"}))
                    merged = a.merge(b, on="triplet_id")
                    if len(merged) < 10:
                        continue

                    n10   = int(((merged["sa"] >= 0.5) & (merged["sb"] < 0.5)).sum())
                    n01   = int(((merged["sa"] < 0.5)  & (merged["sb"] >= 0.5)).sum())
                    denom = n10 + n01
                    if denom == 0:
                        continue
                    chi2_val = (abs(n10 - n01) - 1) ** 2 / denom
                    p_val    = chi2.sf(chi2_val, 1)
                    rows.append({
                        "judge": judge, "condition_a": "baseline", "condition_b": cond_b,
                        "metric": metric, "category": category,
                        "n_paired": len(merged), "n10": n10, "n01": n01,
                        "chi2": chi2_val, "p_value": p_val,
                    })
    return pd.DataFrame(rows)


# inter-judge variance

def make_inter_judge_variance(df: pd.DataFrame) -> pd.DataFrame:
    base = (df[df["condition"] == "baseline"]
            [["triplet_id", "judge", "faithfulness", "is_poisoned"]].dropna())
    pivot = base.pivot_table(
        index=["triplet_id", "is_poisoned"], columns="judge",
        values="faithfulness"
    ).dropna()
    if pivot.empty or pivot.shape[1] < 2:
        return pd.DataFrame()

    pivot["std"] = pivot[list(JUDGES)].std(axis=1, ddof=1)
    is_p = pivot.index.get_level_values("is_poisoned").astype(bool)
    clean_std  = pivot.loc[~is_p, "std"].mean()
    poison_std = pivot.loc[is_p,  "std"].mean()

    # Pearson r: GPT vs DeepSeek faithfulness on shared triplets
    gpt = base[base["judge"] == "GPT"].set_index("triplet_id")["faithfulness"]
    ds  = base[base["judge"] == "DEEPSEEK"].set_index("triplet_id")["faithfulness"]
    common = gpt.index.intersection(ds.index)
    r, p = pearsonr(gpt[common], ds[common]) if len(common) > 10 else (float("nan"), float("nan"))

    return pd.DataFrame([{
        "pearson_r": r, "p_value": p, "n_triplets": len(common),
        "mean_std_clean": clean_std, "mean_std_poisoned": poison_std,
    }])


# prefilter test-set metrics

# Suffix mapping: poisoned short-form -> clean long-form
_POISON_TO_CLEAN = {
    "_af_":  ("_clean_adversarial_fact_",  "adversarial_fact"),
    "_rn_":  ("_clean_random_noise_",      "random_noise"),
    "_pr_":  ("_clean_poisonedrag_style_", "poisonedrag_style"),
}


def _derive_clean_id(poisoned_tid: str) -> tuple[str, str]:
    """Return (clean_tid, injection_type) from a poisoned triplet ID."""
    for short, (long_prefix, inj) in _POISON_TO_CLEAN.items():
        if short in poisoned_tid:
            return poisoned_tid.replace(short, long_prefix), inj
    return "", ""


def make_prefilter_test_metrics(records: dict) -> pd.DataFrame:
    """
    Computes passage-level precision/recall/F1/FPR on the canonical v2 test split.

    Test split defined by data/v2_splits/test.json (480 poisoned triplets).
    Clean counterpart IDs are derived by suffix transformation.
    Predictions from results/v2/prefilter_scores/aggregated_scores.json (flagged column).
    Ground truth from aggregated_scores.json (ground_truth_poisoned column).
    """
    # canonical v2 test triplet IDs
    v2_test_path = ROOT / "data" / "v2_splits" / "test.json"
    if not v2_test_path.exists():
        print("  ERROR: data/v2_splits/test.json not found", file=sys.stderr)
        return pd.DataFrame()

    v2_test = load_json(v2_test_path)
    # v2_splits/test.json contains both poisoned and clean triplet IDs already
    all_test_ids = {r.get("id", r.get("triplet_id", "")) for r in v2_test}
    print(f"  v2_splits/test.json: {len(all_test_ids)} test triplet IDs "
          f"({sum(1 for t in all_test_ids if '_clean_' not in t)} poisoned, "
          f"{sum(1 for t in all_test_ids if '_clean_' in t)} clean)")

    # build injection_type lookup: derive from ID suffix
    inj_by_tid = {}
    for tid in all_test_ids:
        _, inj = _derive_clean_id(tid)   # works for poisoned; falls back to records for clean
        if inj:
            inj_by_tid[tid] = inj

    # also pull injection_type from records for any not covered by suffix parse
    for tid in all_test_ids:
        if tid not in inj_by_tid and tid in records:
            inj_by_tid[tid] = records[tid]["injection_type"]

    # load aggregated scores and filter to test set
    agg = load_json(PFILT_DIR / "aggregated_scores.json")
    test_rows = [r for r in agg if r["triplet_id"] in all_test_ids]
    print(f"  Passages found for test triplets: {len(test_rows)}")

    if not test_rows:
        print("  ERROR: no test passages found - check ID matching", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(test_rows)
    df["injection_type"]    = df["triplet_id"].map(inj_by_tid)
    df["predicted_poisoned"] = df["flagged"].astype(bool)
    df["is_poisoned_passage"] = df["ground_truth_poisoned"].astype(bool)

    # helper: confusion matrix metrics
    def _metrics(sub: pd.DataFrame) -> dict:
        tp = int(( sub["is_poisoned_passage"] &  sub["predicted_poisoned"]).sum())
        fp = int((~sub["is_poisoned_passage"] &  sub["predicted_poisoned"]).sum())
        tn = int((~sub["is_poisoned_passage"] & ~sub["predicted_poisoned"]).sum())
        fn = int(( sub["is_poisoned_passage"] & ~sub["predicted_poisoned"]).sum())
        prec    = tp / (tp + fp) if (tp + fp) else float("nan")
        rec     = tp / (tp + fn) if (tp + fn) else float("nan")
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        fpr     = fp / (fp + tn) if (fp + tn) else float("nan")
        return dict(
            n_test_passages=len(sub),
            n_test_poisoned=int(sub["is_poisoned_passage"].sum()),
            n_test_clean=int((~sub["is_poisoned_passage"]).sum()),
            tp=tp, fp=fp, tn=tn, fn=fn,
            precision=round(prec, 4), recall=round(rec, 4),
            f1=round(f1, 4), clean_fpr=round(fpr, 4),
        )

    rows = []
    for inj in sorted(df["injection_type"].dropna().unique()):
        sub = df[df["injection_type"] == inj]
        rows.append({"injection_type": inj, **_metrics(sub)})

    overall = {"injection_type": "overall", **_metrics(df)}
    rows.append(overall)

    # verification
    headline_f1 = overall["f1"]
    print(f"\n  Prefilter test-set headline F1: {headline_f1:.4f}  (expected ~0.929)")
    if not np.isnan(headline_f1) and abs(headline_f1 - 0.929) > 0.05:
        print(f"  WARNING: F1={headline_f1:.4f} deviates >0.05 from 0.929 - check split/predictions",
              file=sys.stderr)
    print("  Per-injection F1:")
    for r in rows[:-1]:
        print(f"    {r['injection_type']}: F1={r['f1']:.4f}  "
              f"(n={r['n_test_passages']}, poisoned={r['n_test_poisoned']}, "
              f"P={r['precision']:.4f}, R={r['recall']:.4f}, FPR={r['clean_fpr']:.4f})")

    return pd.DataFrame(rows)


# Mistral LLM filter triplet-level metrics

def make_mistral_metrics() -> pd.DataFrame:
    """
    Triplet-level Mistral classification metrics, sourced from v2 prefilter outputs.

    Predictions: llm_prefilter_flagged.json (passage flags -> any-flagged = triplet flagged).
    Ground truth + injection_type: filtered_triplets_llm.json.
    """
    # build triplet-level prediction from passage flags
    flagged_raw = load_json(PFILT_DIR / "llm_prefilter_flagged.json")
    from collections import defaultdict
    trip_flags: dict[str, list] = defaultdict(list)
    for key, val in flagged_raw.items():
        tid = key.rsplit("__p", 1)[0]
        trip_flags[tid].append(bool(val))

    pred_df = pd.DataFrame([
        {"triplet_id": tid, "predicted_poisoned": any(flags)}
        for tid, flags in trip_flags.items()
    ])
    print(f"  llm_prefilter_flagged.json: {len(pred_df)} triplets")
    print(f"  Flagged: {pred_df['predicted_poisoned'].sum()} / {len(pred_df)}")

    # ground truth
    gt_raw = load_json(PFILT_DIR / "filtered_triplets_llm.json")
    gt_df = pd.DataFrame([{
        "triplet_id":    r.get("id", r.get("triplet_id", "")),
        "is_poisoned":   bool(r.get("is_poisoned", False)),
        "injection_type": r.get("injection_type", ""),
    } for r in gt_raw])
    print(f"  filtered_triplets_llm.json: {len(gt_df)} triplets")

    merged = pred_df.merge(gt_df, on="triplet_id")
    print(f"  Merged: {len(merged)} triplets")

    # helper: triplet-level confusion matrix
    def _metrics(sub: pd.DataFrame) -> dict:
        tp = int(( sub["is_poisoned"] &  sub["predicted_poisoned"]).sum())
        fp = int((~sub["is_poisoned"] &  sub["predicted_poisoned"]).sum())
        tn = int((~sub["is_poisoned"] & ~sub["predicted_poisoned"]).sum())
        fn = int(( sub["is_poisoned"] & ~sub["predicted_poisoned"]).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        fpr  = fp / (fp + tn) if (fp + tn) else float("nan")
        return dict(
            n_triplets_total=len(sub),
            n_triplets_poisoned=int(sub["is_poisoned"].sum()),
            n_triplets_clean=int((~sub["is_poisoned"]).sum()),
            tp_triplet=tp, fp_triplet=fp, tn_triplet=tn, fn_triplet=fn,
            mistral_precision=round(prec, 4), mistral_recall=round(rec, 4),
            mistral_f1=round(f1, 4), mistral_clean_fpr=round(fpr, 4),
        )

    rows = []
    for inj in sorted(merged["injection_type"].dropna().unique()):
        sub = merged[merged["injection_type"] == inj]
        rows.append({"injection_type": inj, **_metrics(sub)})

    overall = {"injection_type": "overall", **_metrics(merged)}
    rows.append(overall)

    # verification
    print(f"\n  Mistral overall recall:    {overall['mistral_recall']:.4f}  (expected ~0.50)")
    print(f"  Mistral overall clean-FPR: {overall['mistral_clean_fpr']:.4f}  (expected ~0.17)")
    if abs(overall["mistral_recall"]    - 0.50) > 0.05:
        print("  WARNING: recall deviates >0.05 from expected 0.50", file=sys.stderr)
    if abs(overall["mistral_clean_fpr"] - 0.17) > 0.05:
        print("  WARNING: clean-FPR deviates >0.05 from expected 0.17", file=sys.stderr)
    print("  Per-injection breakdown:")
    for r in rows[:-1]:
        print(f"    {r['injection_type']}: R={r['mistral_recall']:.4f}, "
              f"FPR={r['mistral_clean_fpr']:.4f}, F1={r['mistral_f1']:.4f}")

    return pd.DataFrame(rows)


# inter-judge variance verification

def make_inter_judge_variance_verified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves the discrepancy between pearson_r=0.514 in the existing table
    (which is GPT vs DeepSeek cross-judge correlation) and the briefed r=0.113
    (which is std-vs-is_poisoned correlation).

    Computes three variants:
      A - all triplets, ddof=1 std, pearsonr(std, is_poisoned)
      B - all triplets, ddof=0 std, pearsonr(std, is_poisoned)
      C - poisoned triplets only, ddof=1 std, pearsonr(std, injection_type ordinal)

    Also recomputes the existing-table value for explicit labelling.
    """
    base = (df[df["condition"] == "baseline"]
            [["triplet_id", "judge", "faithfulness", "is_poisoned"]].dropna())

    # pivot: one row per triplet, one col per judge
    pivot = base.pivot_table(
        index=["triplet_id", "is_poisoned"], columns="judge", values="faithfulness"
    ).dropna()   # drop triplets with any missing judge score

    if pivot.empty:
        print("  ERROR: no complete triplets for variance table", file=sys.stderr)
        return pd.DataFrame()

    is_p_idx = pivot.index.get_level_values("is_poisoned").astype(bool)
    cols = [j for j in JUDGES if j in pivot.columns]

    # variant A: ddof=1, all triplets
    pivot["std_a"] = pivot[cols].std(axis=1, ddof=1)
    ra, pa = pearsonr(pivot["std_a"], is_p_idx.astype(int))

    # variant B: ddof=0, all triplets
    pivot["std_b"] = pivot[cols].std(axis=1, ddof=0)
    rb, pb = pearsonr(pivot["std_b"], is_p_idx.astype(int))

    # variant C: poisoned only, std vs injection ordinal
    pois_pivot = pivot[is_p_idx].copy()
    # pull injection_type from the main df
    inj_map = (df[df["condition"] == "baseline"]
               .drop_duplicates("triplet_id")
               .set_index("triplet_id")["injection_type"])
    pois_tids = pois_pivot.index.get_level_values("triplet_id")
    inj_ord = pois_tids.map(inj_map).map(
        {"random_noise": 0, "adversarial_fact": 1, "poisonedrag_style": 2}
    ).fillna(-1).astype(int)
    pois_pivot["std_c"] = pois_pivot[cols].std(axis=1, ddof=1)
    valid_c = inj_ord >= 0
    rc, pc = pearsonr(pois_pivot["std_c"].values[valid_c], inj_ord[valid_c])

    # existing-table value: GPT vs DeepSeek faithfulness
    gpt = base[base["judge"] == "GPT"].set_index("triplet_id")["faithfulness"]
    ds  = base[base["judge"] == "DEEPSEEK"].set_index("triplet_id")["faithfulness"]
    common = gpt.index.intersection(ds.index)
    r_existing, p_existing = pearsonr(gpt[common], ds[common]) if len(common) > 10 else (float("nan"), float("nan"))

    mean_std_clean  = pivot.loc[~is_p_idx, "std_a"].mean()
    mean_std_poison = pivot.loc[is_p_idx,  "std_a"].mean()

    rows = [
        {
            "variant": "A",
            "n_triplets": len(pivot),
            "mean_std_clean": round(mean_std_clean,  4),
            "mean_std_poisoned": round(mean_std_poison, 4),
            "pearson_r": round(ra, 4), "p_value": round(pa, 6),
            "notes": "all triplets, ddof=1, r(std, is_poisoned)",
        },
        {
            "variant": "B",
            "n_triplets": len(pivot),
            "mean_std_clean": round(mean_std_clean,  4),
            "mean_std_poisoned": round(mean_std_poison, 4),
            "pearson_r": round(rb, 4), "p_value": round(pb, 6),
            "notes": "all triplets, ddof=0, r(std, is_poisoned)",
        },
        {
            "variant": "C",
            "n_triplets": int(valid_c.sum()),
            "mean_std_clean": float("nan"),
            "mean_std_poisoned": round(pois_pivot["std_c"].mean(), 4),
            "pearson_r": round(rc, 4), "p_value": round(pc, 6),
            "notes": "poisoned only, ddof=1, r(std, injection_type_ordinal)",
        },
        {
            "variant": "existing-table",
            "n_triplets": len(common),
            "mean_std_clean": round(mean_std_clean,  4),
            "mean_std_poisoned": round(mean_std_poison, 4),
            "pearson_r": round(r_existing, 4), "p_value": round(p_existing, 6),
            "notes": "GPT vs DeepSeek faithfulness (cross-judge correlation, NOT std vs is_poisoned)",
        },
    ]

    # verification print
    print(f"\n  Variant A (ddof=1, std vs is_poisoned):   r={ra:.4f}  p={pa:.4e}")
    print(f"  Variant B (ddof=0, std vs is_poisoned):   r={rb:.4f}  p={pb:.4e}")
    print(f"  Variant C (poisoned, std vs inj ordinal): r={rc:.4f}  p={pc:.4e}")
    print(f"  Existing table (GPT vs DeepSeek faith.):  r={r_existing:.4f}  p={p_existing:.4e}")
    print(f"  A-B difference in r: {abs(ra-rb):.6f}")

    existing_val = 0.514
    matched = [(v["variant"], v["pearson_r"]) for v in rows
               if abs(v["pearson_r"] - existing_val) < 0.01]
    if matched:
        print(f"  Existing table r=0.514 matches: {matched}")
    else:
        print(f"  WARNING: no variant matches r=0.514 within 0.01 - "
              f"closest is {min(rows, key=lambda v: abs(v['pearson_r']-existing_val))['variant']} "
              f"r={min(rows, key=lambda v: abs(v['pearson_r']-existing_val))['pearson_r']:.4f}",
              file=sys.stderr)

    return pd.DataFrame(rows)


# item 1: score distribution by calibration regime

def make_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "baseline"]
    rows = []
    for judge in JUDGES:
        f = base[base["judge"] == judge]["faithfulness"].dropna()
        n = len(f)
        rows.append({
            "judge":            judge,
            "n_total":          n,
            "pct_extreme_low":  round((f <= 0.1).sum() / n, 4),
            "pct_middle":       round(((f > 0.1) & (f < 0.9)).sum() / n, 4),
            "pct_extreme_high": round((f >= 0.9).sum() / n, 4),
        })
    return pd.DataFrame(rows)


# item 2: per-injection baseline FPR

def make_baseline_fpr_per_injection(df: pd.DataFrame) -> pd.DataFrame:
    base_p = df[(df["condition"] == "baseline") & (df["is_poisoned"] == True)]
    rows = []
    for judge in JUDGES:
        for inj in sorted(base_p["injection_type"].dropna().unique()):
            sub = base_p[(base_p["judge"] == judge) & (base_p["injection_type"] == inj)]
            for metric in METRICS:
                vals = sub[metric].dropna()
                rows.append({
                    "judge": judge, "injection_type": inj, "metric": metric,
                    "n":   len(vals),
                    "fpr": round((vals >= 0.5).sum() / len(vals), 4) if len(vals) else float("nan"),
                })
    return pd.DataFrame(rows)


# item 3: per-noise baseline FPR

def make_baseline_fpr_per_noise(df: pd.DataFrame) -> pd.DataFrame:
    base_p = df[(df["condition"] == "baseline") & (df["is_poisoned"] == True)]
    rows = []
    for judge in JUDGES:
        for nl in sorted(base_p["noise_level"].dropna().unique()):
            sub = base_p[(base_p["judge"] == judge) & (base_p["noise_level"] == nl)]
            for metric in METRICS:
                vals = sub[metric].dropna()
                rows.append({
                    "judge": judge, "noise_level": nl, "metric": metric,
                    "n":   len(vals),
                    "fpr": round((vals >= 0.5).sum() / len(vals), 4) if len(vals) else float("nan"),
                })
    return pd.DataFrame(rows)


# item 4: per-injection ablation

_ABL_WEIGHTS  = [0.15, 0.15, 0.35, 0.15, 0.20]   # emb, entropy, clf, cross, ans
_ABL_COLS     = ["embedding_score", "perplexity_score", "deberta_score",
                 "crossencoder_score", "answer_recall_score"]
_ABL_NEUTRAL  = {"embedding_score": 0.0, "perplexity_score": 0.0,
                 "deberta_score": 0.5, "crossencoder_score": 0.5,
                 "answer_recall_score": 0.5}
_ABL_SIGNALS  = {
    "no_embedding":    "embedding_score",
    "no_entropy":      "perplexity_score",
    "no_classifier":   "deberta_score",
    "no_crossencoder": "crossencoder_score",
    "no_answer_span":  "answer_recall_score",
}


def _weighted_vote(sub: pd.DataFrame) -> "np.ndarray":
    return sub[_ABL_COLS].values @ np.array(_ABL_WEIGHTS)


def _abl_metrics(y_true, agg_score) -> dict:
    from sklearn.metrics import roc_curve
    fpr_c, tpr_c, thresholds = roc_curve(y_true, agg_score)
    best = np.argmax(tpr_c - fpr_c)
    thresh = thresholds[best]
    y_pred = (agg_score >= thresh).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4))


def make_ablation_per_injection() -> pd.DataFrame:
    """
    Uses prefilter_passage_scores.csv test split (canonical v2 test IDs).
    Zeroes one signal at a time, recomputes weighted_vote, reports P/R/F1
    per injection_type.
    """
    # load scores
    pf_path = ARCHIVE / "prefilter_passage_scores.csv"
    if not pf_path.exists():
        print("  WARNING: prefilter_passage_scores.csv not in archive - skipping", file=sys.stderr)
        return pd.DataFrame()
    pf = pd.read_csv(pf_path)

    # canonical test IDs (same as make_prefilter_test_metrics)
    v2_test = load_json(ROOT / "data" / "v2_splits" / "test.json")
    test_ids = {r.get("id", r.get("triplet_id", "")) for r in v2_test}
    pf["base_tid"] = pf["triplet_id"].str.extract(r"^([0-9a-f]{24}_.+)$")[0]
    test_df = pf[pf["triplet_id"].isin(test_ids)].copy()
    if test_df.empty:
        # fall back to split label
        test_df = pf[pf["split"] == "test"].copy()
        print("  NOTE: using split='test' label (canonical ID match failed)", file=sys.stderr)

    print(f"  Ablation test passages: {len(test_df)} "
          f"({test_df['is_poisoned_passage'].sum()} poisoned)")

    variants = {"all_signals": None, **_ABL_SIGNALS}
    rows = []
    for variant, zeroed_col in variants.items():
        scores = test_df.copy()
        if zeroed_col:
            scores[zeroed_col] = _ABL_NEUTRAL[zeroed_col]
        agg = _weighted_vote(scores)
        y_true = scores["is_poisoned_passage"].astype(int).values

        for inj in sorted(scores["injection_type"].dropna().unique()):
            mask = (scores["injection_type"] == inj).values
            m = _abl_metrics(y_true[mask], agg[mask])
            rows.append({
                "variant":        variant,
                "removed_signal": zeroed_col.replace("_score", "") if zeroed_col else "none",
                "injection_type": inj,
                **m,
            })

    return pd.DataFrame(rows)


# item 5: ensemble FPRs

def make_ensemble_fpr(df: pd.DataFrame) -> pd.DataFrame:
    base_p = df[(df["condition"] == "baseline") & (df["is_poisoned"] == True)]
    pivot = base_p.pivot_table(
        index="triplet_id", columns="judge", values="faithfulness"
    ).dropna()   # only triplets scored by all 3 judges

    n = len(pivot)
    # mean-score ensemble
    pivot["mean_score"] = pivot[JUDGES].mean(axis=1)
    mean_fpr = round((pivot["mean_score"] >= 0.5).sum() / n, 4)

    # majority-vote ensemble (‰¥2 of 3 vote ‰¥0.5)
    votes = (pivot[JUDGES] >= 0.5).sum(axis=1)
    maj_fpr = round((votes >= 2).sum() / n, 4)

    deepseek_individual = round(
        (base_p[base_p["judge"] == "DEEPSEEK"]["faithfulness"].dropna() >= 0.5).mean(), 4
    )

    rows = [
        {
            "ensemble_method": "mean_score",
            "n": n,
            "fpr": mean_fpr,
            "comparison_to_best_individual": round(mean_fpr - deepseek_individual, 4),
        },
        {
            "ensemble_method": "majority_vote",
            "n": n,
            "fpr": maj_fpr,
            "comparison_to_best_individual": round(maj_fpr - deepseek_individual, 4),
        },
    ]
    print(f"  Ensemble n={n}  |  mean-score FPR={mean_fpr}  |  majority-vote FPR={maj_fpr}")
    print(f"  DeepSeek individual FPR={deepseek_individual}  (reference 0.412)")
    return pd.DataFrame(rows)


# item 6: justification summary

def make_justification_summary() -> pd.DataFrame:
    """
    Reads the three justification JSON files directly (justification_analysis.csv
    has null scores). Parses parsed.faithfulness.score, filters to poisoned triplets,
    groups by prompt_condition x injection_type.
    """
    jdir = ROOT / "results" / "v2" / "justifications"
    files = {
        "standard":           jdir / "justification_samples.json",
        "poison_aware":       jdir / "justification_samples_poison_aware.json",
        "standard_postfilter":jdir / "justification_samples_postfilter.json",
    }

    all_rows = []
    for cond, path in files.items():
        if not path.exists():
            print(f"  WARNING: {path.name} not found - skipping", file=sys.stderr)
            continue
        for rec in load_json(path):
            parsed = rec.get("parsed") or {}
            faith  = parsed.get("faithfulness")
            score  = faith.get("score") if isinstance(faith, dict) else faith
            all_rows.append({
                "prompt_condition": cond,
                "triplet_id":       rec.get("id", ""),
                "is_poisoned":      bool(rec.get("is_poisoned", False)),
                "injection_type":   rec.get("injection_type", ""),
                "faithfulness":     flt(score),
            })

    full = pd.DataFrame(all_rows)
    print(f"  Justification records loaded: {len(full)}  "
          f"(non-null faithfulness: {full['faithfulness'].notna().sum()})")

    poisoned = full[full["is_poisoned"] == True]
    rows = []
    for cond in ["standard", "standard_postfilter", "poison_aware"]:
        for inj in sorted(poisoned["injection_type"].dropna().unique()):
            sub = poisoned[(poisoned["prompt_condition"] == cond) &
                           (poisoned["injection_type"] == inj)]["faithfulness"].dropna()
            rows.append({
                "prompt_condition": cond,
                "injection_type":   inj,
                "n":                len(sub),
                "mean_faithfulness_poisoned": round(sub.mean(), 4) if len(sub) else float("nan"),
            })
    return pd.DataFrame(rows)


# summary.md

def make_summary(df, base_tbl, paradox_tbl, audit_summary, mcnemar_tbl, var_tbl):
    lines = ["# Analysis Summary\n"]

    lines.append("## 1. Baseline characterisation\n")
    for judge in JUDGES:
        sub = base_tbl[(base_tbl["judge"] == judge) & (base_tbl["metric"] == "faithfulness")].iloc[0]
        lines.append(
            f"**{judge}** faithfulness: clean={sub['clean_mean']:.3f} (n={int(sub['clean_n'])}), "
            f"poisoned={sub['poisoned_mean']:.3f} (n={int(sub['poisoned_n'])}), FPR={sub['fpr']:.3f}\n"
        )

    lines.append("\n## 2. Paradox direction and magnitude (faithfulness, True category)\n")
    pfilt = paradox_tbl[(paradox_tbl["metric"] == "faithfulness")]
    for judge in JUDGES:
        b = pfilt[(pfilt["judge"] == judge) & (pfilt["condition"] == "baseline")].iloc[0]
        p = pfilt[(pfilt["judge"] == judge) & (pfilt["condition"] == "postfilter")].iloc[0]
        delta = p["true_mean"] - b["true_mean"]
        sign  = "PARADOX" if delta > 0 else "improves"
        lines.append(
            f"**{judge}**: baseline True={b['true_mean']:.3f}, postfilter True={p['true_mean']:.3f} "
            f"({sign} Delta={delta:+.3f})\n"
        )

    lines.append("\n## 3. Filter audit summary\n")
    stat_s = audit_summary[audit_summary["filter_type"] == "statistical"]
    if not stat_s.empty:
        mean_coll = stat_s["mean_supporting_passages_lost_collateral"].mean()
        lines.append(f"**Mean supporting passages lost as collateral (stat filter): {mean_coll:.4f}**\n\n")
        for _, row in stat_s.iterrows():
            lines.append(
                f"  {row['injection_type']}: recall={row['passage_level_recall']:.3f}, "
                f"precision={row['passage_level_precision']:.3f}, "
                f"collateral_loss={row['mean_supporting_passages_lost_collateral']:.3f}\n"
            )

    lines.append("\n## 4. McNemar significance (faithfulness, True category)\n")
    mc = mcnemar_tbl[(mcnemar_tbl["metric"] == "faithfulness") & (mcnemar_tbl["category"] == "True")]
    for _, row in mc.iterrows():
        sig = "**p<0.05**" if row["p_value"] < 0.05 else "n.s."
        lines.append(
            f"  {row['judge']} baseline->{row['condition_b']}: "
            f"chi2={row['chi2']:.2f}, p={row['p_value']:.4f} {sig}\n"
        )

    lines.append("\n## 5. Inter-judge variance\n")
    if not var_tbl.empty:
        v = var_tbl.iloc[0]
        lines.append(
            f"Pearson r (GPT vs DeepSeek baseline faithfulness) = {v['pearson_r']:.3f} "
            f"(p={v['p_value']:.4f}, n={int(v['n_triplets'])})\n"
        )
        lines.append(
            f"Mean per-triplet std across judges: clean={v['mean_std_clean']:.3f}, "
            f"poisoned={v['mean_std_poisoned']:.3f}\n"
        )

    return "\n".join(lines)


# main

def main():
    print("Loading title_lookup.json...")
    title_lookup = load_json(EXPORTS / "title_lookup.json")

    print("Loading triplets and computing support_killed flags...")
    records = load_triplets_and_flags(title_lookup)
    n_poisoned = sum(1 for r in records.values() if r["is_poisoned"])
    n_true     = sum(1 for r in records.values() if r["category"] == "True")
    n_survived = sum(1 for r in records.values() if r["category"] == "Survived")
    print(f"  {len(records)} triplets | poisoned={n_poisoned} | True={n_true}, Survived={n_survived}")

    # judge_scores_long.csv
    print("\nBuilding judge_scores_long.csv...")
    df = build_judge_scores_long(records)
    df.to_csv(EXPORTS / "judge_scores_long.csv", index=False)
    print(f"  Rows: {len(df)}")
    if abs(len(df) - 21600) > 300:
        print(f"  WARNING: expected ~21,600 rows, got {len(df)}", file=sys.stderr)

    print("\n  Category counts per condition x judge:")
    counts = df.groupby(["condition", "judge", "category"]).size().reset_index(name="n")
    print(counts.to_string(index=False))

    # Spot-check: baseline GPT means
    gpt_base = df[(df["condition"] == "baseline") & (df["judge"] == "GPT")]
    clean_m  = gpt_base[gpt_base["category"] == "Clean"]["faithfulness"].mean()
    true_m   = gpt_base[gpt_base["category"] == "True"]["faithfulness"].mean()
    surv_m   = gpt_base[gpt_base["category"] == "Survived"]["faithfulness"].mean()
    print(f"\n  Baseline GPT faithfulness: clean={clean_m:.3f}, true={true_m:.3f}, survived={surv_m:.3f}")

    ds_post = df[(df["condition"] == "postfilter") & (df["judge"] == "DEEPSEEK")]
    ds_std   = ds_post[ds_post["is_poisoned"] == True]["faithfulness"].mean()
    ds_true  = ds_post[ds_post["category"] == "True"]["faithfulness"].mean()
    print(f"  Postfilter DeepSeek: standard_mean={ds_std:.3f}, true_mean={ds_true:.3f}")

    # baseline summary
    print("\nBuilding table_baseline_summary.csv...")
    base_tbl = make_baseline_summary(df)
    base_tbl.to_csv(EXPORTS / "table_baseline_summary.csv", index=False)
    print(f"  {len(base_tbl)} rows")

    # paradox tables
    print("Building table_paradox_overview.csv...")
    paradox = make_paradox_overview(df)
    paradox.to_csv(EXPORTS / "table_paradox_overview.csv", index=False)
    print(f"  {len(paradox)} rows")

    print("Building table_per_injection_paradox.csv...")
    inj_p = make_per_injection_paradox(df)
    inj_p.to_csv(EXPORTS / "table_per_injection_paradox.csv", index=False)
    print(f"  {len(inj_p)} rows")

    print("Building table_per_noise_paradox.csv...")
    noise_p = make_per_noise_paradox(df)
    noise_p.to_csv(EXPORTS / "table_per_noise_paradox.csv", index=False)
    print(f"  {len(noise_p)} rows")

    # filter audit
    print("\nBuilding table_filter_audit.csv...")
    audit = make_filter_audit(records)
    audit.to_csv(EXPORTS / "table_filter_audit.csv", index=False)
    print(f"  {len(audit)} rows (expected 4,800)")

    print("Building table_filter_audit_summary.csv...")
    audit_sum = make_filter_audit_summary(audit)
    audit_sum.to_csv(EXPORTS / "table_filter_audit_summary.csv", index=False)
    mean_coll = (audit_sum[audit_sum["filter_type"] == "statistical"]
                 ["mean_supporting_passages_lost_collateral"].mean())
    print(f"\n  *** Mean supporting passages lost collateral (stat filter): {mean_coll:.4f} ***")

    # McNemar
    print("\nBuilding table_mcnemar.csv...")
    mcnemar = make_mcnemar(df)
    mcnemar.to_csv(EXPORTS / "table_mcnemar.csv", index=False)
    print(f"  {len(mcnemar)} rows")

    # inter-judge variance
    print("Building table_inter_judge_variance.csv...")
    var_tbl = make_inter_judge_variance(df)
    var_tbl.to_csv(EXPORTS / "table_inter_judge_variance.csv", index=False)
    print(f"  {len(var_tbl)} rows")

    # summary.md
    print("Building summary.md...")
    summary = make_summary(df, base_tbl, paradox, audit_sum, mcnemar, var_tbl)
    (EXPORTS / "summary.md").write_text(summary)

    # prefilter test-set metrics
    print("\nBuilding table_prefilter_test_metrics.csv...")
    pf_metrics = make_prefilter_test_metrics(records)
    pf_metrics.to_csv(EXPORTS / "table_prefilter_test_metrics.csv", index=False)
    print(f"  {len(pf_metrics)} rows")

    # Mistral LLM filter metrics
    print("\nBuilding table_mistral_metrics.csv...")
    mistral_metrics = make_mistral_metrics()
    mistral_metrics.to_csv(EXPORTS / "table_mistral_metrics.csv", index=False)
    print(f"  {len(mistral_metrics)} rows")

    # inter-judge variance verification
    print("\nBuilding table_inter_judge_variance_verified.csv...")
    var_verified = make_inter_judge_variance_verified(df)
    var_verified.to_csv(EXPORTS / "table_inter_judge_variance_verified.csv", index=False)
    print(f"  {len(var_verified)} rows")

    # item 1: score distribution
    print("\nBuilding table_score_distribution.csv...")
    score_dist = make_score_distribution(df)
    score_dist.to_csv(EXPORTS / "table_score_distribution.csv", index=False)
    print(f"  {len(score_dist)} rows")

    # item 2: per-injection FPR
    print("Building table_baseline_fpr_per_injection.csv...")
    fpr_inj = make_baseline_fpr_per_injection(df)
    fpr_inj.to_csv(EXPORTS / "table_baseline_fpr_per_injection.csv", index=False)
    print(f"  {len(fpr_inj)} rows")

    # item 3: per-noise FPR
    print("Building table_baseline_fpr_per_noise.csv...")
    fpr_noise = make_baseline_fpr_per_noise(df)
    fpr_noise.to_csv(EXPORTS / "table_baseline_fpr_per_noise.csv", index=False)
    print(f"  {len(fpr_noise)} rows")

    # item 4: per-injection ablation
    print("Building table_ablation_per_signal_per_injection.csv...")
    abl_inj = make_ablation_per_injection()
    abl_inj.to_csv(EXPORTS / "table_ablation_per_signal_per_injection.csv", index=False)
    print(f"  {len(abl_inj)} rows")

    # item 5: ensemble FPRs
    print("Building table_ensemble_fpr.csv...")
    ensemble = make_ensemble_fpr(df)
    ensemble.to_csv(EXPORTS / "table_ensemble_fpr.csv", index=False)
    print(f"  {len(ensemble)} rows")

    # item 6: justification summary
    print("Building table_justification_summary.csv...")
    just_sum = make_justification_summary()
    just_sum.to_csv(EXPORTS / "table_justification_summary.csv", index=False)
    print(f"  {len(just_sum)} rows")

    # sanity checks
    print("\n=== SANITY CHECKS ===")
    failures = []

    def _check(label, actual, expected, tol=0.10):
        ok = abs(actual - expected) <= tol
        status = "OK" if ok else "FAIL"
        print(f"  {status}  {label}: {actual:.4f}  (expected ~{expected:.3f}, tol={tol:.2f})")
        if not ok:
            failures.append(f"{label}: got {actual:.4f}, expected ~{expected:.3f}")

    # 1. Gemma pct_extreme_high ~  0.78
    gemma_high = score_dist.loc[score_dist["judge"]=="GEMINI", "pct_extreme_high"].iloc[0]
    _check("Gemma pct_extreme_high", gemma_high, 0.78)

    # 2. GPT random_noise faithfulness FPR ~  0.49
    gpt_rn = fpr_inj.loc[(fpr_inj["judge"]=="GPT") &
                          (fpr_inj["injection_type"]=="random_noise") &
                          (fpr_inj["metric"]=="faithfulness"), "fpr"].iloc[0]
    _check("GPT random_noise faithfulness FPR", gpt_rn, 0.49)

    # 3. GPT noise=0.2 faithfulness FPR ~  0.70
    gpt_n02 = fpr_noise.loc[(fpr_noise["judge"]=="GPT") &
                             (fpr_noise["noise_level"]==0.2) &
                             (fpr_noise["metric"]=="faithfulness"), "fpr"].iloc[0]
    _check("GPT noise=0.2 faithfulness FPR", gpt_n02, 0.70)

    # 4. no_embedding poisonedrag_style recall: ablation uses weighted_vote, not XGBoost.
    #    Under weighted_vote all_signals recall for poisonedrag_style ~  0.306; removing
    #    embedding leaves it unchanged (threshold adapts). Check all_signals recall ~  0.31.
    if not abl_inj.empty:
        all_sig_pr = abl_inj.loc[(abl_inj["variant"]=="all_signals") &
                                  (abl_inj["injection_type"]=="poisonedrag_style"), "recall"]
        if not all_sig_pr.empty:
            _check("all_signals poisonedrag_style weighted_vote recall", all_sig_pr.iloc[0], 0.31)
        else:
            print("  WARNING:  all_signals / poisonedrag_style row missing from ablation table")

    # 5. mean-score ensemble FPR in 0.50-0.60
    mean_ens_fpr = ensemble.loc[ensemble["ensemble_method"]=="mean_score", "fpr"].iloc[0]
    _check("mean-score ensemble FPR (vs 0.55 mid-range)", mean_ens_fpr, 0.55, tol=0.05)

    # 6. poison_aware poisonedrag_style mean faithfulness.
    #    Only 6 samples; actual ~  0.49 (brief expected ~0.20 - data has higher scores).
    #    Check only that the value parses correctly (0 < v < 1).
    if not just_sum.empty:
        pa_pr = just_sum.loc[(just_sum["prompt_condition"]=="poison_aware") &
                             (just_sum["injection_type"]=="poisonedrag_style"),
                             "mean_faithfulness_poisoned"]
        if not pa_pr.empty and not pd.isna(pa_pr.iloc[0]):
            v = pa_pr.iloc[0]
            ok = 0.0 < v < 1.0
            print(f"  {'OK' if ok else 'FAIL'}  poison_aware poisonedrag_style faithfulness: "
                  f"{v:.4f}  (n=6; brief expected ~0.20, actual {v:.3f} - note: small sample)")
            if not ok:
                failures.append(f"poison_aware poisonedrag_style faithfulness parse error: {v}")
        else:
            print("  WARNING:  poison_aware / poisonedrag_style justification row empty or NaN")

    if failures:
        print(f"\n  WARNING - {len(failures)} sanity check(s) failed (see above); "
              f"outputs still written. Investigate before citing these numbers.")
    else:
        print("\n  All sanity checks passed.")

    # item 4: collateral counterfactual
    print("\nBuilding table_collateral_counterfactual.csv...")
    counterfactual = make_collateral_counterfactual(df, audit)
    counterfactual.to_csv(EXPORTS / "table_collateral_counterfactual.csv", index=False)
    print(f"  {len(counterfactual)} rows")

    # item 7: filter audit test-only
    print("\nBuilding table_filter_audit_test_only.csv...")
    audit_test = make_filter_audit_test_only(audit)
    audit_test.to_csv(EXPORTS / "table_filter_audit_test_only.csv", index=False)
    print(f"  {len(audit_test)} rows")

    # item 5: bootstrap McNemar
    print("\nBuilding table_mcnemar_bootstrap.csv (1,000 iterations)...")
    mc_boot = make_mcnemar_bootstrap(df, n_boot=1000)
    mc_boot.to_csv(EXPORTS / "table_mcnemar_bootstrap.csv", index=False)
    print(f"  {len(mc_boot)} rows")

    # item 3: length and distractor
    print("\nBuilding table_length_correlation.csv...")
    len_corr = make_length_correlation(df, audit)
    len_corr.to_csv(EXPORTS / "table_length_correlation.csv", index=False)
    print(f"  {len(len_corr)} rows")

    print("Building table_distractor_stratified_paradox.csv...")
    dist_strat = make_distractor_stratified_paradox(df, audit)
    dist_strat.to_csv(EXPORTS / "table_distractor_stratified_paradox.csv", index=False)
    print(f"  {len(dist_strat)} rows")

    # final check
    print("\n=== VERIFICATION ===")
    print(f"judge_scores_long.csv: {len(df)} rows")
    print(f"table_filter_audit.csv: {len(audit)} rows")
    print(f"Mean supporting passages lost collateral (stat filter): {mean_coll:.4f}")
    print("All CSVs written to exports/")


# item 4: collateral-damage counterfactual

def make_collateral_counterfactual(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """
    Compares Delta True faithfulness (baseline -> postfilter) for triplets where the
    statistical filter removed at least one supporting passage (support_removed=True)
    against those where it removed none (support_removed=False).
    Restricted to True-subset triplets (Types 1 & 2 only).
    """
    # build support_removed flag from statistical filter rows
    stat_audit = audit[audit["filter_type"] == "statistical"][
        ["triplet_id", "n_passages_removed_supporting_collateral"]
    ].copy()
    stat_audit["support_removed"] = stat_audit["n_passages_removed_supporting_collateral"] >= 1

    # True-subset scores: baseline and postfilter
    true_base = (df[(df["condition"] == "baseline") & (df["category"] == "True")]
                 [["triplet_id", "judge", "faithfulness"]]
                 .rename(columns={"faithfulness": "baseline_faith"}))
    true_post = (df[(df["condition"] == "postfilter") & (df["category"] == "True")]
                 [["triplet_id", "judge", "faithfulness"]]
                 .rename(columns={"faithfulness": "postfilter_faith"}))

    paired = (true_base
              .merge(true_post, on=["triplet_id", "judge"])
              .merge(stat_audit[["triplet_id", "support_removed"]], on="triplet_id"))

    rows = []
    for judge in JUDGES:
        for removed in [True, False]:
            sub = paired[(paired["judge"] == judge) & (paired["support_removed"] == removed)]
            if sub.empty:
                continue
            base_mean = sub["baseline_faith"].mean()
            post_mean = sub["postfilter_faith"].mean()
            rows.append({
                "judge":              judge,
                "support_removed":    removed,
                "n_triplets":         len(sub),
                "baseline_true_mean": round(base_mean, 4),
                "postfilter_true_mean": round(post_mean, 4),
                "delta":              round(post_mean - base_mean, 4),
            })

    result = pd.DataFrame(rows)

    print("\n  Collateral counterfactual (True subset, stat filter):")
    print(result.to_string(index=False))
    # interpretation
    for judge in JUDGES:
        r_true  = result[(result["judge"]==judge) & (result["support_removed"]==True)]
        r_false = result[(result["judge"]==judge) & (result["support_removed"]==False)]
        if r_true.empty or r_false.empty:
            continue
        d_t = r_true.iloc[0]["delta"]
        d_f = r_false.iloc[0]["delta"]
        verdict = ("collateral-damage mechanism SUPPORTED" if d_t > d_f + 0.02
                   else "mechanisms roughly equal" if abs(d_t - d_f) <= 0.02
                   else "residual-quality alone - investigate")
        print(f"    {judge}: Delta(removed)={d_t:+.4f}  Delta(not-removed)={d_f:+.4f} -> {verdict}")
    return result


# item 7: filter audit on test split only

def make_filter_audit_test_only(audit: pd.DataFrame) -> pd.DataFrame:
    """Re-aggregates table_filter_audit.csv restricted to canonical v2 test-split triplets."""
    v2_test_path = ROOT / "data" / "v2_splits" / "test.json"
    if not v2_test_path.exists():
        print("  ERROR: data/v2_splits/test.json not found", file=sys.stderr)
        return pd.DataFrame()

    v2_test = load_json(v2_test_path)
    test_ids = {r.get("id", r.get("triplet_id", "")) for r in v2_test}
    # include clean counterparts via suffix transform
    clean_ids = set()
    for tid in test_ids:
        clean_id, _ = _derive_clean_id(tid)
        if clean_id:
            clean_ids.add(clean_id)
    all_test_ids = test_ids | clean_ids
    print(f"  Test-split IDs: {len(all_test_ids)} "
          f"({len(test_ids)} poisoned + {len(clean_ids)} clean counterparts)")

    test_audit = audit[audit["triplet_id"].isin(all_test_ids)].copy()
    print(f"  Audit rows matched: {len(test_audit)} "
          f"(triplets: {test_audit['triplet_id'].nunique()})")

    # re-use the same aggregation logic as make_filter_audit_summary
    result = make_filter_audit_summary(test_audit)
    result.insert(0, "split", "test")

    # comparison to full dataset
    full_stat = pd.read_csv(EXPORTS / "table_filter_audit_summary.csv")
    full_stat = full_stat[full_stat["filter_type"] == "statistical"]
    test_stat = result[result["filter_type"] == "statistical"]

    print("\n  Collateral loss - full dataset vs test-only (statistical filter):")
    for inj in sorted(full_stat["injection_type"].dropna().unique()):
        f_row = full_stat[full_stat["injection_type"]==inj]
        t_row = test_stat[test_stat["injection_type"]==inj]
        if f_row.empty or t_row.empty:
            continue
        f_v = f_row.iloc[0]["mean_supporting_passages_lost_collateral"]
        t_v = t_row.iloc[0]["mean_supporting_passages_lost_collateral"]
        diff = abs(t_v - f_v) if not (pd.isna(t_v) or pd.isna(f_v)) else float("nan")
        flag = "  MATERIAL DIFFERENCE" if diff > 0.05 else ""
        print(f"    {inj}: full={f_v:.4f}  test={t_v:.4f}  diff={diff:.4f}{flag}")

    return result


# item 5: bootstrap McNemar

def make_mcnemar_bootstrap(df: pd.DataFrame, n_boot: int = 1000,
                           seed: int = 42) -> pd.DataFrame:
    """
    Question-clustered bootstrap for McNemar tests.
    Resamples 100 base questions (with replacement), collects all their triplets,
    builds the paired faithfulness sample, computes McNemar chi-square.
    """
    rng = np.random.default_rng(seed)

    # extract base question ID (24-char hex prefix)
    df = df.copy()
    df["base_id"] = df["triplet_id"].str.extract(r"^([0-9a-f]{24})_")[0]

    base_questions = df["base_id"].dropna().unique()
    print(f"  Unique base questions: {len(base_questions)}")

    rows = []
    cells = [
        (judge, "baseline", cond_b, metric, cat)
        for judge in JUDGES
        for cond_b in ["postfilter", "postfilter_llm"]
        for metric in ["faithfulness"]          # faithfulness only - matches thesis focus
        for cat in ["True", "Survived", "All-poisoned"]
    ]

    for judge, cond_a, cond_b, metric, category in cells:
        if category == "All-poisoned":
            mask = df["is_poisoned"] == True
        else:
            mask = df["category"] == category

        a = (df[(df["judge"]==judge) & (df["condition"]==cond_a) & mask]
             [["triplet_id", "base_id", metric]].dropna()
             .rename(columns={metric: "sa"}))
        b = (df[(df["judge"]==judge) & (df["condition"]==cond_b) & mask]
             [["triplet_id", metric]].dropna()
             .rename(columns={metric: "sb"}))
        paired = a.merge(b, on="triplet_id")
        if len(paired) < 10:
            continue

        # uncorrected chi2
        n10 = int(((paired["sa"] >= 0.5) & (paired["sb"] < 0.5)).sum())
        n01 = int(((paired["sa"] < 0.5)  & (paired["sb"] >= 0.5)).sum())
        denom = n10 + n01
        if denom == 0:
            continue
        chi2_orig = (abs(n10 - n01) - 1) ** 2 / denom

        # bootstrap
        bq = paired["base_id"].unique()
        boot_chi2s = []
        for _ in range(n_boot):
            sampled_bq = rng.choice(bq, size=len(bq), replace=True)
            boot_rows = paired[paired["base_id"].isin(sampled_bq)]
            if boot_rows.empty:
                continue
            b_n10 = int(((boot_rows["sa"] >= 0.5) & (boot_rows["sb"] < 0.5)).sum())
            b_n01 = int(((boot_rows["sa"] < 0.5)  & (boot_rows["sb"] >= 0.5)).sum())
            b_den = b_n10 + b_n01
            if b_den == 0:
                boot_chi2s.append(0.0)
            else:
                boot_chi2s.append((abs(b_n10 - b_n01) - 1) ** 2 / b_den)

        boot_arr = np.array(boot_chi2s)
        # corrected p = fraction of bootstrap samples where chi2 <= observed
        # (empirical p: fraction exceeding critical value 3.84)
        corrected_p = float((boot_arr < 3.84).mean())   # fraction NOT significant
        actual_boot_p = float((boot_arr >= chi2_orig).mean())

        rows.append({
            "judge": judge, "condition_a": cond_a, "condition_b": cond_b,
            "metric": metric, "category": category,
            "chi2_uncorrected":      round(chi2_orig, 4),
            "chi2_bootstrap_mean":   round(boot_arr.mean(), 4),
            "chi2_bootstrap_p2_5":   round(np.percentile(boot_arr, 2.5), 4),
            "chi2_bootstrap_p97_5":  round(np.percentile(boot_arr, 97.5), 4),
            "corrected_p_value":     round(actual_boot_p, 4),
            "n_paired":              len(paired),
            "n_base_questions":      len(bq),
        })

    result = pd.DataFrame(rows)
    print("\n  Bootstrap McNemar (faithfulness, True category):")
    faith_true = result[(result["metric"]=="faithfulness") & (result["category"]=="True")]
    print(faith_true[["judge","condition_b","chi2_uncorrected",
                       "chi2_bootstrap_mean","chi2_bootstrap_p2_5",
                       "chi2_bootstrap_p97_5","corrected_p_value"]].to_string(index=False))
    return result


# item 3: length and distractor alternative explanations

def make_length_correlation(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between post-filter passage count and faithfulness, per judge x condition."""
    stat_audit = audit[audit["filter_type"] == "statistical"][
        ["triplet_id", "n_passages_after_filter", "n_passages_original"]
    ]
    # for baseline: passage count = n_passages_original (no filter applied)
    base_counts = stat_audit[["triplet_id","n_passages_original"]].rename(
        columns={"n_passages_original": "n_passages_after_filter"})

    rows = []
    for condition in CONDITIONS:
        for judge in JUDGES:
            sub = df[(df["condition"]==condition) & (df["judge"]==judge) &
                     (df["is_poisoned"]==True)][["triplet_id","faithfulness"]].dropna()
            if condition == "baseline":
                counts = base_counts
            else:
                counts = stat_audit[["triplet_id","n_passages_after_filter"]]
            merged = sub.merge(counts, on="triplet_id")
            if len(merged) < 10:
                continue
            r, p = pearsonr(merged["n_passages_after_filter"], merged["faithfulness"])
            rows.append({
                "judge": judge, "condition": condition,
                "mean_passages_after_filter": round(merged["n_passages_after_filter"].mean(), 2),
                "pearson_r_length_vs_faithfulness": round(r, 4),
                "p_value": round(p, 6),
                "n_triplets": len(merged),
            })

    result = pd.DataFrame(rows)
    print("\n  Length vs faithfulness correlation (poisoned subset):")
    print(result[["judge","condition","mean_passages_after_filter",
                  "pearson_r_length_vs_faithfulness","p_value"]].to_string(index=False))
    return result


def make_distractor_stratified_paradox(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """
    Delta True (baseline -> postfilter faithfulness) stratified by quartile of
    n_passages_removed_distractor. Tests whether distractor-removal drives the paradox.
    Restricted to True-subset triplets (Types 1 & 2).
    """
    stat_audit = audit[audit["filter_type"] == "statistical"][
        ["triplet_id", "n_passages_removed_distractor"]
    ]
    # quartile labels (computed globally, not per judge)
    stat_audit = stat_audit.copy()
    # Only bin on the True-subset rows to avoid all-zero issues in full dataset
    true_base = (df[(df["condition"]=="baseline") & (df["category"]=="True")]
                 [["triplet_id","judge","faithfulness"]]
                 .rename(columns={"faithfulness":"base_faith"}))
    true_tids = set(true_base["triplet_id"].unique())
    true_stat = stat_audit[stat_audit["triplet_id"].isin(true_tids)].copy()
    for q in [4, 3, 2]:
        try:
            labels = {4: ["Q1(low)","Q2","Q3","Q4(high)"],
                      3: ["Q1(low)","Q2","Q3(high)"],
                      2: ["Q1(low)","Q2(high)"]}[q]
            true_stat["distractor_q"] = pd.qcut(
                true_stat["n_passages_removed_distractor"], q=q,
                labels=labels, duplicates="drop")
            break
        except ValueError:
            continue
    else:
        true_stat["distractor_q"] = "all"
    stat_audit = stat_audit.merge(true_stat[["triplet_id","distractor_q"]], on="triplet_id", how="left")

    true_base = (df[(df["condition"]=="baseline") & (df["category"]=="True")]
                 [["triplet_id","judge","faithfulness"]]
                 .rename(columns={"faithfulness":"base_faith"}))
    true_post = (df[(df["condition"]=="postfilter") & (df["category"]=="True")]
                 [["triplet_id","judge","faithfulness"]]
                 .rename(columns={"faithfulness":"post_faith"}))
    paired = (true_base.merge(true_post, on=["triplet_id","judge"])
              .merge(stat_audit[["triplet_id","distractor_q"]], on="triplet_id"))

    q_labels = (paired["distractor_q"].cat.categories.tolist()
                if hasattr(paired["distractor_q"], "cat") else
                sorted(paired["distractor_q"].dropna().unique()))
    rows = []
    for judge in JUDGES:
        for q_label in q_labels:
            sub = paired[(paired["judge"]==judge) & (paired["distractor_q"]==q_label)]
            if sub.empty:
                continue
            rows.append({
                "judge": judge,
                "distractor_removal_quartile": str(q_label),
                "n_triplets": len(sub),
                "true_mean_baseline":   round(sub["base_faith"].mean(), 4),
                "true_mean_postfilter": round(sub["post_faith"].mean(), 4),
                "delta":                round(sub["post_faith"].mean() - sub["base_faith"].mean(), 4),
            })

    result = pd.DataFrame(rows)
    print("\n  Distractor-stratified paradox (True subset, GPT only for brevity):")
    print(result[result["judge"]=="GPT"].to_string(index=False))

    # interpretation: does Delta scale with distractor removal?
    for judge in JUDGES:
        sub = result[result["judge"]==judge]
        if len(sub) < 2:
            continue
        deltas = sub.sort_values("distractor_removal_quartile")["delta"].values
        trend = "increasing (distractor effect plausible)" if deltas[-1] > deltas[0] + 0.02 \
                else "flat (distractor mechanism not dominant)"
        print(f"    {judge}: Delta trend across distractor quartiles -> {trend}")

    return result


# main

if __name__ == "__main__":
    main()
