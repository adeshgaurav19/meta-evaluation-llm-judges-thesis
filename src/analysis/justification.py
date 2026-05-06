"""
Qualitative justification analysis on 30 GPT-scored samples.

Sample selection:
  - 10 clean (random)
  - 10 worst false positives (poisoned, highest faithfulness scores)
  - 10 random poisoned (stratified by injection type)
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def select_justification_samples(
    triplets: list,
    baseline_scores: list[dict],
    n_total: int = 30,
    seed: int = 42,
) -> list:
    """Select 30 samples for justification analysis."""
    rng = random.Random(seed)

    # Index triplets
    triplet_idx = {t.id: t for t in triplets}

    # Index baseline scores by triplet_id
    score_idx: dict = {}
    for s in baseline_scores:
        tid = s.get("triplet_id") or s.get("id", "")
        if tid not in score_idx:
            score_idx[tid] = s

    # Split into clean and poisoned with valid scores
    clean_ids = [
        tid
        for tid, s in score_idx.items()
        if tid in triplet_idx and not getattr(triplet_idx[tid], "is_poisoned", False)
    ]
    poisoned_ids = [
        tid
        for tid, s in score_idx.items()
        if tid in triplet_idx and getattr(triplet_idx[tid], "is_poisoned", False)
    ]

    n_clean = n_total // 3
    n_worst_fp = n_total // 3
    n_random_poisoned = n_total - n_clean - n_worst_fp

    # 10 clean (random)
    selected_clean = rng.sample(clean_ids, min(n_clean, len(clean_ids)))

    # 10 worst false positives: poisoned with highest faithfulness
    poisoned_with_scores = [
        (
            tid,
            score_idx[tid].get(
                "faithfulness", score_idx[tid].get("scores", {}).get("faithfulness", 0.0)
            ),
        )
        for tid in poisoned_ids
        if tid in score_idx
    ]
    poisoned_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Spread across injection types
    fp_by_type: dict = {}
    for tid, faith in poisoned_with_scores:
        t = triplet_idx.get(tid)
        if t is None:
            continue
        inj = getattr(t, "injection_type", "unknown")
        fp_by_type.setdefault(inj, []).append(tid)

    selected_worst_fp = []
    per_type = max(1, n_worst_fp // max(len(fp_by_type), 1))
    for inj_type, ids in fp_by_type.items():
        selected_worst_fp.extend(ids[:per_type])
    selected_worst_fp = selected_worst_fp[:n_worst_fp]

    # 10 random poisoned (stratified by injection type)
    remaining_poisoned = [tid for tid in poisoned_ids if tid not in selected_worst_fp]
    by_type: dict = {}
    for tid in remaining_poisoned:
        t = triplet_idx.get(tid)
        if t is None:
            continue
        inj = getattr(t, "injection_type", "unknown")
        by_type.setdefault(inj, []).append(tid)

    selected_random_poisoned = []
    per_type = max(1, n_random_poisoned // max(len(by_type), 1))
    for ids in by_type.values():
        rng.shuffle(ids)
        selected_random_poisoned.extend(ids[:per_type])
    selected_random_poisoned = selected_random_poisoned[:n_random_poisoned]

    all_selected_ids = selected_clean + selected_worst_fp + selected_random_poisoned
    return [triplet_idx[tid] for tid in all_selected_ids if tid in triplet_idx]


def parse_justification(raw: dict) -> dict:
    """Parse GPT justification response into structured dict."""
    content = raw.get("content", "")
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {
            "faithfulness": {"score": None, "reason": content},
            "answer_relevance": {"score": None, "reason": ""},
            "context_relevance": {"score": None, "reason": ""},
            "parse_error": True,
        }

    result = {}
    for metric in ("faithfulness", "answer_relevance", "context_relevance"):
        if isinstance(parsed.get(metric), dict):
            result[metric] = {
                "score": parsed[metric].get("score"),
                "reason": parsed[metric].get("reason", ""),
            }
        else:
            result[metric] = {"score": parsed.get(metric), "reason": ""}
    return result


def save_justifications(
    raw_results: list[dict],
    config: dict,
    out_path: Path | None = None,
) -> Path:
    """Parse and save justification results."""
    out_dir = Path(config["experiment"]["results_dir"]) / "justifications"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = out_dir / "justification_samples.json"

    parsed = []
    for raw in raw_results:
        entry = {
            "id": raw["id"],
            "is_poisoned": raw.get("is_poisoned", False),
            "injection_type": raw.get("injection_type"),
            "noise_level": raw.get("noise_level"),
            "raw_response": raw.get("content", ""),
            "parsed": parse_justification(raw),
        }
        parsed.append(entry)

    out_path.write_text(json.dumps(parsed, indent=2))
    print(f"Saved {len(parsed)} justifications to {out_path}")

    _generate_error_analysis(parsed, out_dir)
    return out_path


def _generate_error_analysis(parsed: list[dict], out_dir: Path) -> None:
    """Auto-generate a markdown summary of failure patterns."""
    lines = ["# Error Analysis Summary\n"]
    lines.append(f"Total samples: {len(parsed)}\n")

    fp_samples = [
        p
        for p in parsed
        if p["is_poisoned"]
        and ((p["parsed"].get("faithfulness", {}) or {}).get("score") or 0) > 0.5
    ]
    lines.append(f"False positives (poisoned, faithfulness > 0.5): {len(fp_samples)}\n\n")

    lines.append("## False Positive Justifications\n")
    for p in fp_samples[:10]:
        reason = (p["parsed"].get("faithfulness", {}) or {}).get("reason", "N/A")
        lines.append(
            f"- **ID**: {p['id']} | **Type**: {p.get('injection_type')} | "
            f"**Noise**: {p.get('noise_level')}\n"
        )
        lines.append(f"  > {reason[:300]}\n\n")

    lines.append("## Common Failure Patterns\n")
    lines.append("- (Fill in manually after reviewing justification_samples.json)\n")

    (out_dir / "error_analysis_summary.md").write_text("".join(lines))
    print("Saved error_analysis_summary.md")
