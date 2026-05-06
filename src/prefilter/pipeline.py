"""
Full pre-filter pipeline.

Modes:
  - filter: remove flagged passages, reconstruct context
  - flag:   keep all passages, attach scores
  - audit:  log everything, don't modify
"""

from __future__ import annotations

import json
from pathlib import Path

from src.dataset.schema import PrefilterScore
from src.prefilter import answer_span_signal, classifier_signal, embedding_signal, entropy_signal
from src.prefilter.aggregator import aggregate


def run_pipeline(
    triplets: list,
    config: dict,
    mode: str = "filter",
) -> tuple[list, list[list[PrefilterScore]]]:
    """
    Run the full pre-filter pipeline.

    Returns:
        (filtered_triplets, all_scores)
        filtered_triplets: modified copies (mode='filter') or originals (mode='flag'/'audit')
        all_scores: per-triplet list of PrefilterScore
    """
    separator = config["dataset"]["context_separator"]
    method = config["prefilter"]["aggregation"]["primary"]

    all_scores: list[list[PrefilterScore]] = []
    filtered_triplets = []

    for triplet in triplets:
        emb_scores = embedding_signal.score_triplet(triplet, config)
        ent_scores = entropy_signal.score_triplet(triplet, config)
        clf_scores = classifier_signal.score_triplet(triplet, config)
        as_scores = answer_span_signal.score_triplet(triplet, config)

        n = len(emb_scores)
        combined = []
        for i in range(n):
            ps = PrefilterScore(
                triplet_id=triplet.id,
                passage_index=i,
                embedding_score=emb_scores[i].embedding_score if i < len(emb_scores) else 0.0,
                entropy_score=ent_scores[i].entropy_score if i < len(ent_scores) else 0.0,
                classifier_score=clf_scores[i].classifier_score if i < len(clf_scores) else 0.5,
                answer_span_score=as_scores[i].answer_span_score if i < len(as_scores) else 0.5,
                aggregated_score=0.0,
                flagged=False,
                ground_truth_poisoned=(
                    emb_scores[i].ground_truth_poisoned if i < len(emb_scores) else False
                ),
            )
            combined.append(ps)

        combined = aggregate(combined, config, method=method)
        all_scores.append(combined)

        if mode == "filter":
            ctx = (
                triplet.poisoned_context
                if hasattr(triplet, "poisoned_context")
                else triplet.context
            )
            passages = [p.strip() for p in ctx.split(separator) if p.strip()]
            kept_passages = [p for p, s in zip(passages, combined) if not s.flagged]
            new_ctx = separator.join(kept_passages) if kept_passages else ctx

            if hasattr(triplet, "poisoned_context"):
                filtered = triplet.model_copy(update={"poisoned_context": new_ctx})
            else:
                filtered = triplet.model_copy(update={"context": new_ctx})
            filtered_triplets.append(filtered)

        else:
            filtered_triplets.append(triplet)

    return filtered_triplets, all_scores


def save_prefilter_scores(
    all_scores: list[list[PrefilterScore]],
    config: dict,
    signal: str = "aggregated",
) -> None:
    out_dir = Path(config["experiment"]["results_dir"]) / "prefilter_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = [s.model_dump() for group in all_scores for s in group]
    path = out_dir / f"{signal}_scores.json"
    path.write_text(json.dumps(flat, indent=2))
    print(f"Saved {len(flat)} prefilter scores to {path}")
