"""
Signal 1: Embedding cosine similarity outlier detection.

Flags passages that are either irrelevant (sim < mean - N*std) or suspiciously
over-matched (sim > 0.95). The over-match threshold catches poisonedrag-style passages
that are crafted to look highly relevant. Score is the normalised z-distance from mean.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.dataset.schema import PrefilterScore

_model: SentenceTransformer | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def score_triplet(
    triplet,
    config: dict,
) -> list[PrefilterScore]:
    """One PrefilterScore per passage; embedding_score only, other fields zeroed."""
    emb_cfg = config["prefilter"]["embedding"]
    model = _get_model(emb_cfg["model"])
    separator = config["dataset"]["context_separator"]
    std_thresh = emb_cfg["outlier_std_threshold"]
    high_sim_thresh = emb_cfg["high_sim_threshold"]
    flag_high = emb_cfg.get("flag_high_sim", True)

    ctx = triplet.poisoned_context if hasattr(triplet, "poisoned_context") else triplet.context
    passages = [p.strip() for p in ctx.split(separator) if p.strip()]

    if not passages:
        return []

    q_emb = model.encode(triplet.question, convert_to_tensor=True)
    p_embs = model.encode(passages, convert_to_tensor=True)

    sims = [float(util.cos_sim(q_emb, p_emb)) for p_emb in p_embs]
    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims)) if len(sims) > 1 else 0.1

    ground_truth_indices = set(getattr(triplet, "poisoned_passage_indices", []))
    scores = []

    for i, (passage, sim) in enumerate(zip(passages, sims)):
        low_outlier = sim < mean_sim - std_thresh * std_sim
        high_outlier = flag_high and sim > high_sim_thresh

        flagged = low_outlier or high_outlier

        # Normalise score: 0 = normal, 1 = very suspicious
        z = abs(sim - mean_sim) / (std_sim + 1e-9)
        raw_score = min(z / (std_thresh * 2), 1.0)
        if high_outlier:
            raw_score = max(raw_score, 0.8)

        scores.append(
            PrefilterScore(
                triplet_id=triplet.id,
                passage_index=i,
                embedding_score=float(raw_score),
                entropy_score=0.0,
                classifier_score=0.0,
                aggregated_score=0.0,
                flagged=flagged,
                ground_truth_poisoned=(
                    i in ground_truth_indices and getattr(triplet, "is_poisoned", False)
                ),
            )
        )

    return scores


def score_dataset(triplets: list, config: dict) -> list[list[PrefilterScore]]:
    """Score all triplets. Returns list-of-lists (one per triplet)."""
    return [score_triplet(t, config) for t in triplets]
