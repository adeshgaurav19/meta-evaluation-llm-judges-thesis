"""
Classifier-based prefilter signal.

The model scores each passage independently using checkpoints trained from the
prepared triplet splits.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.dataset.schema import PrefilterScore

_model = None
_tokenizer = None
_checkpoint_dir: str | None = None


def load_classifier(checkpoint_dir: str) -> None:
    """Load the classifier once and reuse it across triplets."""
    global _model, _tokenizer, _checkpoint_dir
    if _model is not None and _checkpoint_dir == checkpoint_dir:
        return
    _tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    _model.eval()
    _checkpoint_dir = checkpoint_dir


def _score_passages(passages: list[str], max_length: int = 512) -> list[float]:
    """Return the model's poisoned-passage probability for each passage."""
    if _model is None:
        raise RuntimeError("Call load_classifier() before scoring.")
    scores = []
    for passage in passages:
        enc = _tokenizer(
            passage,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = _model(**enc).logits
        prob_poisoned = float(torch.softmax(logits, dim=-1)[0, 1])
        scores.append(prob_poisoned)
    return scores


def score_triplet(triplet, config: dict) -> list[PrefilterScore]:
    clf_cfg = config["prefilter"]["classifier"]
    separator = config["dataset"]["context_separator"]

    if not Path(clf_cfg["checkpoint_dir"]).exists():
        # Keep downstream aggregation neutral if the classifier has not been trained yet.
        ctx = triplet.poisoned_context if hasattr(triplet, "poisoned_context") else triplet.context
        passages = [p.strip() for p in ctx.split(separator) if p.strip()]
        ground_truth_indices = set(getattr(triplet, "poisoned_passage_indices", []))
        return [
            PrefilterScore(
                triplet_id=triplet.id,
                passage_index=i,
                embedding_score=0.0,
                entropy_score=0.0,
                classifier_score=0.5,
                aggregated_score=0.0,
                flagged=False,
                ground_truth_poisoned=(
                    i in ground_truth_indices and getattr(triplet, "is_poisoned", False)
                ),
            )
            for i in range(len(passages))
        ]

    load_classifier(clf_cfg["checkpoint_dir"])
    ctx = triplet.poisoned_context if hasattr(triplet, "poisoned_context") else triplet.context
    passages = [p.strip() for p in ctx.split(separator) if p.strip()]
    ground_truth_indices = set(getattr(triplet, "poisoned_passage_indices", []))

    clf_scores = _score_passages(passages, clf_cfg["max_length"])
    threshold = config["prefilter"]["aggregation"]["threshold"]

    return [
        PrefilterScore(
            triplet_id=triplet.id,
            passage_index=i,
            embedding_score=0.0,
            entropy_score=0.0,
            classifier_score=float(s),
            aggregated_score=0.0,
            flagged=float(s) > threshold,
            ground_truth_poisoned=(
                i in ground_truth_indices and getattr(triplet, "is_poisoned", False)
            ),
        )
        for i, s in enumerate(clf_scores)
    ]


def score_dataset(triplets: list, config: dict) -> list[list[PrefilterScore]]:
    """Score a list of triplets with the classifier signal."""
    return [score_triplet(t, config) for t in triplets]
