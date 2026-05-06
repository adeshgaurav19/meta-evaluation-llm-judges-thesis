from __future__ import annotations

import re

from tqdm.auto import tqdm

from src.dataset.schema import PrefilterScore

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "it",
    "its",
}
_BINARY_ANSWERS = {"yes", "no"}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _answer_token_recall(answer: str, passage: str) -> float:
    """Fraction of meaningful answer tokens found in passage. Returns 0.5 for yes/no."""
    if answer.lower().strip() in _BINARY_ANSWERS:
        return 0.5  # token matching is meaningless for binary answers

    answer_tokens = _tokenize(answer) - _STOPWORDS
    if not answer_tokens:
        return 0.5

    passage_tokens = _tokenize(passage)
    return len(answer_tokens & passage_tokens) / len(answer_tokens)


def score_triplet(triplet, config: dict) -> list[PrefilterScore]:
    recall_threshold = config["prefilter"].get("answer_span", {}).get("recall_threshold", 0.3)
    separator = config["dataset"]["context_separator"]

    ctx = triplet.poisoned_context if hasattr(triplet, "poisoned_context") else triplet.context
    passages = [p.strip() for p in ctx.split(separator) if p.strip()]
    ground_truth_indices = set(getattr(triplet, "poisoned_passage_indices", []))
    answer = getattr(triplet, "answer", "") or ""

    return [
        PrefilterScore(
            triplet_id=triplet.id,
            passage_index=i,
            embedding_score=0.0,
            entropy_score=0.0,
            classifier_score=0.0,
            answer_span_score=float(1.0 - _answer_token_recall(answer, passage)),
            aggregated_score=0.0,
            flagged=_answer_token_recall(answer, passage) < recall_threshold,
            ground_truth_poisoned=(
                i in ground_truth_indices and getattr(triplet, "is_poisoned", False)
            ),
        )
        for i, passage in enumerate(passages)
    ]


def score_dataset(triplets: list, config: dict) -> list[list[PrefilterScore]]:
    return [score_triplet(t, config) for t in tqdm(triplets, desc="Answer span scoring")]
