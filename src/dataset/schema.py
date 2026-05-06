from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


class RAGTriplet(BaseModel):
    id: str = Field(default_factory=_new_id)
    question: str
    context: str
    answer: str
    ground_truth: Optional[str] = None


class PoisonedTriplet(BaseModel):
    """One clean/poisoned pair for a configured injection type."""

    id: str
    question: str
    original_context: str
    poisoned_context: str
    answer: str
    ground_truth: Optional[str] = None
    injection_type: str
    noise_level: float
    is_poisoned: bool
    poisoned_passage_indices: list[int]
    # Optional metadata: populated only for poisonedrag_style poisoned triplets.
    # Not read by downstream scoring scripts; kept for debugging and transparency.
    target_wrong_answer: str = ""


class JudgeScore(BaseModel):
    triplet_id: str
    judge_model: str
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    is_poisoned: bool
    injection_type: Optional[str] = None
    noise_level: Optional[float] = None
    prefilter_applied: bool = False
    latency_ms: float = 0.0


class JudgeJustification(BaseModel):
    """Used for qualitative analysis on the 28-sample subset."""

    triplet_id: str
    judge_model: str
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    justification: str
    is_poisoned: bool
    injection_type: str
    noise_level: float


class PrefilterScore(BaseModel):
    """Scores for a single passage. crossencoder_score is 0.0 locally (Kaggle-only signal)."""

    triplet_id: str
    passage_index: int
    embedding_score: float
    entropy_score: float
    classifier_score: float
    crossencoder_score: float = 0.0
    answer_span_score: float = 0.0
    aggregated_score: float
    flagged: bool
    ground_truth_poisoned: bool
