from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseJudge(ABC):

    @abstractmethod
    def score(self, question: str, context: str, answer: str) -> dict[str, float]:
        """Returns {faithfulness, answer_relevance, context_relevance} in [0, 1]."""
        ...

    @abstractmethod
    def score_batch(self, triplets: list[Any]) -> list[dict]: ...
