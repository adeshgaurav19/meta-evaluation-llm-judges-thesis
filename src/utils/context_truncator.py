"""
Truncates all text fields to token limits from config BEFORE any API calls.
Uses tiktoken for OpenAI-compatible token counting.
"""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken


def _enc(model: str = "gpt-4o") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    return len(_enc(model).encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    enc = _enc(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def truncate_triplet(triplet, config: dict) -> None:
    """In-place truncation of all text fields in a triplet."""
    ctx_max = config["dataset"]["context_max_tokens"]
    passage_max = config["dataset"]["passage_max_tokens"]
    answer_max = config["dataset"]["answer_max_tokens"]
    separator = config["dataset"]["context_separator"]

    # Identify context field
    if hasattr(triplet, "poisoned_context"):
        ctx = triplet.poisoned_context
    else:
        ctx = triplet.context

    # Truncate individual passages first
    passages = [p.strip() for p in ctx.split(separator) if p.strip()]
    truncated_passages = [truncate_to_tokens(p, passage_max) for p in passages]
    truncated_ctx = separator.join(truncated_passages)

    # Cap total context
    truncated_ctx = truncate_to_tokens(truncated_ctx, ctx_max)

    if hasattr(triplet, "poisoned_context"):
        triplet.poisoned_context = truncated_ctx
    else:
        triplet.context = truncated_ctx

    triplet.answer = truncate_to_tokens(triplet.answer, answer_max)


def truncate_dataset(triplets: list, config: dict) -> dict:
    """Truncate all triplets and return token stats."""
    stats: dict = {"before": [], "after": [], "truncated_count": 0}

    for t in triplets:
        ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
        before = count_tokens(ctx)
        stats["before"].append(before)

        truncate_triplet(t, config)

        ctx_after = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
        after = count_tokens(ctx_after)
        stats["after"].append(after)

        if after < before:
            stats["truncated_count"] += 1

    n = len(triplets)
    print(f"Context tokens  before: mean={sum(stats['before'])/n:.0f}, max={max(stats['before'])}")
    print(f"Context tokens  after:  mean={sum(stats['after'])/n:.0f}, max={max(stats['after'])}")
    print(f"Truncated {stats['truncated_count']}/{n} triplets")
    return stats


def save_truncation_stats(stats: dict, config: dict) -> None:
    out = Path(config["experiment"]["results_dir"]) / "truncation_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"Saved truncation stats to {out}")
