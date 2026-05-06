"""Async HTTP client with semaphore, backoff, and retry."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Coroutine


async def with_retry(
    fn: Callable[[], Coroutine[Any, Any, Any]],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Retry an async callable with exponential backoff + jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)


async def bounded_gather(
    coros: list[Coroutine],
    max_concurrent: int = 10,
) -> list[Any]:
    """Run coroutines with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _wrapped(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[_wrapped(c) for c in coros], return_exceptions=True)
