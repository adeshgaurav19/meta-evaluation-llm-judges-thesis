"""Stratified train/val/test splitting of poisoned triplets."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.dataset.schema import PoisonedTriplet


def stratified_split(
    triplets: list[PoisonedTriplet],
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
    seed: int = 42,
) -> dict[str, list[PoisonedTriplet]]:
    """
    Stratified split by (injection_type, is_poisoned).
    Returns {"train": [...], "val": [...], "test": [...]}
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    rng = random.Random(seed)
    groups: dict[tuple, list[PoisonedTriplet]] = defaultdict(list)
    for t in triplets:
        key = (t.injection_type, t.is_poisoned)
        groups[key].append(t)

    splits: dict[str, list[PoisonedTriplet]] = {"train": [], "val": [], "test": []}

    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train)
        n_val = int(n * val)

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    for k in splits:
        rng.shuffle(splits[k])

    return splits


def save_splits(splits: dict[str, list[PoisonedTriplet]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (out / f"{name}.json").write_text(json.dumps([t.model_dump() for t in items], indent=2))
    print(f"Saved splits to {out}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


def load_splits(splits_dir: str | Path) -> dict[str, list[PoisonedTriplet]]:
    d = Path(splits_dir)
    result = {}
    for name in ("train", "val", "test"):
        p = d / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            result[name] = [PoisonedTriplet(**r) for r in data]
    return result
