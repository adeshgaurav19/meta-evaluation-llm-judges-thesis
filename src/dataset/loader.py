"""Load RAG evaluation logs from CSV, JSON, or Excel files."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from src.dataset.schema import RAGTriplet


def load_raw(path: str | Path) -> list[RAGTriplet]:
    """
    Load RAG triplets from a file or directory.

    Supported formats: .csv, .json, .jsonl, .xlsx
    Required columns/keys: question, context, answer
    Optional: id, ground_truth
    """
    p = Path(path)

    if p.is_dir():
        triplets: list[RAGTriplet] = []
        for f in sorted(p.iterdir()):
            if f.suffix not in {".csv", ".json", ".jsonl", ".xlsx"}:
                continue
            if not _looks_like_data_file(f):
                print(f"  Skipping {f.name} (not a triplet data file)")
                continue
            triplets.extend(_load_file(f))
        return triplets

    return _load_file(p)


def _load_file(path: Path) -> list[RAGTriplet]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        return _load_json(path)
    elif suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return _df_to_triplets(df)


def _load_json(path: Path) -> list[RAGTriplet]:
    text = path.read_text()
    # Try JSONL first
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) > 1:
        try:
            records = [json.loads(l) for l in lines]
        except json.JSONDecodeError:
            records = json.loads(text)
    else:
        records = json.loads(text)

    if isinstance(records, dict):
        records = [records]

    return [_record_to_triplet(r) for r in records]


def _df_to_triplets(df: pd.DataFrame) -> list[RAGTriplet]:
    _require_columns(df, ["question", "context", "answer"])
    return [_record_to_triplet(row.to_dict()) for _, row in df.iterrows()]


def _record_to_triplet(r: dict) -> RAGTriplet:
    return RAGTriplet(
        id=str(r.get("id", uuid.uuid4())),
        question=str(r["question"]),
        context=str(r["context"]),
        answer=str(r["answer"]),
        ground_truth=r.get("ground_truth"),
    )


def _looks_like_data_file(path: Path) -> bool:
    """Return False for known non-triplet files (metadata, raw downloads, README)."""
    name = path.name.lower()
    skip_patterns = (
        "metadata",  # *_metadata.json
        "distractor_v1",  # hotpotqa_dev_distractor_v1.json (raw download)
        "readme",
    )
    return not any(pat in name for pat in skip_patterns)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
