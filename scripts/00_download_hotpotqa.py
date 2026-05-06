"""
Download HotpotQA and convert it to the local RAGTriplet schema.

Downloads the HotpotQA distractor validation split from Hugging Face,
samples 100 hard multi-hop questions, and writes them to data/raw/ in
the format expected by script 01.

Usage:
    python scripts/00_download_hotpotqa.py --config config/base.yaml [--force]

Output:
    data/raw/hotpotqa_base.json       - RAGTriplet-compatible list
    data/raw/hotpotqa_metadata.json   - provenance and citation info
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.context_truncator import count_tokens

# HotpotQA constants

HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"
TARGET_LEVEL = "hard"

CITATION = (
    "Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop "
    "Question Answering. EMNLP 2018. https://hotpotqa.github.io/"
)
LICENSE = "CC BY-SA 4.0"


# Conversion helpers


def _build_context(context_raw: list, separator: str) -> tuple[str, list[str]]:
    """
    Reconstruct multi-passage context from HotpotQA's raw JSON format.

    Raw JSON stores context as a list of [title, sentences] pairs:
        [["Title1", ["sent1", "sent2", ...]], ["Title2", [...]], ...]

    Each pair becomes one passage; passages are joined with `separator`.
    Returns (joined_context, list_of_titles).
    """
    passages: list[str] = []
    titles: list[str] = []

    for item in context_raw:
        title = item[0]
        sents = item[1]
        passage = " ".join(s.strip() for s in sents if s.strip())
        if passage:
            passages.append(passage)
            titles.append(title)

    return separator.join(passages), titles


def _build_ground_truth(answer: str, supporting_facts: list) -> str:
    """
    Combine answer with supporting fact titles for richer ground truth.

    Raw JSON stores supporting_facts as [[title, sent_id], ...].
    """
    sf_titles = list(dict.fromkeys(item[0] for item in supporting_facts))
    return json.dumps({"answer": answer, "supporting_titles": sf_titles})


def convert_example(example: dict, separator: str) -> dict:
    """
    Convert one HotpotQA raw-JSON example to a RAGTriplet-compatible dict.

    Raw JSON fields used: _id, question, context, answer, supporting_facts, level, type
    """
    context_str, passage_titles = _build_context(example["context"], separator)
    ground_truth = _build_ground_truth(example["answer"], example["supporting_facts"])

    return {
        "id": example["_id"],
        "question": example["question"].strip(),
        "context": context_str,
        "answer": example["answer"].strip(),
        "ground_truth": ground_truth,
        # Provenance fields; ignored by the RAGTriplet loader.
        "_source": "hotpotqa/hotpot_qa",
        "_split": "validation_distractor",
        "_level": example.get("level", TARGET_LEVEL),
        "_type": example.get("type", "unknown"),
        "_passage_titles": passage_titles,
    }


# Statistics


def _compute_stats(triplets: list[dict], separator: str) -> dict:
    q_lens, ctx_lens, ans_lens, n_passages = [], [], [], []
    types: dict[str, int] = {}

    for t in triplets:
        q_lens.append(count_tokens(t["question"]))
        ctx_lens.append(count_tokens(t["context"]))
        ans_lens.append(count_tokens(t["answer"]))
        n_passages.append(len([p for p in t["context"].split(separator) if p.strip()]))
        qtype = t.get("_type", "unknown")
        types[qtype] = types.get(qtype, 0) + 1

    def _stats(vals: list[int]) -> dict:
        import statistics

        return {
            "min": min(vals),
            "max": max(vals),
            "mean": round(statistics.mean(vals), 1),
            "median": statistics.median(vals),
        }

    return {
        "n_triplets": len(triplets),
        "question_tokens": _stats(q_lens),
        "context_tokens": _stats(ctx_lens),
        "answer_tokens": _stats(ans_lens),
        "passages_per_item": _stats(n_passages),
        "question_types": types,
    }


def _print_stats(stats: dict, config: dict) -> None:
    ctx_cap = config["dataset"]["context_max_tokens"]

    print("\nDataset statistics")
    print("-" * 58)
    print(f"  Triplets:          {stats['n_triplets']}")
    print(
        f"  Question tokens:   min={stats['question_tokens']['min']}, "
        f"max={stats['question_tokens']['max']}, "
        f"mean={stats['question_tokens']['mean']}"
    )
    print(
        f"  Context tokens:    min={stats['context_tokens']['min']}, "
        f"max={stats['context_tokens']['max']}, "
        f"mean={stats['context_tokens']['mean']}  (cap={ctx_cap})"
    )
    print(
        f"  Answer tokens:     min={stats['answer_tokens']['min']}, "
        f"max={stats['answer_tokens']['max']}, "
        f"mean={stats['answer_tokens']['mean']}"
    )
    print(
        f"  Passages per item: min={stats['passages_per_item']['min']}, "
        f"max={stats['passages_per_item']['max']}, "
        f"mean={stats['passages_per_item']['mean']}"
    )
    if stats["question_types"]:
        print(f"  Question types:    {stats['question_types']}")
    if stats["context_tokens"]["max"] > ctx_cap:
        print(
            f"\n  WARNING: Some contexts exceed the {ctx_cap}-token cap; "
            "script 01 will truncate them."
        )
    print("-" * 58 + "\n")


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if output file already exists"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["experiment"]["seed"]
    n_target = config["dataset"]["n_base_triplets"]
    separator = config["dataset"]["context_separator"]
    out_dir = Path(config["dataset"]["raw_path"])
    out_json = out_dir / "hotpotqa_base.json"
    out_meta = out_dir / "hotpotqa_metadata.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and out_json.exists():
        existing = json.loads(out_json.read_text())
        print(
            f"hotpotqa_base.json already exists ({len(existing)} triplets). "
            f"Use --force to re-download."
        )
        return

    # Download the official JSON directly. This avoids occasional parquet
    # compatibility problems in the Hugging Face mirror.
    # HotpotQA publishes the distractor dev set as a plain JSON file.
    # This is the canonical source cited in the paper.
    HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    cache_path = out_dir / "hotpotqa_dev_distractor_v1.json"

    if not cache_path.exists():
        print(f"Downloading from {HOTPOTQA_URL} ...")
        import urllib.request

        try:
            urllib.request.urlretrieve(HOTPOTQA_URL, cache_path)
        except Exception as e:
            print(f"ERROR: Download failed: {e}")
            sys.exit(1)
        print(f"Saved raw file -> {cache_path}")
    else:
        print(f"Using cached file: {cache_path}")

    raw = json.loads(cache_path.read_text())
    print(f"Loaded {len(raw)} examples from raw JSON")

    # Filter to hard difficulty, then sample n_target.
    hard_pool = [ex for ex in raw if ex.get("level") == TARGET_LEVEL]
    print(f"Hard examples: {len(hard_pool)}")

    if len(hard_pool) < n_target:
        print(f"WARNING: only {len(hard_pool)} hard examples; using all of them.")

    rng = random.Random(seed)
    rng.shuffle(hard_pool)
    selected_examples = hard_pool[:n_target]
    selected_indices = [ex["_id"] for ex in selected_examples]
    print(f"Sampled {len(selected_examples)} examples (seed={seed})")

    # Convert to the local schema.
    print("Converting to RAGTriplet schema...")
    triplets: list[dict] = []
    skipped = 0

    for ex in selected_examples:
        try:
            t = convert_example(ex, separator)
            # Basic sanity checks
            if not t["question"] or not t["context"] or not t["answer"]:
                skipped += 1
                continue
            if len(t["context"]) < 50:
                skipped += 1
                continue
            triplets.append(t)
        except (KeyError, TypeError) as e:
            skipped += 1
            print(f"  Skipping {ex.get('id', '?')}: {e}")

    if skipped:
        print(f"Skipped {skipped} malformed examples")

    if not triplets:
        print("ERROR: No valid triplets after conversion.")
        sys.exit(1)

    print(f"Converted {len(triplets)} triplets")

    # Print statistics for a quick sanity check.
    stats = _compute_stats(triplets, separator)
    _print_stats(stats, config)

    # Save data.
    out_json.write_text(json.dumps(triplets, indent=2, ensure_ascii=False))
    print(f"Saved {len(triplets)} triplets -> {out_json}")

    # Save metadata.
    metadata = {
        "source": HF_DATASET,
        "hf_config": HF_CONFIG,
        "hf_split": HF_SPLIT,
        "target_level": TARGET_LEVEL,
        "n_requested": n_target,
        "n_saved": len(triplets),
        "n_skipped": skipped,
        "seed": seed,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "citation": CITATION,
        "license": LICENSE,
        "url": "https://hotpotqa.github.io/",
        "paper": "https://arxiv.org/abs/1809.09600",
        "context_separator": separator,
        "stats": stats,
        "selected_ids": selected_indices,  # HotpotQA IDs of sampled examples
    }
    out_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved metadata  -> {out_meta}")

    print("\nNext step:")
    print("  python scripts/01_prepare_dataset.py --config config/base.yaml")


if __name__ == "__main__":
    main()
