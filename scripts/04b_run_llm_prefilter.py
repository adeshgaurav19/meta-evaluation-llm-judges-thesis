"""
Phase 4b: optional LLM-based prefilter using the configured LLM provider.

The script scores the prepared triplets and writes filtered_triplets_llm.json
for the matching post-filter judge phase.

Usage:
    cp .env.example .env
    # add MISTRAL_API_KEY or the key named in config.llm_prefilter.api_key_env
    python scripts/04b_run_llm_prefilter.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.dataset.splitter import load_splits
from src.prefilter.llm_prefilter import run_llm_prefilter
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = (
        Path(config["experiment"]["results_dir"])
        / "prefilter_scores"
        / "filtered_triplets_llm.json"
    )

    if not args.force and out_path.exists():
        print("LLM prefilter already done. Use --force to rerun.")
        return

    splits = load_splits(config["dataset"]["splits_path"])
    all_triplets = splits["train"] + splits["val"] + splits["test"]
    print(
        f"Loaded {len(all_triplets)} triplets ({sum(1 for t in all_triplets if t.is_poisoned)} poisoned)"
    )

    run_llm_prefilter(all_triplets, config)
    print("Phase 4b complete.")


if __name__ == "__main__":
    main()
