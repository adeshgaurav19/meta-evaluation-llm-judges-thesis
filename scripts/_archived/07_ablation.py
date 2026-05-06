"""
Phase 4c: Signal ablation studies. Reuses existing prefilter scores — no re-inference.

Usage:
    python scripts/07_ablation.py --config config/base.yaml [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.ablation import save_ablation_tables
from src.dataset.schema import PrefilterScore
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["experiment"]["results_dir"]) / "tables"

    if not args.force and (out_dir / "table_4_13_ablation.csv").exists():
        print("Ablation tables already exist. Use --force to rerun.")
        return

    scores_path = (
        Path(config["experiment"]["results_dir"]) / "prefilter_scores" / "aggregated_scores.json"
    )
    if not scores_path.exists():
        print("ERROR: Prefilter scores not found. Run script 04 first.")
        sys.exit(1)

    raw = json.loads(scores_path.read_text())
    all_scores = [PrefilterScore(**r) for r in raw]
    print(f"Loaded {len(all_scores)} prefilter scores")

    # Build triplet_id → injection_type map from splits
    from src.dataset.splitter import load_splits

    splits = load_splits(config["dataset"]["splits_path"])
    all_triplets = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    injection_map = {t.id: t.injection_type for t in all_triplets}
    print(f"Loaded injection types for {len(injection_map)} triplets")

    # Load judge scores for prefilter strategy comparison (optional)
    raw_dir = Path(config["experiment"]["results_dir"]) / "raw_scores"

    def _load(path):
        return json.loads(path.read_text()) if path.exists() else []

    baseline_scores = (
        _load(raw_dir / "baseline_gpt.json")
        + _load(raw_dir / "baseline_gemini.json")
        + _load(raw_dir / "baseline_deepseek.json")
    )
    postfilter_scores = (
        _load(raw_dir / "postfilter_gpt.json")
        + _load(raw_dir / "postfilter_gemini.json")
        + _load(raw_dir / "postfilter_deepseek.json")
    )
    postfilter_llm_scores = (
        _load(raw_dir / "postfilter_llm_gpt.json")
        + _load(raw_dir / "postfilter_llm_gemini.json")
        + _load(raw_dir / "postfilter_llm_deepseek.json")
    )

    save_ablation_tables(
        all_scores,
        config,
        injection_map,
        baseline_scores=baseline_scores or None,
        postfilter_scores=postfilter_scores or None,
        postfilter_llm_scores=postfilter_llm_scores or None,
    )
    print("Phase 7 complete.")


if __name__ == "__main__":
    main()
