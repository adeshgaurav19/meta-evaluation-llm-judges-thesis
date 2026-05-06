"""
Phase 6: GPT justification on 30 selected samples.

Without --postfilter: scores original triplets, selects worst false positives.
With --postfilter:    scores the same triplet IDs but from filtered_triplets.json,
                      allowing direct before/after comparison.

Usage:
    python scripts/06_run_justifications.py [--force]
    python scripts/06_run_justifications.py --postfilter [--force]
    python scripts/06_run_justifications.py --postfilter --llm-prefilter [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.analysis.justification import save_justifications, select_justification_samples
from src.dataset.schema import PoisonedTriplet
from src.dataset.splitter import load_splits
from src.judges.runner import run_justification_phase
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--postfilter",
        action="store_true",
        help="Score filtered triplets using same sample IDs as baseline run",
    )
    parser.add_argument(
        "--llm-prefilter",
        action="store_true",
        help="Use LLM-filtered triplets (requires --postfilter)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    just_dir = Path(config["experiment"]["results_dir"]) / "justifications"

    if args.postfilter:
        suffix = "_postfilter_llm" if args.llm_prefilter else "_postfilter"
        out_path = just_dir / f"justification_samples{suffix}.json"
    else:
        out_path = just_dir / "justification_samples.json"

    if not args.force and out_path.exists():
        print(f"{out_path.name} already exists. Use --force to rerun.")
        return

    splits = load_splits(config["dataset"]["splits_path"])
    if not splits:
        print("ERROR: No splits found. Run script 01 first.")
        sys.exit(1)

    all_triplets = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])

    if args.postfilter:
        # Load same IDs from baseline justification run
        baseline_just = just_dir / "justification_samples.json"
        if not baseline_just.exists():
            print("ERROR: justification_samples.json not found. Run without --postfilter first.")
            sys.exit(1)
        selected_ids = {r["id"] for r in json.loads(baseline_just.read_text())}

        # Load filtered triplets
        fname = "filtered_triplets_llm.json" if args.llm_prefilter else "filtered_triplets.json"
        filtered_path = Path(config["experiment"]["results_dir"]) / "prefilter_scores" / fname
        if not filtered_path.exists():
            print(f"ERROR: {fname} not found.")
            sys.exit(1)
        filtered_map = {
            d["id"]: PoisonedTriplet(**d) for d in json.loads(filtered_path.read_text())
        }
        selected = [filtered_map[tid] for tid in selected_ids if tid in filtered_map]
        print(f"Loaded {len(selected)} filtered triplets matching baseline sample IDs")

    else:
        baseline_path = (
            Path(config["experiment"]["results_dir"]) / "raw_scores" / "baseline_gpt.json"
        )
        if not baseline_path.exists():
            print("ERROR: Baseline scores not found. Run script 02 first.")
            sys.exit(1)
        baseline_scores = json.loads(baseline_path.read_text())
        print(f"Loaded {len(baseline_scores)} baseline scores")

        n_samples = config["judges"]["justification"]["n_samples"]
        selected = select_justification_samples(all_triplets, baseline_scores, n_total=n_samples)
        print(f"Selected {len(selected)} samples for justification analysis")

    raw_results = run_justification_phase(selected, config)
    save_justifications(raw_results, config, out_path=out_path)
    print("Phase 6 complete.")


if __name__ == "__main__":
    main()
