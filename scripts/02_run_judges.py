"""
Phase 2: Baseline judge scoring via Batch API.

Usage:
    python scripts/02_run_judges.py --config config/base.yaml [--no-batch] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.dataset.schema import JudgeScore
from src.dataset.splitter import load_splits
from src.judges.runner import run_scoring_deepseek, run_scoring_phase, run_scoring_realtime
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--no-batch", action="store_true", help="Use real-time API instead of batch"
    )
    parser.add_argument(
        "--judge",
        choices=["gpt", "gemini", "deepseek"],
        default=None,
        help="Run only one judge (default: both)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry only failed triplets from a previous run (requires --judge)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    phase_name = "baseline"
    out_dir = Path(config["experiment"]["results_dir"]) / "raw_scores"

    sentinel = f"{phase_name}_{args.judge}.json" if args.judge else f"{phase_name}_gpt.json"
    if not args.force and not args.retry_errors and (out_dir / sentinel).exists():
        print("Baseline scores already exist. Use --force to rerun.")
        return

    splits = load_splits(config["dataset"]["splits_path"])
    if not splits:
        print("ERROR: No splits found. Run script 01 first.")
        sys.exit(1)

    all_triplets = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    triplet_map = {t.id: t for t in all_triplets}
    print(f"Loaded {len(all_triplets)} triplets for scoring")

    if args.retry_errors:
        if not args.judge:
            print("ERROR: --retry-errors requires --judge gemini, --judge gpt, or --judge deepseek")
            sys.exit(1)
        fname = f"{phase_name}_{args.judge}.json"
        prev_path = out_dir / fname
        if not prev_path.exists():
            print(f"ERROR: No previous results at {prev_path}. Run without --retry-errors first.")
            sys.exit(1)
        prev = json.loads(prev_path.read_text())
        succeeded_ids = {
            r.get("triplet_id") or r.get("id")
            for r in prev
            if r.get("faithfulness") is not None and (r.get("triplet_id") or r.get("id"))
        }
        all_triplets = [t for t in all_triplets if t.id not in succeeded_ids]
        print(
            f"Retrying {len(all_triplets)} failed/missing triplets (skipping {len(succeeded_ids)} successes)"
        )

    models = config["judges"]["scoring"]["models"]
    use_batch = config["judges"]["scoring"]["use_batch_api"] and not args.no_batch

    if args.judge == "deepseek":
        print("Using DeepSeek real-time API...")
        raw_deepseek = run_scoring_deepseek(all_triplets, phase_name, config)
        results = {"deepseek": raw_deepseek}
    elif use_batch:
        print("Using Batch API (~1-3h turnaround)...")
        results = run_scoring_phase(all_triplets, phase_name, config, only_judge=args.judge)
    else:
        print("Using real-time API...")
        results = run_scoring_realtime(all_triplets, phase_name, config, only_judge=args.judge)

    judge_name_map = {m["provider"]: m["name"] for m in models}
    judge_key_map = {"gpt": "openai", "gemini": "google", "deepseek": "deepseek"}

    out_dir.mkdir(parents=True, exist_ok=True)
    for judge_key, raw_results in results.items():
        provider = judge_key_map.get(judge_key, judge_key)
        judge_name = judge_name_map.get(provider, judge_key)
        fname = f"{phase_name}_{judge_key}.json"

        existing_map: dict[str, dict] = {}
        if args.retry_errors and (out_dir / fname).exists():
            for s in json.loads((out_dir / fname).read_text()):
                key = s.get("triplet_id") or s.get("id")
                if key:
                    existing_map[key] = s

        new_scores = []
        for r in raw_results:
            if r.get("scores"):
                s = r["scores"]
                triplet = triplet_map.get(r["id"])
                if triplet is None:
                    continue
                score = JudgeScore(
                    triplet_id=r["id"],
                    judge_model=judge_name or judge_key,
                    faithfulness=float(s.get("faithfulness", 0)),
                    answer_relevance=float(s.get("answer_relevance", 0)),
                    context_relevance=float(s.get("context_relevance", 0)),
                    is_poisoned=triplet.is_poisoned,
                    injection_type=triplet.injection_type,
                    noise_level=triplet.noise_level,
                    prefilter_applied=False,
                    latency_ms=r.get("latency_ms", 0),
                )
                existing_map[r["id"]] = score.model_dump()
                new_scores.append(score.model_dump())

        final_scores = list(existing_map.values())
        (out_dir / fname).write_text(json.dumps(final_scores, indent=2))
        print(f"Saved {len(final_scores)} scores to {fname} ({len(new_scores)} new)")

    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
