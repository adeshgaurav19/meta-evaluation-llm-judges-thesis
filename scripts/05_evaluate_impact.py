"""
Phase 5: Re-run judges on pre-filtered data.

Usage:
    python scripts/05_evaluate_impact.py --config config/base.yaml [--no-batch] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.dataset.schema import JudgeScore, PoisonedTriplet
from src.judges.runner import run_scoring_deepseek, run_scoring_phase, run_scoring_realtime
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument(
        "--judge",
        choices=["gpt", "gemini", "deepseek"],
        default=None,
        help="Run only one judge (default: all)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--llm-prefilter",
        action="store_true",
        help="Use LLM-filtered triplets (filtered_triplets_llm.json) instead of statistical",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    phase_name = "postfilter_llm" if args.llm_prefilter else "postfilter"
    out_dir = Path(config["experiment"]["results_dir"]) / "raw_scores"

    sentinel = f"{phase_name}_{args.judge}.json" if args.judge else f"{phase_name}_gpt.json"
    if not args.force and (out_dir / sentinel).exists():
        print("Post-filter scores already exist. Use --force to rerun.")
        return

    # Load pre-filtered triplets
    fname = "filtered_triplets_llm.json" if args.llm_prefilter else "filtered_triplets.json"
    filtered_path = Path(config["experiment"]["results_dir"]) / "prefilter_scores" / fname
    if not filtered_path.exists():
        script = "04b_run_llm_prefilter.py" if args.llm_prefilter else "04_run_prefilter.py"
        print(f"ERROR: {fname} not found. Run {script} first.")
        sys.exit(1)

    filtered_data = json.loads(filtered_path.read_text())
    filtered_triplets = [PoisonedTriplet(**d) for d in filtered_data]
    triplet_map = {t.id: t for t in filtered_triplets}
    print(f"Loaded {len(filtered_triplets)} pre-filtered triplets")

    models = config["judges"]["scoring"]["models"]
    use_batch = config["judges"]["scoring"]["use_batch_api"] and not args.no_batch

    # Run scoring
    if args.judge == "deepseek":
        print("Using DeepSeek real-time API...")
        raw_deepseek = run_scoring_deepseek(filtered_triplets, phase_name, config)
        results = {"deepseek": raw_deepseek}
    elif use_batch:
        print("Using Batch API...")
        results = run_scoring_phase(filtered_triplets, phase_name, config, only_judge=args.judge)
    else:
        print("Using real-time API...")
        results = run_scoring_realtime(filtered_triplets, phase_name, config, only_judge=args.judge)

    # Convert and save  merge with any existing results to avoid data loss
    judge_name_map = {m["provider"]: m["name"] for m in models}
    judge_key_map = {"gpt": "openai", "gemini": "google", "deepseek": "deepseek"}

    out_dir.mkdir(parents=True, exist_ok=True)
    for judge_key, raw_results in results.items():
        provider = judge_key_map.get(judge_key, judge_key)
        judge_name = judge_name_map.get(provider, judge_key)
        fname = f"{phase_name}_{judge_key}.json"

        # Load existing to merge into (safe rerun)
        existing_map: dict[str, dict] = {}
        if (out_dir / fname).exists():
            for s in json.loads((out_dir / fname).read_text()):
                key = s.get("triplet_id") or s.get("id")
                if key:
                    existing_map[key] = s

        new_count = 0
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
                    prefilter_applied=True,
                    latency_ms=r.get("latency_ms", 0),
                )
                existing_map[r["id"]] = score.model_dump()
                new_count += 1

        final = list(existing_map.values())
        (out_dir / fname).write_text(json.dumps(final, indent=2))
        print(f"Saved {len(final)} scores to {fname} ({new_count} new)")

    print("Phase 5 complete.")


if __name__ == "__main__":
    main()
