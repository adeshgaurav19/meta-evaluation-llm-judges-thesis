"""
Phase 1 (v2): Load HotpotQA -> inject poison -> split -> save.

Changes from v1:
  - Noise levels: [0.2, 0.4, 0.6, 0.8] (was 8 levels)
  - PoisonedRAG (Type 3): fixed prompt using full context + target wrong answer
  - Output: data/v2_fixed_poisonedrag/ (original data/ preserved)
  - target_wrong_answer stored as metadata field on Type 3 triplets only

Usage:
    python scripts/01_prepare_dataset.py --config config/v2.yaml [--force]
    python scripts/01_prepare_dataset.py --config config/v2.yaml --spot-check-type3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from tqdm.auto import tqdm

from src.dataset.loader import load_raw
from src.dataset.poisoner import (
    generate_crafted_passages,
    generate_rewrites,
    generate_target_wrong_answers,
    inject_adversarial_facts,
    inject_poisonedrag_style,
    inject_random_noise,
    make_clean_copy,
    precompute_embeddings,
)
from src.dataset.schema import PoisonedTriplet
from src.dataset.splitter import save_splits, stratified_split
from src.utils.config import load_config
from src.utils.context_truncator import save_truncation_stats, truncate_dataset


def _n_malicious(n_passages: int, noise_level: float) -> int:
    return max(1, math.ceil(n_passages * noise_level))


def spot_check_type3(
    all_triplets: list[PoisonedTriplet],
    target_wrong_map: dict[str, str],
    n: int = 20,
    seed: int = 42,
) -> None:
    """Print n random Type 3 poisoned triplets for manual validation."""
    rng = random.Random(seed)
    type3 = [t for t in all_triplets if t.is_poisoned and t.injection_type == "poisonedrag_style"]
    sample = rng.sample(type3, min(n, len(type3)))
    print(f"\n{'='*80}")
    print(f"SPOT CHECK - {len(sample)} random poisonedrag_style triplets")
    print(f"{'='*80}")
    for t in sample:
        # base question id is the triplet id before the _pr_ suffix
        base_id = t.id.split("_pr_")[0] if "_pr_" in t.id else t.id
        print(f"\nID: {t.id}  noise={t.noise_level}")
        print(f"Question:       {t.question}")
        print(f"Correct answer: {t.answer}")
        print(f"Target wrong:   {t.target_wrong_answer or target_wrong_map.get(base_id, 'N/A')}")
        passages = t.poisoned_context.split("\n\n")
        poisoned_passages = [passages[i] for i in t.poisoned_passage_indices if i < len(passages)]
        for i, p in enumerate(poisoned_passages[:2]):
            print(f"Poisoned passage {i+1}: {p[:300]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/v2.yaml")
    parser.add_argument("--force", action="store_true", help="Regenerate even if outputs exist")
    parser.add_argument(
        "--spot-check-type3",
        action="store_true",
        help="Print Type 3 spot-check samples (requires dataset already generated)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rng = random.Random(config["experiment"]["seed"])

    splits_dir = Path(config["dataset"]["splits_path"])
    out_base = Path(config["dataset"]["output_path"])

    if args.spot_check_type3:
        level3_dir = out_base / "level3_poisonedrag"
        twa_path = out_base / "checkpoints" / "target_wrong_answers.json"
        all_triplets: list[PoisonedTriplet] = []
        for f in sorted(level3_dir.glob("noise_*.json")):
            all_triplets.extend([PoisonedTriplet(**t) for t in json.loads(f.read_text())])
        target_wrong_map = json.loads(twa_path.read_text()) if twa_path.exists() else {}
        seed = config["experiment"]["seed"]
        spot_check_type3(all_triplets, target_wrong_map, n=20, seed=seed)
        return

    if not args.force and (splits_dir / "train.json").exists():
        print(f"Splits already exist at {splits_dir}. Use --force to regenerate.")
        return

    # 1. Load raw triplets
    raw_path = Path(config["dataset"]["raw_path"])
    print(f"Loading RAG logs from {raw_path}...")
    triplets = load_raw(raw_path)

    if not triplets:
        print("ERROR: No RAG logs found.")
        sys.exit(1)

    n_target = config["dataset"]["n_base_triplets"]
    if len(triplets) > n_target:
        rng.shuffle(triplets)
        triplets = triplets[:n_target]
    print(f"Loaded {len(triplets)} base triplets")

    separator = config["dataset"]["context_separator"]
    noise_levels = config["dataset"]["noise_levels"]
    injection_types = config["dataset"]["injection_types"]
    seed = config["experiment"]["seed"]
    max_concurrent = config["poison_generation"].get("max_concurrent", 10)

    # 2. Pre-compute embeddings
    all_passages = [p.strip() for t in triplets for p in t.context.split(separator) if p.strip()]
    print(f"Pre-computing embeddings for {len(triplets)} questions + {len(all_passages)} passages...")
    precompute_embeddings([t.question for t in triplets] + all_passages)

    # 3. Generate target wrong answers (Type 3 only, 100 total)
    target_wrong_map: dict[str, str] = {}
    if "poisonedrag_style" in injection_types:
        print("Generating target wrong answers for Type 3 (100 questions)...")
        twa_checkpoint = str(out_base / "checkpoints" / "target_wrong_answers.json")
        Path(twa_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        target_wrong_map = asyncio.run(
            generate_target_wrong_answers(triplets, config, checkpoint_path=twa_checkpoint)
        )

    # 4. Pre-generate LLM content
    rewritten_map: dict[str, str] = {}
    crafted_by_job: list[list[str]] = []
    prag_jobs: list[tuple[str, str, int, str, str]] = []

    if "adversarial_fact" in injection_types:
        unique_passages = list(dict.fromkeys(all_passages))
        print(f"Pre-generating {len(unique_passages)} adversarial rewrites...")
        af_checkpoint = str(out_base / "checkpoints" / "rewrites.json")
        Path(af_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        rewritten_map = asyncio.run(
            generate_rewrites(unique_passages, config, max_concurrent=max_concurrent,
                              checkpoint_path=af_checkpoint)
        )

    if "poisonedrag_style" in injection_types:
        # Loop order must match consumption order in step 5: noise_level outer, triplet inner
        for nl in noise_levels:
            for t in triplets:
                passages = [p.strip() for p in t.context.split(separator) if p.strip()]
                twa = target_wrong_map.get(t.id, "unknown")
                n = _n_malicious(len(passages), nl)
                prag_jobs.append((t.question, t.context, n, twa, t.answer))
        print(f"Pre-generating PoisonedRAG passages for {len(prag_jobs)} jobs...")
        pr_checkpoint = str(out_base / "checkpoints" / "crafted.json")
        crafted_by_job = asyncio.run(
            generate_crafted_passages(prag_jobs, config, max_concurrent=max_concurrent,
                                      checkpoint_path=pr_checkpoint)
        )

    # 5. Inject poison
    all_poisoned: list[PoisonedTriplet] = []
    prag_job_idx = 0

    inj_pbar = tqdm(injection_types, desc="Injection types", position=0)
    for inj_type in inj_pbar:
        inj_pbar.set_description(f"Injecting [{inj_type}]")
        level_dir = out_base / {
            "random_noise": "level1_random",
            "adversarial_fact": "level2_adversarial",
            "poisonedrag_style": "level3_poisonedrag",
        }[inj_type]
        level_dir.mkdir(parents=True, exist_ok=True)

        for nl in tqdm(noise_levels, desc=f"  noise levels", position=1, leave=False):
            batch: list[PoisonedTriplet] = []

            for t in tqdm(triplets, desc=f"    triplets @ noise={nl}", position=2, leave=False):
                if inj_type == "random_noise":
                    poisoned = inject_random_noise(t, triplets, nl, separator, seed)
                elif inj_type == "adversarial_fact":
                    poisoned = inject_adversarial_facts(t, nl, rewritten_map, separator, seed)
                else:  # poisonedrag_style
                    crafted = crafted_by_job[prag_job_idx]
                    prag_job_idx += 1
                    poisoned = inject_poisonedrag_style(t, nl, crafted, separator, seed)
                    # Attach target_wrong_answer as metadata (not read by downstream scripts)
                    poisoned = poisoned.model_copy(
                        update={"target_wrong_answer": target_wrong_map.get(t.id, "")}
                    )
                batch.append(poisoned)
                batch.append(make_clean_copy(t, inj_type, nl))

            out_file = level_dir / f"noise_{nl:.1f}.json"
            out_file.write_text(json.dumps([b.model_dump() for b in batch], indent=2))
            all_poisoned.extend(batch)

    print(f"Generated {len(all_poisoned)} total triplets (poisoned + clean)")

    # 6. Spot-check Type 3 (optional, exits after printing)
    if args.spot_check_type3:
        spot_check_type3(all_poisoned, target_wrong_map, n=20, seed=seed)
        return

    # 7. Truncate contexts
    print("Truncating contexts to token limits...")
    stats = truncate_dataset(all_poisoned, config)
    save_truncation_stats(stats, config)

    # 8. Split
    print("Creating stratified splits...")
    splits = stratified_split(
        all_poisoned,
        train=config["dataset"]["splits"]["train"],
        val=config["dataset"]["splits"]["val"],
        test=config["dataset"]["splits"]["test"],
        seed=seed,
    )
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_splits(splits, splits_dir)

    total_poisoned = sum(1 for t in all_poisoned if t.is_poisoned)
    total_clean = sum(1 for t in all_poisoned if not t.is_poisoned)
    print(
        f"\nPhase 1 complete. Dataset: {len(all_poisoned)} triplets "
        f"({total_poisoned} poisoned, {total_clean} clean)\n"
        f"Noise levels: {noise_levels}\n"
        f"Output: {out_base}\n"
        f"Splits: {splits_dir}"
    )


if __name__ == "__main__":
    main()
