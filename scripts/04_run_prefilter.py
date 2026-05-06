"""
Phase 3b: Score all passages with pre-filter pipeline.

Usage:
    python scripts/04_run_prefilter.py --config config/base.yaml [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.dataset.schema import PrefilterScore
from src.dataset.splitter import load_splits
from src.prefilter import answer_span_signal, classifier_signal, embedding_signal, entropy_signal
from src.prefilter.aggregator import aggregate
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["experiment"]["results_dir"]) / "prefilter_scores"

    if not args.force and (out_dir / "aggregated_scores.json").exists():
        print("Prefilter scores already exist. Use --force to rerun.")
        return

    splits = load_splits(config["dataset"]["splits_path"])
    if not splits:
        print("ERROR: No splits found. Run script 01 first.")
        sys.exit(1)

    all_triplets = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    print(f"Scoring {len(all_triplets)} triplets with pre-filter...")

    from tqdm.auto import tqdm

    all_emb, all_ent, all_clf, all_ans, all_agg = [], [], [], [], []

    for t in tqdm(all_triplets, desc="Scoring all triplets"):
        emb = embedding_signal.score_triplet(t, config)
        ent = entropy_signal.score_triplet(t, config)
        clf = classifier_signal.score_triplet(t, config)
        ans = answer_span_signal.score_triplet(t, config)

        combined = []
        for j in range(len(emb)):
            ps = PrefilterScore(
                triplet_id=t.id,
                passage_index=j,
                embedding_score=emb[j].embedding_score if j < len(emb) else 0.0,
                entropy_score=ent[j].entropy_score if j < len(ent) else 0.0,
                classifier_score=clf[j].classifier_score if j < len(clf) else 0.5,
                answer_span_score=ans[j].answer_span_score if j < len(ans) else 0.5,
                aggregated_score=0.0,
                flagged=False,
                ground_truth_poisoned=emb[j].ground_truth_poisoned if j < len(emb) else False,
            )
            combined.append(ps)

        aggregate(combined, config)

        all_emb.extend(
            [ps.model_copy(update={"aggregated_score": ps.embedding_score}) for ps in emb]
        )
        all_ent.extend([ps.model_copy(update={"aggregated_score": ps.entropy_score}) for ps in ent])
        all_clf.extend(
            [ps.model_copy(update={"aggregated_score": ps.classifier_score}) for ps in clf]
        )
        all_ans.extend(
            [ps.model_copy(update={"aggregated_score": ps.answer_span_score}) for ps in ans]
        )
        all_agg.extend(combined)

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, scores in [
        ("embedding", all_emb),
        ("entropy", all_ent),
        ("classifier", all_clf),
        ("answer_span", all_ans),
        ("aggregated", all_agg),
    ]:
        path = out_dir / f"{name}_scores.json"
        path.write_text(json.dumps([s.model_dump() for s in scores], indent=2))
        print(f"Saved {len(scores)} {name} scores")

    # Save filtered triplets (passages removed) for phase 5
    from src.prefilter.pipeline import run_pipeline

    print("\nApplying filter and saving filtered triplets...")
    filtered_triplets, _ = run_pipeline(all_triplets, config, mode="filter")
    filtered_path = out_dir / "filtered_triplets.json"
    filtered_path.write_text(json.dumps([t.model_dump() for t in filtered_triplets], indent=2))
    n_flagged = sum(
        1
        for orig, filt in zip(all_triplets, filtered_triplets)
        if (
            getattr(orig, "poisoned_context", orig.context)
            != getattr(filt, "poisoned_context", filt.context)
        )
    )
    print(f"Saved {len(filtered_triplets)} filtered triplets ({n_flagged} had passages removed)")

    print("Phase 3b complete.")


if __name__ == "__main__":
    main()
