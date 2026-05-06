"""
Phase 3: Train DeBERTa classifier + XGBoost aggregator, then evaluate aggregation methods.

Usage:
    python scripts/03_train_prefilter.py --config config/base.yaml [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.dataset.splitter import load_splits
from src.prefilter.train_classifier import train_classifier
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_dir = Path(config["prefilter"]["classifier"]["checkpoint_dir"])

    if not args.force and (checkpoint_dir / "config.json").exists():
        print("Classifier already trained. Use --force to retrain.")
        return

    splits = load_splits(config["dataset"]["splits_path"])
    if not splits:
        print("ERROR: No splits found. Run script 01 first.")
        sys.exit(1)

    train_triplets = splits.get("train", [])
    val_triplets = splits.get("val", [])

    if not train_triplets:
        print("ERROR: Empty training split.")
        sys.exit(1)

    print(f"Training on {len(train_triplets)} train, {len(val_triplets)} val triplets...")
    train_classifier(train_triplets, val_triplets, config)

    # Also train XGBoost aggregator on val set scores
    print("\nTraining XGBoost aggregator on val set pre-filter scores...")
    from tqdm.auto import tqdm

    from src.dataset.schema import PrefilterScore
    from src.prefilter import (
        answer_span_signal,
        classifier_signal,
        embedding_signal,
        entropy_signal,
    )
    from src.prefilter.aggregator import train_xgboost

    all_val_scores = []
    for t in tqdm(val_triplets, desc="Scoring val set for XGBoost"):
        emb = embedding_signal.score_triplet(t, config)
        ent = entropy_signal.score_triplet(t, config)
        clf = classifier_signal.score_triplet(t, config)
        ans = answer_span_signal.score_triplet(t, config)
        for i in range(len(emb)):
            combined = PrefilterScore(
                triplet_id=t.id,
                passage_index=i,
                embedding_score=emb[i].embedding_score if i < len(emb) else 0.0,
                entropy_score=ent[i].entropy_score if i < len(ent) else 0.0,
                classifier_score=clf[i].classifier_score if i < len(clf) else 0.5,
                answer_span_score=ans[i].answer_span_score if i < len(ans) else 0.5,
                aggregated_score=0.0,
                flagged=False,
                ground_truth_poisoned=emb[i].ground_truth_poisoned if i < len(emb) else False,
            )
            all_val_scores.append(combined)

    if all_val_scores:
        train_xgboost(all_val_scores, config)

    # Phase 3c - evaluate aggregation methods on val set before Phase 4
    print("\nEvaluating aggregation methods on val set...")
    import copy

    from sklearn.metrics import f1_score, precision_score, recall_score

    from src.prefilter.aggregator import aggregate

    methods = ["weighted_vote", "xgboost", "majority_vote"]
    print(f"\n{'Method':<20} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 48)
    best_method, best_f1 = "weighted_vote", 0.0
    for method in methods:
        scores_copy = copy.deepcopy(all_val_scores)
        aggregate(scores_copy, config, method=method)
        y_true = [int(s.ground_truth_poisoned) for s in scores_copy]
        y_pred = [int(s.flagged) for s in scores_copy]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        note = "  (in-sample)" if method == "xgboost" else ""
        print(f"{method:<20} {f1:>6.3f} {p:>10.3f} {r:>8.3f}{note}")
        if f1 > best_f1 and method != "xgboost":
            best_f1, best_method = f1, method

    print(f"\nBest out-of-sample method: {best_method} (F1={best_f1:.3f})")
    print(f"Primary in config is: '{config['prefilter']['aggregation']['primary']}'")
    print("Update config/base.yaml -> prefilter.aggregation.primary before running Phase 4.")
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
