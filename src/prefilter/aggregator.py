"""
Aggregation methods for combining passage-level prefilter scores.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np

from src.dataset.schema import PrefilterScore

AggMethod = Literal["weighted_vote", "xgboost", "majority_vote"]


def aggregate(
    scores: list[PrefilterScore],
    config: dict,
    method: AggMethod | None = None,
) -> list[PrefilterScore]:
    """Mutates scores in-place, setting aggregated_score and flagged on each entry."""
    agg_cfg = config["prefilter"]["aggregation"]
    if method is None:
        method = agg_cfg["primary"]

    threshold = agg_cfg["threshold"]

    if method == "weighted_vote":
        w = agg_cfg["weights"]  # [emb, ent, clf, cross, answer_span]
        for s in scores:
            s.aggregated_score = (
                w[0] * s.embedding_score
                + w[1] * s.entropy_score
                + w[2] * s.classifier_score
                + w[3] * s.crossencoder_score
                + w[4] * s.answer_span_score
            )
            s.flagged = s.aggregated_score > threshold

    elif method == "xgboost":
        xgb_path = agg_cfg["xgboost"]["checkpoint_path"]
        if Path(xgb_path).exists():
            clf = pickle.loads(Path(xgb_path).read_bytes())
            # Feature order must match train_xgboost below.
            X = np.array(
                [
                    [
                        s.embedding_score,
                        s.entropy_score,
                        s.classifier_score,
                        s.crossencoder_score,
                        s.answer_span_score,
                    ]
                    for s in scores
                ]
            )
            preds = clf.predict_proba(X)[:, 1]
            for s, p in zip(scores, preds):
                s.aggregated_score = float(p)
                s.flagged = float(p) > threshold
        else:
            return aggregate(scores, config, method="weighted_vote")

    elif method == "majority_vote":
        for s in scores:
            votes = sum(
                [
                    s.embedding_score > 0.5,
                    s.entropy_score > 0.5,
                    s.classifier_score > 0.5,
                    s.crossencoder_score > 0.5,
                    s.answer_span_score > 0.5,
                ]
            )
            s.aggregated_score = votes / 5.0
            s.flagged = votes >= 3

    return scores


def train_xgboost(
    all_scores: list[PrefilterScore],
    config: dict,
) -> object:
    """Train the XGBoost aggregator and save it for later inference."""
    import xgboost as xgb

    X = np.array(
        [
            [
                s.embedding_score,
                s.entropy_score,
                s.classifier_score,
                s.crossencoder_score,
                s.answer_span_score,
            ]
            for s in all_scores
        ]
    )
    y = np.array([int(s.ground_truth_poisoned) for s in all_scores])

    xgb_cfg = config["prefilter"]["aggregation"]["xgboost"]
    clf = xgb.XGBClassifier(
        max_depth=xgb_cfg["max_depth"],
        n_estimators=xgb_cfg["n_estimators"],
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=config["experiment"]["seed"],
    )
    clf.fit(X, y)

    out_path = Path(xgb_cfg["checkpoint_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pickle.dumps(clf))
    print(f"Saved XGBoost aggregator to {out_path}")
    return clf
