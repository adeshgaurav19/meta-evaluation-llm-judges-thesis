"""
GPT-2 token entropy signal for passage-level anomaly scoring.

This is a lightweight baseline signal: it looks for unusually high or low token
surprisal and for passages whose surprisal varies strongly across chunks.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.schema import PrefilterScore

_model = None
_tokenizer = None


def _load_model(model_name: str = "gpt2"):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name)
        _model.eval()


def _passage_entropy(
    passage: str, chunk_size: int, model_name: str = "gpt2"
) -> tuple[float, float]:
    """Return mean token surprisal and chunk-level variance for one passage."""
    _load_model(model_name)
    enc = _tokenizer.encode(passage, return_tensors="pt")
    if enc.shape[1] < 2:
        return 0.0, 0.0

    with torch.no_grad():
        logits = _model(enc).logits

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    target_ids = enc[0, 1:]
    token_neg_log_probs = -log_probs[:-1].gather(1, target_ids.unsqueeze(1)).squeeze(1)
    token_entropies = token_neg_log_probs.cpu().numpy()

    if len(token_entropies) == 0:
        return 0.0, 0.0

    mean_entropy = float(np.mean(token_entropies))

    n_chunks = max(1, len(token_entropies) // chunk_size)
    chunks = np.array_split(token_entropies, n_chunks)
    chunk_means = [float(np.mean(c)) for c in chunks if len(c) > 0]
    entropy_variance = float(np.var(chunk_means)) if len(chunk_means) > 1 else 0.0

    return mean_entropy, entropy_variance


def score_triplet(triplet, config: dict) -> list[PrefilterScore]:
    ent_cfg = config["prefilter"]["entropy"]
    separator = config["dataset"]["context_separator"]
    model_name = ent_cfg["model"]
    chunk_size = ent_cfg["chunk_size"]
    high_thresh = ent_cfg["high_entropy_threshold"]
    low_thresh = ent_cfg["low_entropy_threshold"]
    var_thresh = ent_cfg["variance_threshold"]

    ctx = triplet.poisoned_context if hasattr(triplet, "poisoned_context") else triplet.context
    passages = [p.strip() for p in ctx.split(separator) if p.strip()]
    ground_truth_indices = set(getattr(triplet, "poisoned_passage_indices", []))

    scores = []
    for i, passage in enumerate(passages):
        ent, var = _passage_entropy(passage, chunk_size, model_name)

        entropy_anomaly = 0.0
        if ent > high_thresh:
            entropy_anomaly = min((ent - high_thresh) / high_thresh, 1.0)
        elif ent < low_thresh:
            entropy_anomaly = min((low_thresh - ent) / low_thresh, 1.0)

        variance_anomaly = min(var / var_thresh, 1.0) if var > var_thresh else 0.0

        raw_score = max(entropy_anomaly, variance_anomaly)
        flagged = ent > high_thresh or ent < low_thresh or var > var_thresh

        scores.append(
            PrefilterScore(
                triplet_id=triplet.id,
                passage_index=i,
                embedding_score=0.0,
                entropy_score=float(raw_score),
                classifier_score=0.0,
                aggregated_score=0.0,
                flagged=flagged,
                ground_truth_poisoned=(
                    i in ground_truth_indices and getattr(triplet, "is_poisoned", False)
                ),
            )
        )

    return scores


def score_dataset(triplets: list, config: dict) -> list[list[PrefilterScore]]:
    return [score_triplet(t, config) for t in tqdm(triplets, desc="GPT-2 entropy scoring")]
