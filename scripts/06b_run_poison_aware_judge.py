"""
Phase 6b: Poison-aware judge prompting on the same 28 justification samples.

Runs GPT-5.4-nano with an enhanced prompt that describes corpus poisoning patterns,
enabling direct before/after comparison against the standard prompt (Phase 6).

Usage:
    python scripts/06b_run_poison_aware_judge.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.dataset.splitter import load_splits
from src.judges.prompts import POISON_AWARE_JUSTIFICATION_PROMPT
from src.utils.config import load_config


def run_poison_aware_phase(selected_triplets: list, config: dict) -> list[dict]:
    from openai import OpenAI

    client = OpenAI()

    model = config["judges"]["justification"]["model"]
    max_tokens = config["judges"]["justification"]["max_output_tokens"]
    results = []

    for t in selected_triplets:
        ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": POISON_AWARE_JUSTIFICATION_PROMPT.format(
                            question=t.question,
                            context=ctx,
                            answer=t.answer,
                        ),
                    }
                ],
            )
            content = response.choices[0].message.content
        except Exception as e:
            content = json.dumps({"error": str(e)})

        results.append(
            {
                "id": t.id,
                "content": content,
                "is_poisoned": getattr(t, "is_poisoned", False),
                "injection_type": getattr(t, "injection_type", None),
                "noise_level": getattr(t, "noise_level", None),
            }
        )

    return results


def parse_poison_aware(raw: dict) -> dict:
    content = raw.get("content", "")
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"parse_error": True, "raw": content[:200]}

    result = {
        "poisoning_detected": parsed.get("poisoning_detected"),
        "suspicious_patterns": parsed.get("suspicious_patterns", ""),
    }
    for metric in ("faithfulness", "answer_relevance", "context_relevance"):
        if isinstance(parsed.get(metric), dict):
            result[metric] = {
                "score": parsed[metric].get("score"),
                "reason": parsed[metric].get("reason", ""),
            }
        else:
            result[metric] = {"score": parsed.get(metric), "reason": ""}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    just_dir = Path(config["experiment"]["results_dir"]) / "justifications"
    out_path = just_dir / "justification_samples_poison_aware.json"

    if not args.force and out_path.exists():
        print(f"{out_path.name} already exists. Use --force to rerun.")
        return

    baseline_just = just_dir / "justification_samples.json"
    if not baseline_just.exists():
        print("ERROR: justification_samples.json not found. Run script 06 first.")
        sys.exit(1)

    selected_ids = {r["id"] for r in json.loads(baseline_just.read_text())}

    splits = load_splits(config["dataset"]["splits_path"])
    all_triplets = splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    triplet_map = {t.id: t for t in all_triplets}
    selected = [triplet_map[tid] for tid in selected_ids if tid in triplet_map]
    print(f"Running poison-aware judge on {len(selected)} samples")

    raw_results = run_poison_aware_phase(selected, config)

    parsed = []
    for raw in raw_results:
        entry = {
            "id": raw["id"],
            "is_poisoned": raw.get("is_poisoned", False),
            "injection_type": raw.get("injection_type"),
            "noise_level": raw.get("noise_level"),
            "raw_response": raw.get("content", ""),
            "parsed": parse_poison_aware(raw),
        }
        parsed.append(entry)

    just_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed, indent=2))
    print(f"Saved {len(parsed)} poison-aware justifications to {out_path}")

    # Quick summary
    poisoned = [p for p in parsed if p["is_poisoned"]]
    detected = [p for p in poisoned if p["parsed"].get("poisoning_detected")]
    fp = [
        p for p in poisoned if (p["parsed"].get("faithfulness", {}) or {}).get("score") or 0 > 0.5
    ]
    print(f"\nPoison detection rate: {len(detected)}/{len(poisoned)} poisoned samples flagged")
    print(f"False positives (faith>0.5): {len(fp)}/{len(poisoned)}")
    print("Phase 6b complete.")


if __name__ == "__main__":
    main()
