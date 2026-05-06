"""
Phase 4d: Computational benchmarking — latency, memory, throughput.

Usage:
    python scripts/08_benchmark.py --config config/base.yaml [--n-samples 50] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.splitter import load_splits
from src.metrics.compute_metrics import profile_full_pipeline, save_compute_tables
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of triplets to benchmark")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    bench_dir = Path(config["experiment"]["results_dir"]) / "benchmarks"

    if not args.force and (bench_dir / "latency_profile.json").exists():
        print("Benchmark results already exist. Use --force to rerun.")
        return

    splits = load_splits(config["dataset"]["splits_path"])
    if not splits:
        print("ERROR: No splits found. Run script 01 first.")
        sys.exit(1)

    test_triplets = splits.get("test", [])[: args.n_samples]
    print(f"Benchmarking on {len(test_triplets)} triplets...")

    profiles = profile_full_pipeline(test_triplets, config)
    save_compute_tables(profiles, config)

    print("\nLatency summary:")
    for component, stats in profiles.items():
        if "latency_mean_ms" in stats:
            print(
                f"  {component}: {stats['latency_mean_ms']:.1f}ms avg, "
                f"{stats['throughput_triplets_per_sec']:.1f} triplets/sec"
            )
        elif "total_latency_ms" in stats:
            print(f"  {component}: {stats['total_latency_ms']:.0f}ms total")

    print("Phase 4d complete.")


if __name__ == "__main__":
    main()
