"""
Computational profiling: latency, memory, throughput for each pre-filter component.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd


def profile_component(
    fn,
    triplets: list,
    config: dict,
    n_warmup: int = 2,
) -> dict:
    """
    Profile a scoring function (one of the three signals).
    fn(triplet, config) -> list[PrefilterScore]
    Returns latency stats (ms) and peak memory (MB).
    """
    # Warmup
    for t in triplets[:n_warmup]:
        fn(t, config)

    latencies_ms = []
    peak_mbs = []

    for triplet in triplets:
        tracemalloc.start()
        t0 = time.perf_counter()
        fn(triplet, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latencies_ms.append(elapsed_ms)
        peak_mbs.append(peak / 1e6)

    ctx = triplet.poisoned_context if hasattr(triplet, "poisoned_context") else triplet.context
    separator = config["dataset"]["context_separator"]
    n_passages = len([p for p in ctx.split(separator) if p.strip()])

    return {
        "n_triplets": len(triplets),
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_std_ms": float(np.std(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_p99_ms": float(np.percentile(latencies_ms, 99)),
        "peak_memory_mean_mb": float(np.mean(peak_mbs)),
        "peak_memory_max_mb": float(np.max(peak_mbs)),
        "throughput_triplets_per_sec": 1000.0 / max(np.mean(latencies_ms), 0.01),
        "avg_passages_per_triplet": n_passages,
    }


def profile_full_pipeline(
    triplets: list,
    config: dict,
) -> dict:
    """Profile each pipeline component separately and combined."""
    from src.prefilter import classifier_signal, embedding_signal, entropy_signal
    from src.prefilter.pipeline import run_pipeline

    components = {
        "embedding": embedding_signal.score_triplet,
        "entropy": entropy_signal.score_triplet,
        "classifier": classifier_signal.score_triplet,
    }

    component_profiles = {}
    for name, fn in components.items():
        print(f"Profiling {name}...")
        component_profiles[name] = profile_component(fn, triplets, config)

    # Full pipeline
    print("Profiling full pipeline...")
    tracemalloc.start()
    t0 = time.perf_counter()
    _, _ = run_pipeline(triplets, config)
    total_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    component_profiles["full_pipeline"] = {
        "total_latency_ms": total_ms,
        "peak_memory_mb": peak / 1e6,
        "throughput_triplets_per_sec": len(triplets) / (total_ms / 1000),
    }

    return component_profiles


def save_compute_tables(profiles: dict, config: dict) -> None:
    out_dir = Path(config["experiment"]["results_dir"]) / "tables"
    bench_dir = Path(config["experiment"]["results_dir"]) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Latency table
    latency_rows = []
    memory_rows = []
    efficiency_rows = []

    for component, stats in profiles.items():
        if component == "full_pipeline":
            continue
        latency_rows.append(
            {
                "component": component,
                "mean_ms": stats.get("latency_mean_ms"),
                "p50_ms": stats.get("latency_p50_ms"),
                "p95_ms": stats.get("latency_p95_ms"),
                "p99_ms": stats.get("latency_p99_ms"),
                "throughput_tps": stats.get("throughput_triplets_per_sec"),
            }
        )
        memory_rows.append(
            {
                "component": component,
                "peak_mean_mb": stats.get("peak_memory_mean_mb"),
                "peak_max_mb": stats.get("peak_memory_max_mb"),
            }
        )

    pd.DataFrame(latency_rows).to_csv(out_dir / "table_4_10_latency.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(out_dir / "table_4_11_memory.csv", index=False)

    # Save raw benchmarks
    (bench_dir / "latency_profile.json").write_text(json.dumps(profiles, indent=2))
    print("Saved compute tables and benchmark profiles.")
