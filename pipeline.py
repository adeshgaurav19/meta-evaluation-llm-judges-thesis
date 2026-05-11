"""
Run the thesis experiment pipeline end-to-end, or resume from a later phase.

Usage:
    python pipeline.py                          # run all phases
    python pipeline.py --from-phase 3           # resume from phase 3
    python pipeline.py --only-phases 1 2        # run only phases 1 and 2
    python pipeline.py --force                  # re-run everything even if outputs exist
    python pipeline.py --no-batch               # use real-time API (no 24h wait)
    python pipeline.py --config path/to/cfg.yaml
"""
import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


# Phase definitions

class Phase(NamedTuple):
    number: int
    name: str
    script: str
    description: str
    sentinel: str          # File whose existence means this phase is done


PHASES: list[Phase] = [
    Phase(0, "download_hotpotqa",  "scripts/00_download_hotpotqa.py",
          "Download HotpotQA and sample 100 hard questions into data/raw/",
          "data/raw/hotpotqa_base.json"),
    Phase(1, "prepare_dataset",    "scripts/01_prepare_dataset.py",
          "Load RAG logs, inject three poison levels, truncate, and split",
          "data/v2_splits/train.json"),
    Phase(2, "baseline_scoring",   "scripts/02_run_judges.py",
          "Baseline judge scoring via Batch API (GPT + Gemini)",
          "results/raw_scores/baseline_gpt.json"),
    Phase(3, "train_prefilter",    "scripts/03_train_prefilter.py",
          "Fine-tune classifier and train XGBoost aggregator",
          "results/v2/models/xgboost_aggregator.pkl"),
    Phase(4, "run_prefilter",      "scripts/04_run_prefilter.py",
          "Score all passages with embedding / entropy / classifier signals",
          "results/prefilter_scores/aggregated_scores.json"),
    Phase(5, "evaluate_impact",    "scripts/05_evaluate_impact.py",
          "Re-run judges on pre-filtered data via Batch API",
          "results/raw_scores/postfilter_gpt.json"),
    Phase(6, "justifications",     "scripts/06_run_justifications.py",
          "GPT justification on 30 selected samples (real-time)",
          "results/justifications/justification_samples.json"),
    Phase(7, "analyze_exports",    "scripts/analyze.py",
          "Generate thesis-ready CSV tables in exports/",
          "exports/judge_scores_long.csv"),
    Phase(8, "plot_exports",       "scripts/plot.py",
          "Generate thesis figures from exports/ CSVs",
          "exports/figures/fig01_baseline_calibration.png"),
]


# Helpers

def _phase_done(phase: Phase) -> bool:
    return Path(phase.sentinel).exists()


def _print_plan(phases_to_run: list[Phase], force: bool) -> None:
    print(f"\n{'='*65}")
    print(f"  META-EVAL PIPELINE    {len(phases_to_run)} phase(s) queued")
    print(f"{'='*65}")
    for p in phases_to_run:
        done = _phase_done(p) and not force
        status = "SKIP" if done else "RUN "
        print(f"  {status}  Phase {p.number}: {p.name:<22} {p.description}")
    print(f"{'='*65}\n")


def _run_phase(phase: Phase, extra_args: list[str], config_path: str) -> bool:
    cmd = [sys.executable, phase.script, "--config", config_path] + extra_args
    start = time.time()

    print(f"\nPhase {phase.number}: {phase.name}")
    print(f"  {phase.description}")
    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  Phase {phase.number} complete ({elapsed:.0f}s)")
        return True
    else:
        print(f"\n  Phase {phase.number} FAILED (exit {result.returncode})")
        return False


def _print_summary(
    completed: list[int],
    skipped: list[int],
    failed: int | None,
    wall_seconds: float,
) -> None:
    print(f"\n{'='*65}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*65}")
    print(f"  Completed phases : {completed}")
    print(f"  Skipped (done)   : {skipped}")
    if failed is not None:
        print(f"  Failed at phase  : {failed}")
    print(f"  Wall time        : {wall_seconds/60:.1f} min")

    exports = sorted(Path("exports").glob("*.csv")) if Path("exports").exists() else []
    if exports:
        print(f"\n  Exports : {len(exports)} CSV files -> exports/")

    print(f"{'='*65}\n")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full meta-eval experiment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config/base.yaml", help="Path to base.yaml")
    parser.add_argument("--from-phase", type=int, default=0, metavar="N",
                        help="Start from phase N (0-9). Skips earlier phases.")
    parser.add_argument("--only-phases", type=int, nargs="+", metavar="N",
                        help="Run only these phase numbers, e.g. --only-phases 1 2 9")
    parser.add_argument("--force", action="store_true",
                        help="Re-run phases even if sentinel files already exist")
    parser.add_argument("--no-batch", action="store_true",
                        help="Use real-time API for phases 2 and 5.")
    parser.add_argument("--stop-after-phase", type=int, metavar="N",
                        help="Stop after completing phase N")
    args = parser.parse_args()

    # Determine which phases to run
    if args.only_phases:
        phases_to_run = [p for p in PHASES if p.number in args.only_phases]
    else:
        phases_to_run = [p for p in PHASES if p.number >= args.from_phase]

    if args.stop_after_phase:
        phases_to_run = [p for p in phases_to_run if p.number <= args.stop_after_phase]

    if not phases_to_run:
        print("No phases selected.")
        sys.exit(0)

    _print_plan(phases_to_run, args.force)

    # Build extra args to pass down to each script
    extra_args: list[str] = []
    if args.force:
        extra_args.append("--force")
    if args.no_batch:
        extra_args.append("--no-batch")

    # Run phases
    wall_start = time.time()
    completed: list[int] = []
    skipped: list[int] = []
    failed_phase = None

    for phase in phases_to_run:
        # Skip if already done and not forced
        if _phase_done(phase) and not args.force:
            print(f"  {_color(f'Phase {phase.number} ({phase.name}): already done; skipping', DIM)}")
            skipped.append(phase.number)
            continue

        success = _run_phase(phase, extra_args, args.config)
        if success:
            completed.append(phase.number)
        else:
            failed_phase = phase.number
            print(f"\n{_color(f'Pipeline halted at phase {phase.number}.', RED)}")
            print(f"Fix the error and resume with:")
            print(f"  python pipeline.py --from-phase {phase.number} --config {args.config}")
            print(f"  (or skip this phase: --from-phase {phase.number + 1})")
            break

    _print_summary(completed, skipped, failed_phase, time.time() - wall_start)

    if failed_phase is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
