# Meta-Evaluation of LLM Judges Under Poisoned RAG Context

This repository contains the code, configuration, saved outputs, analysis exports, and report source for evaluating LLM-as-a-Judge robustness under poisoned RAG context.

The project tests two questions:

1. How do LLM judges behave when the retrieved context is poisoned?
2. Do lightweight pre-filters improve or worsen downstream judge reliability?

The final experiments use three judges, three poisoning types, two pre-filter strategies, and an end-to-end analysis of how filtering changes the context seen by the judge.

## Repository Map

| Path | What it contains |
|---|---|
| `config/` | Experiment configuration. `config/v2.yaml` is the final v2 config. |
| `src/` | Reusable package code: dataset construction/poisoning, judges, pre-filters, metrics, analysis helpers, and utilities. |
| `scripts/` | Main runnable pipeline scripts and final analysis scripts. |
| `notebooks/phase3_4_kaggle.ipynb` | Kaggle notebook used for GPU-heavy Phase 3 and Phase 4 work. The classifier/pre-filter GPU code can be inspected there. |
| `results/v2/` | Saved v2 experiment outputs used by the final analysis: raw judge scores, pre-filter scores, filtered triplets, justification samples, and audit files. |
| `exports/` | Final thesis-ready CSV tables, figures, and summaries generated from `scripts/analyze.py` and `scripts/plot.py`. |
| `report/` | LaTeX report source, bibliography, report images, and report-specific README. |
| `docs/` | Supporting documentation. `docs/code-flow/` explains the end-to-end code flow. |
| `data/` | Local dataset workspace. Large source/intermediate data is not intended to be the main source of truth in Git. |

## Experiment Phases

| Phase | Purpose | Main code |
|---|---|---|
| Phase 1 | Build the poisoned evaluation dataset from HotpotQA. | `scripts/00_download_hotpotqa.py`, `scripts/01_prepare_dataset.py`, `src/dataset/` |
| Phase 2 | Run baseline LLM judges on clean and poisoned triplets. | `scripts/02_run_judges.py`, `src/judges/` |
| Phase 3 | Train and run statistical and LLM pre-filters. | `scripts/03_train_prefilter.py`, `scripts/04_run_prefilter.py`, `scripts/04b_run_llm_prefilter.py`, `src/prefilter/`, `notebooks/phase3_4_kaggle.ipynb` |
| Phase 4 | Score post-filter contexts and run end-to-end analysis. | `scripts/05_evaluate_impact.py`, `scripts/06_run_justifications.py`, `scripts/06b_run_poison_aware_judge.py`, `scripts/analyze.py`, `scripts/plot.py` |

Phase 3 and parts of Phase 4 used Kaggle because the classifier and cross-encoder steps benefit from GPU. The notebook is included so the GPU-side workflow is visible in the repository. A short phase-by-phase code walkthrough is in `docs/code-flow/`.

## Code Map

Main scripts:

| Script | Role |
|---|---|
| `scripts/00_download_hotpotqa.py` | Downloads/prepares the HotpotQA base data. |
| `scripts/01_prepare_dataset.py` | Builds the poisoned evaluation dataset. |
| `scripts/02_run_judges.py` | Runs baseline judge scoring. |
| `scripts/03_train_prefilter.py` | Trains the statistical pre-filter components. |
| `scripts/04_run_prefilter.py` | Applies the statistical pre-filter to triplets. |
| `scripts/04b_run_llm_prefilter.py` | Runs the Mistral triplet-level pre-filter. |
| `scripts/05_evaluate_impact.py` | Scores post-filter contexts and compares impact. |
| `scripts/06_run_justifications.py` | Runs justification prompt experiments. |
| `scripts/06b_run_poison_aware_judge.py` | Runs poison-aware judge prompt experiments. |
| `scripts/analyze.py` | Regenerates final analysis tables from saved v2 outputs. |
| `scripts/plot.py` | Regenerates final figures from exported tables. |
| `scripts/ablation_minimal.py` | Produces the minimal ablation report used in the appendix. |
| `scripts/_archived/` | Older scripts kept for traceability; not part of the final v2 pipeline. |

Source modules:

| Module | Role |
|---|---|
| `src/dataset/` | Dataset loading, poisoning, schema handling, and split creation. |
| `src/judges/` | Judge clients, prompts, batching, and scoring runners. |
| `src/prefilter/` | Statistical signals, aggregation, classifier training, and LLM pre-filter logic. |
| `src/metrics/` | Judge, detection, and system metric helpers. |
| `src/analysis/` | Supporting analysis utilities. |
| `src/utils/` | Shared configuration, logging, async HTTP, and context truncation helpers. |

## Main Pipeline

All scripts are run sequentially. The pipeline is configured through `config/v2.yaml`.

```bash
# Phase 1 — Build the poisoned evaluation dataset
python scripts/00_download_hotpotqa.py --config config/v2.yaml
python scripts/01_prepare_dataset.py --config config/v2.yaml

# Phase 2 — Baseline judge scoring (run each judge separately)
python scripts/02_run_judges.py --config config/v2.yaml --judge gpt
python scripts/02_run_judges.py --config config/v2.yaml --judge gemini
python scripts/02_run_judges.py --config config/v2.yaml --judge deepseek

# Phase 3 — Train and run pre-filters
# Note: 03 and 04 use GPU; run via notebooks/phase3_4_kaggle.ipynb on Kaggle for classifier/cross-encoder steps
python scripts/03_train_prefilter.py --config config/v2.yaml
python scripts/04_run_prefilter.py --config config/v2.yaml
python scripts/04b_run_llm_prefilter.py --config config/v2.yaml

# Phase 4 — Score post-filter contexts (run each judge separately)
python scripts/05_evaluate_impact.py --config config/v2.yaml --judge gpt
python scripts/05_evaluate_impact.py --config config/v2.yaml --judge gemini
python scripts/05_evaluate_impact.py --config config/v2.yaml --judge deepseek

# Phase 4 (cont.) — Justification and poison-aware judge experiments
python scripts/06_run_justifications.py --config config/v2.yaml
python scripts/06b_run_poison_aware_judge.py --config config/v2.yaml

# Analysis — Regenerate tables and figures from saved outputs
python scripts/analyze.py
python scripts/plot.py
```

Some steps require provider API keys and can incur cost. The saved v2 outputs in `results/v2/` are the outputs used for the final analysis, so the analysis tables and figures can be regenerated without rerunning judges.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,judges,poison]"
```

Copy `.env.example` to `.env` and add local API keys if running judge or generation jobs. `.env` is ignored by Git.

## Regenerating Final Analysis

From the repository root:

```bash
python scripts/analyze.py
python scripts/plot.py
```

Outputs are written to `exports/`.

This step expects the local v2 dataset and split files under `data/v2_fixed_poisonedrag/`
and `data/v2_splits/`. Those files can be regenerated with Phase 1 or restored from the
dataset artifact used for the thesis submission; judge API calls are not needed for this
analysis-only step.

Key outputs:

| Output | Purpose |
|---|---|
| `exports/table_baseline_summary.csv` | Baseline judge false-positive rates. |
| `exports/table_mistral_metrics.csv` | Mistral filter metrics. |
| `exports/table_filter_audit.csv` | What the statistical filter removed by attack type. |
| `exports/table_paradox_overview.csv` | Clean / Survived / True decomposition. |
| `exports/table_mcnemar_bootstrap.csv` | Question-clustered significance checks. |
| `exports/figures/` | Final report figures. |

The code-flow notes in `docs/code-flow/` give a phase-by-phase walkthrough of how the saved outputs, exports, and report connect.

## Report

The report source is in `report/`.

Compile from inside `report/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

`report/README.md` contains report-specific notes.

## What Is Tracked

The repository is intended to track:

- source code in `src/` and `scripts/`
- final configuration in `config/`
- final report source in `report/`
- final compact analysis exports in `exports/`
- final v2 saved experiment outputs needed to reproduce the submitted analysis
- documentation in `docs/`
- Kaggle notebook/code used for GPU-based Phase 3 and Phase 4 steps

The repository is not intended to track:

- API keys or `.env`
- generated LaTeX build files
- Python cache files
- local virtual environments
- model checkpoint binaries
- legacy result folders and archived outputs
