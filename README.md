# Meta-Evaluation of LLM Judges Under Poisoned RAG Context

This repository contains the code, experiment configuration, analysis outputs, and LaTeX thesis source for Adesh Gaurav's master thesis:

**Meta-Evaluation of LLM Judges: Quantifying Robustness to Poisoned Context and Lightweight Pre-filtering**

The project studies whether contemporary LLM-as-a-Judge pipelines remain reliable when retrieved context is poisoned, and whether lightweight pre-filtering improves or worsens downstream judge reliability.

## Current Findings

The final thesis results support four main claims:

- LLM judges fail at substantial rates on poisoned RAG context. Baseline faithfulness FPR ranges from 0.412 to 0.699 across the three judges studied.
- Judge behaviour is structured by calibration regime. GPT behaves like a continuous scorer, Gemma is highly lenient, and DeepSeek is near-binary.
- The statistical pre-filter is strong as a passage-level classifier on the held-out test split (F1 = 0.929, precision = 0.955, recall = 0.904), but this does not translate cleanly into better end-to-end judge reliability.
- The main end-to-end failure mechanism is context clarification through distractor removal: filtering shortens and cleans the residual context, and judges can over-trust that cleaner-looking residual even when poison remains.

## Repository Layout

| Path | Purpose | Push policy |
|---|---|---|
| `src/` | Dataset construction, judge runners, pre-filter modules, metrics, analysis helpers | Push |
| `scripts/` | Reproducible pipeline entry points | Push |
| `config/` | Experiment configuration files | Push |
| `report/` | LaTeX thesis source, bibliography, report figures, final manuscript assets | Push source; treat generated PDFs as release artifacts |
| `report/Images/` | Methodology figures used in Chapter 3 | Push |
| `exports/` | Final compact tables, figures, and summaries used by the thesis | Push selected final exports |
| `data/` | HotpotQA source, poisoned datasets, splits, batch inputs, checkpoints | Do not push by default; use Git LFS or an external artifact release |
| `results/` | Raw judge outputs, pre-filter outputs, model checkpoints, generated tables/figures | Push compact tables/figures only; keep raw outputs/checkpoints out of normal git |
| `docs/` | Repository documentation and push strategy | Push |

See [docs/GIT_PUSH_STRATEGY.md](docs/GIT_PUSH_STRATEGY.md) for the exact first-push manifest.

## Setup

Create and activate a Python environment, then install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,judges,poison]"
```

Copy `.env.example` to `.env` and fill in local API keys when running judge or generation jobs. The real `.env` file is intentionally ignored by git.

## Main Pipeline

The pipeline is configured through `config/v2.yaml`.

```bash
python scripts/00_download_hotpotqa.py --config config/v2.yaml
python scripts/01_prepare_dataset.py --config config/v2.yaml
python scripts/02_run_judges.py --config config/v2.yaml
python scripts/03_train_prefilter.py --config config/v2.yaml
python scripts/04_run_prefilter.py --config config/v2.yaml
python scripts/04b_run_llm_prefilter.py --config config/v2.yaml
python scripts/05_evaluate_impact.py --config config/v2.yaml
python scripts/analyze.py
python scripts/plot.py
```

Some steps require provider API keys and can incur cost. Large raw outputs and trained model checkpoints are treated as artifacts rather than normal git-tracked source.

## Report Build

Compile the thesis from the `report/` directory:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The generated `main.pdf` is useful for review, but the push strategy treats PDFs as release artifacts unless you explicitly decide to version the final submitted PDF.

## Documentation

- [docs/GIT_PUSH_STRATEGY.md](docs/GIT_PUSH_STRATEGY.md): what to include in the separate thesis repository, what to exclude, and the exact push sequence.
- [report/README.md](report/README.md): LaTeX-specific notes.

## Reproducibility Notes

The repository is designed to be source-first. The initial git push should contain the code, configuration, LaTeX source, compact final exports, and documentation. Large datasets, raw provider outputs, and model checkpoints should be distributed through Git LFS, GitHub Releases, Zenodo, Google Drive, or another artifact store with checksums.
