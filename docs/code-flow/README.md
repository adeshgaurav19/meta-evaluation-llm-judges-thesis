# Code Flow

This folder explains how the repository code moves from the raw HotpotQA data to
the final thesis tables, figures, and PDF.

The main path is:

1. Build the poisoned dataset.
2. Run baseline judges.
3. Train and run the pre-filters.
4. Score the filtered contexts and collect justification samples.
5. Regenerate the final analysis tables and figures.
6. Compile the LaTeX report.

The saved v2 outputs under `results/v2/` are the final experiment outputs used
for the submitted thesis. The analysis step can be rerun from those files without
rerunning judge APIs.

Files in this folder:

| File | Covers |
|---|---|
| `01_dataset.md` | Phase 1 dataset construction and splits. |
| `02_judge_scoring.md` | Baseline judge scoring. |
| `03_prefilters.md` | Statistical and LLM pre-filter code. |
| `04_end_to_end.md` | Post-filter scoring, justifications, and audits. |
| `05_analysis_exports.md` | `scripts/analyze.py`, `scripts/plot.py`, and `exports/`. |
| `06_report_build.md` | LaTeX report build and final PDF. |

