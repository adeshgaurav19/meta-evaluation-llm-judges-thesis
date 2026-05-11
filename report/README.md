# Thesis Report

This directory contains the LaTeX source for the master thesis:

**Meta-Evaluation of LLM Judges: Quantifying Robustness to Poisoned Context and Lightweight Pre-filtering**

Author: Adesh Gaurav  
Programme: MSc in Data Science and Artificial Intelligence Strategy, emlyon business school

## Structure

| File or directory | Purpose |
|---|---|
| `main.tex` | Master LaTeX file; compile this file from inside `report/`. |
| `00_frontmatter.tex` | Title page, abstract, resume, acknowledgements, lists. |
| `01_introduction.tex` | Chapter 1. |
| `02_literature_review.tex` | Chapter 2. |
| `03_methodology.tex` | Chapter 3, including methodology figures. |
| `04_results.tex` | Chapter 4, including result tables and figure references. |
| `05_discussion.tex` | Chapter 5. |
| `06_conclusion.tex` | Chapter 6. |
| `07_references.tex` | Bibliography wrapper. |
| `08_appendices.tex` | Appendices. |
| `references.bib` | BibTeX bibliography database. |
| `Images/` | Report images used by the LaTeX source. |

Generated LaTeX files such as `.aux`, `.bbl`, `.log`, `.out`, `.toc`, `.lof`, `.lot`, and generated PDFs are ignored by git.

## Compile

From this directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The output is `main.pdf`. The repository strategy treats the PDF as a release artifact unless it is deliberately force-added for a submitted snapshot.

## Editing Workflow

For thesis text, edit the chapter `.tex` files directly.

When adding figures:

1. Put the image in `Images/`.
2. Reference it from the relevant chapter with `\includegraphics`.
3. Add a `\caption{...}` and `\label{...}`.
4. Recompile twice after BibTeX so cross-references settle.

## Current Figure Assets

The methodology figures currently tracked in `Images/` are:

- `OverallExperimentalPipeline.png`
- `PoisonedDatasetConstructionPipeline.png`
- `PoisonedRAG-styleInjectionProcess.png`
- `StatisticalPre-filterArchitecture.png`
- `LLMPre-filterStrategy.png`
- `ResidualAwareTrueSurvivedDecomposition.png`

## Notes

The report source is the authoritative thesis text. Root-level documentation explains the code and experiment pipeline; this file only covers the LaTeX report workflow.
