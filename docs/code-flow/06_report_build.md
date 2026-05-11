# Report Build

Goal: compile the final thesis PDF from the LaTeX source.

Report source:

- `report/main.tex`
- `report/00_frontmatter.tex`
- `report/01_introduction.tex`
- `report/02_literature_review.tex`
- `report/03_methodology.tex`
- `report/04_results.tex`
- `report/05_discussion.tex`
- `report/06_conclusion.tex`
- `report/07_references.tex`
- `report/08_appendices.tex`
- `report/references.bib`

Build command from inside `report/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Alternative:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Output:

- `report/main.pdf`

Current formatting choices:

- 12 pt Times-like font through `mathptmx`
- double spacing
- A4 paper
- 1 inch margins
- footer with candidate name and page number after the cover page
- cover page without footer

Notes:

- Tables and figures in the report use files from `exports/` and
  `report/images/`.
- Appendix G links the GitHub repository.

