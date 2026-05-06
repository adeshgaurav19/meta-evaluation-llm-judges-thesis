# Git Push Strategy

This project should become a separate thesis repository. It is currently inside a broader `Documents/Projects` git worktree, so the first step is to initialize git inside `Thesis/` and push only the thesis files.

## Goal

Create a clean repository that someone can clone to:

- read the thesis source and final documentation;
- inspect the implementation of dataset poisoning, judge scoring, filtering, and analysis;
- regenerate compact tables and figures when the required artifacts are available;
- understand where large datasets and raw model outputs live.

The repository should not become a 3.2 GB archive of every intermediate output.

## Push In The First Commit

Track these files and directories:

```text
README.md
.gitignore
.env.example
pyproject.toml
pipeline.py
config/
src/
scripts/
docs/
report/*.tex
report/*.bib
report/README.md
report/Images/
report/emlyon_logo.*
report/human_rewrite_from_methodology.md
exports/*.csv
exports/*.md
exports/*.json
exports/figures/
data/raw/README.md
data/raw/hotpotqa_base.json
data/raw/hotpotqa_metadata.json
```

Optional, depending on repo size and supervisor preference:

```text
report/main.pdf
notebooks/
```

If `report/main.pdf` is included, treat it as the submitted manuscript artifact. Otherwise attach it to a GitHub Release named `thesis-final`.

## Do Not Push In Normal Git

Keep these out of the normal repository:

```text
PROJECT_CONTEXT.md
v2_runbook.txt
methodology.docx
methodology.txt
.env
.DS_Store
__pycache__/
.pytest_cache/
.ruff_cache/
report/*.aux
report/*.bbl
report/*.blg
report/*.fdb_latexmk
report/*.fls
report/*.lof
report/*.log
report/*.lot
report/*.out
report/*.toc
report/verify_methodology_figures.pdf
data/batch_inputs/
data/checkpoints/
data/poisoned/
data/splits/
data/v2_splits/
data/raw/hotpotqa_dev_distractor_v1.json
results/models/
results/v2/models/
results/prefilter_scores/
results/v2/prefilter_scores/
results/*backup*/
kaggle_phase34/
```

These files are either secrets, caches, LaTeX build outputs, reproducible generated outputs, raw provider batch payloads, raw datasets, or model checkpoints.

## Large Artifact Policy

The current project is about 3.2 GB:

- `results/`: about 2.6 GB;
- `data/`: about 511 MB;
- `report/`: about 69 MB;
- `exports/`: about 23 MB.

Use one of these strategies for large artifacts:

1. **Recommended for thesis submission:** keep git source-first and upload large artifacts to a release or external archive.
2. **Recommended for reproducible research release:** use Git LFS for selected final datasets and raw result JSON files.
3. **Not recommended:** commit all of `data/` and `results/` directly to git.

If using Git LFS, track only the final artifacts needed for reproducibility:

```bash
git lfs track "data/v2_fixed_poisonedrag/**/*.json"
git lfs track "data/v2_splits/*.json"
git lfs track "results/v2/raw_scores/*.json"
git lfs track "results/v2/prefilter_scores/*.json"
git lfs track "results/v2/models/**/*"
git add .gitattributes
```

Do this only if the remote supports LFS and the storage quota is acceptable.

## Pre-Push Checks

Run these before the first commit:

```bash
git status --short
find . -type f -size +25M -print
rg -n "OPENAI_API_KEY|GEMINI_API_KEY|DEEPSEEK_API_KEY|MISTRAL_API_KEY|api_key|secret|password" . --glob '!data/**' --glob '!results/**' --glob '!.env'
cd report && pdflatex -interaction=nonstopmode main.tex
```

Expected large files after `.gitignore` should be reviewed manually before adding them. `.env` must never appear in `git status --short` as a tracked file.

## First Push Sequence

From `/Users/adeshgaurav/Documents/Projects/Thesis`:

```bash
git init
git branch -M main
git status --short
git add README.md .gitignore .env.example pyproject.toml pipeline.py
git add config src scripts docs report exports data/raw/README.md data/raw/hotpotqa_base.json data/raw/hotpotqa_metadata.json
git status --short
git commit -m "Initial thesis code and report archive"
git remote add origin <REMOTE_URL>
git push -u origin main
```

If including the compiled thesis PDF:

```bash
git add -f report/main.pdf
git commit -m "Add compiled thesis manuscript"
git push
```

Prefer adding the PDF as a release artifact instead of versioning every future rebuild.

## Branch And Tag Strategy

Use a simple strategy:

- `main`: clean, reproducible thesis source.
- `docs/*`: documentation-only edits.
- `analysis/*`: future analysis changes.
- `report/*`: thesis text changes.

Useful tags:

```bash
git tag thesis-source-v1
git tag thesis-final-pdf
git push --tags
```

## Remote Naming

Recommended repository names:

- `thesis-llm-judge-poisoned-rag`
- `llm-judge-poisoned-rag-thesis`
- `meta-evaluation-llm-judges-thesis`

Use a private repository if API-adjacent prompts, generated data, or unpublished thesis material should not be public yet. Make it public only after checking institutional and supervisor requirements.
