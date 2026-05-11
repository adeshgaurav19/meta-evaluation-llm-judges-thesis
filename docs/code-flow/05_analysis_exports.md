# Analysis Tables and Figures

Goal: regenerate the final thesis tables and figures from saved v2 outputs.

Main commands:

```bash
python scripts/analyze.py
python scripts/plot.py
```

Main scripts:

| Script | Reads | Writes |
|---|---|---|
| `scripts/analyze.py` | `results/v2/`, dataset labels, saved score files | `exports/*.csv`, `exports/summary.md` |
| `scripts/plot.py` | `exports/*.csv`, selected saved results | `exports/figures/*.png`, `exports/figures/captions.md` |

Main output folder:

- `exports/`

Important exported tables:

| File | Use |
|---|---|
| `table_baseline_summary.csv` | Baseline FPR and mean scores. |
| `table_score_distribution.csv` | Judge calibration regimes. |
| `table_paradox_overview.csv` | Clean/True/Survived score means. |
| `table_filter_audit_summary.csv` | What the statistical filter removed. |
| `table_mistral_metrics.csv` | Mistral triplet-level filter metrics. |
| `table_mcnemar_bootstrap.csv` | Question-clustered significance checks. |
| `table_length_correlation.csv` | Context length correlations. |
| `table_justification_summary.csv` | Small poison-aware prompt analysis. |

Important figure folder:

- `exports/figures/`

Notes:

- This step does not rerun judges.
- It uses the saved v2 outputs as the source of truth.
- The README lists the main exported tables; this file explains where they come from in the pipeline.
