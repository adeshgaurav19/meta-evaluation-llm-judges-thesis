# Phase 4: End-to-End Evaluation

Goal: measure how judge behaviour changes after the filters modify the context.

Main commands:

```bash
python scripts/05_evaluate_impact.py --config config/v2.yaml
python scripts/06_run_justifications.py --config config/v2.yaml
python scripts/06b_run_poison_aware_judge.py --config config/v2.yaml
```

Main code:

| Path | Role |
|---|---|
| `src/metrics/judge_metrics.py` | Judge-level metric helpers. |
| `src/metrics/detection_metrics.py` | Detection/filter metrics. |
| `src/metrics/system_metrics.py` | End-to-end system metrics. |
| `src/analysis/justification.py` | Justification analysis helpers. |
| `src/judges/prompts.py` | Standard and poison-aware prompts. |

Inputs:

- raw baseline judge scores
- statistical pre-filter outputs
- Mistral filter outputs
- filtered contexts/triplets

Outputs:

- post-filter judge scores
- justification sample outputs
- poison-aware judge outputs
- intermediate result files under `results/v2/`

Important analysis split:

- `True`: original supporting passages were overwritten by the attack.
- `Survived`: original supporting passages remained intact.
- `Clean`: matched clean controls.

The True/Survived split is computed from dataset labels. It is used for analysis,
not as something available in a real deployment.

