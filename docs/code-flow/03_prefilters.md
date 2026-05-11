# Phase 3: Pre-Filters

Goal: run two upstream filters before judge scoring.

The two filters are:

1. Statistical passage-level filter.
2. Mistral triplet-level LLM filter.

Main commands:

```bash
python scripts/03_train_prefilter.py --config config/v2.yaml
python scripts/04_run_prefilter.py --config config/v2.yaml
python scripts/04b_run_llm_prefilter.py --config config/v2.yaml
```

Main code:

| Path | Role |
|---|---|
| `src/prefilter/embedding_signal.py` | Embedding cosine signal. |
| `src/prefilter/entropy_signal.py` | Token-level entropy signal. |
| `src/prefilter/classifier_signal.py` | DeBERTa classifier signal. |
| `src/prefilter/crossencoder_signal.py` | Cross-encoder relevance signal. |
| `src/prefilter/answer_span_signal.py` | Answer-span recall signal. |
| `src/prefilter/aggregator.py` | Weighted vote, majority vote, and XGBoost aggregation. |
| `src/prefilter/pipeline.py` | Statistical filter pipeline. |
| `src/prefilter/llm_prefilter.py` | Mistral filter logic. |
| `src/prefilter/train_classifier.py` | Classifier training. |

Inputs:

- train/validation/test splits
- poisoned dataset files
- `config/v2.yaml`
- model/API credentials where required

Outputs:

- trained classifier/checkpoints under `results/v2/models/`
- statistical pre-filter scores under `results/v2/prefilter_scores/`
- Mistral filter outputs under `results/v2/`

Kaggle/GPU note:

- Phase 3 and parts of Phase 4 used Kaggle because DeBERTa and cross-encoder
  steps benefit from GPU.
- The notebook/code used for this path is in:
  - `notebooks/phase3_4_kaggle.ipynb`
  - `kaggle_phase34/`

