# Phase 1: Dataset Construction

Goal: build the Poisoned Evaluation Dataset from HotpotQA.

Main commands:

```bash
python scripts/00_download_hotpotqa.py --config config/v2.yaml
python scripts/01_prepare_dataset.py --config config/v2.yaml
```

Main code:

| Path | Role |
|---|---|
| `src/dataset/loader.py` | Loads HotpotQA source data. |
| `src/dataset/schema.py` | Defines triplet/data structures. |
| `src/dataset/splitter.py` | Builds train/validation/test splits. |
| `src/dataset/poisoner.py` | Creates poisoned triplets. |
| `src/dataset/poisoner.py` | Attack construction and poisoned-context creation. |

Inputs:

- `config/v2.yaml`
- HotpotQA raw files under `data/raw/`

Outputs:

- fixed v2 poisoned dataset under `data/v2_fixed_poisonedrag/`
- split files under `data/v2_splits/`

Notes:

- The final dataset uses 100 base HotpotQA questions.
- Each base question is expanded across three injection types and four noise
  levels.
- Clean controls are paired with poisoned triplets.
- PoisonedRAG-style triplets are additive: malicious passages are inserted
  without overwriting the original supporting passages.
