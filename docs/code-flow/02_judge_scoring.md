# Phase 2: Baseline Judge Scoring

Goal: score clean and poisoned triplets before any filter is applied.

Main command:

```bash
python scripts/02_run_judges.py --config config/v2.yaml
```

Main code:

| Path | Role |
|---|---|
| `src/judges/runner.py` | Runs judge scoring over triplets. |
| `src/judges/base.py` | Shared judge client logic. |
| `src/judges/batch_openai.py` | OpenAI batch/client helpers. |
| `src/judges/batch_gemini.py` | Google/Gemini batch/client helpers. |
| `src/judges/prompts.py` | Scoring and justification prompts. |
| `src/utils/context_truncator.py` | Applies context length limits before scoring. |

Inputs:

- dataset/split files from Phase 1
- judge model configuration in `config/v2.yaml`
- provider API keys in environment variables

Outputs:

- baseline judge scores under `results/v2/raw_scores/`
- run logs and saved scoring artefacts under `results/v2/`

Notes:

- The three judges are GPT-5.4-nano, Gemma-4-26B-A4B-IT, and DeepSeek-V3.2.
- Judges score faithfulness, answer relevance, and context relevance.
- Saved v2 baseline scores are treated as final experiment artefacts.

