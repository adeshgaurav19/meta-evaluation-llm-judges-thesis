# RAG Evaluation Logs

Place your RAG evaluation logs here as CSV or JSON.

## Expected Format

### CSV
Columns: `question`, `context`, `answer` (and optionally `id`, `ground_truth`)

```
question,context,answer
"What is X?","Passage 1 text\n\nPassage 2 text","X is ..."
```

### JSON
List of objects with the same fields:

```json
[
  {
    "id": "optional-custom-id",
    "question": "What is X?",
    "context": "Passage 1 text\n\nPassage 2 text",
    "answer": "X is ...",
    "ground_truth": "optional ground truth answer"
  }
]
```

## Notes
- Context passages should be separated by `\n\n` (configurable in `config/base.yaml`)
- Aim for ~100 triplets
- `id` is auto-generated (uuid) if not provided
- `ground_truth` is optional but useful for downstream evaluation
