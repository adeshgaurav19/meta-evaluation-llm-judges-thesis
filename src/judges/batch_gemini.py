"""
Gemini Batch API helper for judge scoring.

JSONL format: one JSON object per line with:
  - key: triplet ID (for matching results back)
  - request: standard GenerateContent request body
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def write_gemini_batch_file(
    triplets: list,
    prompt_template: str,
    output_path: str,
    max_tokens: int = 50,
) -> Path:
    """Write JSONL batch input file for Gemini."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in triplets:
            ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
            request = {
                "key": t.id,
                "request": {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt_template.format(
                                        question=t.question,
                                        context=ctx,
                                        answer=t.answer,
                                    )
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0,
                        "responseMimeType": "application/json",
                    },
                },
            }
            f.write(json.dumps(request) + "\n")
    print(f"Wrote {len(triplets)} requests to {path}")
    return path


def submit_gemini_batch(input_path: str, model: str, description: str = "") -> str:
    """Upload file and create batch job. Returns batch job name."""
    from google import genai

    client = genai.Client()
    uploaded_file = client.files.upload(file=input_path, config={"mime_type": "application/jsonl"})
    print(f"Uploaded file: {uploaded_file.name}")

    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": description},
    )
    print(f"Created batch: {batch_job.name}")
    return batch_job.name


def poll_gemini_batch(batch_name: str, poll_interval: int = 60) -> object:
    """Poll until batch completes. Returns batch job object."""
    from google import genai

    client = genai.Client()

    while True:
        job = client.batches.get(name=batch_name)
        print(f"[Gemini] {job.state.name}")

        if job.state.name == "JOB_STATE_SUCCEEDED":
            return job
        if job.state.name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(f"Batch {job.state.name}")

        time.sleep(poll_interval)


def download_gemini_results(batch_job, output_path: str) -> list[dict]:
    """Download results and parse into list of dicts."""
    from google import genai

    client = genai.Client()
    result_bytes = client.files.download(file=batch_job.dest.file_name)
    result_text = result_bytes.decode("utf-8")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(result_text)

    results = []
    for line in result_text.strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        triplet_id = r["key"]

        try:
            text = r["response"]["candidates"][0]["content"]["parts"][0]["text"]
            scores = json.loads(text)
            results.append({"id": triplet_id, "scores": scores, "error": None})
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            results.append({"id": triplet_id, "scores": None, "error": str(e)})

    n_err = sum(1 for r in results if r["error"])
    print(f"Parsed {len(results)} results, {n_err} errors")
    return results


def _chunk_triplets(
    triplets: list, prompt_template: str, max_tokens: int, token_limit: int = 1_400_000
) -> list[list]:
    """Split triplets into chunks that each stay under token_limit."""
    sample = triplets[0]
    ctx = sample.poisoned_context if hasattr(sample, "poisoned_context") else sample.context
    prompt_len = len(
        prompt_template.format(question=sample.question, context=ctx, answer=sample.answer)
    )
    tokens_per_request = prompt_len // 4 + max_tokens
    chunk_size = max(1, token_limit // tokens_per_request)
    return [triplets[i : i + chunk_size] for i in range(0, len(triplets), chunk_size)]


def run_gemini_batch(
    triplets: list, prompt_template: str, phase_name: str, config: dict
) -> list[dict]:
    """Write, submit, poll, download, and parse one or more batch jobs."""
    model = config.get("model_id", config.get("name", "gemini-2.5-pro"))
    max_tokens = config.get("max_output_tokens", 50)

    chunks = _chunk_triplets(triplets, prompt_template, max_tokens)
    if len(chunks) > 1:
        print(f"[Gemini] Splitting {len(triplets)} requests into {len(chunks)} batches")

    all_results: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_phase = f"{phase_name}_chunk{i}" if len(chunks) > 1 else phase_name
        input_path = f"data/batch_inputs/gemini_{chunk_phase}.jsonl"
        output_path = f"results/raw_scores/{chunk_phase}_gemini_raw.jsonl"

        write_gemini_batch_file(chunk, prompt_template, input_path, max_tokens)
        batch_name = submit_gemini_batch(input_path, model, description=chunk_phase)
        job = poll_gemini_batch(batch_name)
        results = download_gemini_results(job, output_path)
        all_results.extend(results)

    return all_results
