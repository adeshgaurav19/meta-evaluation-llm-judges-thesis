"""
OpenAI Batch API helper for judge scoring.

JSONL format: one JSON object per line with:
  - custom_id: triplet ID (for matching results back)
  - method: "POST"
  - url: "/v1/chat/completions"
  - body: standard chat completion params
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def write_openai_batch_file(
    triplets: list,
    prompt_template: str,
    output_path: str,
    model: str = "gpt-4o",
    max_tokens: int = 50,
) -> Path:
    """Write JSONL batch input file for OpenAI."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in triplets:
            ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
            request = {
                "custom_id": t.id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "max_completion_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_template.format(
                                question=t.question,
                                context=ctx,
                                answer=t.answer,
                            ),
                        }
                    ],
                },
            }
            f.write(json.dumps(request) + "\n")
    print(f"Wrote {len(triplets)} requests to {path}")
    return path


def submit_openai_batch(input_path: str, description: str = "") -> str:
    """Upload file and create batch job. Returns batch job ID."""
    from openai import OpenAI

    client = OpenAI()

    batch_file = client.files.create(
        file=open(input_path, "rb"),
        purpose="batch",
    )
    print(f"Uploaded file: {batch_file.id}")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"Created batch: {batch_job.id}")
    return batch_job.id


def poll_openai_batch(batch_id: str, poll_interval: int = 60) -> object:
    """Poll until batch completes. Returns batch status object."""
    from openai import OpenAI

    client = OpenAI()

    while True:
        status = client.batches.retrieve(batch_id)
        counts = status.request_counts
        print(
            f"[OpenAI] {status.status} - "
            f"{counts.completed}/{counts.total} done, "
            f"{counts.failed} failed"
        )

        if status.status == "completed":
            return status
        if status.status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Batch {status.status}: {status.errors}")

        time.sleep(poll_interval)


def _parse_json_content(content: str) -> dict:
    """Extract JSON from model response, handling markdown code blocks."""
    content = content.strip()
    # Strip ```json ... ``` or ``` ... ``` wrappers
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    # Find first { ... } block if there's surrounding text
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]
    return json.loads(content)


def download_openai_results(batch_status, output_path: str) -> list[dict]:
    """Download results JSONL and parse into list of dicts."""
    from openai import OpenAI

    client = OpenAI()

    if batch_status.output_file_id is None:
        # All requests failed; try to surface the error file for debugging.
        if batch_status.error_file_id:
            try:
                err_content = client.files.content(batch_status.error_file_id)
                first_line = err_content.text.strip().splitlines()[0]
                sample_err = json.loads(first_line)
                print(f"[OpenAI] Batch error sample: {sample_err}")
            except Exception:
                pass
        raise RuntimeError(
            f"Batch completed with 0 successes; all {batch_status.request_counts.failed} requests failed. "
            "Check response_format compatibility with this model."
        )

    result_content = client.files.content(batch_status.output_file_id)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(result_content.text)

    results = []
    for line in result_content.text.strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        triplet_id = r["custom_id"]
        response = r["response"]

        if response["status_code"] == 200:
            content = response["body"]["choices"][0]["message"]["content"]
            try:
                scores = _parse_json_content(content)
                results.append({"id": triplet_id, "scores": scores, "error": None})
            except (json.JSONDecodeError, ValueError):
                results.append(
                    {"id": triplet_id, "scores": None, "error": "parse_fail", "raw": content[:200]}
                )
        else:
            results.append({"id": triplet_id, "scores": None, "error": response["status_code"]})

    n_err = sum(1 for r in results if r["error"])
    print(f"Parsed {len(results)} results, {n_err} errors")
    return results


def _estimate_tokens(triplets: list, prompt_template: str, max_tokens: int) -> int:
    """Rough token estimate: 1 token is about 4 chars, plus max_tokens output."""
    sample = triplets[0]
    ctx = sample.poisoned_context if hasattr(sample, "poisoned_context") else sample.context
    prompt_len = len(
        prompt_template.format(question=sample.question, context=ctx, answer=sample.answer)
    )
    avg_input_tokens = prompt_len // 4
    return (avg_input_tokens + max_tokens) * len(triplets)


def _chunk_triplets(
    triplets: list, prompt_template: str, max_tokens: int, token_limit: int = 900_000
) -> list[list]:
    """Split triplets into chunks that each stay under token_limit."""
    sample = triplets[0]
    ctx = sample.poisoned_context if hasattr(sample, "poisoned_context") else sample.context
    prompt_len = len(
        prompt_template.format(question=sample.question, context=ctx, answer=sample.answer)
    )
    tokens_per_request = prompt_len // 4 + max_tokens
    chunk_size = max(1, token_limit // tokens_per_request)

    chunks = []
    for i in range(0, len(triplets), chunk_size):
        chunks.append(triplets[i : i + chunk_size])
    return chunks


def run_openai_batch(
    triplets: list, prompt_template: str, phase_name: str, config: dict
) -> list[dict]:
    """Write, submit, poll, download, and parse one or more batch jobs."""
    from openai import OpenAI

    client = OpenAI()

    model = config.get("model_id", config.get("name", "gpt-4o"))
    max_tokens = config.get("max_output_tokens", 50)

    estimated = _estimate_tokens(triplets, prompt_template, max_tokens)
    chunks = (
        _chunk_triplets(triplets, prompt_template, max_tokens)
        if estimated > 900_000
        else [triplets]
    )

    if len(chunks) > 1:
        print(
            f"[OpenAI] Splitting {len(triplets)} requests into {len(chunks)} batches "
            f"(estimated {estimated:,} tokens > 1.4M limit)"
        )

    all_results: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_phase = f"{phase_name}_chunk{i}" if len(chunks) > 1 else phase_name
        input_path = f"data/batch_inputs/openai_{chunk_phase}.jsonl"
        output_path = f"results/raw_scores/{chunk_phase}_gpt_raw.jsonl"

        write_openai_batch_file(chunk, prompt_template, input_path, model, max_tokens)
        batch_id = submit_openai_batch(input_path, description=chunk_phase)
        status = poll_openai_batch(batch_id)
        results = download_openai_results(status, output_path)
        all_results.extend(results)

        try:
            client.files.delete(status.input_file_id)
        except Exception:
            pass

    # Write merged results
    merged_path = f"results/raw_scores/{phase_name}_gpt_raw.jsonl"
    if len(chunks) > 1:
        Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

    return all_results
