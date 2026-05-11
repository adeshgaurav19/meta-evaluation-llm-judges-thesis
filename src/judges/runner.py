import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.judges.batch_gemini import run_gemini_batch
from src.judges.batch_openai import run_openai_batch
from src.judges.prompts import JUSTIFICATION_PROMPT, SCORING_PROMPT


def _load_checkpoint(checkpoint_path: Path, triplets: list, label: str) -> tuple[dict, list]:
    results_map: dict[str, dict] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().strip().splitlines():
            if line.strip():
                r = json.loads(line)
                results_map[r["id"]] = r
        already_done = {tid for tid, r in results_map.items() if not r.get("error")}
        triplets_to_run = [t for t in triplets if t.id not in already_done]
        print(f"[{label}] Resuming  {len(already_done)} already done, {len(triplets_to_run)} remaining")
    else:
        triplets_to_run = triplets
    return results_map, triplets_to_run


def run_scoring_phase(
    triplets: list,
    phase_name: str,
    config: dict,
    only_judge: str | None = None,  # "gpt" | "gemini" | None (both)
) -> dict[str, list[dict]]:
    """
    Run judges via batch API.
    only_judge: "gpt" or "gemini" to run a single judge; None runs both in parallel.
    Returns: {"gpt": [...], "gemini": [...]}
    """
    max_out = config["judges"]["scoring"]["max_output_tokens"]
    openai_cfg = {**config["judges"]["scoring"]["models"][0], "max_output_tokens": max_out}
    gemini_cfg = {**config["judges"]["scoring"]["models"][1], "max_output_tokens": max_out}

    out_dir = Path(config["experiment"]["results_dir"]) / "raw_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_gpt = only_judge in (None, "gpt")
    run_gemini = only_judge in (None, "gemini")

    openai_results: list[dict] = []
    gemini_results: list[dict] = []

    if not run_gpt:
        p = out_dir / f"{phase_name}_gpt.json"
        openai_results = json.loads(p.read_text()) if p.exists() else []
        print(f"[runner] Skipping GPT  {len(openai_results)} existing results loaded")

    if not run_gemini:
        p = out_dir / f"{phase_name}_gemini.json"
        gemini_results = json.loads(p.read_text()) if p.exists() else []
        print(f"[runner] Skipping Gemini  {len(gemini_results)} existing results loaded")

    workers = sum([run_gpt, run_gemini])
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        future_gpt = (
            executor.submit(run_openai_batch, triplets, SCORING_PROMPT, phase_name, openai_cfg)
            if run_gpt
            else None
        )
        future_gem = (
            executor.submit(run_gemini_batch, triplets, SCORING_PROMPT, phase_name, gemini_cfg)
            if run_gemini
            else None
        )
        if future_gpt:
            openai_results = future_gpt.result()
        if future_gem:
            gemini_results = future_gem.result()

    (out_dir / f"{phase_name}_gpt.json").write_text(json.dumps(openai_results, indent=2))
    (out_dir / f"{phase_name}_gemini.json").write_text(json.dumps(gemini_results, indent=2))

    return {"gpt": openai_results, "gemini": gemini_results}


def _score_gemma_one(
    t, gemini_client, model_id: str, prompt_template: str, max_tokens: int, max_retries: int = 5
) -> dict:
    import json as _json

    ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
    prompt = prompt_template.format(question=t.question, context=ctx, answer=t.answer)
    t0 = time.perf_counter()
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            scores = _json.loads(resp.text)
            return {
                "id": t.id,
                "scores": scores,
                "error": None,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
            )
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return {"id": t.id, "scores": None, "error": err_str, "latency_ms": 0}
    return {"id": t.id, "scores": None, "error": "max_retries_exceeded", "latency_ms": 0}


def run_scoring_realtime(
    triplets: list,
    phase_name: str,
    config: dict,
    only_judge: str | None = None,
    gemini_concurrency: int = 3,
) -> dict[str, list[dict]]:
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from google import genai
    from openai import OpenAI

    max_tokens = config["judges"]["scoring"]["max_output_tokens"]
    run_gpt = only_judge in (None, "gpt")
    run_gemini = only_judge in (None, "gemini")

    openai_client = OpenAI() if run_gpt else None
    gemini_client = None
    gemini_model_id = None
    if run_gemini:
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        gemini_model_id = config["judges"]["scoring"]["models"][1]["model_id"]
        print(f"[Gemma] Scoring {len(triplets)} triplets with concurrency={gemini_concurrency} ...")

    gpt_model = config["judges"]["scoring"]["models"][0]["model_id"]
    gpt_results = []
    gemini_results_map: dict[str, dict] = {}

    for t in triplets:
        if not run_gpt:
            break
        ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
        prompt = SCORING_PROMPT.format(question=t.question, context=ctx, answer=t.answer)
        t0 = time.perf_counter()
        try:
            resp = openai_client.chat.completions.create(
                model=gpt_model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            scores = json.loads(resp.choices[0].message.content)
            gpt_results.append(
                {
                    "id": t.id,
                    "scores": scores,
                    "error": None,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                }
            )
        except Exception as e:
            gpt_results.append({"id": t.id, "scores": None, "error": str(e), "latency_ms": 0})

    if run_gemini:
        from tqdm import tqdm

        out_dir_tmp = Path(config["experiment"]["results_dir"]) / "raw_scores"
        out_dir_tmp.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir_tmp / f"{phase_name}_gemini_checkpoint.jsonl"
        gemini_results_map, triplets_to_run = _load_checkpoint(checkpoint_path, triplets, "Gemma")

        ckpt_file = open(checkpoint_path, "a")
        try:
            with ThreadPoolExecutor(max_workers=gemini_concurrency) as executor:
                futures = {
                    executor.submit(
                        _score_gemma_one,
                        t,
                        gemini_client,
                        gemini_model_id,
                        SCORING_PROMPT,
                        max_tokens,
                    ): t
                    for t in triplets_to_run
                }
                with tqdm(total=len(triplets_to_run), desc="[Gemma]", unit="req") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        gemini_results_map[result["id"]] = result
                        ckpt_file.write(json.dumps(result) + "\n")
                        ckpt_file.flush()
                        n_err = sum(1 for r in gemini_results_map.values() if r.get("error"))
                        pbar.set_postfix(errors=n_err)
                        pbar.update(1)
        finally:
            ckpt_file.close()

        gemini_results = list(gemini_results_map.values())
        n_err = sum(1 for r in gemini_results if r.get("error"))
        print(f"[Gemma] Complete  {len(gemini_results)} scored, {n_err} errors")
    else:
        gemini_results = []

    out_dir = Path(config["experiment"]["results_dir"]) / "raw_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{phase_name}_gpt.json").write_text(json.dumps(gpt_results, indent=2))
    (out_dir / f"{phase_name}_gemini.json").write_text(json.dumps(gemini_results, indent=2))

    return {"gpt": gpt_results, "gemini": gemini_results}


def _score_deepseek_one(
    t, client, model_id: str, prompt_template: str, max_tokens: int, max_retries: int = 5
) -> dict:
    ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
    prompt = prompt_template.format(question=t.question, context=ctx, answer=t.answer)
    t0 = time.perf_counter()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            scores = json.loads(resp.choices[0].message.content)
            return {
                "id": t.id,
                "scores": scores,
                "error": None,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return {"id": t.id, "scores": None, "error": err_str, "latency_ms": 0}
    return {"id": t.id, "scores": None, "error": "max_retries_exceeded", "latency_ms": 0}


def run_scoring_deepseek(
    triplets: list,
    phase_name: str,
    config: dict,
    concurrency: int = 10,
) -> list[dict]:
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import OpenAI
    from tqdm import tqdm

    model_cfg = next(
        m for m in config["judges"]["scoring"]["models"] if m["provider"] == "deepseek"
    )
    model_id = model_cfg["model_id"]
    max_tokens = config["judges"]["scoring"]["max_output_tokens"]

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    out_dir = Path(config["experiment"]["results_dir"]) / "raw_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{phase_name}_deepseek_checkpoint.jsonl"

    results_map, triplets_to_run = _load_checkpoint(checkpoint_path, triplets, "DeepSeek")
    if not checkpoint_path.exists():
        print(f"[DeepSeek] Scoring {len(triplets_to_run)} triplets with concurrency={concurrency} ...")

    ckpt_file = open(checkpoint_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _score_deepseek_one, t, client, model_id, SCORING_PROMPT, max_tokens
                ): t
                for t in triplets_to_run
            }
            with tqdm(total=len(triplets_to_run), desc="[DeepSeek]", unit="req") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results_map[result["id"]] = result
                    ckpt_file.write(json.dumps(result) + "\n")
                    ckpt_file.flush()
                    n_err = sum(1 for r in results_map.values() if r.get("error"))
                    pbar.set_postfix(errors=n_err)
                    pbar.update(1)
    finally:
        ckpt_file.close()

    results = list(results_map.values())
    n_err = sum(1 for r in results if r.get("error"))
    print(f"[DeepSeek] Complete  {len(results)} scored, {n_err} errors")

    (out_dir / f"{phase_name}_deepseek.json").write_text(json.dumps(results, indent=2))
    return results


def run_justification_phase(selected_triplets: list, config: dict) -> list[dict]:
    """
    Run GPT with full justification on 30 selected samples.
    Uses real-time API (not batch) since it's only 30 calls.
    """
    from openai import OpenAI

    client = OpenAI()

    model = config["judges"]["justification"]["model"]
    max_tokens = config["judges"]["justification"]["max_output_tokens"]
    results = []

    for t in selected_triplets:
        ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": JUSTIFICATION_PROMPT.format(
                            question=t.question,
                            context=ctx,
                            answer=t.answer,
                        ),
                    }
                ],
            )
            content = response.choices[0].message.content
        except Exception as e:
            content = json.dumps({"error": str(e)})

        results.append(
            {
                "id": t.id,
                "content": content,
                "is_poisoned": getattr(t, "is_poisoned", False),
                "injection_type": getattr(t, "injection_type", None),
                "noise_level": getattr(t, "noise_level", None),
            }
        )

    return results
