"""Generate open-model answers with vLLM. Runs on a Colab GPU, not on CPU.

Used by colab/02_generate_open_model_answers.ipynb to fill gaps that public
benchmarks don't cover. Output rows follow tine.schema with `success` left
pending (NaN): answers still have to be graded (tests for code) before training.

Not exercised by the CPU test suite (vLLM needs a GPU); run the notebook's
50-prompt debug cell first.
"""

import time

import numpy as np
import pandas as pd

from tine.schema import RESULT_COLUMNS


def load_vllm(model_name: str, tensor_parallel_size=1):
    """Load a vLLM engine once (prefix caching on) and reuse it for every chunk."""
    from vllm import LLM  # imported lazily: GPU-only dependency

    return LLM(model=model_name, enable_prefix_caching=True,
               tensor_parallel_size=tensor_parallel_size)


def generate_with_vllm(llm, tasks: pd.DataFrame, model_name: str, *, max_tokens=1024,
                       temperature=0.0, gpu_usd_per_hour=None, use_chat_template=True):
    """Answer every prompt in `tasks` (columns tenant, task_id, prompt, task_type) in one
    batched call to `llm` (from `load_vllm`); vLLM batches continuously inside it.

    Returns (results, answers): result rows (success pending) and a frame of
    task_id, model, answer for grading. Cost is the GPU time for the batch split by
    token share, when `gpu_usd_per_hour` is given; otherwise NaN.
    """
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    prompts = tasks["prompt"].tolist()
    start = time.perf_counter()
    if use_chat_template:
        outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    else:
        outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    input_tokens = np.array([len(o.prompt_token_ids) for o in outputs], dtype=float)
    output_tokens = np.array([len(o.outputs[0].token_ids) for o in outputs], dtype=float)
    cached = np.array([getattr(o, "num_cached_tokens", None) or np.nan for o in outputs])
    share = (input_tokens + output_tokens) / max(1.0, (input_tokens + output_tokens).sum())
    cost = (gpu_usd_per_hour * elapsed / 3600 * share if gpu_usd_per_hour is not None
            else np.full(len(outputs), np.nan))
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    results = pd.DataFrame({
        "tenant": tasks["tenant"].to_numpy(),
        "task_id": tasks["task_id"].to_numpy(),
        "prompt": prompts,
        "task_type": tasks["task_type"].to_numpy(),
        "model": model_name,
        "success": np.nan,  # pending until graded
        "cost_usd": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached,
        "latency_ms": 1000 * elapsed / len(outputs),  # batch time per request, amortized
        "timestamp": stamp,
    }, columns=list(RESULT_COLUMNS))
    answers = pd.DataFrame({"task_id": tasks["task_id"].to_numpy(), "model": model_name,
                            "answer": [o.outputs[0].text for o in outputs]})
    return results, answers
