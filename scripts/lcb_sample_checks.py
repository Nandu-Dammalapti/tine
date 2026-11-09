#!/usr/bin/env python3
"""Run recorded LiveCodeBench answers against the problem's *public* sample tests.

Benchmark grades come from hidden tests. A deployed router only has the visible
examples, so this measures how often those examples catch a wrong answer (and how
often they reject a right one). No model is called: the code is the answer each
model already produced in LLMRouterBench.

The code is untrusted. Each (model, task) runs in its own temporary directory in a
separate `python -I` process with no network (`unshare -n`), CPU, memory and file
size limits, and a wall-clock timeout. Linux only.

Output CSV: task_id, model, mode, n_samples, passed, reason.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SAMPLE_SECONDS = 4
MEMORY_BYTES = 4 * 1024 ** 3

PREAMBLE = """\
import sys, math, re, string, heapq, bisect, itertools, functools, collections, operator
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
from math import *
sys.setrecursionlimit(10 ** 6)
"""

HARNESS = r'''
import io, json, math, signal, sys

spec = json.load(open("spec.json"))
PREAMBLE, CODE = spec["preamble"], spec["code"]

class Timeout(Exception):
    pass

def alarm(*_):
    raise Timeout()

signal.signal(signal.SIGALRM, alarm)

def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

def tokens_match(got, want):
    g, w = got.split(), want.split()
    if len(g) != len(w):
        return False
    for x, y in zip(g, w):
        if x.lower() == y.lower():
            continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except ValueError:
            return False
    return True

results = []
for sample in spec["samples"]:
    ok = False
    signal.alarm(spec["seconds"])
    try:
        if spec["mode"] == "stdin":
            sys.stdin = io.StringIO(sample["input"])
            out = io.StringIO()
            real = sys.stdout
            sys.stdout = out
            try:
                exec(compile(PREAMBLE + CODE, "answer", "exec"), {"__name__": "__main__"})
            except SystemExit:
                pass
            finally:
                sys.stdout = real
            ok = tokens_match(out.getvalue(), sample["output"])
        else:
            namespace = {"__name__": "answer"}
            exec(compile(PREAMBLE + CODE, "answer", "exec"), namespace)
            method = getattr(namespace["Solution"](), spec["method"])
            ok = same(method(**eval(sample["kwargs"])), eval(sample["expected"]))
    except BaseException:
        ok = False
    finally:
        signal.alarm(0)
    results.append(ok)
print(json.dumps(results))
'''


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def stdin_samples(query: str):
    """Sample Input k / Sample Output k blocks. Blank lines inside inputs are a
    formatting artifact and are dropped; an output ends at its first blank line
    (explanations follow)."""
    query = _clean(query)
    samples = []
    for match in re.finditer(r"Sample Input (\d+):?\n(.*?)Sample Output \1:?\n", query, re.S):
        start = match.end()
        following = re.search(r"Sample Input \d+:?\n", query[start:])
        block = query[start:start + following.start()] if following else query[start:]
        lines = block.lstrip("\n").split("\n")
        output = []
        for line in lines:
            if not line.strip():
                break
            output.append(line)
        inp = "\n".join(line for line in match.group(2).split("\n") if line.strip()) + "\n"
        if output:
            samples.append({"input": inp, "output": "\n".join(output)})
    return samples


_LITERALS = {"true": "True", "false": "False", "null": "None"}


def _python_literal(text: str) -> str:
    text = re.sub(r"\b(true|false|null)\b", lambda m: _LITERALS[m.group(1)], text.strip())
    ast.literal_eval(text)  # validate: only plain literals
    return text


def functional_samples(query: str, prompt: str):
    """LeetCode-style 'Input: a = ..., b = ...' / 'Output: ...' examples."""
    method = re.search(r"def (\w+)\(self", prompt)
    if not method:
        return None, []
    samples = []
    for match in re.finditer(r"Input:\s*(.+?)\nOutput:\s*(.+?)\n", _clean(query) + "\n", re.S):
        try:
            call = ast.parse(f"f({match.group(1).strip()})", mode="eval").body
            if call.args or not call.keywords:
                continue
            kwargs = {kw.arg: _python_literal(ast.unparse(kw.value)) for kw in call.keywords}
            expected = _python_literal(match.group(2))
        except (SyntaxError, ValueError):
            continue
        samples.append({"kwargs": "{" + ", ".join(f"{k!r}: {v}" for k, v in kwargs.items()) + "}",
                        "expected": expected})
    return method.group(1), samples


def check_answer(code: str, mode: str, samples, method: str | None = None) -> list[bool] | None:
    """Run one answer against its samples in an isolated process. None = harness failure."""
    spec = {"preamble": PREAMBLE, "code": code, "mode": mode, "samples": samples,
            "method": method, "seconds": SAMPLE_SECONDS}
    with tempfile.TemporaryDirectory(prefix="tine-check-") as work:
        Path(work, "spec.json").write_text(json.dumps(spec))
        Path(work, "harness.py").write_text(HARNESS)
        command = ["unshare", "-n", "prlimit", f"--as={MEMORY_BYTES}",
                   f"--cpu={SAMPLE_SECONDS * len(samples) + 5}", "--fsize=10000000",
                   sys.executable, "-I", "harness.py"]
        try:
            done = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                  timeout=SAMPLE_SECONDS * len(samples) + 10)
        except subprocess.TimeoutExpired:
            return [False] * len(samples)
        lines = done.stdout.strip().splitlines()
        try:
            return json.loads(lines[-1]) if lines else [False] * len(samples)
        except json.JSONDecodeError:
            return [False] * len(samples)


def _latest(model_dir: Path) -> Path | None:
    runs = sorted(model_dir.glob("*.json"))
    return runs[-1] if runs else None


def collect(bench_dir: Path, models):
    jobs = []
    for model_dir in sorted(p for p in (bench_dir / "test").iterdir() if p.is_dir()):
        if models and model_dir.name not in models:
            continue
        path = _latest(model_dir)
        if path is None:
            continue
        for record in json.loads(path.read_text())["records"]:
            query, prompt = record.get("origin_query") or "", record.get("prompt") or ""
            code = record.get("prediction") or ""
            if "Sample Input 1" in query:
                mode, method, samples = "stdin", None, stdin_samples(query)
            else:
                mode = "functional"
                method, samples = functional_samples(query, prompt)
            jobs.append({"task_id": f"livecodebench:test:{record['index']}", "model": model_dir.name,
                         "mode": mode, "method": method, "samples": samples, "code": code})
    return jobs


def run(job):
    if not job["samples"]:
        return job, None, "no_parsed_samples"
    if not job["code"].strip():
        return job, [False] * len(job["samples"]), "empty_answer"
    results = check_answer(job["code"], job["mode"], job["samples"], job["method"])
    return job, results, "ran"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench-dir", type=Path, required=True,
                        help="the extracted livecodebench folder (contains test/<model>/...)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="check only the first N jobs (debugging)")
    args = parser.parse_args(argv)
    jobs = collect(args.bench_dir, set(args.models or []))[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle, futures.ThreadPoolExecutor(args.workers) as pool:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "model", "mode", "n_samples", "passed", "reason"])
        for count, (job, results, reason) in enumerate(pool.map(run, jobs), 1):
            passed = "" if results is None else all(results)
            writer.writerow([job["task_id"], job["model"], job["mode"], len(job["samples"]), passed, reason])
            if count % 1000 == 0:
                print(f"{count}/{len(jobs)} checked", flush=True)
    print(f"Wrote {len(jobs)} rows to {args.output}")


if __name__ == "__main__":
    main()
