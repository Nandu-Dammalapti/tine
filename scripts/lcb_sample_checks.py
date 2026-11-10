#!/usr/bin/env python3
"""Run recorded LiveCodeBench answers against the problem's *public* sample tests.

Benchmark grades come from hidden tests. A deployed router only has the visible
examples, so this measures how often those examples catch a wrong answer (and how
often they reject a right one). No model is called: the code is the answer each
model already produced in LLMRouterBench.

The code is untrusted. Each (model, task) runs in its own temporary directory in a
separate `python -I` process as the unprivileged `nobody` user (`setpriv`), with an
empty environment, no network (`unshare -n`), CPU, memory and file size limits, and a
wall-clock timeout. Linux only; must start as root to switch user.

Output CSV: task_id, model, mode, n_samples, passed, reason.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

SAMPLE_SECONDS = 2  # samples are tiny; correct answers finish far sooner
MEMORY_BYTES = 4 * 1024 ** 3
WORK_ROOT = Path("/tmp/tine-checks")
SYSTEM_PYTHON = shutil.which("python3", path="/usr/bin:/bin") or "/usr/bin/python3"
CLEAN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp"}

PREAMBLE = """\
import sys, math, re, string, heapq, bisect, itertools, functools, collections, operator
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
from math import *
from builtins import *  # undo shadowing, e.g. math.pow over the 3-argument pow
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
    if not ok:
        break  # one failed sample already rejects the answer
print(json.dumps(results))
'''


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def stdin_samples(query: str):
    """Sample Input k / Sample Output k blocks. Blank lines inside inputs are a
    formatting artifact and are dropped. One blank line follows each output header;
    the output then runs to the next blank line (explanations follow), so a second
    blank line straight away means the expected output is empty."""
    query = _clean(query)
    samples = []
    for match in re.finditer(r"Sample Input (\d+):?\n(.*?)Sample Output \1:?\n", query, re.S):
        start = match.end()
        following = re.search(r"Sample Input \d+:?\n", query[start:])
        block = query[start:start + following.start()] if following else query[start:]
        if block.startswith("\n"):
            block = block[1:]
        output = []
        for line in block.split("\n"):
            if not line.strip():
                break
            output.append(line)
        inp = "\n".join(line for line in match.group(2).split("\n") if line.strip()) + "\n"
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


def tokens_match(got: str, want: str) -> bool:
    """Whitespace-token comparison; case-insensitive; numbers within 1e-6."""
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


def _sandbox(script: str, seconds: int) -> list[str]:
    return ["unshare", "-n", "prlimit", f"--as={MEMORY_BYTES}", f"--cpu={seconds}",
            "--fsize=10000000", "setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
            SYSTEM_PYTHON, "-I", script]


def _workdir(files: dict[str, str]):
    WORK_ROOT.mkdir(mode=0o711, exist_ok=True)
    work = tempfile.TemporaryDirectory(prefix="tine-check-", dir=WORK_ROOT)
    for name, text in files.items():
        Path(work.name, name).write_text(text)
        Path(work.name, name).chmod(0o644)
    os.chmod(work.name, 0o777)  # the unprivileged user may write scratch files here
    return work


def check_answer(code: str, mode: str, samples, method: str | None = None) -> list[bool]:
    """Run one answer against its samples in isolated processes.

    stdin problems run the answer as a real program, one process per sample, with
    the sample on standard input. Function problems call the method in a harness.
    """
    if mode == "stdin":
        results = []
        with _workdir({"answer.py": PREAMBLE + code}) as work:
            for sample in samples:
                try:
                    done = subprocess.run(_sandbox("answer.py", SAMPLE_SECONDS + 1), cwd=work,
                                          input=sample["input"], capture_output=True, text=True,
                                          env=CLEAN_ENV, timeout=SAMPLE_SECONDS + 3)
                    results.append(done.returncode == 0 and tokens_match(done.stdout, sample["output"]))
                except (subprocess.TimeoutExpired, UnicodeDecodeError):
                    results.append(False)
                if not results[-1]:
                    break  # one failed sample already rejects the answer
        return results
    return _check_functional(code, samples, method)


def _check_functional(code: str, samples, method: str | None) -> list[bool]:
    spec = {"preamble": PREAMBLE, "code": code, "mode": "functional", "samples": samples,
            "method": method, "seconds": SAMPLE_SECONDS}
    with _workdir({"spec.json": json.dumps(spec), "harness.py": HARNESS}) as work:
        command = _sandbox("harness.py", SAMPLE_SECONDS * len(samples) + 5)
        try:
            done = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                  env=CLEAN_ENV, timeout=SAMPLE_SECONDS * len(samples) + 10)
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
