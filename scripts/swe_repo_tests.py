#!/usr/bin/env python3
"""Run each SWE-bench repository's tests on recorded model patches, without Docker.

THIS EXECUTES MODEL-WRITTEN CODE. Run it only on a disposable Linux machine (a
Colab CPU runtime), as root, never on a workstation. Tests run as an unprivileged
user with a cleared environment, and the script refuses to start if that user can
list a protected path such as a mounted Drive. Network stays on by default, as in
the official harness, because the evaluation script reinstalls the package;
``--isolate-network`` turns it off for the test step where the kernel allows it.

The official SWE-bench harness runs every task in a Docker image. Colab has no
Docker daemon, so this script replays the harness's own recipes natively: the conda
environment script, the repository install script and the evaluation script that
``swebench`` (pinned to 3.0.17) generates for each task. It grades logs with the
harness's own parser. Because the environment is rebuilt outside the official
images, **every task is validated first with its reference ("gold") patch**; a task
whose gold patch does not resolve here is reported as invalid and no model patch is
run on it.

Only answers that already passed ``scripts/swe_patch_checks.py`` are run. Each row
of the output says whether the patch applied, whether the task's hidden tests pass
(``resolved``, to compare with the benchmark's grade) and whether the tests that
passed before still pass (``p2p_failed == 0``: the check a customer could run at
request time). Results are appended as they finish, so a rerun resumes.

``--workers N`` runs N tasks at once, each with its own conda environment
(``testbed1``, ...), checkout (``/testbed1``, ...) and home directories. The work is
unit tests on CPU cores: a GPU or TPU does not speed it up, more cores do.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

CONDA = "/opt/miniconda3"
GOLD = "__gold__"
FIELDS = ["task_id", "model", "instance_id", "patch_sha", "status", "applied", "resolved",
          "f2p_passed", "f2p_failed", "p2p_passed", "p2p_failed", "seconds"]
# status: ok | apply_failed | timeout | no_test_output | env_failed | install_failed | gold_unresolved
CLEAN_ENV = {"PATH": f"{CONDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
             "TZ": "Etc/UTC", "DEBIAN_FRONTEND": "noninteractive"}


class Shell:
    """Runs bash scripts as root or as the unprivileged test user."""

    def __init__(self, run_user: str, work_dir: Path, isolate_network: bool = False,
                 name: str = "") -> None:
        self.run_user, self.work_dir, self.isolate_network = run_user, Path(work_dir), isolate_network
        self.work_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.work_dir, 0o755)
        # Parallel workers get their own home directories: the harness scripts write
        # $HOME/requirements.txt and the global git configuration.
        self.root_home = f"/root/tine-swe-{name}" if name else "/root"
        self.user_home = f"/home/{run_user}/{name}" if name else f"/home/{run_user}"
        if name:
            os.makedirs(self.root_home, exist_ok=True)
            os.makedirs(self.user_home, exist_ok=True)
            shutil.chown(self.user_home, user=run_user)

    def script(self, text: str, *, as_user: bool = False, isolate: bool = False,
               timeout: float | None = None, log_path: Path | None = None) -> tuple[int | None, str]:
        """Returns (exit code, output). The exit code is ``None`` after a timeout."""
        handle, name = tempfile.mkstemp(suffix=".sh", dir=self.work_dir)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(name, 0o755)
        home = self.user_home if as_user else self.root_home
        command = ["env", "-i", f"HOME={home}", *[f"{k}={v}" for k, v in CLEAN_ENV.items()], "bash", name]
        if as_user:
            command = ["runuser", "-u", self.run_user, "--", *command]
            if isolate and self.isolate_network:
                command = ["unshare", "-n", *command]
        log_path = Path(log_path) if log_path else Path(name + ".log")
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = None
            return code, log_path.read_text(encoding="utf-8", errors="replace")
        finally:
            os.unlink(name)
            if log_path.name.endswith(".sh.log"):
                log_path.unlink(missing_ok=True)


def load_harness():
    """The pinned SWE-bench functions and markers this script relies on."""
    import dataclasses

    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, TESTS_TIMEOUT
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.create_scripts import (
        make_env_script_list, make_eval_script_list, make_repo_script_list)
    from swebench.harness.test_spec.test_spec import make_test_spec

    def make_spec(instance, env_name="testbed", repo_dir="/testbed"):
        """The harness's test spec, with its scripts regenerated for another
        environment name and checkout directory when a worker needs its own."""
        spec = make_test_spec(instance)
        if (env_name, repo_dir) == ("testbed", "/testbed"):
            return spec
        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
        return dataclasses.replace(
            spec,
            repo_script_list=make_repo_script_list(
                specs, instance["repo"], repo_dir, instance["base_commit"], env_name),
            env_script_list=make_env_script_list(instance, specs, env_name),
            eval_script_list=make_eval_script_list(
                instance, specs, env_name, repo_dir, instance["base_commit"], instance["test_patch"]))

    return make_spec, get_eval_report, TESTS_TIMEOUT


def patch_sha(patch: str) -> str:
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()[:16]


def plan(bench_dir: Path, static_checks: Path, instances: dict, *, repos: set[str],
         models: set[str], limit_instances: int | None = None) -> dict[str, list[dict]]:
    """instance_id -> answers to run: those that passed the static checks, for the
    chosen repositories and models. Instances keep the dataset's order."""
    passed = set()
    with Path(static_checks).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["passed"].strip().lower() == "true":
                passed.add((row["task_id"], row["model"]))
    jobs: dict[str, list[dict]] = {}
    for model_dir in sorted(path for path in Path(bench_dir).iterdir() if path.is_dir()):
        if models and model_dir.name not in models:
            continue
        runs = sorted(model_dir.glob("*.json"))
        if not runs:
            continue
        data = json.loads(runs[-1].read_text(encoding="utf-8"))
        for record in data["records"]:
            instance = instances.get(record.get("instance_id"))
            task_id = f"swe-bench:{data['split']}:{record['index']}"
            if instance is None or (repos and instance["repo"] not in repos):
                continue
            if (task_id, data["model_name"]) not in passed or not isinstance(record.get("prediction"), str):
                continue
            jobs.setdefault(record["instance_id"], []).append({
                "task_id": task_id, "model": data["model_name"], "patch": record["prediction"]})
    ordered = [iid for iid in instances if iid in jobs]
    return {iid: jobs[iid] for iid in ordered[:limit_instances]}


def read_done(output: Path) -> dict[tuple[str, str], dict]:
    if not Path(output).exists():
        return {}
    with Path(output).open(newline="", encoding="utf-8") as stream:
        return {(row["instance_id"], row["model"]): row for row in csv.DictReader(stream)}


WRITE_LOCK = threading.Lock()
ENV_LOCK = threading.Lock()  # conda's package cache is shared: build one environment at a time


def append_row(output: Path, row: dict) -> None:
    output = Path(output)
    with WRITE_LOCK, _open_for_append(output) as (stream, new):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
        stream.flush()
        os.fsync(stream.fileno())


class _open_for_append:
    def __init__(self, output: Path) -> None:
        self.output = output

    def __enter__(self):
        self.new = not self.output.exists()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output.open("a", newline="", encoding="utf-8")
        return self.stream, self.new

    def __exit__(self, *error) -> None:
        self.stream.close()


class Runner:
    def __init__(self, shell: Shell, harness, *, mirror_dir: Path | None, log_dir: Path,
                 timeout: int = 1800, env_name: str = "testbed", repo_dir: str = "/testbed") -> None:
        self.shell, self.mirror_dir, self.log_dir, self.timeout = shell, mirror_dir, Path(log_dir), timeout
        self.env_name, self.repo_dir = env_name, repo_dir
        self.make_test_spec, self.get_eval_report, self.timeout_marker = harness
        self.env_key = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def ensure_env(self, spec) -> bool:
        """Build the task's conda environment unless the current one already matches."""
        if spec.env_image_key == self.env_key:
            return True
        self.env_key = None
        env_path = f"{CONDA}/envs/{self.env_name}"
        with ENV_LOCK:
            self.shell.script(f"source {CONDA}/bin/activate; conda env remove -n {self.env_name} -y; "
                              f"rm -rf {env_path} {self.repo_dir}", timeout=900)
            code, output = self.shell.script(spec.setup_env_script, timeout=3600)
        if code != 0 or not Path(env_path).is_dir():
            (self.log_dir / f"env-{spec.instance_id}.log").write_text(output[-20000:])
            return False
        # The evaluation script reinstalls the package, so the test user owns the env.
        self.shell.script(f"chown -R {self.shell.run_user} {env_path}", timeout=900)
        self.env_key = spec.env_image_key
        return True

    def install_repo(self, spec) -> str | None:
        """Clone and install the repository at the base commit; returns the commit to
        reset to between patches (the install may add its own commit)."""
        script = spec.install_repo_script
        mirror = self.mirror_dir / (spec.repo.replace("/", "__") + ".git") if self.mirror_dir else None
        if mirror is not None and mirror.is_dir():
            script = script.replace(f"https://github.com/{spec.repo} ", f"{mirror} ")
        self.shell.script(f"rm -rf {self.repo_dir}", timeout=300)
        code, output = self.shell.script(script, timeout=3600)
        if code != 0:
            (self.log_dir / f"install-{spec.instance_id}.log").write_text(output[-20000:])
            return None
        self.shell.script(f"chown -R {self.shell.run_user} {self.repo_dir}", timeout=600)
        code, output = self.shell.script(f"cd {self.repo_dir} && git rev-parse HEAD", as_user=True, timeout=60)
        return output.strip().splitlines()[-1] if code == 0 and output.strip() else None

    def run_patch(self, spec, head: str, patch: str, log_path: Path) -> dict:
        started = time.time()
        row = {"patch_sha": patch_sha(patch), "applied": False, "resolved": False}
        patch_file = self.shell.work_dir / "answer.patch"
        patch_file.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
        os.chmod(patch_file, 0o644)
        # Same order as the harness, with no reset between attempts.
        code, _ = self.shell.script(
            f"cd {self.repo_dir} && git reset -q --hard {head} && git clean -fdq && "
            f"(git apply --verbose {patch_file} || git apply --verbose --reject {patch_file} || "
            f"patch --batch --fuzz=5 -p1 -i {patch_file})", as_user=True, timeout=300)
        if code != 0:
            return {**row, "status": "apply_failed", "seconds": round(time.time() - started, 1)}
        row["applied"] = True
        code, output = self.shell.script(spec.eval_script, as_user=True, isolate=True,
                                         timeout=self.timeout, log_path=log_path)
        if code is None:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n{self.timeout_marker}\n")
            return {**row, "status": "timeout", "seconds": round(time.time() - started, 1)}
        report = self.get_eval_report(
            spec, {"instance_id": spec.instance_id, "model_patch": patch, "model_name_or_path": "tine"},
            str(log_path), True)[spec.instance_id]
        status = report.get("tests_status")
        if not status:
            return {**row, "status": "no_test_output", "seconds": round(time.time() - started, 1)}
        return {**row, "status": "ok", "resolved": bool(report["resolved"]),
                "f2p_passed": len(status["FAIL_TO_PASS"]["success"]),
                "f2p_failed": len(status["FAIL_TO_PASS"]["failure"]),
                "p2p_passed": len(status["PASS_TO_PASS"]["success"]),
                "p2p_failed": len(status["PASS_TO_PASS"]["failure"]),
                "seconds": round(time.time() - started, 1)}

    def run_instance(self, instance: dict, answers: list[dict], output: Path, done: dict,
                     gold_only: bool = False) -> None:
        iid = instance["instance_id"]
        spec = self.make_test_spec(instance, self.env_name, self.repo_dir)
        gold = done.get((iid, GOLD))
        pending = [] if gold_only else [a for a in answers if (iid, a["model"]) not in done]
        if gold is not None and (not pending or gold["status"] != "ok" or gold["resolved"] != "True"):
            return  # finished, or already known to be invalid here

        def record(model, task_id, fields):
            row = {"task_id": task_id, "model": model, "instance_id": iid, **fields}
            append_row(output, row)
            done[(iid, model)] = {key: str(value) for key, value in row.items()}

        task_id = answers[0]["task_id"] if answers else ""
        if not self.ensure_env(spec):
            return record(GOLD, task_id, {"status": "env_failed"})
        head = self.install_repo(spec)
        if head is None:
            return record(GOLD, task_id, {"status": "install_failed"})
        if gold is None:
            result = self.run_patch(spec, head, instance["patch"], self.log_dir / f"{iid}.gold.log")
            if result["status"] == "ok" and not result["resolved"]:
                result["status"] = "gold_unresolved"
            record(GOLD, task_id, result)
            if result["status"] != "ok":
                return
            (self.log_dir / f"{iid}.gold.log").unlink(missing_ok=True)
        cache: dict[str, dict] = {}
        for answer in pending:
            sha = patch_sha(answer["patch"])
            if sha not in cache:  # several models often give the same patch
                log_path = self.log_dir / f"{iid}.{sha}.log"
                cache[sha] = self.run_patch(spec, head, answer["patch"], log_path)
                log_path.unlink(missing_ok=True)
            record(answer["model"], answer["task_id"], cache[sha])


def partition(ordered: list[str], jobs: dict, instances: dict, workers: int) -> list[list[str]]:
    """Split tasks between workers, keeping each repository version with one worker so
    its environment is built once. Costs are rough: sympy's test files take about ten
    times longer than django's. Largest groups are placed first on the idlest worker."""
    groups: dict[tuple, list[str]] = {}
    for iid in ordered:
        groups.setdefault((instances[iid]["repo"], str(instances[iid]["version"])), []).append(iid)

    def cost(key):
        per_run = 50 if key[0] == "sympy/sympy" else 5
        return sum(30 + (1 + len(jobs[iid])) * per_run for iid in groups[key])

    shares = [[] for _ in range(max(1, workers))]
    loads = [0.0] * len(shares)
    for key in sorted(groups, key=cost, reverse=True):
        target = loads.index(min(loads))
        shares[target].extend(groups[key])
        loads[target] += cost(key)
    return [share for share in shares if share]


def preflight(run_user: str, protect: list[str]) -> dict:
    """Facts about the machine, and refusal conditions, before any code is run."""
    def ok(command):
        return subprocess.run(command, capture_output=True).returncode == 0

    facts = {
        "linux": sys.platform.startswith("linux"), "root": hasattr(os, "geteuid") and os.geteuid() == 0,
        "cpus": os.cpu_count(), "free_disk_gb": round(shutil.disk_usage("/").free / 1e9, 1),
        "conda": Path(f"{CONDA}/bin/conda").exists(),
        "run_user": ok(["id", run_user]),
        "network_isolation": ok(["unshare", "-n", "true"]),
        "gnu_patch": "GNU" in subprocess.run(["patch", "--version"], capture_output=True, text=True).stdout,
    }
    facts["protected_paths_hidden_from_run_user"] = facts["run_user"] and all(
        not ok(["runuser", "-u", run_user, "--", "ls", path]) for path in protect if Path(path).exists())
    facts["ready"] = all(facts[key] for key in (
        "linux", "root", "conda", "run_user", "gnu_patch", "protected_paths_hidden_from_run_user"))
    return facts


def write_checks(results: Path, static_checks: Path, checks_output: Path) -> dict:
    """A ``tine cascade --checks`` file: an answer passes if it passed the static
    checks, applied, and left every previously passing test passing. Answers that
    were not run (other repositories or models, invalid tasks) stay unknown."""
    done = read_done(results)
    valid = {iid for (iid, model), row in done.items()
             if model == GOLD and row["status"] == "ok" and row["resolved"] == "True"}
    counts = {"passed": 0, "failed": 0, "unknown": 0}
    checks_output = Path(checks_output)
    checks_output.parent.mkdir(parents=True, exist_ok=True)
    with Path(static_checks).open(newline="", encoding="utf-8") as source, \
            checks_output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["task_id", "model", "passed", "reason"])
        for row in csv.DictReader(source):
            run = done.get((row["instance_id"], row["model"]))
            if row["passed"].strip().lower() != "true":
                verdict, reason = "False", "static:" + row["reason"]
            elif row["instance_id"] not in valid or run is None:
                verdict, reason = "", "not_run"
            elif run["status"] == "ok" and run["applied"] == "True" and run["p2p_failed"] == "0":
                verdict, reason = "True", "existing_tests_pass"
            elif run["status"] in ("ok", "apply_failed", "timeout", "no_test_output"):
                verdict, reason = "False", "tests:" + run["status"]
            else:
                verdict, reason = "", "not_run"
            counts["passed" if verdict == "True" else "failed" if verdict == "False" else "unknown"] += 1
            writer.writerow([row["task_id"], row["model"], verdict, reason])
    return counts


def summarize(results: Path, static_checks: Path, bench_dir: Path) -> dict:
    """Environment validity, agreement with the benchmark's grades, and how many
    wrong answers each check catches, over the answers on validated tasks."""
    done = read_done(results)
    gold = {iid: row for (iid, model), row in done.items() if model == GOLD}
    valid = {iid for iid, row in gold.items() if row["status"] == "ok" and row["resolved"] == "True"}
    grades = {}
    for model_dir in sorted(path for path in Path(bench_dir).iterdir() if path.is_dir()):
        runs = sorted(model_dir.glob("*.json"))
        if runs:
            data = json.loads(runs[-1].read_text(encoding="utf-8"))
            for record in data["records"]:
                if record.get("score") is not None:
                    grades[(record["instance_id"], data["model_name"])] = float(record["score"]) == 1.0
    models = sorted({model for (iid, model) in done if model != GOLD})
    per_model, agree, compared = {}, 0, 0
    with Path(static_checks).open(newline="", encoding="utf-8") as stream:
        static = [row for row in csv.DictReader(stream)
                  if row["instance_id"] in valid and row["model"] in models]
    for model in models:
        counts = {"answers": 0, "right": 0, "wrong": 0, "wrong_caught_static": 0,
                  "wrong_caught_static_or_tests": 0, "right_rejected": 0, "accepted": 0, "accepted_right": 0}
        for row in static:
            key = (row["instance_id"], row["model"])
            if row["model"] != model or key not in grades:
                continue
            static_pass = row["passed"].strip().lower() == "true"
            run = done.get(key)
            if static_pass and run is None:
                continue  # not run yet
            tests_pass = static_pass and run["status"] == "ok" and run["p2p_failed"] == "0"
            right = grades[key]
            counts["answers"] += 1
            counts["right" if right else "wrong"] += 1
            counts["wrong_caught_static"] += (not right) and not static_pass
            counts["wrong_caught_static_or_tests"] += (not right) and not tests_pass
            counts["right_rejected"] += right and not tests_pass
            counts["accepted"] += tests_pass
            counts["accepted_right"] += tests_pass and right
            if static_pass and run["status"] == "ok":
                compared += 1
                agree += (run["resolved"] == "True") == right
        per_model[model] = counts
    statuses: dict[str, int] = {}
    for row in gold.values():
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    return {"tasks_attempted": len(gold), "tasks_valid": len(valid), "gold_statuses": statuses,
            "runs_compared_with_benchmark_grade": compared,
            "agreement_with_benchmark_grade": round(agree / compared, 4) if compared else None,
            "per_model": per_model}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench-dir", type=Path, help="extracted swe-bench split folder")
    parser.add_argument("--static-checks", type=Path, help="CSV from scripts/swe_patch_checks.py")
    parser.add_argument("--dataset", type=Path, help="SWE-bench Verified rows, one JSON object per line")
    parser.add_argument("--output", type=Path, help="results CSV, appended to and resumed from")
    parser.add_argument("--mirror-dir", type=Path, help="bare clones named <owner>__<name>.git")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/tine-swe-logs"))
    parser.add_argument("--repos", nargs="*", default=["django/django", "sympy/sympy"])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--run-user", default="tineswe")
    parser.add_argument("--protect", nargs="*", default=["/content/drive"],
                        help="paths the test user must not be able to list")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per test run")
    parser.add_argument("--isolate-network", action="store_true",
                        help="run the test step without network if the kernel allows it")
    parser.add_argument("--max-hours", type=float, help="stop starting new tasks after this long")
    parser.add_argument("--workers", type=int, default=1,
                        help="tasks to run at once, each in its own environment (use the CPU count)")
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--gold-only", action="store_true", help="only validate environments")
    parser.add_argument("--preflight", action="store_true", help="print machine checks and exit")
    parser.add_argument("--checks-output", type=Path,
                        help="write a tine cascade --checks file from --output and exit")
    parser.add_argument("--summary", action="store_true",
                        help="print validity, agreement and catch counts from --output and exit")
    args = parser.parse_args(argv)
    if args.preflight:
        facts = preflight(args.run_user, args.protect)
        print(json.dumps(facts, indent=2))
        raise SystemExit(0 if facts["ready"] else 1)
    if args.summary:
        print(json.dumps(summarize(args.output, args.static_checks, args.bench_dir), indent=2))
        return
    if args.checks_output:
        print(json.dumps(write_checks(args.output, args.static_checks, args.checks_output)))
        return
    facts = preflight(args.run_user, args.protect)
    if not facts["ready"]:
        raise SystemExit(f"Refusing to run model-written code on this machine: {json.dumps(facts)}")
    instances = {}
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            instances[row["instance_id"]] = row
    jobs = plan(args.bench_dir, args.static_checks, instances, repos=set(args.repos),
                models=set(args.models), limit_instances=args.limit_instances)
    isolate = args.isolate_network and facts["network_isolation"]
    setup = Shell(args.run_user, Path("/tmp/tine-swe-work"), isolate)
    setup.script("git config --system --add safe.directory '*'", timeout=60)
    # Some evaluation scripts generate this locale themselves, which needs root.
    setup.script("sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen", timeout=600)
    harness = load_harness()
    done = read_done(args.output)
    # Group by environment so each conda environment is built once.
    ordered = sorted(jobs, key=lambda iid: (instances[iid]["repo"], str(instances[iid]["version"])))
    shares = partition(ordered, jobs, instances, args.workers) if args.workers > 1 else [ordered]
    started = time.time()
    progress = {"count": 0}

    def work(index: int, share: list[str]) -> None:
        name = f"w{index}" if len(shares) > 1 else ""
        suffix = str(index) if len(shares) > 1 else ""
        shell = Shell(args.run_user, Path(f"/tmp/tine-swe-work{suffix}"), isolate, name=name)
        runner = Runner(shell, harness, mirror_dir=args.mirror_dir, log_dir=args.log_dir,
                        timeout=args.timeout, env_name=f"testbed{suffix}", repo_dir=f"/testbed{suffix}")
        for iid in share:
            if args.max_hours and time.time() - started > args.max_hours * 3600:
                print(f"Worker {index} stopped after {args.max_hours} hours; rerun to resume.", flush=True)
                break
            try:
                runner.run_instance(instances[iid], jobs[iid], args.output, done, gold_only=args.gold_only)
            except Exception as error:  # one task must not end the whole run
                print(f"{iid}: runner error {type(error).__name__}: {error}", flush=True)
            gold = done.get((iid, GOLD), {})
            progress["count"] += 1
            print(f"[{progress['count']}/{len(ordered)}] {iid}: gold {gold.get('status')} "
                  f"resolved={gold.get('resolved')} ({(time.time() - started) / 60:.0f} min)", flush=True)

    threads = [threading.Thread(target=work, args=(index + 1, share)) for index, share in enumerate(shares)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
