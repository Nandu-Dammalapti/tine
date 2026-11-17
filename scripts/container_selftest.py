#!/usr/bin/env python3
"""Build the Tine image and test a running container of it, with one command.

    python3 scripts/container_selftest.py

It needs Docker running and nothing else: no provider key, no ``deploy/tine.env``.
NO MODEL IS CALLED AND NOTHING IS PAID FOR. The proxy under test is given a stand-in
provider (``scripts/container_standin.py``) in a second container, and a tenant token
this script makes up. It uses ``docker-compose.yml`` with
``docker-compose.selftest.yml`` on top, under the project name ``tine-selftest``, so
the image and the hardening settings are the real ones while the env file, the config
folder, the host port and the ledger volume are its own.

What it checks, one line each, PASS or FAIL:

- the image builds and both containers reach "healthy";
- ``GET /health`` answers, and a request without a token is refused with 401;
- a request and a streamed request go through to the stand-in and back, are written
  to the ledger and have a cost; ``GET /v1/savings`` shows them;
- ``scripts/savings_report.py``, fed to the container's Python, makes the savings
  page from the ledger there;
- the process runs as uid 10001, cannot write outside ``/data`` and ``/tmp``, and
  the image holds only the package;
- after the mounted policy file is edited and the container restarted, ``/health``
  shows the policy and a request is served through it;
- ``docker compose stop`` ends the container in time although a stream is still
  open, and the ledger is whole when the container is started again.

At the end it removes its containers, network and ledger volume (``--keep`` leaves
them). It never touches the operator's own containers, ``deploy/tine.env`` or
``deploy/config``. The image stays, tagged ``tine-proxy:dev``, the one
``docker compose build`` makes.

Exit status: 0 when every check passed, 1 when one failed, 2 when Docker could not
be used at all. Standard library only, so the Mac's own ``python3`` runs it.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import container_smoke as smoke  # noqa: E402
import container_standin as standin  # noqa: E402

PROJECT = "tine-selftest"
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.selftest.yml")
TENANT, FRONTIER, CHEAP = "selftest", "selftest-frontier", "selftest-cheap"
PASS, FAIL = smoke.PASS, smoke.FAIL
NAME_WIDTH = len("read-only except /data, /tmp")
# Where Docker Desktop puts its command-line tools on a Mac when they are not on PATH.
DOCKER_FOLDERS = (Path.home() / ".docker" / "bin", Path("/Applications/Docker.app/Contents/Resources/bin"),
                  Path("/usr/local/bin"), Path("/opt/homebrew/bin"))
# docker-compose.yml gives the proxy 30 seconds to stop before Docker kills it. The
# proxy cuts what is still open after 20 (tine serve --shutdown-seconds).
STOP_LIMIT_SECONDS = 28.0
REPORT_PATH = "/tmp/selftest-report.html"


class Unusable(Exception):
    """Docker cannot be used here. The message says what to do."""


def find_docker(path=None) -> str:
    """The docker command: on PATH, else where Docker Desktop keeps it."""
    found = shutil.which("docker", path=path)
    if found:
        return found
    for folder in DOCKER_FOLDERS:
        if os.access(folder / "docker", os.X_OK):
            return str(folder / "docker")
    raise Unusable("the docker command was not found on PATH or in ~/.docker/bin. Install Docker Desktop, start "
                   "it, and open a new terminal")


def policy(stages: list) -> dict:
    """A route policy for the test tenant. Made up: nothing in it was measured."""
    return {TENANT: {"anthropic": {
        "version": 1, "tenant": TENANT, "task_type": "selftest", "frontier": FRONTIER, "stages": stages,
        "guarantee": {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03,
                      "statement": "SELF-TEST ONLY. Written by scripts/container_selftest.py. Nothing was measured."}}}}


def env_text(token: str) -> str:
    prices = {f"anthropic:{model}": {"input_per_million": rate, "output_per_million": rate * 5}
              for model, rate in ((FRONTIER, 5.0), (CHEAP, 1.0))}
    return "\n".join([
        "# Written by scripts/container_selftest.py. The token is made up for one test run and the",
        "# provider key is not a key: the provider is the stand-in container.",
        "TINE_TENANT_KEYS=" + json.dumps({token: TENANT}, separators=(",", ":")),
        "TINE_ANTHROPIC_API_KEY=selftest-not-a-provider-key",
        f"TINE_ANTHROPIC_FRONTIER_MODEL={FRONTIER}",
        "TINE_ANTHROPIC_BASE_URL=http://standin:9000",
        "TINE_LEDGER_PATH=/data/ledger.sqlite3",
        "TINE_MODEL_PRICES=" + json.dumps(prices, separators=(",", ":")),
        "TINE_ROUTE_POLICIES=/config/route-policies.json",
        "TINE_TIMEOUT_SECONDS=10",
        # Docker Desktop can hand containers the host's proxy settings. The stand-in is
        # reached by its service name and must not be sent to that proxy.
        "NO_PROXY=standin,localhost,127.0.0.1",
        "no_proxy=standin,localhost,127.0.0.1",
    ]) + "\n"


def write_files(work: Path, token: str) -> None:
    """The env file and config folder the test container is given. Starts from nothing."""
    if work.exists():
        shutil.rmtree(work)
    (work / "config").mkdir(parents=True)
    os.chmod(work / "config", 0o755)
    descriptor = os.open(work / "selftest.env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(env_text(token))
    write_policy(work, [])


def write_policy(work: Path, stages: list) -> None:
    path = work / "config" / "route-policies.json"
    path.write_text(json.dumps(policy(stages), indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)  # read inside the container by uid 10001


class Docker:
    """Runs docker commands for the self-test project and nothing else."""

    def __init__(self, binary: str, work: Path, port: int, out) -> None:
        self.binary, self.out = binary, out
        folder = str(Path(binary).parent)
        # The folder also holds docker-credential-desktop, which a build needs on PATH
        # to pull the base image.
        self.env = {**os.environ, "PATH": folder + os.pathsep + os.environ.get("PATH", ""),
                    "TINE_SELFTEST_DIR": str(work), "TINE_SELFTEST_PORT": str(port)}

    def run(self, *arguments: str, timeout: float = 120, show: bool = False,
            stdin: Path | None = None) -> subprocess.CompletedProcess:
        """One docker command. ``show`` lets its output through to the terminal;
        ``stdin`` is a file to feed it."""
        source = open(stdin, "rb") if stdin is not None else subprocess.DEVNULL
        try:
            return subprocess.run([self.binary, *arguments], cwd=ROOT, env=self.env, timeout=timeout,
                                  stdin=source, stdout=None if show else subprocess.PIPE,
                                  stderr=None if show else subprocess.STDOUT, text=True)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(arguments, 124, f"no result after {timeout:g} s")
        except OSError as error:
            raise Unusable(f"{self.binary} could not be run: {type(error).__name__}") from None
        finally:
            if stdin is not None:
                source.close()

    def compose(self, *arguments: str, **options) -> subprocess.CompletedProcess:
        files = [part for name in COMPOSE_FILES for part in ("-f", name)]
        return self.run("compose", "-p", PROJECT, *files, *arguments, **options)

    def state(self, service: str) -> dict:
        """The service's container: status, exit code and health, or {} when there is none."""
        found = self.compose("ps", "-a", "-q", service)
        identifier = (found.stdout or "").strip().splitlines()[-1:] if found.returncode == 0 else []
        if not identifier:
            return {}
        shown = self.run("inspect", "--format", "{{json .State}}", identifier[0])
        try:
            state = json.loads((shown.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {}
        return {"status": state.get("Status"), "exit_code": state.get("ExitCode"),
                "health": (state.get("Health") or {}).get("Status")}


def tail(text, lines: int = 6) -> str:
    """The end of a command's output, on one line."""
    kept = [line.strip() for line in (text or "").splitlines() if line.strip()][-lines:]
    return " | ".join(kept)[:600] or "no output"


class HeldStream(threading.Thread):
    """A streamed request the stand-in never finishes, read until the proxy cuts it."""

    def __init__(self, base_url: str, token: str) -> None:
        super().__init__(daemon=True)
        self.base, self.token = urlsplit(base_url), token
        self.opened, self.status, self.ended = threading.Event(), None, None

    def run(self) -> None:
        connection = http.client.HTTPConnection(self.base.hostname, self.base.port, timeout=90)
        try:
            body = json.dumps({"model": "selftest", "max_tokens": 16, "stream": True, "messages": [
                {"role": "user", "content": f"{standin.HOLD_MARKER}: keep this stream open"}]})
            connection.request("POST", "/v1/messages", body=body, headers={
                "authorization": f"Bearer {self.token}", "content-type": "application/json",
                "anthropic-version": "2023-06-01"})
            response = connection.getresponse()
            self.status = response.status
            response.read(1)
            self.opened.set()
            while response.read(256):
                pass
        except (OSError, http.client.HTTPException):
            pass
        finally:
            self.opened.set()
            self.ended = time.monotonic()
            connection.close()


def message(proxy, text: str, stream: bool = False):
    body = {"model": "selftest", "max_tokens": 16, "messages": [{"role": "user", "content": text}]}
    if stream:
        body["stream"] = True
    return proxy.call("POST", "/v1/messages", body=body)


def served(reply) -> str:
    headers = reply.headers
    return (f"status {reply.status}, served by {headers.get('x-tine-model', '-')}, route "
            f"{headers.get('x-tine-route', '-')}, ledger {headers.get('x-tine-ledger', '-')}")


def savings(proxy) -> dict:
    reply = proxy.call("GET", "/v1/savings")
    report = reply.json()
    return report if reply.status == 200 and isinstance(report, dict) else {}


def checks(docker: Docker, proxy, work: Path, add, wait: float) -> None:
    """Every check on the running container, in order. ``add(name, passed, detail)``."""
    nonce = secrets.token_hex(4)

    state = docker.state("tine")
    add("container healthy", state.get("status") == "running" and state.get("health") == "healthy",
        f"status {state.get('status')}, health {state.get('health')}")

    health = smoke.wait_for_health(proxy, wait)
    described = health.json() if isinstance(health.json(), dict) else {}
    add("health", health.status == 200 and described.get("status") == "ok" and described.get("routing") == "frontier-default",
        f"status {health.status}, routing {described.get('routing')}, shadow {described.get('shadow')}")

    absent = proxy.call("GET", "/v1/savings", token=None)
    wrong = proxy.call("GET", "/v1/savings", token="wrong-" + secrets.token_urlsafe(16))
    add("token needed", absent.status == wrong.status == 401,
        f"{absent.status} with no token, {wrong.status} with a wrong one"
        + ("" if absent.status == wrong.status == 401 else "; expected 401 for both"))

    before = savings(proxy)
    reply = message(proxy, f"Self-test {nonce}-1. Reply with the one word: ok")
    payload = reply.json() if isinstance(reply.json(), dict) else {}
    add("request", reply.status == 200 and payload.get("id") == standin.REPLY_ID
        and reply.headers.get("x-tine-model") == FRONTIER and reply.headers.get("x-tine-ledger") == "recorded",
        served(reply) + ("" if reply.status == 200 else f": {reply.error()}"))

    streamed = message(proxy, f"Self-test {nonce}-2. Reply with the one word: ok", stream=True)
    add("streamed request", streamed.status == 200 and b"message_stop" in streamed.body
        and streamed.headers.get("x-tine-ledger") == "recorded", served(streamed))

    # A streamed request's cost is written when its stream ends, a moment after the reply.
    deadline = time.monotonic() + smoke.SETTLE_SECONDS
    while True:
        after = savings(proxy)
        grew = smoke.count(after, "requests") - smoke.count(before, "requests")
        unknown = (smoke.count(after, "unknown_answer_cost_requests")
                   - smoke.count(before, "unknown_answer_cost_requests"))
        if (grew >= 2 and unknown <= 0) or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    add("savings", grew == 2 and unknown == 0 and after.get("tenant") == TENANT,
        f"tenant {after.get('tenant')!r}, {grew} new request(s), answer cost unknown for {unknown}, known answer "
        f"cost {smoke.usd(after.get('known_answer_cost_total_usd'))}")

    # The image has no scripts/ folder, so the report script is fed to the container's
    # Python on standard input. It opens the ledger read-only while the proxy runs.
    made = docker.compose("exec", "-T", "tine", "python", "-", "--ledger", "/data/ledger.sqlite3", "--tenant", TENANT,
                          "--output", REPORT_PATH, stdin=ROOT / "scripts" / "savings_report.py")
    page = docker.compose("exec", "-T", "tine", "cat", REPORT_PATH) if made.returncode == 0 else made
    html = page.stdout or ""
    add("savings report page", made.returncode == 0 and page.returncode == 0 and "<html" in html[:400].lower()
        and "</html>" in html[-400:].lower() and TENANT in html,
        f"a page of {len(html):,} characters was made inside the container" if made.returncode == 0
        else f"the report script ended with status {made.returncode}: {tail(made.stdout, 2)}")

    user = docker.compose("exec", "-T", "tine", "id", "-u")
    add("runs as uid 10001", user.returncode == 0 and (user.stdout or "").strip() == "10001",
        f"id -u printed {tail(user.stdout, 1)!r}")

    probe = ("import os\n"
             "def writable(folder):\n"
             "    try:\n"
             "        path = os.path.join(folder, 'selftest-write-probe')\n"
             "        open(path, 'w').close(); os.remove(path); return True\n"
             "    except OSError:\n"
             "        return False\n"
             "print(' '.join(f'{folder}={writable(folder)}' for folder in "
             "('/app', '/config', '/var/tmp', '/data', '/tmp')))\n"
             "print('app:' + ','.join(sorted(os.listdir('/app'))))\n")
    wrote = docker.compose("exec", "-T", "tine", "python", "-c", probe)
    lines = (wrote.stdout or "").strip().splitlines()
    # /var/tmp is open to every user in the base image, so only a read-only root
    # filesystem stops uid 10001 writing there. /app and /config are root's as well.
    expected = "/app=False /config=False /var/tmp=False /data=True /tmp=True"
    add("read-only except /data, /tmp", wrote.returncode == 0 and lines[:1] == [expected],
        f"writable: {lines[0] if lines else tail(wrote.stdout)}")
    listing = lines[1][4:].split(",") if len(lines) > 1 and lines[1].startswith("app:") else []
    add("image holds only the package", sorted(listing) == ["pyproject.toml", "requirements-dev.lock", "requirements.txt", "tine"],
        f"/app holds {', '.join(listing) or 'nothing readable'}")

    write_policy(work, [{"models": [CHEAP], "check": "none"}])
    restarted = docker.compose("restart", "tine", timeout=90)
    routed, described = None, {}
    if restarted.returncode == 0:
        try:
            described = smoke.wait_for_health(proxy, wait).json() or {}
            routed = message(proxy, f"Self-test {nonce}-3. Reply with the one word: ok")
        except smoke.Unreachable:
            pass
    add("policy edit, then restart", routed is not None and described.get("routing") == "policy"
        and routed.headers.get("x-tine-route") == "stage-0" and routed.headers.get("x-tine-model") == CHEAP,
        (f"/health routing {described.get('routing')}; " + served(routed)) if routed is not None
        else f"the container did not come back: {tail(restarted.stdout)}")
    kept = smoke.count(savings(proxy), "requests")
    add("ledger kept over a restart", kept >= smoke.count(after, "requests") + 1,
        f"{kept} request(s) in the ledger after the restart, {smoke.count(after, 'requests')} before it")

    held = HeldStream(proxy.base, proxy.token)
    held.start()
    held.opened.wait(15)
    started = time.monotonic()
    stopped = docker.compose("stop", "tine", timeout=90)
    took = time.monotonic() - started
    state = docker.state("tine")
    add("stop with a stream open", held.status == 200 and stopped.returncode == 0 and took < STOP_LIMIT_SECONDS
        and state.get("status") == "exited" and state.get("exit_code") != 137,
        f"stream status {held.status}; stopped in {took:.1f} s (Docker kills at 30), exit code {state.get('exit_code')}"
        + (" (137 means Docker had to kill it)" if state.get("exit_code") == 137 else ""))

    again = docker.compose("start", "tine", timeout=90)
    count_after = None
    if again.returncode == 0:
        try:
            smoke.wait_for_health(proxy, wait)
            count_after = smoke.count(savings(proxy), "requests")
        except smoke.Unreachable:
            pass
    add("ledger whole after the stop", count_after is not None and count_after >= kept,
        f"{count_after} request(s) in the ledger after the stop and start" if count_after is not None
        else f"the container did not start again: {tail(again.stdout)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=18000,
                        help="host port for the test container, on 127.0.0.1 only; default 18000")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts" / "container-selftest",
                        help="folder for the generated env file and config; default artifacts/container-selftest, "
                             "which git ignores. It is emptied first")
    parser.add_argument("--wait", type=float, default=120.0,
                        help="seconds to wait for the containers to become healthy; default 120")
    parser.add_argument("--build-timeout", type=float, default=1800.0,
                        help="seconds the image build may take; default 1800")
    parser.add_argument("--skip-build", action="store_true", help="use the tine-proxy:dev image that already exists")
    parser.add_argument("--keep", action="store_true",
                        help="leave the test containers and their ledger volume running for a look around")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.wait <= 0 or args.build_timeout <= 0:
        parser.error("--port must be from 1 to 65535, --wait and --build-timeout above 0")
    work = args.work_dir.resolve()

    def out(text: str = "") -> None:
        print(text, flush=True)

    rows = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append((name, PASS if passed else FAIL, detail.replace(token, "<token>")))
        out(f"{name:<{NAME_WIDTH}}  {rows[-1][1]}  {rows[-1][2]}")

    token = "selftest-" + secrets.token_urlsafe(24)
    try:
        binary = find_docker()
        docker = Docker(binary, work, args.port, out)
        version = docker.run("version", "--format", "{{.Server.Version}}", timeout=30)
        if version.returncode != 0:
            raise Unusable(f"{binary} is installed but the Docker daemon did not answer. Start Docker Desktop, wait "
                           f"until it says it is running, and run this again. Docker said: {tail(version.stdout, 2)}")
    except Unusable as error:
        out(f"container_selftest: {error}")
        return 2
    out(f"Docker {(version.stdout or '').strip().splitlines()[-1]} at {binary}. Project {PROJECT}, proxy on "
        f"http://127.0.0.1:{args.port}. No model is called: the provider is a stand-in container.")
    out()
    write_files(work, token)
    proxy = smoke.Proxy(f"http://127.0.0.1:{args.port}", token, 30.0)
    started = False
    try:
        built = None
        if not args.skip_build:
            out("Building the image (the first build downloads the base image and the packages) ...")
            built = docker.compose("build", "tine", timeout=args.build_timeout, show=True)
            add("image builds", built.returncode == 0, "docker compose build tine" + (
                "" if built.returncode == 0 else f" ended with status {built.returncode}; its output is above"))
        if built is None or built.returncode == 0:
            started = True
            up = docker.compose("up", "-d", "--wait", "--wait-timeout", str(int(args.wait)), timeout=args.wait + 60)
            add("containers start", up.returncode == 0, "docker compose up -d --wait" + (
                "" if up.returncode == 0 else f" ended with status {up.returncode}: {tail(up.stdout)}"))
            if up.returncode == 0:
                try:
                    checks(docker, proxy, work, add, min(args.wait, 60.0))
                except smoke.Unreachable as error:
                    add("connection", False, f"the proxy stopped answering on port {args.port}: {error}")
            if any(result == FAIL for _, result, _ in rows):
                logs = docker.compose("logs", "--no-color", "--tail", "40", "tine", timeout=30)
                out()
                out("Last log lines of the proxy container:")
                out((logs.stdout or "").replace(token, "<token>").rstrip() or "(none)")
    except Unusable as error:
        add("docker", False, str(error))
    finally:
        if started and args.keep:
            out()
            out(f"Left running (--keep). The token is in {work / 'selftest.env'}. Remove everything with:")
            out(f"  docker compose -p {PROJECT} -f {COMPOSE_FILES[0]} -f {COMPOSE_FILES[1]} down --volumes"
                f"   (with TINE_SELFTEST_DIR={work} set)")
        elif started:
            down = docker.compose("down", "--volumes", "--remove-orphans", timeout=120)
            add("clean up", down.returncode == 0, "containers, network and test ledger volume removed"
                if down.returncode == 0 else f"docker compose down ended with status {down.returncode}: {tail(down.stdout)}")
            shutil.rmtree(work, ignore_errors=True)
    failed = [name for name, result, _ in rows if result == FAIL]
    out()
    out(f"{len(failed)} check(s) failed: {', '.join(failed)}." if failed
        else "All checks passed. No model was called and nothing was paid for.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
