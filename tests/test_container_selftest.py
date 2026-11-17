"""scripts/container_selftest.py and scripts/container_standin.py, without Docker.

Docker is not available where these tests run. The self-test is run against a fake
``docker`` command that starts the real proxy (``python -m tine serve``, the image's
command) and the real stand-in provider as local processes, and answers the few
docker commands the script sends. So the requests, the policy edit and restart, and
the stop with a stream open are real; the build, the image, uid 10001 and the
read-only filesystem are not, and the fake only says what a good container would.
No model is called and nothing leaves this machine.
"""

import importlib.util
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tine.proxy import ProxyConfig, create_app

ROOT = Path(__file__).resolve().parents[1]
DOCKER_CLIENT = Path("/Applications/Docker.app/Contents/Resources/bin/docker")


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selftest = load("container_selftest")
standin = load("container_standin")

FAKE_DOCKER = r'''
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

STATE = Path(os.environ["FAKE_DOCKER_STATE"])
ROOT = os.environ["FAKE_DOCKER_ROOT"]
WORK = Path(os.environ.get("TINE_SELFTEST_DIR", "/nonexistent"))
PORT = os.environ.get("TINE_SELFTEST_PORT", "0")
STANDIN_PORT = os.environ["FAKE_DOCKER_STANDIN_PORT"]
arguments = sys.argv[1:]
with open(STATE / "commands.log", "a") as log:
    log.write(json.dumps(arguments) + "\n")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_of(name):
    try:
        return int((STATE / f"{name}.pid").read_text())
    except (OSError, ValueError):
        return None


def healthy(port):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except OSError:
        return False


def spawn(name, command, env):
    with open(STATE / f"{name}.log", "a") as output:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (STATE / f"{name}.pid").write_text(str(process.pid))


def start_standin():
    spawn("standin", [sys.executable, os.path.join(ROOT, "scripts", "container_standin.py"),
                      "--port", STANDIN_PORT], dict(os.environ))


def start_tine():
    env = {name: value for name, value in os.environ.items() if not name.startswith("TINE_")}
    for line in (WORK / "selftest.env").read_text().splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            env[name] = value.replace("/config/", str(WORK / "config") + "/")
    assert env["TINE_ANTHROPIC_BASE_URL"] == "http://standin:9000"
    env.update(TINE_ANTHROPIC_BASE_URL=f"http://127.0.0.1:{STANDIN_PORT}",
               TINE_LEDGER_PATH=str(STATE / "ledger.sqlite3"), PYTHONPATH=ROOT)
    spawn("tine", [sys.executable, "-m", "tine", "serve", "--host", "127.0.0.1", "--port", PORT,
                   "--shutdown-seconds", "3"], env)
    (STATE / "tine.exit").unlink(missing_ok=True)


def wait_healthy(port, seconds=40):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if healthy(port):
            return True
        time.sleep(0.2)
    return False


def stop(name, grace=30):
    """SIGTERM, then SIGKILL after ``grace`` seconds, as Docker does."""
    pid = pid_of(name)
    code = 0
    if pid and alive(pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        code = 143
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
            code = 137
    (STATE / f"{name}.exit").write_text(str(code))


if arguments[:1] == ["version"]:
    if (STATE / "daemon-down").exists():
        print("Cannot connect to the Docker daemon")
        sys.exit(1)
    print("99.0-fake")
elif arguments[:1] == ["inspect"]:
    pid = pid_of("tine")
    running = bool(pid and alive(pid))
    exit_file = STATE / "tine.exit"
    print(json.dumps({"Status": "running" if running else "exited",
                      "ExitCode": int(exit_file.read_text()) if exit_file.exists() else 0,
                      "Health": {"Status": "healthy" if running and healthy(PORT) else "unhealthy"}}))
elif arguments[:1] == ["compose"]:
    assert arguments[1:7] == ["-p", "tine-selftest", "-f", "docker-compose.yml", "-f", "docker-compose.selftest.yml"]
    command = arguments[7:]
    if command[0] == "build":
        if (STATE / "build-fails").exists():
            print("failed to solve: no such image")
            sys.exit(1)
        print("fake build of", command[1])
    elif command[0] == "up":
        start_standin()
        ok = wait_healthy(STANDIN_PORT)
        start_tine()
        sys.exit(0 if ok and wait_healthy(PORT) else 1)
    elif command[0] == "ps":
        if pid_of("tine"):
            print("fake-tine-container-id")
    elif command[0] == "exec":
        if command[3:] == ["id", "-u"]:
            print("10001")
        elif command[3:7] == ["python", "-m", "tine", "backup-ledger"]:
            # Run for real against the ledger the proxy has open, /data mapped to this fake's folder.
            mapped = [f"{STATE}/{value.split('/', 2)[2]}" if value.startswith("/data/") else value
                      for value in command[7:]]
            sys.exit(subprocess.run([sys.executable, "-m", "tine", "backup-ledger", *mapped],
                                    env={**os.environ, "PYTHONPATH": ROOT,
                                         "TINE_LEDGER_PATH": str(STATE / "ledger.sqlite3")}, cwd=STATE).returncode)
        elif command[3:5] == ["rm", "-f"]:
            (STATE / Path(command[5]).name).unlink(missing_ok=True)
        elif command[3:5] == ["python", "-"]:
            # The report script arrives on standard input and is run for real, with the
            # container's /data and /tmp mapped to this fake's folder.
            mapped = [f"{STATE}/{value.split('/', 2)[2]}" if value.startswith(("/data/", "/tmp/"))
                      else str(WORK / value[1:]) if value.startswith("/config/") else value
                      for value in command[5:]]
            sys.exit(subprocess.run([sys.executable, "-", *mapped], env={**os.environ, "PYTHONPATH": ROOT},
                                    cwd=STATE).returncode)
        elif command[3] == "cat":
            sys.stdout.write((STATE / Path(command[4]).name).read_text())
        else:
            # What a container with a read-only root prints. The fake cannot be one.
            print("/app=False /config=False /var/tmp=False /data=True /tmp=True")
            print("app:pyproject.toml,requirements-dev.lock,requirements.txt,tine")
    elif command[0] == "cp":
        # Only what docker cp can read: a file on the /data volume, not the /tmp tmpfs.
        assert command[1].startswith("tine:/data/"), command
        import shutil
        shutil.copyfile(STATE / Path(command[1]).name, command[2])
    elif command[0] == "restart":
        stop("tine")
        start_tine()
    elif command[0] == "stop":
        stop("tine")
    elif command[0] == "start":
        start_tine()
    elif command[0] == "logs":
        print((STATE / "tine.log").read_text()[-2000:])
    elif command[0] == "down":
        for name in ("tine", "standin"):
            stop(name, grace=5)
            (STATE / f"{name}.pid").unlink(missing_ok=True)
    else:
        sys.exit(f"fake docker: no compose {command[0]}")
else:
    sys.exit(f"fake docker: no {arguments[:1]}")
'''


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    """A ``docker`` on PATH that runs the proxy and the stand-in as local processes."""
    state, folder = tmp_path / "state", tmp_path / "bin"
    state.mkdir()
    folder.mkdir()
    script = folder / "docker"
    script.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("FAKE_DOCKER_ROOT", str(ROOT))
    monkeypatch.setenv("FAKE_DOCKER_STANDIN_PORT", str(free_port()))
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    yield state
    for name in ("tine", "standin"):
        try:
            os.kill(int((state / f"{name}.pid").read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass


def commands(state: Path) -> list[list[str]]:
    """What the script asked docker to do, without the project and file options."""
    sent = [json.loads(line) for line in (state / "commands.log").read_text().splitlines()]
    return [line[7:] if line[0] == "compose" else line for line in sent]


def run_selftest(tmp_path, capsys, *options):
    status = selftest.main(["--port", str(free_port()), "--work-dir", str(tmp_path / "work"), "--wait", "40", *options])
    return status, capsys.readouterr().out


def test_the_whole_selftest_passes_against_the_real_proxy_and_cleans_up(fake_docker, tmp_path, capsys):
    status, output = run_selftest(tmp_path, capsys)
    assert status == 0, output
    assert "All checks passed. No model was called" in output
    rows = dict(re.findall(r"^(\S.*?)\s{2,}(PASS|FAIL)\s", output, flags=re.MULTILINE))
    assert list(rows) == ["image builds", "containers start", "container healthy", "health", "token needed",
                          "request", "streamed request", "savings", "ledger copy", "savings report page",
                          "runs as uid 10001",
                          "read-only except /data, /tmp", "image holds only the package",
                          "policy edit, then restart", "ledger kept over a restart", "stop with a stream open",
                          "ledger whole after the stop", "clean up"]
    assert set(rows.values()) == {"PASS"}
    assert "served by selftest-frontier, route -, ledger recorded" in output
    assert "served by selftest-cheap, route stage-0, ledger recorded" in output
    # The stop is real: SIGTERM to the proxy while the stand-in holds a stream open. The
    # fake starts the proxy with --shutdown-seconds 3, so it must be gone well inside 15.
    took = float(re.search(r"stopped in ([0-9.]+) s", output).group(1))
    assert 2.0 < took < 15.0, output
    # The ledger leaves the container the documented way: the backup command, one file
    # copied out of /data, and the page made from that copy and the proxy's policies file.
    assert "the copy on the host holds 2 of 2 request(s)" in output
    sent = commands(fake_docker)
    backup = ["exec", "-T", "tine", "python", "-m", "tine", "backup-ledger", "--output", "/data/ledger-copy.sqlite3",
              "--force"]
    copy_out = ["cp", "tine:/data/ledger-copy.sqlite3", str(tmp_path / "work" / "ledger-copy.sqlite3")]
    assert sent.index(backup) < sent.index(copy_out)
    report = next(line for line in sent if line[:5] == ["exec", "-T", "tine", "python", "-"])
    assert report[5:] == ["--ledger", "/data/ledger-copy.sqlite3", "--tenant", "selftest", "--policy",
                          "/config/route-policies.json", "--output", "/tmp/selftest-report.html"]
    assert sent.index(copy_out) < sent.index(report) < sent.index(["exec", "-T", "tine", "rm", "-f",
                                                                 "/data/ledger-copy.sqlite3"])
    assert not (fake_docker / "ledger-copy.sqlite3").exists()
    assert sent[0][0] == "version" and sent[1] == ["build", "tine"]
    assert sent[2] == ["up", "-d", "--wait", "--wait-timeout", "40"]
    assert ["restart", "tine"] in sent and ["stop", "tine"] in sent and ["start", "tine"] in sent
    assert sent[-1] == ["down", "--volumes", "--remove-orphans"]
    # Nothing is left behind: no process, no env file with the token.
    assert not (tmp_path / "work").exists()
    assert not (fake_docker / "tine.pid").exists() and not (fake_docker / "standin.pid").exists()
    assert not re.search(r"selftest-[A-Za-z0-9_-]{20,}", output), "the made-up tenant token was printed"


def test_a_failed_build_stops_before_anything_is_started(fake_docker, tmp_path, capsys):
    (fake_docker / "build-fails").touch()
    status, output = run_selftest(tmp_path, capsys)
    assert status == 1 and "1 check(s) failed: image builds." in output
    assert [line[0] for line in commands(fake_docker)] == ["version", "build"]


def test_a_daemon_that_does_not_answer_is_reported_with_what_to_do(fake_docker, tmp_path, capsys):
    (fake_docker / "daemon-down").touch()
    status, output = run_selftest(tmp_path, capsys)
    assert status == 2 and "Start Docker Desktop" in output
    assert [line[0] for line in commands(fake_docker)] == ["version"]
    assert not (tmp_path / "work").exists(), "nothing is written before Docker answers"


def test_docker_is_found_where_docker_desktop_keeps_it(tmp_path, monkeypatch):
    """A new Docker Desktop install may not be on PATH yet. Its own folder also holds the
    credential helper a build needs, so that folder goes first on PATH for every command."""
    folder = tmp_path / "docker-bin"
    folder.mkdir()
    (folder / "docker").write_text("#!/bin/sh\n", encoding="utf-8")
    (folder / "docker").chmod(0o755)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(selftest, "DOCKER_FOLDERS", (tmp_path / "missing", folder))
    assert selftest.find_docker(path=str(empty)) == str(folder / "docker")
    docker = selftest.Docker(str(folder / "docker"), tmp_path / "work", 18000, print)
    assert docker.env["PATH"].split(os.pathsep)[0] == str(folder)
    monkeypatch.setattr(selftest, "DOCKER_FOLDERS", (tmp_path / "missing",))
    with pytest.raises(selftest.Unusable, match="not found"):
        selftest.find_docker(path=str(empty))


@pytest.mark.parametrize("stages,routing", [([], "frontier-default"),
                                            ([{"models": [selftest.CHEAP], "check": "none"}], "policy")])
def test_the_generated_configuration_is_one_the_proxy_accepts(tmp_path, monkeypatch, stages, routing):
    token = "selftest-" + "t0" * 12
    selftest.write_files(tmp_path / "work", token)
    selftest.write_policy(tmp_path / "work", stages)
    assert stat.S_IMODE((tmp_path / "work" / "selftest.env").stat().st_mode) == 0o600
    for name in list(os.environ):
        if name.startswith("TINE_"):
            monkeypatch.delenv(name)
    for line in (tmp_path / "work" / "selftest.env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            assert value == value.strip() and value[0] not in "'\"" and "$" not in value and " #" not in value, name
            monkeypatch.setenv(name, value.replace("/config/", f"{tmp_path / 'work' / 'config'}/"))
    monkeypatch.setenv("TINE_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))
    config = ProxyConfig.from_env()
    config.validate()
    assert config.tenant_keys == {token: selftest.TENANT} and config.anthropic_base_url == "http://standin:9000"
    assert {f"anthropic:{selftest.FRONTIER}", f"anthropic:{selftest.CHEAP}"} <= set(config.prices)
    with TestClient(create_app(config)) as client:
        assert client.get("/health").json()["routing"] == routing


def test_the_compose_pair_resolves_without_the_operators_files():
    """``docker compose config`` needs no daemon. The self-test pair must resolve with no
    deploy/tine.env, keep the real file's hardening and publish on loopback only."""
    if not DOCKER_CLIENT.exists():
        pytest.skip("no docker client on this machine")
    env = {**os.environ, "TINE_SELFTEST_DIR": "/nonexistent/selftest", "TINE_SELFTEST_PORT": "18001"}
    files = [part for name in selftest.COMPOSE_FILES for part in ("-f", name)]
    done = subprocess.run([str(DOCKER_CLIENT), "compose", "-p", selftest.PROJECT, *files, "config", "--format", "json",
                           "--no-env-resolution"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-800:]
    resolved = json.loads(done.stdout)
    tine, provider = resolved["services"]["tine"], resolved["services"]["standin"]
    # One env file, the generated one: deploy/tine.env is not asked for.
    assert [entry["path"] for entry in tine["env_file"]] == ["/nonexistent/selftest/selftest.env"]
    assert [(port["host_ip"], port["published"], port["target"]) for port in tine["ports"]] == [("127.0.0.1", "18001", 8000)]
    mounts = {mount["target"]: mount for mount in tine["volumes"]}
    assert set(mounts) == {"/data", "/config"}
    assert mounts["/config"]["source"] == "/nonexistent/selftest/config" and mounts["/config"]["read_only"] is True
    assert mounts["/data"]["type"] == "volume"
    for service in (tine, provider):
        assert service["read_only"] is True and service["user"] == "10001:10001" and service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"] and service["restart"] == "no"
        assert service["image"] == "tine-proxy:dev"
    assert tine["depends_on"]["standin"]["condition"] == "service_healthy"
    assert "ports" not in provider and provider["pull_policy"] == "never"
    assert tine.get("command") is None, "the proxy under test runs the image's own command"


@pytest.fixture
def provider():
    server = standin.Server(("127.0.0.1", 0), standin.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(url: str, body: dict, timeout: float = 5):
    request = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def test_the_standin_answers_like_the_messages_api_and_calls_nothing(provider):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    assert json.loads(opener.open(provider + "/health", timeout=5).read()) == {"status": "ok", "standin": True}
    body = {"model": "any-model", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    reply = json.loads(post(provider, body).read())
    assert reply["id"] == standin.REPLY_ID and reply["model"] == "any-model"
    assert reply["usage"] == {"input_tokens": 12, "output_tokens": 3} and reply["content"][0]["text"] == "ok"
    stream = post(provider, {**body, "stream": True})
    assert stream.headers["content-type"] == "text/event-stream"
    text = stream.read().decode()
    assert text.count("event: ") == 6 and text.rstrip().endswith('{"type": "message_stop"}')
    source = (ROOT / "scripts" / "container_standin.py").read_text(encoding="utf-8")
    assert "urllib" not in source and "http.client" not in source, "the stand-in makes no outbound call"


def test_the_standin_holds_a_marked_stream_open(provider):
    held = post(provider, {"model": "m", "stream": True, "messages": [
        {"role": "user", "content": [{"type": "text", "text": f"{standin.HOLD_MARKER} please"}]}]}, timeout=1.5)
    assert held.readline().startswith(b"event: message_start")
    lines, deadline = [], time.monotonic() + 2.5
    while time.monotonic() < deadline:
        lines.append(held.readline())
    held.close()
    assert lines.count(b": still open\n") >= 2 and not any(b"message_stop" in line for line in lines)
