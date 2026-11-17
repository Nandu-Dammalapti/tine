"""Checks on the deployment files that can be made without Docker.

No Docker daemon is available where these tests run, so nothing here builds or starts
the image. The tests read the files as text and data: the example configuration must
load into the real ``ProxyConfig`` and validate, the env example must name exactly
the variables the proxy reads, and no committed deploy file may hold something that
looks like a credential. Three things are run rather than read: the Dockerfile's
command that writes the requirements file, its health check command against the app
on a local port, and ``tine serve`` with the Dockerfile's arguments. When the docker
client is installed, ``docker compose config``, which needs no daemon, is run too.
"""

import ast
import importlib.metadata
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest
import uvicorn
import yaml
from fastapi.testclient import TestClient
from packaging.requirements import Requirement

from tine import __main__ as cli
from tine.proxy import ProxyConfig, create_app

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
ENV_EXAMPLE = DEPLOY / "tine.env.example"

# Shapes of real credentials. Placeholders in the examples must match none of them.
CREDENTIAL_SHAPES = [
    r"sk-[A-Za-z0-9_-]{16,}",             # OpenAI and Anthropic style keys
    r"AKIA[0-9A-Z]{16}",                  # AWS access key id
    r"gh[pousr]_[A-Za-z0-9]{20,}",        # GitHub tokens
    r"xox[abprs]-[A-Za-z0-9-]{10,}",      # Slack tokens
    r"AIza[0-9A-Za-z_-]{35}",             # Google API keys
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",  # JWT
    r"\b[0-9a-fA-F]{32,}\b",              # long hex strings
    r"\b(?=[A-Za-z0-9+/_-]*[0-9])(?=[A-Za-z0-9+/_-]*[a-z])(?=[A-Za-z0-9+/_-]*[A-Z])[A-Za-z0-9+/_-]{40,}\b",
    r"://[^/\s:@]+:[^/\s@]+@",            # user:password@host in a URL
]
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
# The fields tine.runtime_checks reads from a step. Anything else in the example
# would be ignored by the proxy, and is where an inline credential could hide.
STEP_FIELDS = {"webhook": {"type", "url", "bearer_env", "timeout_seconds"},
               "judge": {"type", "model", "effort", "threshold", "max_tokens", "repository"}}


def ignored_locally(path: Path) -> bool:
    """True for a file under deploy/ that deploy/.gitignore keeps out of git.

    An operator who follows docs/DEPLOY.md has a real deploy/tine.env and a
    deploy/config/ directory in the checkout. They hold credentials and real
    policies, so these tests must neither fail on them nor read them.
    """
    relative = path.relative_to(DEPLOY)
    for pattern in (DEPLOY / ".gitignore").read_text(encoding="utf-8").split("\n"):
        pattern = pattern.strip()
        if not pattern or pattern.startswith("#"):
            continue
        if pattern.endswith("/"):
            if any(fnmatch(part, pattern[:-1]) for part in relative.parts[:-1]):
                return True
        elif any(fnmatch(part, pattern) for part in relative.parts):
            return True
    return False


def committed_files() -> list[Path]:
    """The deploy files that go into git: everything under deploy/ that is not ignored."""
    return [ROOT / "Dockerfile", ROOT / "docker-compose.yml", ROOT / "docker-compose.selftest.yml", ROOT / ".dockerignore",
            ROOT / "docs" / "DEPLOY.md",
            *sorted(path for path in DEPLOY.rglob("*") if path.is_file() and not ignored_locally(path))]


def env_example() -> tuple[dict[str, str], dict[str, str]]:
    """The example's variables: those set, and those shown commented out."""
    active, commented = {}, {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        found = re.fullmatch(r"(#\s?)?([A-Z][A-Z0-9_]*)=(.*)", line)
        if found:
            (commented if found.group(1) else active)[found.group(2)] = found.group(3)
    return active, commented


def proxy_variables() -> set[str]:
    """Every TINE_* name that appears in the proxy's source."""
    return set(re.findall(r"TINE_[A-Z0-9_]+", (ROOT / "tine" / "proxy.py").read_text(encoding="utf-8")))


def set_example_environment(monkeypatch, tmp_path, **overrides) -> None:
    """Set the environment the container would have: the example env file, with the
    in-container /config paths pointed at the committed example files. An override of
    None removes a variable, as commenting its line out does."""
    active, _ = env_example()
    active.update(overrides)
    for name in proxy_variables() | set(active):
        monkeypatch.delenv(name, raising=False)
    for name, value in active.items():
        if value is None:
            continue
        found = re.fullmatch(r"/config/(.+)\.json", value)
        if found:
            value = str(DEPLOY / f"{found.group(1)}.example.json")
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TINE_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))


def load_example_config(monkeypatch, tmp_path, **overrides) -> ProxyConfig:
    """Build a ProxyConfig the way the container would, from the example env file."""
    set_example_environment(monkeypatch, tmp_path, **overrides)
    return ProxyConfig.from_env()


def test_env_example_names_exactly_the_variables_the_proxy_reads():
    active, commented = env_example()
    assert not set(active) & set(commented), "a variable is both set and commented out"
    named = {name for name in active | commented if name.startswith("TINE_")}
    assert named == proxy_variables()


def test_every_assignment_in_the_env_example_is_seen_by_these_tests():
    """An indented, lower-case or "export NAME=" line would be skipped by env_example(),
    and with it the placeholder and name checks. Docker would still read some of them."""
    active, commented = env_example()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        loose = re.match(r"[#\s]*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
        if loose:
            assert loose.group(1) in active | commented, line.split("=")[0]
            assert re.fullmatch(r"(#\s?)?[A-Z][A-Z0-9_]*=.*", line), line.split("=")[0]
        else:
            assert not line.strip() or line.startswith("#"), "a line that is neither a comment nor NAME=value"


def test_env_example_follows_its_own_format_rules():
    active, commented = env_example()
    for name, value in (active | commented).items():
        # Booleans first: a failing assert prints what it compared, and a value may be a key.
        trimmed = bool(value) and value == value.strip()
        assert trimmed, name
        plain = value[0] not in "'\"" and "$" not in value and " #" not in value
        assert plain, name
    for name in ("TINE_TENANT_KEYS", "TINE_MODEL_PRICES", "TINE_ALLOWED_MODELS"):
        assert isinstance(json.loads((active | commented)[name]), dict)


def test_example_configuration_validates_as_shipped(monkeypatch, tmp_path):
    config = load_example_config(monkeypatch, tmp_path)
    config.validate()
    assert config.shadow_sample_rate == 0, "the example must not start sampling by itself"
    assert config.shadow_max_cost_usd is not None and config.shadow_max_cost_usd > 0
    assert config.route_policies == {}
    assert set(config.shadow_policies) <= set(config.tenant_keys.values())


def test_example_configuration_validates_with_shadow_sampling_on(monkeypatch, tmp_path):
    config = load_example_config(monkeypatch, tmp_path, TINE_SHADOW_SAMPLE_RATE="0.05",
                                 TINE_ANSWER_PATTERN=env_example()[1]["TINE_ANSWER_PATTERN"])
    config.validate()
    assert config.shadow_policies and config.checks


def test_shadow_sampling_without_a_cap_is_refused(monkeypatch, tmp_path):
    config = load_example_config(monkeypatch, tmp_path, TINE_SHADOW_SAMPLE_RATE="0.05", TINE_SHADOW_MAX_COST_USD="")
    with pytest.raises(ValueError, match="shadow_max_cost_usd"):
        config.validate()


def test_changing_the_frontier_needs_the_example_policy_switched_off(monkeypatch, tmp_path):
    """The first edit an operator makes is the frontier model name. The example policy
    names the example frontier, so the proxy then refuses to start until the policy
    and checks lines are commented out (docs/DEPLOY.md, "Run it")."""
    with pytest.raises(ValueError, match="frontier"):
        load_example_config(monkeypatch, tmp_path, TINE_ANTHROPIC_FRONTIER_MODEL="A-REAL-MODEL-NAME").validate()
    config = load_example_config(monkeypatch, tmp_path, TINE_ANTHROPIC_FRONTIER_MODEL="A-REAL-MODEL-NAME",
                                 TINE_CHECKS="", TINE_SHADOW_POLICIES="")
    config.validate()
    assert config.shadow_policies == {} and config.checks == {}


def test_every_commented_out_line_validates_when_switched_on(monkeypatch, tmp_path):
    """The commented lines are what an operator uncomments later. Each must be a value
    the proxy accepts. The route policy line has no example file of its own: the shadow
    example serves, moved over, since a tenant cannot have both."""
    _, commented = env_example()
    assert len(commented) >= 15, "the commented lines were not read"
    route = tmp_path / "route-policies.json"
    shutil.copy(DEPLOY / "shadow-policies.example.json", route)
    config = load_example_config(monkeypatch, tmp_path, **{**commented, "TINE_SHADOW_POLICIES": None,
                                                         "TINE_ROUTE_POLICIES": str(route)})
    config.validate()
    assert config.route_policies and config.tenant_checks and config.openai_api_key
    assert config.ledger_required is True and config.audit_rate == 0.05 and config.max_body_bytes == 10_485_760
    assert config.denied_fields and config.anthropic_beta_allowlist and config.allowed_models
    with TestClient(create_app(config)) as client:
        assert client.get("/health").json()["routing"] == "policy"
    # With shadow sampling on as well, for the shadow-only lines.
    config = load_example_config(monkeypatch, tmp_path, **{**commented, "TINE_ROUTE_POLICIES": None,
                                                         "TINE_SHADOW_SAMPLE_RATE": "0.05"})
    config.validate()
    assert (config.shadow_max_concurrent_per_tenant, config.shadow_max_calls_per_minute) == (4, 60)


def test_the_example_policy_as_a_route_policy_needs_an_answer_reader(monkeypatch, tmp_path):
    """Routing compares answers. A policy with agreement stages and no reader is refused,
    which is why the example carries one and the env example says so beside the line."""
    example = json.loads((DEPLOY / "shadow-policies.example.json").read_text(encoding="utf-8"))
    for by_provider in example.values():
        for policy in by_provider.values():
            assert policy.pop("answer_pattern"), "the example policy carries its own answer reader"
    route = tmp_path / "route-policies.json"
    route.write_text(json.dumps(example), encoding="utf-8")
    switched = {"TINE_SHADOW_POLICIES": None, "TINE_ROUTE_POLICIES": str(route)}
    with pytest.raises(ValueError, match="agreement stages but no answer reader"):
        load_example_config(monkeypatch, tmp_path, **switched).validate()
    load_example_config(monkeypatch, tmp_path, **switched,
                        TINE_ANSWER_PATTERN=env_example()[1]["TINE_ANSWER_PATTERN"]).validate()


def test_a_webhook_token_the_env_file_does_not_set_is_refused(monkeypatch, tmp_path):
    """Removing the CI_WEBHOOK_TOKEN line and leaving TINE_CHECKS used to start cleanly:
    the webhook step then failed every time it ran."""
    with pytest.raises(ValueError, match="bearer_env 'CI_WEBHOOK_TOKEN' is not set"):
        load_example_config(monkeypatch, tmp_path, CI_WEBHOOK_TOKEN=None).validate()
    with pytest.raises(ValueError, match="bearer_env 'CI_WEBHOOK_TOKEN' is not set"):
        load_example_config(monkeypatch, tmp_path, CI_WEBHOOK_TOKEN="").validate()
    load_example_config(monkeypatch, tmp_path, CI_WEBHOOK_TOKEN=None, TINE_CHECKS=None,
                        TINE_SHADOW_POLICIES=None).validate()


def test_example_policy_is_priced_and_plainly_marked_as_fake(monkeypatch, tmp_path):
    config = load_example_config(monkeypatch, tmp_path)
    assert config.shadow_policies, "the example env must point at the example policy, or this test checks nothing"
    for by_provider in config.shadow_policies.values():
        for provider, policy in by_provider.items():
            assert "EXAMPLE ONLY" in policy["guarantee"]["statement"]
            models = {policy["frontier"]} | {model for stage in policy["stages"] for model in stage["models"]}
            judges = {step["model"] for stage in policy["stages"] for step in config.checks.get(stage["check"], [])
                      if step["type"] == "judge"}
            assert all("EXAMPLE" in model for model in models | judges)
            # A model without a price has unknown cost, and unknown cost defeats the spending cap.
            assert {f"{provider}:{model}" for model in models} <= set(config.prices)
            assert {f"anthropic:{model}" for model in judges} <= set(config.prices)
    assert all(rate > 0 for rates in config.prices.values() for rate in rates.values())


def test_example_webhook_token_is_passed_by_name_only():
    active, _ = env_example()
    steps = [step for check in json.loads((DEPLOY / "checks.example.json").read_text(encoding="utf-8")).values()
             for step in check]
    names = {step["bearer_env"] for step in steps if step["type"] == "webhook"}
    assert names and names <= set(active)
    assert all(urlhost(step["url"]).endswith(".example") for step in steps if step["type"] == "webhook")
    for step in steps:
        assert set(step) <= STEP_FIELDS[step["type"]], "a field the proxy does not read"


def urlhost(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).hostname or ""


def test_health_route_the_healthcheck_calls_exists(monkeypatch, tmp_path):
    """The Dockerfile's HEALTHCHECK asks for /health and expects 200."""
    with TestClient(create_app(load_example_config(monkeypatch, tmp_path))) as client:
        reply = client.get("/health")
    assert reply.status_code == 200 and reply.json()["status"] == "ok"


def healthcheck_command() -> list[str]:
    checks = [arguments for name, arguments in dockerfile_instructions() if name == "HEALTHCHECK"]
    assert len(checks) == 1
    return json.loads(checks[0][checks[0].index("CMD") + 3:])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_healthcheck_command_runs_as_written(monkeypatch, tmp_path):
    """The command from the Dockerfile, with only the port changed, against the app built
    from the example configuration and served on a local port."""
    command = healthcheck_command()
    server = uvicorn.Server(uvicorn.Config(create_app(load_example_config(monkeypatch, tmp_path)), host="127.0.0.1",
                                           port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            assert time.monotonic() < deadline and thread.is_alive(), "the app did not start"
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]

        def probe(port: int, **extra: str) -> int:
            env = {name: value for name, value in os.environ.items() if "proxy" not in name.lower()}
            return subprocess.run([sys.executable, "-c", command[2].replace(":8000/", f":{port}/")],
                                  env={**env, **extra}, capture_output=True, timeout=30).returncode

        assert probe(port) == 0
        # A container given an egress proxy: the loopback check must not go to it.
        dead = f"http://127.0.0.1:{free_port()}"
        assert probe(port, HTTP_PROXY=dead, HTTPS_PROXY=dead, ALL_PROXY=dead, http_proxy=dead) == 0
        assert probe(free_port()) == 1, "nothing listens there: the check must fail"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def credential_lines(path: Path) -> list[int]:
    """Line numbers in ``path`` that hold something shaped like a credential.

    Only numbers are returned, so a failing test reports where and never what:
    the match may be a real key, and test output ends up in CI logs.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [number for number, line in enumerate(lines, start=1)
            if any(re.search(shape, line) for shape in CREDENTIAL_SHAPES)]


def test_no_credential_shaped_values_in_committed_deploy_files():
    committed = committed_files()
    assert ENV_EXAMPLE in committed and len(committed) >= 7
    for path in committed:
        assert credential_lines(path) == [], f"{path.relative_to(ROOT)} holds something shaped like a credential"


def test_credential_shapes_catch_keys_and_spare_the_placeholders(tmp_path):
    made_up = ["sk-ant-" + "Ab1" * 8, "AKIA" + "A1" * 8, "AIza" + "b2C" * 11 + "dE", "ghp_" + "x9" * 12,
               "https://user:" + "pw" + "@host.example/v1", "0f" * 16]
    sample = tmp_path / "sample.env"
    sample.write_text("FIRST=fine\n" + "".join(f"NAME_{index}={value}\n" for index, value in enumerate(made_up)),
                      encoding="utf-8")
    assert credential_lines(sample) == list(range(2, len(made_up) + 2))


def test_local_configuration_is_not_read_as_a_committed_file():
    assert ignored_locally(DEPLOY / "tine.env") and ignored_locally(DEPLOY / "config" / "checks.json")
    assert ignored_locally(DEPLOY / "config" / "nested" / "policy.json")
    assert not ignored_locally(ENV_EXAMPLE) and not ignored_locally(DEPLOY / "checks.example.json")
    assert not ignored_locally(DEPLOY / "config.json") and not ignored_locally(DEPLOY / ".gitignore")


def test_what_the_local_test_kit_writes_is_kept_out_of_git():
    """scripts/container_env.py writes the provider key and a working tenant token under
    deploy/. Each file, and the temporary name it has while it is checked, is ignored."""
    for name in ("tine.env", "client-token.txt", "tine.env.1a2b3c4d.tmp.env", "client-token.txt.1a2b3c4d.tmp.env"):
        assert ignored_locally(DEPLOY / name), name
    assert ignored_locally(DEPLOY / "config" / "route-policies.json")
    assert ignored_locally(DEPLOY / "config" / "route-policies.json.1a2b3c4d.tmp.env")
    assert not ignored_locally(DEPLOY / "client-token.example.txt")
    # The repository's own ignore file says the same, for a checkout whose deploy/.gitignore was lost.
    top = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for rule in ("deploy/*.env", "deploy/config/", "deploy/client-token.txt"):
        assert rule in top, rule
    # Neither file can reach the image: the build context leaves out all of deploy/.
    assert "deploy" in (ROOT / ".dockerignore").read_text(encoding="utf-8").split()


def test_secret_variables_in_the_example_hold_placeholders():
    active, commented = env_example()
    for name, value in (active | commented).items():
        if SECRET_NAME.search(name):
            # A boolean first: a failing assert prints what it compared, and this may be a key.
            placeholder = "REPLACE_WITH" in value
            assert placeholder, name


def test_only_examples_live_in_deploy_and_real_config_is_ignored():
    # Whatever git would take from deploy/ is an example. Real configuration made by
    # following docs/DEPLOY.md may sit beside it, but only where git ignores it.
    names = {path.relative_to(DEPLOY).as_posix() for path in committed_files() if path.is_relative_to(DEPLOY)}
    assert all(name == ".gitignore" or ".example" in name for name in names), sorted(names)
    ignored = (DEPLOY / ".gitignore").read_text(encoding="utf-8").split()
    assert "*.env" in ignored and "config/" in ignored and "client-token.txt" in ignored


def dockerfile_instructions() -> list[tuple[str, str]]:
    """(instruction, arguments) pairs, with line continuations joined."""
    text = re.sub(r"\\\n", " ", (ROOT / "Dockerfile").read_text(encoding="utf-8"))
    return [(line.split(None, 1)[0].upper(), line.split(None, 1)[1] if " " in line else "")
            for line in (raw.strip() for raw in text.splitlines()) if line and not line.startswith("#")]


def test_dockerfile_base_user_and_command():
    steps = dockerfile_instructions()
    bases = [arguments for name, arguments in steps if name == "FROM"]
    assert len(bases) == 1 and re.fullmatch(r"python:3\.\d+(\.\d+)?-slim[a-z-]*(@sha256:[0-9a-f]{64})?", bases[0])
    minimum = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    assert tuple(map(int, re.search(r"python:(\d+)\.(\d+)", bases[0]).groups())) >= tuple(
        map(int, re.search(r"(\d+)\.(\d+)", minimum).groups()))
    users = [arguments for name, arguments in steps if name == "USER"]
    assert users and users[-1].split(":")[0].isdigit() and int(users[-1].split(":")[0]) > 0
    assert steps[-1][0] == "CMD"
    assert json.loads(steps[-1][1]) == ["python", "-m", "tine", "serve", "--host", "0.0.0.0", "--port", "8000"]
    assert not any(name == "ENTRYPOINT" for name, _ in steps)
    assert ("VOLUME", '["/data"]') in steps
    environment = " ".join(arguments for name, arguments in steps if name == "ENV")
    assert "TINE_LEDGER_PATH=/data/ledger.sqlite3" in environment


def test_dockerfile_healthcheck_uses_the_standard_library():
    command = healthcheck_command()
    assert command[:2] == ["python", "-c"] and "urllib.request" in command[2]
    assert "http://127.0.0.1:8000/health" in command[2]
    # An HTTP_PROXY variable in the container must not send the loopback check to the proxy.
    assert "ProxyHandler({})" in command[2] and "urlopen" not in command[2]
    compile(command[2], "<healthcheck>", "exec")
    assert not any(re.search(r"\b(curl|wget|apt-get)\b", arguments) for _, arguments in dockerfile_instructions())


def test_dockerfile_carries_no_secret_and_copies_only_the_package():
    steps = dockerfile_instructions()
    assert not any(name == "ARG" for name, _ in steps)
    for name, arguments in steps:
        if name == "ENV":
            # "ENV NAME=value ..." and the older "ENV NAME value".
            variables = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", arguments)
            if "=" not in arguments.split()[0]:
                variables.append(arguments.split()[0])
            for variable in variables:
                assert not SECRET_NAME.search(variable), variable
    copied = [source for name, arguments in steps if name in ("COPY", "ADD") for source in arguments.split()[:-1]]
    assert sorted(copied) == ["pyproject.toml", "requirements-dev.lock", "tine"]
    assert not any(name == "ADD" for name, _ in steps)
    installs = [arguments for name, arguments in steps if name == "RUN" and "pip install" in arguments]
    assert len(installs) == 1 and "tomllib" in installs[0] and "['project']['dependencies']" in installs[0]
    # The lock is a constraints file: the image gets the versions the tests ran on, and
    # none of the lock's development tools.
    assert "-r /app/requirements.txt -c /app/requirements-dev.lock" in installs[0]
    assert "-r /app/requirements-dev.lock" not in installs[0]


def lock_pins() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements-dev.lock").read_text(encoding="utf-8").splitlines():
        # pip reads a constraints file line by line: each must be a plain "name==version".
        found = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.!+_-]+)", line)
        assert found, f"not a plain pin, so not usable as a pip constraint: {line.split('=')[0]}"
        pins[normal(found.group(1))] = found.group(2)
    return pins


def normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_requirements_command_runs_and_every_dependency_is_pinned_by_the_lock(tmp_path):
    """The first half of the Dockerfile's install step, run through sh as Docker runs it,
    on a copy of pyproject.toml. Each line it writes must be pinned by the lock, inside
    the declared range, or the constraints file would not decide the version."""
    install = next(arguments for name, arguments in dockerfile_instructions()
                   if name == "RUN" and "pip install" in arguments)
    extract = install.split("&&")[0].strip()
    assert extract.startswith('python -c "import tomllib') and extract.endswith("> /app/requirements.txt")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    command = extract.replace("/app/", f"{tmp_path}/").replace("python -c", f"{sys.executable} -c", 1)
    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    lines = (tmp_path / "requirements.txt").read_text(encoding="utf-8").splitlines()
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert lines == declared and len(lines) >= 8
    pins = lock_pins()
    for line in lines:
        requirement = Requirement(line)
        assert normal(requirement.name) in pins, f"{requirement.name} has no pin in requirements-dev.lock"
        assert requirement.specifier.contains(pins[normal(requirement.name)]), requirement.name


def test_everything_the_package_imports_is_installed_by_its_declared_dependencies():
    """The image installs pyproject's dependencies and what they require, nothing else.
    A module the package imports from outside that set would be missing in the image
    while every test here passes, because the development environment has more."""
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    installed, queue = set(), [Requirement(line).name for line in declared]
    while queue:
        name = normal(queue.pop())
        if name in installed:
            continue
        installed.add(name)
        for line in importlib.metadata.requires(name) or []:
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                queue.append(requirement.name)
    owners = importlib.metadata.packages_distributions()
    imported = set()
    for path in (ROOT / "tine").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    third_party = {name for name in imported if name not in sys.stdlib_module_names and name != "tine"}
    assert {"fastapi", "httpx", "pydantic", "starlette", "uvicorn", "numpy"} <= third_party
    # tabulate is optional: pandas asks for it only to print a table, and the one use is guarded.
    for module in sorted(third_party - {"tabulate"}):
        assert module in owners, f"{module} is imported by the package and not installed here"
        assert {normal(owner) for owner in owners[module]} & installed, (
            f"tine imports {module}, which no dependency in pyproject.toml installs")
    assert normal("pytest") not in installed and normal("ruff") not in installed


def test_dockerignore_admits_only_the_package():
    lines = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    assert lines[0] == "*"
    assert {line for line in lines if line.startswith("!")} == {"!pyproject.toml", "!requirements-dev.lock", "!tine"}
    # Exclusions come after the allow-list, because the last matching line wins.
    last_admit = max(index for index, line in enumerate(lines) if line.startswith("!"))
    for name in ("artifacts", ".tine-dev", ".git", ".env", ".env.*", "deploy", "tests", "results", "notebooks"):
        assert name in lines[last_admit + 1:], name


def seconds(duration: str) -> int:
    found = re.fullmatch(r"(\d+)s", str(duration))
    assert found, f"write the duration in whole seconds: {duration}"
    return int(found.group(1))


def test_compose_file():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert list(compose["services"]) == ["tine"]
    service = compose["services"]["tine"]

    assert service["env_file"] == ["deploy/tine.env"]
    assert "environment" not in service, "configuration belongs in the uncommitted env file"

    mounts = {mount.split(":")[1]: mount for mount in service["volumes"]}
    assert mounts["/config"].endswith(":ro") and mounts["/config"].startswith("./deploy/config:")
    assert mounts["/data"].split(":")[0] in compose["volumes"], "the ledger needs a named volume"
    active, commented = env_example()
    assert active["TINE_LEDGER_PATH"].startswith("/data/")
    for name in ("TINE_CHECKS", "TINE_TENANT_CHECKS", "TINE_SHADOW_POLICIES", "TINE_ROUTE_POLICIES"):
        assert (active | commented)[name].startswith("/config/")

    assert service["read_only"] is True
    assert any(str(entry).startswith("/tmp") for entry in service["tmpfs"])
    assert service["cap_drop"] == ["ALL"] and "cap_add" not in service
    assert "no-new-privileges:true" in service["security_opt"]
    assert service.get("privileged") is not True
    assert str(service["user"]).split(":")[0].isdigit() and int(str(service["user"]).split(":")[0]) > 0
    assert service["restart"] in ("unless-stopped", "always", "on-failure")
    # The host port may be moved with TINE_HOST_PORT. The address may not.
    assert service["ports"] == ["127.0.0.1:${TINE_HOST_PORT:-8000}:8000"], (
        "publish on loopback only: there is no TLS here")
    # Docker kills the container after stop_grace_period. The proxy must be gone first,
    # with the ledger closed, so the wait is longer than tine serve's own limit.
    assert seconds(service["stop_grace_period"]) >= cli.SERVE_SHUTDOWN_SECONDS + 5
    assert service["init"] is True


def serve_from_the_dockerfile(monkeypatch) -> list[dict]:
    """Run ``tine serve`` with the Dockerfile's arguments and return what it handed to
    uvicorn, without starting a server."""
    command = json.loads(dockerfile_instructions()[-1][1])
    assert command[:3] == ["python", "-m", "tine"]
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(options))
    log = logging.getLogger("tine")
    kept = (list(log.handlers), log.level, log.propagate)
    try:
        cli.main(command[3:])
    finally:
        # serve attaches its own handler and stops propagation: undo it for the tests that follow.
        log.handlers[:], log.propagate = kept[0], kept[2]
        log.setLevel(kept[1])
    return calls


def test_serve_with_the_dockerfiles_arguments_stops_inside_the_grace_period(monkeypatch, tmp_path):
    """Without a limit uvicorn waits for every open connection, and a stream that never
    ends holds a stop until Docker kills the container with the ledger still open."""
    set_example_environment(monkeypatch, tmp_path)
    calls = serve_from_the_dockerfile(monkeypatch)
    assert calls == [{"host": "0.0.0.0", "port": 8000, "timeout_graceful_shutdown": cli.SERVE_SHUTDOWN_SECONDS}]
    assert 1 <= cli.SERVE_SHUTDOWN_SECONDS <= 25
    with pytest.raises(SystemExit):
        cli.main(["serve", "--shutdown-seconds", "0"])


def test_a_refused_configuration_ends_serve_with_one_line(monkeypatch, tmp_path):
    """In a container this line is the end of the log before a restart."""
    set_example_environment(monkeypatch, tmp_path, TINE_TIMEOUT_SECONDS="30s")
    with pytest.raises(SystemExit) as stopped:
        serve_from_the_dockerfile(monkeypatch)
    assert str(stopped.value) == ("tine serve: the configuration was refused. TINE_TIMEOUT_SECONDS must be a "
                                  "number, with no unit or other text")
    assert stopped.value.__suppress_context__, "no traceback of the ValueError above the line"


def docker_client() -> str | None:
    """A docker command that can run ``compose config``, or None. No daemon is needed."""
    for candidate in (shutil.which("docker"), str(Path.home() / ".docker" / "bin" / "docker"),
                      "/Applications/Docker.app/Contents/Resources/bin/docker"):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


def test_docker_compose_reads_the_env_example_as_these_tests_do(tmp_path):
    """Everything above reads the env example with this file's own pattern. Compose has
    its own rules for quotes, "$" and " #". Where the docker client is installed, ask
    compose itself (``config`` needs no daemon) and compare."""
    docker = docker_client()
    if docker is None:
        pytest.skip("no docker client on this machine")
    (tmp_path / "deploy" / "config").mkdir(parents=True)
    shutil.copy(ROOT / "docker-compose.yml", tmp_path / "docker-compose.yml")
    shutil.copy(ENV_EXAMPLE, tmp_path / "deploy" / "tine.env")
    env = {name: value for name, value in os.environ.items() if not name.startswith("TINE_")}
    done = subprocess.run([docker, "compose", "config", "--format", "json"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    service = json.loads(done.stdout)["services"]["tine"]
    active, _ = env_example()
    # Compared as a boolean: a failing assert must not print the values.
    same = service["environment"] == active
    assert same, sorted(set(service["environment"]) ^ set(active))
    assert [(port["host_ip"], port["published"], port["target"]) for port in service["ports"]] == [
        ("127.0.0.1", "8000", 8000)]
    # Without the env file every compose command stops, build included.
    (tmp_path / "deploy" / "tine.env").unlink()
    missing = subprocess.run([docker, "compose", "config", "-q"], cwd=tmp_path, env=env, capture_output=True,
                             text=True, timeout=60)
    assert missing.returncode != 0 and "tine.env" in missing.stderr


def test_deploy_doc_states_what_was_not_done():
    text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
    # No daemon had built the image when this was written. This sentence, the same claim at
    # the top of the Dockerfile and both compose files, and docs/PROXY.md change together
    # when someone has built it.
    assert "No Docker daemon has built or run the image yet" in text
    assert "NOT YET BUILT BY A DOCKER DAEMON" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for name in ("docker-compose.yml", "docker-compose.selftest.yml"):
        assert "BY A DOCKER DAEMON" in (ROOT / name).read_text(encoding="utf-8"), name
    assert "This script has not been run against a real Docker daemon" in text
    assert "—" not in text
    for name in ("TINE_SHADOW_SAMPLE_RATE", "TINE_SHADOW_MAX_COST_USD", "SQLite", "TLS", "backup"):
        assert name in text, name


# Which Dockerfile steps were run by hand outside Docker, and which were not. The three
# files that say it use these words. Both sentences go when a daemon has built the image.
RUN_BY_HAND = ("Run by hand outside Docker on 2026-11-17, on a Mac: the command that writes requirements.txt, "
               "a pip dry-run resolve of its lines for Linux (nothing was installed), the image's command "
               "(on 127.0.0.1, not 0.0.0.0), the health check command, requests and a stop.")
NOT_RUN = ("Not run: the base image, the pip install itself, the two COPY steps, the user-creation step "
           "(groupadd, useradd, chown, chmod), USER and VOLUME.")


def plain_words(text: str) -> str:
    """``text`` without comment marks, code marks and line breaks."""
    return " ".join(re.sub(r"(?m)^#+ ?", "", text).replace("`", "").split())


@pytest.mark.parametrize("name", ["Dockerfile", "docs/DEPLOY.md", "docs/PROXY.md"])
def test_the_steps_run_by_hand_are_stated_the_same_everywhere(name):
    """These files once said that each step of the Dockerfile was run by hand. The
    dependency step was a dry run, and the COPY, user, USER and VOLUME steps need a
    daemon: nothing ran them."""
    words = plain_words((ROOT / name).read_text(encoding="utf-8"))
    assert RUN_BY_HAND in words and NOT_RUN in words
    assert words.index(RUN_BY_HAND) < words.index(NOT_RUN)
    for claim in (r"each step[^.]*\brun by hand", r"its steps were run by hand", r"every step[^.]*\bwas run\b"):
        assert not re.search(claim, words, flags=re.IGNORECASE), claim
    assert re.search(r"not every step [^.]* has been run", words)


def test_the_steps_named_as_not_run_are_the_dockerfiles():
    """The sentence names steps of the Dockerfile. It must change when they do."""
    steps = dockerfile_instructions()
    assert sum(name == "COPY" for name, _ in steps) == 2
    assert sum(name == "USER" for name, _ in steps) == 1 and sum(name == "VOLUME" for name, _ in steps) == 1
    creating = [arguments for name, arguments in steps if name == "RUN" and "useradd" in arguments]
    assert len(creating) == 1
    named = re.search(r"user-creation step \(([^)]*)\)", NOT_RUN).group(1).split(", ")
    assert all(re.search(rf"\b{command}\b", creating[0]) for command in named), named
    # Every RUN is one of the two the sentences speak of: the install and the user step.
    runs = [arguments for name, arguments in steps if name == "RUN"]
    assert len(runs) == 2 and "pip install" in runs[0] and runs[1] == creating[0]


def test_deploy_doc_restarts_with_force_recreate_after_a_file_edit():
    """Compose does not recreate a container because a mounted file changed. A plain
    "up -d" given as the restart step after a policy edit would leave routing on."""
    text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "restart" in line.lower() and "docker compose up -d" in line:
            assert "docker compose up -d --force-recreate" in line, line
    off = text[text.index("\n## Turn routing off\n"):]
    off = off[:off.index("\n## ", 1)]
    assert "docker compose up -d --force-recreate" in off and '"routing": "frontier-default"' in off
    # The first-start steps come before the first compose command, which needs the env file.
    run = text[text.index("\n## Run it\n"):]
    assert run.index("cp deploy/tine.env.example deploy/tine.env") < run.index("docker compose build")
    assert "TINE_MODEL_PRICES" in run[:run.index("docker compose build")]
    assert "unknown_answer_cost_requests" in text and "host.docker.internal" in text and "NO_PROXY" in text


def test_deploy_doc_gives_the_one_command_check():
    text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
    start = text.index("\n## Check the image with one command\n")
    section = text[start:text.index("\n## ", start + 1)]
    assert "python3 scripts/container_selftest.py" in section
    assert "No model is called and nothing is paid for" in section
    assert start < text.index("\n## Test the container on your own machine\n")
    known = script_options("container_selftest.py")
    assert len(known) >= 5
    for span in re.findall(r"`(--[^`]+)`", section):
        used = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", span))
        assert used <= known | {"--shutdown-seconds"}, f"container_selftest.py has no {sorted(used - known)}"
    # Every check the script prints is in the sample. tests/test_container_selftest.py pins their order.
    names = re.findall(r'add\("([^"]+)",', (ROOT / "scripts" / "container_selftest.py").read_text(encoding="utf-8"))
    printed = [line.split("  ")[0].strip() for line in section.splitlines() if re.search(r"\s{2}PASS\s{2}", line)]
    assert sorted(printed) == sorted(name for name in names if name not in ("connection", "docker"))


def local_test_section() -> str:
    """The part of docs/DEPLOY.md that walks through a test on the operator's own machine."""
    text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
    start = text.index("\n## Test the container on your own machine\n")
    return text[start:text.index("\n## ", start + 1)]


def script_options(name: str) -> set[str]:
    return set(re.findall(r'add_argument\("(--[a-z-]+)"', (ROOT / "scripts" / name).read_text(encoding="utf-8")))


def test_deploy_doc_gives_the_local_test_in_order():
    section = local_test_section()
    steps = ["scripts/container_env.py", "docker compose build", "docker compose up -d --wait", "docker compose ps",
             "python3 scripts/container_smoke.py", "docker compose logs tine", "docker compose cp tine:/data/",
             "scripts/savings_report.py", "docker compose stop", "docker compose down"]
    places = [section.index(step) for step in steps]
    assert places == sorted(places), "the commands are not in the order an operator runs them"
    # Requests through the container are paid, and the page says so where it shows the command.
    paid = section.index("--yes-paid")
    assert "paid calls" in section[paid:paid + 400].lower()
    # The image was not built when this was written. The sentence goes when someone has built it.
    assert "has still not been built" in section
    # The ledger copy holds prompts: it is written where git ignores it.
    assert "docker compose cp tine:/data/ledger.backup.sqlite3 artifacts/" in section
    assert "/artifacts/" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_deploy_doc_uses_only_options_the_scripts_have():
    section = local_test_section()
    options = {name: script_options(name) for name in ("container_env.py", "container_smoke.py", "savings_report.py")}
    assert all(len(found) >= 5 for found in options.values()), "the option lists were not read"
    blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL)
    commands = [command for block in blocks for command in re.sub(r"\\\n", " ", block).splitlines()]
    seen = set()
    for command in commands:
        for name, known in options.items():
            if f"scripts/{name}" in command:
                seen.add(name)
                used = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", command))
                assert used <= known, f"{name} has no {sorted(used - known)}"
    assert seen == set(options), "a script is missing from the commands"
    # Options named in the text, such as `--route-policy FILE`, exist in one of the scripts.
    every = set().union(*options.values())
    for span in re.findall(r"`(--[^`]+)`", section):
        used = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", span))
        assert used <= every, f"no script has {sorted(used - every)}"
