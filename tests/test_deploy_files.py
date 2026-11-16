"""Checks on the deployment files that can be made without Docker.

Docker is not available where these tests run, so nothing here builds or starts the
image. The tests read the files as text and data: the example configuration must
load into the real ``ProxyConfig`` and validate, the env example must name exactly
the variables the proxy reads, and no committed deploy file may hold something that
looks like a credential.
"""

import json
import re
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest

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
    return [ROOT / "Dockerfile", ROOT / "docker-compose.yml", ROOT / ".dockerignore", ROOT / "docs" / "DEPLOY.md",
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


def load_example_config(monkeypatch, tmp_path, **overrides) -> ProxyConfig:
    """Build a ProxyConfig the way the container would: from the example env file,
    with the in-container /config paths pointed at the committed example files."""
    active, _ = env_example()
    active.update(overrides)
    for name in proxy_variables() | set(active):
        monkeypatch.delenv(name, raising=False)
    for name, value in active.items():
        found = re.fullmatch(r"/config/(.+)\.json", value)
        if found:
            value = str(DEPLOY / f"{found.group(1)}.example.json")
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TINE_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))
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
    from fastapi.testclient import TestClient

    with TestClient(create_app(load_example_config(monkeypatch, tmp_path))) as client:
        reply = client.get("/health")
    assert reply.status_code == 200 and reply.json()["status"] == "ok"


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
    assert "*.env" in ignored and "config/" in ignored


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
    checks = [arguments for name, arguments in dockerfile_instructions() if name == "HEALTHCHECK"]
    assert len(checks) == 1
    command = json.loads(checks[0][checks[0].index("CMD") + 3:])
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
    copied = [arguments.split()[0] for name, arguments in steps if name in ("COPY", "ADD")]
    assert sorted(copied) == ["pyproject.toml", "tine"]
    assert not any(name == "ADD" for name, _ in steps)
    installs = [arguments for name, arguments in steps if name == "RUN" and "pip install" in arguments]
    assert len(installs) == 1 and "tomllib" in installs[0] and "['project']['dependencies']" in installs[0]


def test_dockerignore_admits_only_the_package():
    lines = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    assert lines[0] == "*"
    assert {line for line in lines if line.startswith("!")} == {"!pyproject.toml", "!tine"}
    # Exclusions come after the allow-list, because the last matching line wins.
    last_admit = max(index for index, line in enumerate(lines) if line.startswith("!"))
    for name in ("artifacts", ".tine-dev", ".git", ".env", ".env.*", "deploy", "tests", "results", "notebooks"):
        assert name in lines[last_admit + 1:], name


def test_compose_file():
    yaml = pytest.importorskip("yaml")
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
    for name in ("TINE_CHECKS", "TINE_SHADOW_POLICIES", "TINE_ROUTE_POLICIES"):
        assert (active | commented)[name].startswith("/config/")

    assert service["read_only"] is True
    assert any(str(entry).startswith("/tmp") for entry in service["tmpfs"])
    assert service["cap_drop"] == ["ALL"] and "cap_add" not in service
    assert "no-new-privileges:true" in service["security_opt"]
    assert service.get("privileged") is not True
    assert str(service["user"]).split(":")[0].isdigit() and int(str(service["user"]).split(":")[0]) > 0
    assert service["restart"] in ("unless-stopped", "always", "on-failure")
    assert service["ports"] == ["127.0.0.1:8000:8000"], "publish on loopback only: there is no TLS here"


def test_deploy_doc_states_what_was_not_done():
    text = (ROOT / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
    assert "has not been built or run" in text
    assert "—" not in text
    for name in ("TINE_SHADOW_SAMPLE_RATE", "TINE_SHADOW_MAX_COST_USD", "SQLite", "TLS", "backup"):
        assert name in text, name
