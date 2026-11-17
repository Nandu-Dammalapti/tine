"""scripts/container_env.py against a made-up key file in a temporary folder.

No network and no Docker. The key and the token must never reach the output, the env
file and the token file must be mode 600, and what is written must load into the
real ``ProxyConfig`` as the proxy would load it.
"""

import ast
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from tine.proxy import ProxyConfig

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("container_env", ROOT / "scripts" / "container_env.py")
kit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kit)

# Made up, and built at run time so that no committed line is shaped like a key.
FAKE_KEY = "sk-ant-" + "Zq7" * 12
FRONTIER = "frontier-model"
PRICES = {"anthropic:frontier-model": {"input_per_million": 5.0, "output_per_million": 25.0},
          "anthropic:small-a": {"input_per_million": 1.0, "output_per_million": 5.0},
          "anthropic:small-b": {"input_per_million": 1.0, "output_per_million": 5.0}}
GUARANTEE = {"certified": True, "epsilon": 0.04, "delta": 0.05, "bound": 0.03, "statement": "made up for a test"}


def policy(**changes) -> dict:
    values = {"version": 1, "tenant": "acme", "task_type": "quiz", "frontier": FRONTIER,
              "stages": [{"models": ["small-a"], "check": "none"}], "guarantee": GUARANTEE}
    values.update(changes)
    return values


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "key.txt").write_text(f"# a comment\nOTHER=1\nANTHROPIC_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    return tmp_path


def run(folder, *extra, key_file="key.txt") -> int:
    return kit.main(["--key-file", str(folder / key_file), "--frontier", FRONTIER, "--prices", json.dumps(PRICES),
                     "--deploy-dir", str(folder / "deploy"), *extra])


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def env_of(folder) -> dict:
    return kit.parse_env((folder / "deploy" / "tine.env").read_text(encoding="utf-8"))


def load(folder, monkeypatch) -> ProxyConfig:
    """The written files, loaded as the container would load them."""
    for name in set(re.findall(r"TINE_[A-Z0-9_]+", (ROOT / "tine" / "proxy.py").read_text(encoding="utf-8"))):
        monkeypatch.delenv(name, raising=False)
    for name, value in env_of(folder).items():
        monkeypatch.setenv(name, value.replace("/config/", str(folder / "deploy" / "config") + "/"))
    config = ProxyConfig.from_env()
    config.validate()
    return config


def test_writes_an_env_file_and_a_token_without_printing_either_secret(folder, capsys, monkeypatch):
    assert run(folder) == 0
    printed = "".join(capsys.readouterr())
    deploy = folder / "deploy"
    token = (deploy / "client-token.txt").read_text(encoding="utf-8").strip()
    assert FAKE_KEY not in printed and FAKE_KEY[:12] not in printed
    assert token not in printed and f"...{token[-4:]}" in printed and len(token) >= 40
    assert mode(deploy / "tine.env") == 0o600 and mode(deploy / "client-token.txt") == 0o600
    assert (deploy / "config").is_dir()
    assert sorted(path.name for path in deploy.iterdir()) == ["client-token.txt", "config", "tine.env"]
    assert "pass the proxy's start-up validation" in printed

    values = env_of(folder)
    assert values["TINE_ANTHROPIC_API_KEY"] == FAKE_KEY
    assert json.loads(values["TINE_TENANT_KEYS"]) == {token: "local-test"}
    assert json.loads(values["TINE_MODEL_PRICES"]) == PRICES
    assert values["TINE_LEDGER_PATH"] == "/data/ledger.sqlite3"
    config = load(folder, monkeypatch)
    assert config.anthropic_frontier_model == FRONTIER and config.tenant_keys == {token: "local-test"}
    assert config.route_policies == {} and config.shadow_policies == {}


def test_the_env_file_follows_the_format_rules_of_the_example(folder):
    """One NAME=value per line, no quotes, no '$' and no ' #': the rules in
    deploy/tine.env.example, which compose and 'docker run --env-file' both read alike."""
    assert run(folder, "--answer-pattern", r"Answer: ([A-D])") == 0
    for line in (folder / "deploy" / "tine.env").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*=.+", line)
        value = line.partition("=")[2]
        assert value == value.strip() and value[0] not in "'\"" and "$" not in value and " #" not in value


def test_an_existing_env_file_is_kept_without_force(folder, capsys):
    assert run(folder) == 0
    deploy = folder / "deploy"
    before = {path.name: path.read_bytes() for path in deploy.iterdir() if path.is_file()}
    capsys.readouterr()
    assert run(folder) == 1
    refusal = "".join(capsys.readouterr())
    assert "already exist" in refusal and "--force" in refusal and FAKE_KEY not in refusal
    assert {path.name: path.read_bytes() for path in deploy.iterdir() if path.is_file()} == before

    assert run(folder, "--force") == 0
    assert (deploy / "client-token.txt").read_bytes() != before["client-token.txt"], "--force makes a new token"
    assert mode(deploy / "tine.env") == 0o600 and mode(deploy / "client-token.txt") == 0o600
    assert sorted(path.name for path in deploy.iterdir()) == ["client-token.txt", "config", "tine.env"]
    replaced = "".join(capsys.readouterr())
    assert "docker compose up -d --force-recreate" in replaced and FAKE_KEY not in replaced


def test_force_tightens_a_file_that_was_left_readable(folder):
    deploy = folder / "deploy"
    deploy.mkdir()
    (deploy / "tine.env").write_text("TINE_TENANT_KEYS={}\n", encoding="utf-8")
    (deploy / "tine.env").chmod(0o644)
    assert run(folder, "--force") == 0
    assert mode(deploy / "tine.env") == 0o600


@pytest.mark.parametrize("line,name", [
    ("TINE_ANTHROPIC_API_KEY={key}", "TINE_ANTHROPIC_API_KEY"),
    ("export ANTHROPIC_API_KEY=\"{key}\"", "ANTHROPIC_API_KEY"),
    ("  ANTHROPIC_API_KEY = '{key}'  ", "ANTHROPIC_API_KEY"),
])
def test_key_file_forms(folder, capsys, line, name):
    (folder / "other.txt").write_text(line.format(key=FAKE_KEY) + "\n", encoding="utf-8")
    assert run(folder, key_file="other.txt") == 0
    assert env_of(folder)["TINE_ANTHROPIC_API_KEY"] == FAKE_KEY
    printed = "".join(capsys.readouterr())
    assert f"the {name} line" in printed and FAKE_KEY not in printed


@pytest.mark.parametrize("content,expected", [
    ("OPENAI_API_KEY=some-other-value-1234\n", "has no line"),
    ("ANTHROPIC_API_KEY=\n", "has no line"),
    ("ANTHROPIC_API_KEY=REPLACE_WITH_YOUR_ANTHROPIC_KEY\n", "placeholder"),
    ("ANTHROPIC_API_KEY=bad$key-with-a-dollar-" + "x" * 20 + "\n", "cannot carry"),
    ("ANTHROPIC_API_KEY=two words-" + "x" * 20 + "\n", "cannot carry"),
])
def test_a_key_file_without_a_usable_key_is_refused_and_not_echoed(folder, capsys, content, expected):
    (folder / "bad.txt").write_text(content, encoding="utf-8")
    assert run(folder, key_file="bad.txt") == 1
    printed = "".join(capsys.readouterr())
    assert expected in printed
    value = content.strip().partition("=")[2]
    assert not value or value not in printed
    assert not (folder / "deploy" / "tine.env").exists()
    assert run(folder, key_file="missing.txt") == 1


def test_route_policy_is_written_where_the_container_reads_it(folder, capsys, monkeypatch):
    (folder / "policy.json").write_text(json.dumps(policy()), encoding="utf-8")
    assert run(folder, "--route-policy", str(folder / "policy.json"), "--audit-rate", "0.1") == 0
    printed = "".join(capsys.readouterr())
    written = folder / "deploy" / "config" / "route-policies.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {"acme": {"anthropic": policy()}}
    assert mode(written) == 0o644, "uid 10001 in the container must be able to read it"
    values = env_of(folder)
    assert values["TINE_ROUTE_POLICIES"] == "/config/route-policies.json" and values["TINE_AUDIT_RATE"] == "0.1"
    assert "TINE_SHADOW_POLICIES" not in values
    token = (folder / "deploy" / "client-token.txt").read_text(encoding="utf-8").strip()
    assert json.loads(values["TINE_TENANT_KEYS"]) == {token: "acme"}, "the token is for the policy's tenant"
    config = load(folder, monkeypatch)
    assert config.route_policies == {"acme": {"anthropic": policy()}} and config.audit_rate == 0.1
    assert "pass the proxy's start-up validation" in printed and "route policy for tenant 'acme'" in printed


def test_the_proxys_own_policy_file_form_is_accepted(folder, monkeypatch):
    nested = {"acme": {"anthropic": policy()}, "other": {"openai": policy(tenant="other")}}
    (folder / "nested.json").write_text(json.dumps(nested), encoding="utf-8")
    assert run(folder, "--route-policy", str(folder / "nested.json")) == 0
    assert load(folder, monkeypatch).route_policies == {"acme": {"anthropic": policy()}}


def test_shadow_policy_needs_a_cap_before_it_samples(folder, capsys, monkeypatch):
    (folder / "policy.json").write_text(json.dumps(policy()), encoding="utf-8")
    shadow = ["--shadow-policy", str(folder / "policy.json")]
    assert run(folder, *shadow, "--shadow-sample-rate", "0.5") == 1
    assert "--shadow-max-cost-usd" in "".join(capsys.readouterr())
    assert not (folder / "deploy" / "tine.env").exists()

    assert run(folder, *shadow, "--shadow-sample-rate", "0.5", "--shadow-max-cost-usd", "2") == 0
    config = load(folder, monkeypatch)
    assert config.shadow_policies == {"acme": {"anthropic": policy()}}
    assert config.shadow_sample_rate == 0.5 and config.shadow_max_cost_usd == 2.0
    assert env_of(folder)["TINE_SHADOW_POLICIES"] == "/config/shadow-policies.json"

    assert run(folder, *shadow, "--force") == 0
    printed = "".join(capsys.readouterr())
    assert "shadow mode stays off" in printed and load(folder, monkeypatch).shadow_sample_rate == 0


def test_shadow_sampling_needs_a_price_for_every_stage_model(folder, capsys):
    (folder / "policy.json").write_text(json.dumps(policy(stages=[{"models": ["unpriced"], "check": "none"}])),
                                        encoding="utf-8")
    assert run(folder, "--shadow-policy", str(folder / "policy.json"), "--shadow-sample-rate", "1",
               "--shadow-max-cost-usd", "1") == 1
    assert "anthropic:unpriced" in "".join(capsys.readouterr())
    # A route policy may run with an unpriced model: its costs are then unknown, and the output says so.
    assert run(folder, "--route-policy", str(folder / "policy.json")) == 0
    assert "recorded as unknown" in "".join(capsys.readouterr())


@pytest.mark.parametrize("changes,arguments,expected", [
    ({"frontier": "another-model"}, [], "frontier"),
    ({"tenant": None}, [], "names no tenant"),
    ({}, ["--tenant", "someone-else"], "frozen for tenant 'acme'"),
    ({"stages": [{"models": ["small-a", "small-b"], "check": "agreement"}]}, [], "--answer-pattern"),
    ({"stages": [{"models": ["small-a"], "check": "tests+judge:unconfigured"}]}, [], "would refuse to start"),
    ({"guarantee": {"certified": True, "statement": "no numbers"}}, [], "would refuse to start"),
])
def test_a_policy_the_proxy_would_refuse_changes_nothing(folder, capsys, changes, arguments, expected):
    (folder / "policy.json").write_text(json.dumps(policy(**changes)), encoding="utf-8")
    assert run(folder, "--route-policy", str(folder / "policy.json"), *arguments) == 1
    printed = "".join(capsys.readouterr())
    assert expected in printed and FAKE_KEY not in printed
    deploy = folder / "deploy"
    assert not deploy.exists() or [path.name for path in deploy.rglob("*") if path.is_file()] == []


def test_an_agreement_route_policy_passes_with_an_answer_reader(folder, monkeypatch):
    agreeing = policy(stages=[{"models": ["small-a", "small-b"], "check": "agreement"}])
    (folder / "policy.json").write_text(json.dumps(agreeing), encoding="utf-8")
    assert run(folder, "--route-policy", str(folder / "policy.json"), "--answer-pattern", r"Answer: ([A-D])") == 0
    assert load(folder, monkeypatch).answer_pattern == r"Answer: ([A-D])"
    # A pattern compose would rewrite is refused, not written.
    assert run(folder, "--route-policy", str(folder / "policy.json"), "--answer-pattern", "([A-D])$", "--force") == 1
    assert load(folder, monkeypatch).answer_pattern == r"Answer: ([A-D])", "the refused run changed nothing"


@pytest.mark.parametrize("prices,expected", [
    ("not json", "not a JSON object"),
    ("{}", "must be a JSON object"),
    ('{"frontier-model":{"input_per_million":1}}', "provider:model"),
    ('{"anthropic:frontier-model":{"per_token":1}}', "USD rates"),
    ('{"anthropic:frontier-model":{"input_per_million":-1}}', "USD rates"),
])
def test_prices_are_checked(folder, capsys, prices, expected):
    assert kit.main(["--key-file", str(folder / "key.txt"), "--frontier", FRONTIER, "--prices", prices,
                     "--deploy-dir", str(folder / "deploy")]) == 1
    assert expected in "".join(capsys.readouterr())


def test_prices_may_come_from_a_file(folder, monkeypatch):
    (folder / "prices.json").write_text(json.dumps(PRICES), encoding="utf-8")
    assert kit.main(["--key-file", str(folder / "key.txt"), "--frontier", FRONTIER,
                     "--prices", str(folder / "prices.json"), "--deploy-dir", str(folder / "deploy")]) == 0
    assert load(folder, monkeypatch).prices == PRICES


def test_says_so_when_the_package_cannot_be_imported(folder, capsys, monkeypatch):
    """With a Python that lacks the proxy's dependencies the files are still written,
    and the output says the start-up check was not made."""
    monkeypatch.setitem(sys.modules, "tine.proxy", None)  # makes "from tine.proxy import ..." raise ImportError
    assert run(folder) == 0
    printed = "".join(capsys.readouterr())
    assert "NOT checked" in printed and "pass the proxy's start-up validation" not in printed
    assert mode(folder / "deploy" / "tine.env") == 0o600


def test_the_check_leaves_the_environment_as_it_was(folder, monkeypatch):
    monkeypatch.setenv("TINE_ANTHROPIC_FRONTIER_MODEL", "set-in-the-shell")
    monkeypatch.setenv("UNRELATED", "kept")
    assert run(folder) == 0
    assert os.environ["TINE_ANTHROPIC_FRONTIER_MODEL"] == "set-in-the-shell" and os.environ["UNRELATED"] == "kept"
    assert "TINE_TENANT_KEYS" not in os.environ and "TINE_ANTHROPIC_API_KEY" not in os.environ


@pytest.mark.parametrize("name", ["container_env.py", "container_smoke.py"])
def test_the_kit_imports_only_the_standard_library(name):
    """Both scripts must run with a plain python3, without the development environment."""
    tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
    top_level = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top_level |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert top_level <= set(sys.stdlib_module_names), top_level - set(sys.stdlib_module_names)
    # The Mac's own python3 is 3.9: nothing newer than its grammar.
    ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), feature_version=(3, 9))
