#!/usr/bin/env python3
"""Write the env file and a tenant token for testing the Tine container on this machine.

Writes, under ``deploy/``:

``tine.env``            what ``docker compose`` hands the container (mode 600): the provider
                        key copied from ``--key-file``, a new random tenant token, the
                        frontier model, the prices, and the policy settings asked for.
``client-token.txt``    the tenant token alone (mode 600). ``scripts/container_smoke.py``
                        reads it. Only its last four characters are ever printed.
``config/*.json``       with ``--route-policy`` or ``--shadow-policy``: that policy under the
                        tenant and the Anthropic provider, where the container reads it.

The provider key is never printed. An existing ``deploy/tine.env`` is left alone unless
``--force`` is given. Nothing is put in place until every file has passed: the env file
is read back the way compose reads it and loaded into ``ProxyConfig``, as the proxy
does at start-up. That last step needs the ``tine`` package and its dependencies; with
a Python that cannot import them the files are still written and the output says the
check was skipped.

Standard library only. No request is sent anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The provider key's name in the key file. The first one found is used.
KEY_NAMES = ("TINE_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
RATE_NAMES = {"input_per_million", "output_per_million", "cached_input_per_million", "cache_write_per_million"}
PROVIDER = "anthropic"


class KitError(Exception):
    """Something the operator has to fix. The message never holds a key or a token."""


def read_key(path: Path) -> tuple[str, str]:
    """The provider key in ``path`` and the name of the line it came from."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise KitError(f"--key-file {path} could not be read: {type(error).__name__}") from None
    found = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*", line)
        if match and not line.lstrip().startswith("#"):
            value = match.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            found.setdefault(match.group(1), value)
    for name in KEY_NAMES:
        if found.get(name):
            if found[name].startswith("REPLACE_WITH"):
                raise KitError(f"--key-file {path}: the {name} line still holds a placeholder")
            if not env_safe(found[name]) or re.search(r"\s", found[name]):
                raise KitError(f"--key-file {path}: the {name} value has a character an env file cannot carry "
                               "(a space, a quote, '$' or '#')")
            return name, found[name]
    raise KitError(f"--key-file {path} has no line {KEY_NAMES[1]}=... or {KEY_NAMES[0]}=...")


def env_safe(value: str) -> bool:
    """Whether compose and ``docker run --env-file`` both read ``value`` back unchanged.

    Compose expands ``$NAME``, cuts a value at `` #`` and strips quotes and outer spaces;
    ``docker run`` does none of that. Only values both agree on are written.
    """
    return (bool(value) and value == value.strip() and value[0] not in "'\"" and "$" not in value
            and " #" not in value and value.isprintable())


def load_prices(argument: str) -> dict:
    """Prices from a JSON string or a JSON file, checked against the proxy's rules."""
    text = argument
    if not argument.lstrip().startswith("{"):
        try:
            text = Path(argument).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise KitError(f"--prices {argument}: not a JSON object and not a readable file "
                           f"({type(error).__name__})") from None
    try:
        prices = json.loads(text)
    except ValueError as error:
        raise KitError(f"--prices is not valid JSON: {error}") from None
    example = '{"anthropic:MODEL":{"input_per_million":3.0,"output_per_million":15.0}}'
    if not isinstance(prices, dict) or not prices:
        raise KitError(f"--prices must be a JSON object such as {example}")
    for key, rates in prices.items():
        if not re.fullmatch(r"(anthropic|openai):.+", key):
            raise KitError(f"--prices: {key!r} must be written provider:model, such as {example}")
        if not isinstance(rates, dict) or not rates or set(rates) - RATE_NAMES or any(
                isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0
                for rate in rates.values()):
            raise KitError(f"--prices: {key!r} needs USD rates per million tokens from {sorted(RATE_NAMES)}")
    return prices


def load_policy(path: Path, tenant: str | None, frontier: str) -> tuple[str, dict]:
    """The frozen policy in ``path`` and the tenant it is for.

    ``path`` holds one policy, as ``tine freeze-policy`` writes it, or the proxy's own
    form (tenant, then provider, then policy), from which the tenant's Anthropic
    policy is taken.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise KitError(f"{path} could not be loaded as JSON: {type(error).__name__}: {error}") from None
    if not isinstance(data, dict):
        raise KitError(f"{path} does not hold a policy")
    if "stages" not in data:
        names = [name for name, by_provider in data.items() if isinstance(by_provider, dict) and PROVIDER in by_provider]
        chosen = tenant if tenant is not None else names[0] if len(names) == 1 else None
        if chosen not in names:
            raise KitError(f"{path} holds neither one policy nor a tenant's {PROVIDER} policy"
                           + (f" for tenant {tenant!r}" if tenant is not None else "; name the tenant with --tenant"))
        data = data[chosen][PROVIDER]
        if not isinstance(data, dict) or "stages" not in data:
            raise KitError(f"{path} does not hold a policy for tenant {chosen!r}")
    named = data.get("tenant")
    if not isinstance(named, str) or not named:
        raise KitError(f"{path} names no tenant. The proxy runs a policy only for the tenant it was frozen for: "
                       "freeze it with 'tine freeze-policy --tenant NAME'")
    if tenant is not None and tenant != named:
        raise KitError(f"{path} was frozen for tenant {named!r}, not {tenant!r}")
    if data.get("frontier") != frontier:
        raise KitError(f"{path} has frontier {data.get('frontier')!r}, but --frontier is {frontier!r}. "
                       "A policy's frontier must be the provider's frontier model")
    return named, data


def stage_models(policy: dict) -> list[str]:
    stages = policy.get("stages") if isinstance(policy.get("stages"), list) else []
    return sorted({model for stage in stages if isinstance(stage, dict)
                   for model in stage.get("models") or [] if isinstance(model, str)})


def has_agreement(policy: dict) -> bool:
    stages = policy.get("stages") if isinstance(policy.get("stages"), list) else []
    return any(isinstance(stage, dict) and stage.get("check") == "agreement" for stage in stages)


def rate(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1")
    return number


def amount(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite amount of 0 or more")
    return number


def parse_env(text: str) -> dict[str, str]:
    """``NAME=value`` lines as compose reads this script's own output: the value is
    everything after the first ``=``. Comment and blank lines are skipped."""
    values = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            values[name] = value
    return values


def check_like_the_proxy(env_text: str, config_files: dict[str, Path]) -> str | None:
    """Load the env file's text into ``ProxyConfig`` and validate it, as the proxy does
    at start-up. ``config_files`` maps a path inside the container to the file on this
    machine that will be mounted there.

    Returns None when it passed, or the reason the check could not be made. Raises
    ``KitError`` when the proxy would refuse to start.
    """
    values = parse_env(env_text)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from tine.proxy import ProxyConfig
    except Exception as error:  # noqa: BLE001 - a missing dependency, or a Python too old for the package
        return f"{type(error).__name__}: {error}"
    secret = [value for name, value in values.items() if re.search(r"KEY|TOKEN|SECRET", name)]
    saved = dict(os.environ)
    try:
        for name in list(os.environ):
            if name.startswith("TINE_"):
                del os.environ[name]
        for name, value in values.items():
            os.environ[name] = str(config_files[value]) if value in config_files else value
        ProxyConfig.from_env().validate()
    except Exception as error:  # noqa: BLE001 - whatever stops the proxy from starting
        message = f"{type(error).__name__}: {error}"
        for value in secret:
            message = message.replace(value, "<redacted>")
        for name in json.loads(values.get("TINE_TENANT_KEYS", "{}")):
            message = message.replace(name, "<redacted>")
        raise KitError(f"the proxy would refuse to start with this configuration: {message}") from None
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return None


def write_new(path: Path, text: str, mode: int) -> None:
    """Create ``path`` with exactly ``mode``. It must not exist."""
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(handle, mode)
        os.write(handle, text.encode("utf-8"))
    finally:
        os.close(handle)


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def build(args) -> tuple[list[tuple[Path, str, int]], str, list[str], list[str]]:
    """Everything to write as (path, text, mode), the token, and lines to print and to warn."""
    deploy = args.deploy_dir
    key_name, key = read_key(args.key_file)
    prices = load_prices(args.prices)
    if not args.frontier or not env_safe(args.frontier):
        raise KitError("--frontier must be a model name without quotes, '$' or ' #'")
    policy_path = args.route_policy or args.shadow_policy
    kind = "route" if args.route_policy else "shadow" if args.shadow_policy else None
    tenant, policy = (load_policy(policy_path, args.tenant, args.frontier) if policy_path
                      else (args.tenant or "local-test", None))
    token = secrets.token_urlsafe(32)
    notes, warnings = [], []

    lines = [
        "# Written by scripts/container_env.py for a local test of the container.",
        "# It holds a provider key and a tenant token. Never commit it. Keep its mode at 600.",
        "# One NAME=value per line, no quotes. docs/DEPLOY.md, Test the container on your own machine.",
        "TINE_TENANT_KEYS=" + json.dumps({token: tenant}, separators=(",", ":")),
        f"TINE_ANTHROPIC_API_KEY={key}",
        f"TINE_ANTHROPIC_FRONTIER_MODEL={args.frontier}",
        "TINE_LEDGER_PATH=/data/ledger.sqlite3",
        "TINE_MODEL_PRICES=" + json.dumps(prices, separators=(",", ":")),
    ]
    files: list[tuple[Path, str, int]] = []
    sampling = False
    if kind:
        inside = f"/config/{kind}-policies.json"
        lines.append(f"TINE_{kind.upper()}_POLICIES={inside}")
        files.append((deploy / "config" / f"{kind}-policies.json",
                      json.dumps({tenant: {PROVIDER: policy}}, indent=2) + "\n", 0o644))
        notes.append(f"{kind} policy for tenant {tenant!r} on {PROVIDER}, from {shown(policy_path)}")
        if has_agreement(policy) and not (policy.get("answer_pattern") or args.answer_pattern):
            if kind == "route":
                raise KitError(f"{policy_path} has agreement stages and no answer reader, so the proxy would refuse "
                               "to start. Freeze it with 'tine freeze-policy --answer-pattern REGEX', or pass "
                               "--answer-pattern")
            warnings.append("the shadow policy has agreement stages and no answer reader: whole replies are compared")
    if kind == "route":
        if args.audit_rate is not None:
            lines.append(f"TINE_AUDIT_RATE={args.audit_rate}")
    elif args.audit_rate:
        warnings.append("--audit-rate does nothing without --route-policy; it was left out")
    if kind == "shadow":
        sample = args.shadow_sample_rate or 0.0
        sampling = sample > 0 and bool(policy.get("stages"))
        if sampling and args.shadow_max_cost_usd is None:
            raise KitError("shadow calls cost money: give --shadow-max-cost-usd, the most this tenant's shadow "
                           "calls may spend, with --shadow-sample-rate above 0")
        lines.append(f"TINE_SHADOW_SAMPLE_RATE={sample}")
        if args.shadow_max_cost_usd is not None:
            lines.append(f"TINE_SHADOW_MAX_COST_USD={args.shadow_max_cost_usd}")
        if not sampling:
            warnings.append("shadow mode stays off: pass --shadow-sample-rate above 0 with --shadow-max-cost-usd")
    elif args.shadow_sample_rate or args.shadow_max_cost_usd is not None:
        warnings.append("--shadow-sample-rate and --shadow-max-cost-usd do nothing without --shadow-policy; "
                        "they were left out")
    if args.answer_pattern:
        try:
            re.compile(args.answer_pattern)
        except re.error as error:
            raise KitError(f"--answer-pattern is not a valid regular expression: {error}") from None
        lines.append(f"TINE_ANSWER_PATTERN={args.answer_pattern}")

    priced = {name for name, rates in prices.items() if {"input_per_million", "output_per_million"} <= set(rates)}
    unpriced = [model for model in [args.frontier, *(stage_models(policy) if policy else [])]
                if f"{PROVIDER}:{model}" not in priced]
    if unpriced and sampling:
        raise KitError(f"shadow mode's spending cap counts known costs only: --prices needs an input and an output "
                       f"price for {[f'{PROVIDER}:{model}' for model in unpriced]}")
    if unpriced:
        warnings.append(f"no input and output price for {[f'{PROVIDER}:{model}' for model in unpriced]}: "
                        "their costs will be recorded as unknown")

    for line in lines:
        value = line.partition("=")[2]
        if not line.startswith("#") and not env_safe(value):
            raise KitError(f"{line.partition('=')[0]} has a value an env file cannot carry unchanged: no outer "
                           "spaces, no leading quote, no '$' and no ' #'")
    files.insert(0, (deploy / "tine.env", "\n".join(lines) + "\n", 0o600))
    files.append((deploy / "client-token.txt", token + "\n", 0o600))
    notes.insert(0, f"provider key from the {key_name} line of {shown(args.key_file)} (not shown), "
                    f"tenant {tenant!r}, frontier {args.frontier}")
    return files, token, notes, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--key-file", type=Path, required=True,
                        help="file with a line ANTHROPIC_API_KEY=... or TINE_ANTHROPIC_API_KEY=...; "
                             "the key is copied into the env file and never printed")
    parser.add_argument("--frontier", required=True, help="exact name of the Anthropic frontier model")
    parser.add_argument("--prices", required=True,
                        help="JSON object or a file holding one: provider:model -> USD per million tokens, such as "
                             '\'{"anthropic:MODEL":{"input_per_million":3.0,"output_per_million":15.0}}\'')
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--route-policy", type=Path,
                        help="frozen policy to serve answers through (changes what callers receive)")
    policy.add_argument("--shadow-policy", type=Path, help="frozen policy to run in shadow mode (report only)")
    parser.add_argument("--tenant", help="tenant the token is for; default: the policy's tenant, else local-test")
    parser.add_argument("--audit-rate", type=rate,
                        help="with --route-policy: share of routable requests held out to the frontier")
    parser.add_argument("--shadow-sample-rate", type=rate,
                        help="with --shadow-policy: share of requests to shadow; default 0, shadow mode off")
    parser.add_argument("--shadow-max-cost-usd", type=amount,
                        help="with --shadow-policy: spending cap for shadow calls; needed once the rate is above 0")
    parser.add_argument("--answer-pattern",
                        help="regular expression that reads the comparable answer from a reply, for a policy "
                             "with agreement stages that carries none of its own")
    parser.add_argument("--deploy-dir", type=Path, default=ROOT / "deploy",
                        help="where to write; default: this repository's deploy/")
    parser.add_argument("--force", action="store_true", help="replace an existing tine.env, token and policy file")
    args = parser.parse_args(argv)
    deploy = args.deploy_dir
    pending: list[Path] = []
    try:
        if args.tenant is not None and not args.tenant.strip():
            raise KitError("--tenant must not be empty")
        files, token, notes, warnings = build(args)
        existing = [path for path, _, _ in files if path.exists() or path.is_symlink()]
        if existing and not args.force:
            raise KitError(f"{', '.join(shown(path) for path in existing)} already exist(s). Nothing was changed. "
                           "Pass --force to replace them. A new tenant token is made, so a running container "
                           "must then be recreated: docker compose up -d --force-recreate")
        (deploy / "config").mkdir(parents=True, exist_ok=True)
        # Each file is written beside its place, checked, and only then moved into place,
        # so a refused configuration changes nothing. The temporary names end in ".env",
        # which deploy/.gitignore keeps out of git like the env file itself.
        staged = []
        for path, text, mode in files:
            temporary = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp.env")
            pending.append(temporary)
            write_new(temporary, text, mode)
            staged.append((temporary, path))
        inside = {f"/config/{path.name}": temporary for temporary, path in staged if path.parent.name == "config"}
        skipped = check_like_the_proxy(staged[0][0].read_text(encoding="utf-8"), inside)
        for temporary, path in staged:
            os.replace(temporary, path)
    except KitError as error:
        print(f"container_env: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        # A folder that cannot be written, for instance. The message names a path, never a value.
        print(f"container_env: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        for temporary in pending:
            if temporary.exists():
                temporary.unlink()
    env_path, token_path = files[0][0], files[-1][0]
    print(f"Wrote {shown(env_path)} (mode 600): {notes[0]}")
    print(f"Wrote {shown(token_path)} (mode 600): the tenant token, ending in ...{token[-4:]}")
    for (path, _, _), note in zip(files[1:-1], notes[1:]):
        print(f"Wrote {shown(path)}: {note}")
    if skipped is None:
        print("Checked: the files load into ProxyConfig and pass the proxy's start-up validation.")
    else:
        print("NOT checked against the proxy's start-up validation: this Python cannot import the tine package "
              f"({skipped}). Run this script with the development environment's Python to check.")
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if existing:
        print("The tenant token changed. A container that is already running still has the old one: "
              "docker compose up -d --force-recreate")
    print("Next: docker compose build, docker compose up -d, then python3 scripts/container_smoke.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
