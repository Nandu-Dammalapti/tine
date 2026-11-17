#!/usr/bin/env python3
"""Write a tenant's savings report as one self-contained HTML page.

Reads a ledger SQLite file (``TINE_LEDGER_PATH`` of the proxy) for one tenant, builds
the same two reports the proxy serves at ``GET /v1/savings`` and ``GET /v1/shadow``,
and hands them to ``tine.report_page.render``. The shadow report is built only when
the ledger has a shadow table. A policy JSON file adds a plain-words description of
the policy the figures came from.

``--policy`` takes a single frozen policy file, as ``tine freeze-policy`` writes it,
or the file the proxy itself reads (``TINE_ROUTE_POLICIES`` or
``TINE_SHADOW_POLICIES``: tenant to provider to policy). From the proxy's file it
describes ``--tenant``'s policy. When that tenant has a policy for more than one
provider, ``--provider`` says which.

The ledger is opened read-only. ``Ledger`` and ``ShadowStore`` create tables and
switch the journal mode when constructed, which a report must not do to a database a
running proxy owns, so this script attaches a read-only connection to them without
running their constructors. Their own ``report`` methods still do all the arithmetic.
A running proxy keeps its newest rows in ``ledger.sqlite3-wal``. To report on a copy,
make it with ``python -m tine backup-ledger``: the main file copied alone lacks them.

``--report-json`` builds the page from a saved report instead of a ledger: a JSON
file with a ``shadow_report`` and/or ``ledger_report`` object, such as the output of
``scripts/shadow_replay.py``. No model is called and nothing is sent anywhere.

``--note`` puts one line under the headline, for what the reports cannot say about
themselves (for example that the figures are a replay of benchmark answers). When
a policy is given with a ledger, the script compares it with the policy each
shadowed request was run with and adds a note of its own if they differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tine.ledger import Ledger  # noqa: E402
from tine.policy import FREEZE_EDITED, freeze_state, validate_policy  # noqa: E402
from tine.report_page import render  # noqa: E402
from tine.shadow import ShadowStore  # noqa: E402


def open_read_only(path: Path) -> Ledger:
    """A ``Ledger`` over a read-only connection. Nothing in the file can change."""
    if not path.is_file():
        raise FileNotFoundError(f"No ledger file at {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    ledger = Ledger.__new__(Ledger)
    ledger._lock = threading.RLock()
    ledger._db = connection
    return ledger


def _has_table(ledger: Ledger, name: str) -> bool:
    row = ledger._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def reports_from_ledger(path: Path, tenant: str, epsilon: float | None = None) -> tuple[dict, dict | None]:
    """The tenant's ledger report, and its shadow report if the ledger has shadow rows
    to read. A ledger that never ran shadow mode has no shadow table and gives None.
    ``epsilon``, the described policy's quality limit, is what differing answers are
    checked against, as ``GET /v1/shadow`` does with the policy it runs."""
    ledger = open_read_only(path)
    try:
        if not _has_table(ledger, "requests"):
            raise ValueError(f"{path} is not a Tine ledger: it has no requests table. If it was copied from a "
                             "running proxy, the rows were in the ledger.sqlite3-wal file beside it: make the copy "
                             "with 'python -m tine backup-ledger' instead")
        shadow_report = None
        if _has_table(ledger, "shadow_runs"):
            store = ShadowStore.__new__(ShadowStore)
            store._ledger = ledger
            shadow_report = store.report(tenant, epsilon=epsilon)
        return ledger.report(tenant), shadow_report
    finally:
        ledger.close()


def policy_mismatch_note(path: Path, tenant: str, policy: dict) -> str | None:
    """A sentence for the page when shadowed requests were run with another policy.

    Each shadow row stores a digest of the policy it was run with. The shadow report
    adds all of a tenant's rows together, whatever their policy, so a page that
    describes one policy beside those figures must say when they do not belong to it.
    The digest is computed as ``ShadowStore.record`` computes it.
    """
    ledger = open_read_only(path)
    try:
        if not _has_table(ledger, "shadow_runs"):
            return None
        rows = ledger._db.execute("SELECT policy_sha256, COUNT(*) AS runs FROM shadow_runs WHERE tenant=? "
                                  "GROUP BY policy_sha256", (tenant,)).fetchall()
    finally:
        ledger.close()
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    total = sum(row["runs"] for row in rows)
    other = sum(row["runs"] for row in rows if row["policy_sha256"] != digest)
    if not other:
        return None
    return (f"{other:,} of the {total:,} shadowed requests were run with a different policy file from the one "
            "described on this page. The shadow figures add all of them together.")


def reports_from_json(path: Path) -> tuple[dict | None, dict | None]:
    """Saved reports from a JSON file with ``ledger_report`` and/or ``shadow_report``."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or not ({"ledger_report", "shadow_report"} & set(saved)):
        raise ValueError(f"{path} holds neither a ledger_report nor a shadow_report")
    for name in ("ledger_report", "shadow_report"):
        if saved.get(name) is not None and not isinstance(saved[name], dict):
            raise ValueError(f"{name} in {path} is not a JSON object")
    return saved.get("ledger_report"), saved.get("shadow_report")


def is_deploy_format(document) -> bool:
    """Whether a policy file has the shape the proxy reads: tenant to provider to policy.
    A single policy cannot: its ``version`` and ``stages`` are not objects."""
    return isinstance(document, dict) and all(
        isinstance(by_provider, dict) and all(isinstance(policy, dict) for policy in by_provider.values())
        for by_provider in document.values())


def _names(names) -> str:
    return ", ".join(repr(name) for name in sorted(names))


def select_policy(document: dict, tenant: str, provider: str | None, path: Path) -> tuple[dict, str]:
    """One tenant's policy from a file in the proxy's format, and its provider. Every
    refusal names the tenants or providers the file does hold."""
    shape = "It is in the proxy's format (tenant to provider to policy)"
    if tenant not in document:
        held = f"holds policies for {_names(document)}" if document else "holds no tenants"
        raise ValueError(f"{path} holds no policy for tenant {tenant!r}. {shape} and {held}")
    by_provider = document[tenant]
    if not by_provider:
        raise ValueError(f"{path} holds no policy for tenant {tenant!r}: its entry there is empty")
    if provider is None and len(by_provider) > 1:
        raise ValueError(f"{path} holds {len(by_provider)} policies for tenant {tenant!r}, for providers "
                         f"{_names(by_provider)}. Choose one with --provider")
    provider = next(iter(by_provider)) if provider is None else provider
    if provider not in by_provider:
        raise ValueError(f"{path} holds no {provider!r} policy for tenant {tenant!r}. {shape} and holds that "
                         f"tenant's policy for {_names(by_provider)}")
    return by_provider[provider], provider


def read_policy(path: Path, tenant: str, provider: str | None = None) -> dict:
    """A valid policy that names this tenant, from a single policy file or from the
    proxy's own policies file. A policy for one tenant must not be shown beside
    another tenant's figures. A file edited after freezing still loads; the page says
    so, and so does a warning here. A policy taken from the proxy's file is not
    changed, so its digest is the one shadow rows carry."""
    policy = json.loads(path.read_text(encoding="utf-8"))
    if is_deploy_format(policy):
        policy, provider = select_policy(policy, tenant, provider, path)
        try:
            validate_policy(policy)
        except ValueError as error:
            raise ValueError(f"{path}: the {provider} policy of tenant {tenant!r}: {error}") from None
    elif provider is not None:
        raise ValueError(f"--provider chooses among a tenant's policies in the proxy's policies file (tenant to "
                         f"provider to policy), and {path} is not one")
    elif not isinstance(policy, dict) or "version" not in policy:
        raise ValueError(f"{path} is neither one frozen policy (an object with version, frontier and stages) nor "
                         "the proxy's policies file (tenant to provider to policy)")
    else:
        validate_policy(policy)
    if policy.get("tenant") not in (None, tenant):
        raise ValueError(f"The policy file names tenant {policy.get('tenant')!r}, not {tenant!r}")
    if freeze_state(policy) == FREEZE_EDITED:
        print(f"savings_report: warning: {path} was {FREEZE_EDITED}: it does not match its freeze digest",
              file=sys.stderr)
    return policy


def build(*, tenant: str, ledger: Path | None = None, report_json: Path | None = None,
          policy: Path | None = None, provider: str | None = None, generated_at: str | None = None,
          note: str | None = None) -> str:
    """The HTML page for one tenant from a ledger file or a saved report. ``provider``
    picks the tenant's policy when ``policy`` is the proxy's file and holds several."""
    if (ledger is None) == (report_json is None):
        raise ValueError("Give exactly one of a ledger file and a saved report")
    if provider is not None and policy is None:
        raise ValueError("--provider chooses a policy from the --policy file: give --policy too")
    described = read_policy(policy, tenant, provider) if policy else None
    guarantee = described.get("guarantee") if described and described["stages"] else None
    epsilon = guarantee.get("epsilon") if isinstance(guarantee, dict) else None
    epsilon = epsilon if isinstance(epsilon, (int, float)) and not isinstance(epsilon, bool) else None
    ledger_report, shadow_report = (reports_from_ledger(ledger, tenant, epsilon) if ledger
                                    else reports_from_json(report_json))
    for report in (ledger_report, shadow_report):
        if isinstance(report, dict) and report.get("tenant") not in (None, tenant):
            raise ValueError(f"The saved report is for tenant {report.get('tenant')!r}, not {tenant!r}")
    when = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    notes = [note] if note else []
    if described is not None and ledger:
        mismatch = policy_mismatch_note(ledger, tenant, described)
        if mismatch:
            print(f"savings_report: warning: {mismatch}", file=sys.stderr)
            notes.append(mismatch)
    return render(ledger_report, shadow_report, tenant=tenant, generated_at=when, policy=described,
                  note=" ".join(notes) or None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ledger", type=Path, help="ledger SQLite file; opened read-only")
    source.add_argument("--report-json", type=Path,
                        help="JSON with a shadow_report and/or ledger_report object, in place of a ledger")
    parser.add_argument("--tenant", required=True,
                        help="tenant the page is for; a saved report or policy for another tenant is refused")
    parser.add_argument("--policy", type=Path,
                        help="policy JSON to describe on the page: one frozen policy, or the proxy's own file "
                             "(TINE_ROUTE_POLICIES or TINE_SHADOW_POLICIES), from which --tenant's policy is taken")
    parser.add_argument("--provider",
                        help="with the proxy's policies file: which of the tenant's policies to describe, when it "
                             "has one for more than one provider (openai, anthropic)")
    parser.add_argument("--generated-at", help="time to print on the page; default is now, in UTC")
    parser.add_argument("--note", help="one line to show under the headline, for example where the figures came from")
    parser.add_argument("--output", type=Path, required=True, help="HTML file to write")
    args = parser.parse_args(argv)
    try:
        page = build(tenant=args.tenant, ledger=args.ledger, report_json=args.report_json, policy=args.policy,
                     provider=args.provider, generated_at=args.generated_at, note=args.note)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(page, encoding="utf-8")
    except (OSError, ValueError, RecursionError, sqlite3.Error) as error:
        raise SystemExit(f"savings_report: {error}") from error
    print(f"Wrote {args.output} ({len(page.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
