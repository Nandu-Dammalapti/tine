"""The customer-facing savings report, as one self-contained HTML page.

``render`` turns the two reports the proxy already produces (``Ledger.report`` and
``ShadowStore.report``) and, optionally, the frozen policy into a page a customer can
open from a file or receive by email. The page loads nothing: no script, no font, no
image, no link. Styling is one ``<style>`` block, and the page still reads correctly
where a mail client drops it.

The page adds almost no arithmetic of its own. Every cost, saving and share is one the
reports computed, so the page cannot disagree with ``GET /v1/savings`` or
``GET /v1/shadow``. The exceptions are three differences of counts the reports
already hold (accepted answers that were not compared, confirmed successes left out
of the saving, and 1 - delta). Four rules it holds to:

- A value that is missing or null is shown as "not known", never as 0.
- Every value that comes from data is HTML-escaped. A value that should be a number
  and is not is shown as "not known" rather than printed.
- The limits of the figures travel with them: the scope texts of both reports are
  carried word for word into the last section.
- No percentage is rounded in the policy's favour. Savings and shares are rounded
  down, the loss bound and the observed loss are rounded up. Dollar amounts are
  rounded to the nearest shown digit.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any

NOT_KNOWN = "not known"
SMALLEST_SHOWN_USD = 1e-6
ROUNDING = {"nearest": ROUND_HALF_EVEN, "down": ROUND_FLOOR, "up": ROUND_CEILING}


def _esc(value: Any) -> str:
    """Text made safe for both element content and attribute values."""
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | int | None:
    """A finite real number, or None. Booleans and strings are not numbers here.

    An integer too large for a float (JSON allows one of any length) is not a figure
    this page can format, so it is not a number here either.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(float(value)) else None
    except OverflowError:
        return None


def _whole(value: Any) -> int | None:
    """A count, or None when the value is not a whole number."""
    number = _number(value)
    if number is None or number != int(number):
        return None
    return int(number)


def count(value: Any) -> str:
    """A count with thousands separators, or "not known"."""
    number = _whole(value)
    return NOT_KNOWN if number is None else f"{number:,}"


def percent(fraction: Any, *, rounding: str = "nearest", digits: int = 1) -> str:
    """A fraction as a percentage, or "not known".

    ``rounding`` is "nearest", "down" or "up". A saving or a share is shown rounded
    down and a loss bound rounded up, so that no shown figure is better than the one
    it stands for. The value is first rounded to nine decimals, so that float noise
    (89.99999999999999 for 90) does not move a figure by a whole step.
    """
    number = _number(fraction)
    if number is None:
        return NOT_KNOWN
    try:
        scaled = (Decimal(repr(float(number))) * 100).quantize(Decimal("1e-9"), rounding=ROUND_HALF_EVEN)
        shown = scaled.quantize(Decimal(1).scaleb(-digits), rounding=ROUNDING[rounding])
    except InvalidOperation:  # too large to be a share of anything
        return NOT_KNOWN
    return (f"{abs(shown):.{digits}f}" if shown == 0 else f"{shown:.{digits}f}") + "%"


def money(value: Any) -> str:
    """US dollars, readable for a fraction of a cent and for millions.

    One dollar and above: two decimals with thousands separators. Below a dollar:
    three significant digits, because a per-request cost of $0.000127 must not
    print as $0.00. Below a millionth of a dollar the amount is named as such
    instead of printing a long run of zeros.
    """
    number = _number(value)
    if number is None:
        return NOT_KNOWN
    sign, size = ("-" if number < 0 else ""), abs(number)
    if size == 0:
        return "$0.00"
    if size < SMALLEST_SHOWN_USD:
        return "above -$0.000001" if sign else "under $0.000001"
    if size < 1:
        digits = 2 - math.floor(math.log10(size))
        if round(size, digits) < 1:
            text = f"{size:.{digits}f}"
            while text.endswith("0") and len(text.split(".")[1]) > 2:
                text = text[:-1]
            return f"{sign}${text}"
    return f"{sign}${size:,.2f}"


def _money_with_share(amount: Any, fraction: Any) -> str:
    """A saving with its share in brackets when the share is known. The share is
    rounded down: a saving of 53.46% is shown as 53.4%, never as 53.5%."""
    if _number(amount) is None:
        return NOT_KNOWN
    if _number(fraction) is None:
        return money(amount)
    return f"{money(amount)} ({percent(fraction, rounding='down')})"


def _plural(number: int | None, word: str) -> str:
    """"1 request", "3 requests", and "requests" alone when the count is not known."""
    if number is None:
        return f"{word}s"
    return f"{number:,} {word}" if number == 1 else f"{number:,} {word}s"


def _listed(names: list[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _get(report: dict | None, key: str) -> Any:
    return report.get(key) if isinstance(report, dict) else None


def _shadow_has_data(shadow: dict | None) -> bool:
    return (_whole(_get(shadow, "shadowed_requests")) or 0) > 0


def _ledger_has_requests(ledger: dict | None) -> bool:
    return (_whole(_get(ledger, "requests")) or 0) > 0


def _live(ledger: dict | None) -> bool:
    """True when the ledger holds a measured comparison: at least one confirmed
    successful request with answer, routing and baseline costs all known."""
    return ((_whole(_get(ledger, "comparable_verified_requests")) or 0) > 0
            and _number(_get(ledger, "verified_cost_usd")) is not None
            and _number(_get(ledger, "verified_baseline_cost_usd")) is not None)


def _saving_clause(saving: Any, fraction: Any, *, word: str, against: str) -> str:
    """How a cost difference is put in words. A negative saving is called what it is."""
    amount = _number(saving)
    if amount is None:
        return f"the {word} is not known"
    if amount < 0:
        return f"{money(-amount)} more than {against}, so there is no {word}"
    if amount == 0:
        return f"the same as {against}, so there is no {word}"
    return f"a {word} of {_money_with_share(amount, fraction)}"


def _live_headline(ledger: dict) -> str:
    clause = _saving_clause(ledger.get("verified_savings_usd"), ledger.get("savings_fraction"),
                            word="net saving", against="the baseline")
    comparable, recorded = _whole(ledger.get("comparable_verified_requests")), _whole(ledger.get("requests"))
    # The saving is a subtotal. Saying how many requests it leaves out belongs in the
    # same sentence as the saving, not only in the table below it.
    if recorded is not None and recorded != comparable:
        scope = (f"On {count(comparable)} of {_plural(recorded, 'recorded request')}, those with a confirmed "
                 "successful outcome and known costs")
    else:
        scope = f"On {_plural(comparable, 'request')} with a confirmed successful outcome and known costs"
    return (f"{scope}, answers and routing together cost {money(ledger.get('verified_cost_usd'))} against a "
            f"baseline of {money(ledger.get('verified_baseline_cost_usd'))}: {clause}.")


def _shadow_headline(shadow: dict) -> str:
    share = percent(shadow.get("policy_would_accept_share"), rounding="down")
    share = "a share that is not known" if share == NOT_KNOWN else share
    lead = (f"Of {_plural(_whole(shadow.get('shadowed_requests')), 'shadowed request')}, the policy would have "
            f"answered {share} with cheaper models")
    frontier, policy = shadow.get("frontier_cost_usd"), shadow.get("projected_policy_cost_usd")
    if _number(frontier) is None or _number(policy) is None:
        return f"{lead}; their costs are not known, so no saving is projected."
    covered, left_out = "", _whole(shadow.get("unknown_cost_requests"))
    if left_out is None or left_out > 0:
        known = _whole(shadow.get("cost_known_requests"))
        covered = (" on the requests whose costs are all known" if known is None
                   else f" on the {_plural(known, 'request')} whose costs are all known")
    clause = _saving_clause(shadow.get("projected_savings_usd"), shadow.get("projected_savings_fraction"),
                            word="projected saving", against="the frontier model")
    return (f"{lead}; its projected cost is {money(policy)} against the frontier model's "
            f"{money(frontier)}{covered}: {clause}.")


def headline(ledger_report: dict | None, shadow_report: dict | None) -> str:
    """One plain-text sentence that leads with what was measured.

    Measured routing comes before a shadow projection, because a projection is the
    weaker claim. When neither report has anything to stand on, the sentence says so
    instead of showing a zero.
    """
    if _live(ledger_report):
        return _live_headline(ledger_report)
    if _shadow_has_data(shadow_report):
        return _shadow_headline(shadow_report)
    if _ledger_has_requests(ledger_report):
        return (f"Of {_plural(_whole(ledger_report.get('requests')), 'recorded request')}, none has a confirmed "
                "successful outcome with known answer, routing and baseline costs, so no saving can be stated yet.")
    return ("Nothing has been measured yet: no request has been shadowed and no routed request has a confirmed "
            "successful outcome with known costs.")


def _rows(pairs: list[tuple[str, str]]) -> str:
    """A two-column table of label and value. Labels are fixed text; values are
    already formatted by ``count``, ``money`` or ``percent``, or escaped by the caller."""
    body = "".join(f'<tr><th scope="row">{label}</th><td>{value}</td></tr>' for label, value in pairs)
    return f'<table class="figures"><tbody>{body}</tbody></table>'


def _matching_frontier(shadow: dict) -> str:
    """The share of accepted answers equal to the frontier's, or why there is none."""
    accepted, compared = _whole(shadow.get("policy_would_accept")), _whole(shadow.get("accepted_compared_with_frontier"))
    share, matching = shadow.get("accepted_matching_frontier_share"), shadow.get("accepted_matching_frontier")
    if accepted == 0:
        return "no cheaper answer was accepted"
    if compared == 0:
        return "not compared"
    if _number(share) is None:
        return NOT_KNOWN
    if compared is None:
        return f"{percent(share, rounding='down')} ({count(matching)} answers)"
    return f"{percent(share, rounding='down')} ({count(matching)} of {count(compared)} compared)"


def _not_compared(shadow: dict) -> int | None:
    """Accepted answers that were not compared with the frontier's (code stages)."""
    accepted, compared = _whole(shadow.get("policy_would_accept")), _whole(shadow.get("accepted_compared_with_frontier"))
    if accepted is None or compared is None:
        return None
    return max(accepted - compared, 0)


def _cost_bars(frontier: Any, policy: Any) -> str:
    """Two bars comparing the frontier's cost with the policy's projected cost.

    Decoration only: the table beside it holds the figures, so the bars are hidden
    from screen readers and nothing is lost where styles are dropped.
    """
    frontier, policy = _number(frontier), _number(policy)
    if frontier is None or policy is None or min(frontier, policy) < 0 or max(frontier, policy) <= 0:
        return ""
    top = max(frontier, policy)
    bars = "".join(
        f'<div class="bar-row"><span class="bar-label">{label}</span>'
        f'<span class="bar-track"><span class="bar {kind}" style="width:{100 * value / top:.1f}%"></span></span>'
        f'<span class="bar-value">{money(value)}</span></div>'
        for label, kind, value in (("Frontier model", "bar-frontier", frontier), ("Policy, projected", "bar-policy", policy)))
    return f'<div class="bars" aria-hidden="true">{bars}</div>'


def _shadow_section(shadow: dict | None) -> str:
    if not isinstance(shadow, dict):
        return ""
    if _whole(shadow.get("shadowed_requests")) is None:
        return ("<section><h2>Shadow mode</h2><p>The number of shadowed requests is not known, so there is no "
                "projection to show.</p></section>")
    if not _shadow_has_data(shadow):
        return ('<section><h2>Shadow mode</h2><p>No request has been shadowed for this tenant, so there is no '
                "projection to show.</p></section>")
    accepted = count(shadow.get("policy_would_accept"))
    if accepted != NOT_KNOWN and _number(shadow.get("policy_would_accept_share")) is not None:
        accepted = f"{accepted} ({percent(shadow.get('policy_would_accept_share'), rounding='down')})"
    pairs = [
        ("Requests shadowed", count(shadow.get("shadowed_requests"))),
        ("Policy would have answered with cheaper models", accepted),
        ("Of those, same answer as the frontier model", _matching_frontier(shadow)),
    ]
    skipped = _not_compared(shadow)
    if skipped:
        pairs.append(("Accepted by a code check, not compared", count(skipped)))
    pairs += [
        ("Requests with all costs known", count(shadow.get("cost_known_requests"))),
        ("Requests left out because a cost is not known", count(shadow.get("unknown_cost_requests"))),
        ("Frontier model cost, on requests with known costs", money(shadow.get("frontier_cost_usd"))),
        ("Projected policy cost, on the same requests", money(shadow.get("projected_policy_cost_usd"))),
        ("Projected saving", _money_with_share(shadow.get("projected_savings_usd"),
                                               shadow.get("projected_savings_fraction"))),
        ("What shadow mode itself cost, on calls with a known cost", money(shadow.get("shadow_spend_usd"))),
        ("Shadow calls made", count(shadow.get("shadow_calls"))),
        ("Shadow calls that failed", count(shadow.get("failed_shadow_calls"))),
    ]
    saving = _number(shadow.get("projected_savings_usd"))
    warning = ('<p class="warn">On these requests the policy would have cost more than the frontier model.</p>'
               if saving is not None and saving < 0 else "")
    return ("<section><h2>Shadow mode: what the policy would have done</h2>"
            "<p>Each of these requests was answered by the frontier model, as before. The policy's cheaper "
            "models then ran in the background. Their answers were never returned to the caller.</p>"
            f"{_cost_bars(shadow.get('frontier_cost_usd'), shadow.get('projected_policy_cost_usd'))}"
            f"{warning}{_rows(pairs)}"
            "<p class=\"small\">The projected policy cost is every cheaper call and check the policy made, plus "
            "the frontier model's cost wherever no cheaper answer was accepted. The cost of shadow mode is real "
            "spend on the tenant's own provider account, on top of the frontier model's bill. It is not taken off "
            "the projected saving. It sums only calls whose cost is known, so it is too low, and can read as zero, "
            "when a model has no configured price.</p></section>")


def _left_out(ledger: dict) -> str:
    """Confirmed successes the saving does not count. With no baseline estimate these
    are the requests a cheaper model answered, so the reader must see the number."""
    successes, comparable = _whole(ledger.get("verified_successes")), _whole(ledger.get("comparable_verified_requests"))
    if successes is None or comparable is None or comparable > successes:
        return NOT_KNOWN
    return count(successes - comparable)


def _ledger_section(ledger: dict | None) -> str:
    if not isinstance(ledger, dict) or not _ledger_has_requests(ledger):
        return ""
    unknown = count(ledger.get("unknown_cost_requests"))
    if unknown != NOT_KNOWN:
        unknown += (f" (answer cost missing on {count(ledger.get('unknown_answer_cost_requests'))}, "
                    f"routing cost missing on {count(ledger.get('unknown_routing_cost_requests'))})")
    pairs = [
        ("Requests recorded", count(ledger.get("requests"))),
        ("Confirmed successful", count(ledger.get("verified_successes"))),
        ("Confirmed failed", count(ledger.get("failures"))),
        ("Outcome still pending", count(ledger.get("pending_outcomes"))),
        ("Requests with a cost that is not known", unknown),
        ("Answered from the verified cache", count(ledger.get("cache_hits"))),
        ("Requests the saving is computed on", count(ledger.get("comparable_verified_requests"))),
        ("Confirmed successful but left out, because a cost or the baseline is not known", _left_out(ledger)),
        ("Verified answer cost", money(ledger.get("verified_answer_cost_usd"))),
        ("Routing cost (second opinions, escalated attempts, checks)", money(ledger.get("verified_routing_cost_usd"))),
        ("Answer and routing cost together", money(ledger.get("verified_cost_usd"))),
        ("Baseline: always using the frontier model", money(ledger.get("verified_baseline_cost_usd"))),
        ("Net saving", _money_with_share(ledger.get("verified_savings_usd"), ledger.get("savings_fraction"))),
        ("Known spend on all recorded requests", money(ledger.get("known_cost_total_usd"))),
    ]
    if _live(ledger):
        saving = _number(ledger.get("verified_savings_usd"))
        note = ('<p class="warn">On these requests routing cost more than the baseline.</p>'
                if saving is not None and saving < 0 else "")
    else:
        note = ("<p>No request has a confirmed successful outcome together with known answer, routing and "
                "baseline costs. Until one does, no saving is stated.</p>")
    return ("<section><h2>Ledger: requests served through the proxy</h2>"
            "<p>The saving below is net of routing cost. It counts only requests whose outcome was confirmed "
            "successful and whose answer, routing and baseline costs are all known.</p>"
            f"{note}{_rows(pairs)}"
            "<p class=\"small\">Known spend sums only measured costs across every recorded request. It is "
            "incomplete while any cost is not known.</p></section>")


def _models(names: Any) -> list[str]:
    return [f"<code>{_esc(name)}</code>" for name in names] if isinstance(names, list) else []


def _stage_words(stage: Any) -> str:
    """One stage of a policy as a sentence."""
    if not isinstance(stage, dict):
        return "A stage this page cannot read."
    members, check = _models(stage.get("models")), stage.get("check")
    if not members:
        return "A stage this page cannot read."
    asked = f"Ask {_listed(members)}."
    if check == "agreement":
        return f"{asked} Accept the answer only if all {len(members)} give the same one."
    if check == "none":
        return f"{asked} Accept the answer as it is, with no check."
    return f"{asked} Accept the answer only if the check named <code>{_esc(check)}</code> passes."


def _guarantee_rows(guarantee: dict) -> list[tuple[str, str]]:
    epsilon, delta = _number(guarantee.get("epsilon")), _number(guarantee.get("delta"))
    limit, confidence = NOT_KNOWN, NOT_KNOWN
    if epsilon is not None:
        limit = (f"{percent(epsilon)} (epsilon = {_esc(epsilon)}): the most the policy may lose, as a share of "
                 "tasks, where the frontier model succeeds and the policy does not")
    if delta is not None:
        confidence = f"{percent(1 - delta)} (delta = {_esc(delta)})"
    certified = guarantee.get("certified")
    state = "yes" if certified is True else "no" if certified is False else NOT_KNOWN
    return [
        ("Quality limit", limit),
        ("Confidence", confidence),
        ("Certified at this limit", state),
        ("Upper bound on the loss, from the certify tasks", percent(guarantee.get("bound"), rounding="up", digits=2)),
        ("Graded tasks the bound rests on", count(guarantee.get("certify_tasks"))),
        ("Loss observed on those tasks",
         percent(guarantee.get("certify_degradation_rate"), rounding="up", digits=2)),
    ]


def _policy_section(policy: dict | None) -> str:
    if not isinstance(policy, dict):
        return ""
    stages = policy.get("stages") if isinstance(policy.get("stages"), list) else []
    frontier = f"<code>{_esc(policy.get('frontier'))}</code>" if policy.get("frontier") else "the frontier model"
    if stages:
        steps = "".join(f"<li>{_stage_words(stage)}</li>" for stage in stages)
        steps += f"<li>If no stage accepted an answer, send the request to {frontier}.</li>"
        plan = f"<ol>{steps}</ol>"
    else:
        plan = f"<p>No cheaper stage is in this policy. Every request goes to {frontier}.</p>"
    about = []
    if policy.get("task_type"):
        about.append(f"Kind of work it was frozen for: <code>{_esc(policy.get('task_type'))}</code>.")
    if policy.get("tenant"):
        about.append(f"Tenant it was frozen for: <code>{_esc(policy.get('tenant'))}</code>.")
    guarantee = policy.get("guarantee") if isinstance(policy.get("guarantee"), dict) else {}
    parts = ["<section><h2>The policy in plain words</h2>",
             "<p class=\"small\">This describes the policy file supplied when the page was made. The reports do not "
             "name the policy behind each figure, so the page cannot confirm that every figure above came from "
             "it.</p>",
             f"<p>{' '.join(about)}</p>" if about else "", plan, _rows(_guarantee_rows(guarantee))]
    if isinstance(guarantee.get("statement"), str):
        parts.append(f"<p class=\"small\">The guarantee as the policy file states it: "
                     f"<q>{_esc(guarantee['statement'])}</q></p>")
    estimate = policy.get("estimate") if isinstance(policy.get("estimate"), dict) else {}
    if _number(estimate.get("savings")) is not None:
        parts.append(f"<p class=\"small\">When the policy was frozen, its estimated saving on "
                     f"{count(estimate.get('test_tasks'))} held-out test tasks was "
                     f"{percent(estimate.get('savings'), rounding='down')}. "
                     "That is an offline estimate from recorded answers, not a measurement of this tenant's "
                     "traffic.</p>")
    return "".join(parts) + "</section>"


def _caveats(ledger: dict | None, shadow: dict | None, policy: dict | None) -> list[str]:
    """The limits of the figures, in plain words. Fixed text, safe to place as is."""
    items = []
    if isinstance(shadow, dict):
        items += [
            "<strong>Equal answers are not correctness.</strong> Shadow mode compares the policy's answer with the "
            "frontier model's answer. It does not check either one. Two right free-form answers rarely match word "
            "for word, and answers accepted by a code check are not compared at all.",
            "<strong>Shadow figures are projections.</strong> They cover only requests whose costs are all known. "
            "They are not a billing amount.",
            "<strong>Only some requests are shadowed.</strong> A request is shadowed only if the frontier model "
            "answered it without streaming, without a cache hit and without an error, and only if the sample rate "
            "and the spending cap let it through. The figures describe the shadowed requests, not all traffic.",
            "<strong>Shadow mode is not free.</strong> Its calls are real calls on the tenant's own provider "
            "account, and they send the tenant's prompt to the cheaper models on that account.",
        ]
    if isinstance(ledger, dict):
        items += [
            "<strong>The ledger saving is a subtotal.</strong> It is net of routing cost and counts only confirmed "
            "successful requests with known answer, routing and baseline costs. Failed, pending and incompletely "
            "costed requests are left out. It is not a billing amount.",
            "<strong>The baseline may be an estimate.</strong> When a cheaper model answered, the frontier model "
            "was not called, so its cost for that request was not observed. Where the proxy is set to estimate it, "
            "the baseline is the returned answer's token counts at the frontier model's prices. A frontier answer "
            "could have been longer or shorter.",
            "<strong>Outcomes and baseline costs are reported by the caller.</strong> This page does not verify "
            "them.",
        ]
    if isinstance(policy, dict):
        items.append(
            "<strong>The quality limit describes the graded tasks the policy was certified on.</strong> It says "
            "nothing about traffic that differs from them. The quality guarantee comes from that certification, "
            "not from any comparison on this page.")
    items.append("<strong>\"Not known\" means not measured.</strong> A missing cost or outcome is never counted "
                 "as zero.")
    items.append("<strong>Rounding.</strong> Percentages of savings and shares are rounded down. The loss bound "
                 "and the observed loss are rounded up. Dollar amounts are rounded to the nearest shown digit.")
    return items


def _scope_notes(ledger: dict | None, shadow: dict | None, policy: dict | None) -> list[tuple[str, str]]:
    """Every scope text the reports carry, word for word, with where it came from."""
    estimate = _get(policy, "estimate")
    sources = [("Shadow report", _get(shadow, "scope")), ("Ledger", _get(ledger, "scope")),
               ("Ledger costs", _get(ledger, "cost_scope")), ("Ledger savings", _get(ledger, "savings_scope")),
               ("Policy estimate", _get(estimate, "scope"))]
    return [(label, text) for label, text in sources if isinstance(text, str) and text.strip()]


def _limits_section(ledger: dict | None, shadow: dict | None, policy: dict | None) -> str:
    items = "".join(f"<li>{item}</li>" for item in _caveats(ledger, shadow, policy))
    notes = _scope_notes(ledger, shadow, policy)
    carried = ""
    if notes:
        listed = "".join(f"<li><span class=\"source\">{label}:</span> {_esc(text)}</li>" for label, text in notes)
        carried = f"<p class=\"small\">Scope notes carried from the reports, word for word:</p><ul class=\"small\">{listed}</ul>"
    return (f'<section class="limits"><h2>What this does and does not show</h2><ul>{items}</ul>{carried}</section>')


STYLE = """
:root{color-scheme:light dark;--bg:#f6f5f0;--card:#ffffff;--ink:#1c1f23;--muted:#59616b;--rule:#d8d5cb;
--accent:#1f5f5b;--bar:#8d949c;--note:#efece2;--warn:#8a3b12}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--card:#1b1f23;--ink:#e7e5df;--muted:#a0a8b1;
--rule:#30363c;--accent:#7cc4bb;--bar:#6b737c;--note:#22272c;--warn:#f0a676}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:46rem;margin:0 auto;padding:2rem 1rem 3rem}
header p.kicker{margin:0 0 .5rem;color:var(--muted);font-size:.8rem;letter-spacing:.08em;text-transform:uppercase}
h1,h2{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:600;text-wrap:balance}
h1{margin:0 0 .75rem;font-size:1.55rem;line-height:1.3}
h2{margin:0 0 .5rem;font-size:1.2rem;line-height:1.3}
p{margin:.5rem 0}
.meta{color:var(--muted);font-size:.9rem;overflow-wrap:anywhere}
.note{margin-top:.75rem;padding:.6rem .8rem;border:1px solid var(--rule);border-left:4px solid var(--warn);
border-radius:4px;overflow-wrap:anywhere}
section{margin-top:1.5rem;padding:1.25rem;background:var(--card);border:1px solid var(--rule);border-radius:6px}
table.figures{width:100%;border-collapse:collapse;margin:.75rem 0;font-variant-numeric:tabular-nums}
table.figures th,table.figures td{padding:.5rem 0;border-top:1px solid var(--rule);vertical-align:top}
table.figures th{text-align:left;font-weight:400;color:var(--muted);padding-right:1rem;width:58%}
table.figures td{text-align:right;font-weight:600;overflow-wrap:anywhere}
ol,ul{margin:.5rem 0;padding-left:1.25rem}
li{margin:.4rem 0}
code{font:.9em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
.small{font-size:.875rem;color:var(--muted)}
.warn{color:var(--warn);font-weight:600}
.limits{background:var(--note);border-left:4px solid var(--accent)}
.source{font-weight:600}
.bars{margin:1rem 0 .25rem}
.bar-row{display:flex;align-items:center;gap:.6rem;margin:.35rem 0;font-size:.875rem}
.bar-label{flex:0 0 8.5rem;color:var(--muted)}
.bar-track{flex:1 1 auto;min-width:2rem}
.bar{display:block;height:.8rem;min-width:2px;border-radius:2px}
.bar-frontier{background:var(--bar)}
.bar-policy{background:var(--accent)}
.bar-value{flex:0 0 auto;font-weight:600;font-variant-numeric:tabular-nums}
footer{margin-top:1.5rem;color:var(--muted);font-size:.8rem}
@media (max-width:30rem){main{padding:1.25rem .75rem 2rem}section{padding:1rem}h1{font-size:1.3rem}
table.figures th{width:auto}.bar-label{flex-basis:6.5rem}}
@media print{body{background:#fff}section{break-inside:avoid}}
"""


def _when(generated_at: Any) -> str:
    """The generation time as text. A datetime is written in ISO form to the second."""
    if isinstance(generated_at, datetime):
        return generated_at.isoformat(timespec="seconds")
    return str(generated_at)


def render(ledger_report: dict | None, shadow_report: dict | None, *, tenant: str, generated_at: Any,
           policy: dict | None = None, note: str | None = None) -> str:
    """The savings report for one tenant as a complete HTML document.

    ``ledger_report`` is ``Ledger.report(tenant)`` and ``shadow_report`` is
    ``ShadowStore.report(tenant)``; either may be None when that part does not exist.
    ``policy`` is the frozen policy the figures were produced with, if it should be
    described. ``generated_at`` is a datetime or a string and is shown as given.
    ``note`` is one line shown under the headline, for what the reports cannot say
    about themselves: that the figures come from a replay of benchmark answers, or
    that the policy described is not the one some figures were produced with.
    """
    noted = f'<p class="note"><strong>Note.</strong> {_esc(note)}</p>' if isinstance(note, str) and note.strip() else ""
    sections = (_shadow_section(shadow_report), _ledger_section(ledger_report), _policy_section(policy),
                _limits_section(ledger_report, shadow_report, policy))
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>Savings report: {_esc(tenant)}</title><style>{STYLE}</style></head><body><main>"
        '<header><p class="kicker">Tine savings report</p>'
        f"<h1>{_esc(headline(ledger_report, shadow_report))}</h1>"
        f'<p class="meta">Tenant <strong>{_esc(tenant)}</strong>. Generated {_esc(_when(generated_at))}.</p>'
        f"{noted}</header>"
        f"{''.join(sections)}"
        "<footer>This page is self-contained. It loads nothing from the network and runs no script.</footer>"
        "</main></body></html>\n"
    )
