"""The customer-facing savings report, as one self-contained HTML page.

``render`` turns the two reports the proxy already produces (``Ledger.report`` and
``ShadowStore.report``) and, optionally, the frozen policy into a page a customer can
open from a file or receive by email. The page loads nothing: no script, no font, no
image, no link. Styling is one ``<style>`` block, and the page still reads correctly
where a mail client drops it.

The page adds almost no arithmetic of its own. Every cost, saving and share is one the
reports computed, so the page cannot disagree with ``GET /v1/savings`` or
``GET /v1/shadow``. The exceptions are differences of counts the reports already hold
(accepted answers that were not compared, confirmed successes outside the measured
saving, and those of them outside the estimated figure too) and 1 - delta.

A saving measured against the frontier model's measured cost is never mixed with one
whose baseline was estimated: the ledger reports them apart and so does the page. Under
live routing the frontier's cost is measured only for requests that escalated to it,
so when confirmed successes are left out of the measured saving the headline says so
and points to the estimated and projected figures, each labelled as what it is. When
the ledger holds a frontier holdout estimate with a figure, that is the measured saving
under routing: it leads the headline with its interval, and the other figures follow
it, still apart.

Four rules it holds to:

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

from tine.policy import FREEZE_EDITED, freeze_state

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
    article = "an" if word[:1] in "aeiou" else "a"
    return f"{article} {word} of {_money_with_share(amount, fraction)}"


def _holdout(ledger: dict | None) -> dict | None:
    """The ledger's frontier holdout estimate, or None when the report has none."""
    estimate = _get(ledger, "holdout_estimate")
    return estimate if isinstance(estimate, dict) else None


def _interval(estimate: dict) -> str:
    """The interval on the holdout saving. Both ends are rounded down."""
    low, high = _number(estimate.get("saving_fraction_low")), _number(estimate.get("saving_fraction_high"))
    if low is None or high is None:
        return NOT_KNOWN
    return f"{percent(low, rounding='down')} to {percent(high, rounding='down')}"


def _confidence(estimate: dict) -> str:
    confidence = _number(estimate.get("confidence"))
    return "the" if confidence is None else percent(confidence, digits=0)


def _holdout_sentence(ledger: dict | None) -> str | None:
    """The holdout saving as a sentence with its interval, or None when the report has
    no holdout figure (no holdout, or too few requests in a group)."""
    estimate = _holdout(ledger)
    saving = _number(_get(estimate, "saving_fraction"))
    if saving is None:
        return None
    routed, held = _whole(estimate.get("routed_requests")), _whole(estimate.get("holdout_requests"))
    partial = _whole(estimate.get("routed_unknown_cost_requests")) if _upper_limit(estimate) else 0
    # With routed costs not fully known, the routed mean is a lower limit and the
    # saving an upper limit: the sentence says "at most" and how many are partial.
    limit = "at most " if partial else ""
    interval = (f"{_confidence(estimate)} interval for "
                f"{'that upper limit' if partial else 'the saving'}: {_interval(estimate)}")
    if saving > 0:
        result = (f"routed requests cost {limit}{percent(saving, rounding='down')} less per request than requests "
                  f"held out for the frontier model ({interval})")
    elif saving < 0:
        result = (f"routed requests cost {'at least ' if partial else ''}{percent(-saving, rounding='up')} more per "
                  f"request than requests held out for the frontier model, so there is no saving ({interval})")
    elif partial:
        result = (f"routed requests cost at least as much per request as requests held out for the frontier model, "
                  f"so there is no saving ({interval})")
    else:
        result = f"routed requests cost the same per request as requests held out for the frontier model ({interval})"
    if partial:
        total = None if routed is None or partial is None else routed + partial
        groups = (f"on {_plural(total, 'routed request')}, {count(partial)} of them counted at the part of their "
                  f"cost that is known (a cheaper call failed, a check raised or the route deadline ran out; most "
                  f"then escalated to the frontier), and {count(held)} held out")
    else:
        groups = f"on {_plural(routed, 'routed request')} and {count(held)} held out"
    return (f"Measured under routing, from a random frontier holdout: {result}, {groups}. It is an estimate from a "
            "random sample, and it does not measure quality.")


def _upper_limit(estimate: dict) -> bool:
    """Whether the holdout saving is an upper limit, because routed requests whose cost
    is not fully known were counted at their known part."""
    return estimate.get("saving_is_upper_limit") is True


def _estimated(ledger: dict | None) -> int | None:
    """Confirmed successes whose baseline was estimated, not measured."""
    return _whole(_get(ledger, "estimated_baseline_requests"))


def _caller(ledger: dict | None) -> int | None:
    """Confirmed successes whose baseline the caller supplied. A report from before
    baseline sources were recorded has no such count: it counted them as measured."""
    if not isinstance(ledger, dict) or "caller_baseline_requests" not in ledger:
        return 0
    return _whole(ledger["caller_baseline_requests"])


def _outside_measured(ledger: dict | None) -> int | None:
    """Confirmed successes the measured saving does not count, for any reason."""
    successes, comparable = _whole(_get(ledger, "verified_successes")), _whole(_get(ledger, "comparable_verified_requests"))
    if successes is None or comparable is None or comparable > successes:
        return None
    return successes - comparable


def _estimated_sentence(ledger: dict | None) -> str | None:
    """The estimated-baseline figure as a sentence that says it is an estimate, or None
    when no confirmed success has an estimated baseline."""
    requests = _estimated(ledger)
    if not requests:
        return None
    saving = ledger.get("estimated_baseline_savings_usd")
    result = ("the estimated saving is not known" if _number(saving) is None else "the result is " + _saving_clause(
        saving, None, word="estimated saving", against="the estimated baseline"))
    return (f"Estimated, not measured: on {_plural(requests, 'confirmed successful request')} that cheaper "
            f"models answered, with the frontier model's cost estimated from each answer's token counts, {result}.")


def _caller_sentence(ledger: dict | None) -> str | None:
    """The caller-supplied-baseline figure as a sentence that says whose it is, or None
    when no confirmed success has a baseline the caller supplied."""
    requests = _caller(ledger)
    if not requests:
        return None
    saving = ledger.get("caller_baseline_savings_usd")
    result = ("the saving is not known" if _number(saving) is None else "the result is " + _saving_clause(
        saving, None, word="saving", against="the caller's baseline"))
    return (f"Supplied by the caller, not measured: on {_plural(requests, 'confirmed successful request')} whose "
            f"frontier cost the caller supplied through the outcome endpoint, {result}.")


def _projection_sentence(shadow: dict | None) -> str | None:
    """The shadow projection as a pointer that says it is a projection, or None."""
    if not _shadow_has_data(shadow):
        return None
    if _number(shadow.get("projected_savings_usd")) is None:
        return ("Projected, not measured: shadow mode has no projected saving, because the costs of the shadowed "
                "requests are not known.")
    known = _whole(shadow.get("cost_known_requests"))
    where = ("the shadowed requests with all costs known" if known is None
             else f"{_plural(known, 'shadowed request')} with all costs known")
    clause = _saving_clause(shadow.get("projected_savings_usd"), shadow.get("projected_savings_fraction"),
                            word="projected saving", against="the frontier model")
    return f"Projected, not measured: on {where}, shadow mode puts the policy at {clause}."


def _coverage_sentence(shadow: dict | None) -> str | None:
    """What the shadow figures leave out of the sample, or None when they leave out
    nothing, or the report does not count it (a report from before the counts)."""
    if not _shadow_has_data(shadow):
        return None
    sampled, streamed = _whole(shadow.get("sampled_requests")), _whole(shadow.get("streamed_not_shadowed"))
    shadowed = _whole(shadow.get("shadowed_requests"))
    parts = []
    if sampled is not None and shadowed is not None and shadowed < sampled:
        reasons = [f"{count(number)} {why}" for number, why in (
            (_whole(shadow.get("skipped_busy")), "while shadow runs were busy"),
            (_whole(shadow.get("skipped_stopped")), "after shadowing had stopped"),
            (_whole(shadow.get("failed_shadow_runs")), "because the shadow run failed")) if number]
        skipped = f" ({_listed(reasons)})" if reasons else ""
        parts.append(f"Shadow mode ran on {count(shadowed)} of {_plural(sampled, 'sampled request')} "
                     f"({percent(shadow.get('shadow_coverage'), rounding='down')}); the rest were skipped{skipped}, "
                     "so busy periods count for less than they should")
    if streamed:
        parts.append(f"{count(streamed)} more sampled {'request was' if streamed == 1 else 'requests were'} "
                     "streamed, and shadow mode does not run streamed requests")
    return ". ".join(parts) + "." if parts else None


def _drift_warning(shadow: dict | None) -> str | None:
    """The shadow report's warning that its traffic cannot confirm the certification."""
    warning = _get(shadow, "drift_warning")
    return warning.strip() if _shadow_has_data(shadow) and isinstance(warning, str) and warning.strip() else None


def _live_headline(ledger: dict, shadow: dict | None) -> list[str]:
    clause = _saving_clause(ledger.get("verified_savings_usd"), ledger.get("savings_fraction"),
                            word="net saving", against="the baseline")
    comparable, recorded = _whole(ledger.get("comparable_verified_requests")), _whole(ledger.get("requests"))
    costs = (f"answers and routing together cost {money(ledger.get('verified_cost_usd'))} against a "
             f"baseline of {money(ledger.get('verified_baseline_cost_usd'))}: {clause}.")
    outside = _outside_measured(ledger)
    if outside:
        # Confirmed successes are missing from the measured saving. Under live routing
        # these are the requests cheaper models answered, so the measured figure is
        # mostly escalated requests and must not read as the policy's performance.
        successes = _whole(ledger.get("verified_successes"))
        # The title line states the scope; the measured amount, often a loss on these
        # requests, follows it rather than standing as the headline.
        lead = (f"The measured saving covers only {count(comparable)} of "
                f"{_plural(successes, 'confirmed successful request')}, those whose frontier cost was measured.")
        measured = f"On them, {costs}"
        if _holdout_sentence(ledger):
            why = ("Under live routing the frontier model's cost is measured only for requests that escalated to "
                   "it or were held out, so this figure leaves out the requests cheaper models answered and is not "
                   "the policy's saving; the holdout figure above is the measured saving under routing.")
        elif _estimated(ledger):
            why = ("Under live routing the frontier model's cost is measured only for requests that escalated to "
                   "it, so this figure leaves out the requests cheaper models answered and is not the policy's "
                   "saving.")
        else:
            why = ("If the proxy routes through a policy, the frontier model's cost is measured only for requests "
                   "that escalated to it; the requests cheaper models answered are then the ones left out, and "
                   "this figure is not the policy's saving.")
        pointers = [sentence for sentence in (_estimated_sentence(ledger), _caller_sentence(ledger),
                                              _projection_sentence(shadow)) if sentence]
        if not pointers:
            pointers = ["No estimated or projected figure is available for the requests left out: set the proxy to "
                        "estimate the baseline, or run shadow mode."]
        return [lead, measured, why, *pointers]
    # The saving is a subtotal. Saying how many requests it leaves out belongs in the
    # same sentence as the saving, not only in the table below it.
    if recorded is not None and recorded != comparable:
        scope = (f"On {count(comparable)} of {_plural(recorded, 'recorded request')}, those with a confirmed "
                 "successful outcome and known costs")
    else:
        scope = f"On {_plural(comparable, 'request')} with a confirmed successful outcome and known costs"
    return [f"{scope}, {costs}"]


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


def headline_sentences(ledger_report: dict | None, shadow_report: dict | None) -> list[str]:
    """The headline as plain-text sentences. The first is the page's title line; the
    rest qualify it and are shown directly beneath it.

    A frontier holdout estimate with a figure leads: it is the measured saving under
    routing, from a random sample. Measured routing comes before a shadow projection,
    because a projection is the weaker claim, and both come before a saving on
    estimated baselines. When the
    measured saving leaves out confirmed successes, the sentences say so and point to
    the estimated and projected figures. When neither report has anything to stand
    on, the sentence says so instead of showing a zero.

    A shadow report that cannot confirm the policy's certification (too many accepted
    answers differ from the frontier's) puts its warning first, before any saving.
    """
    warning = _drift_warning(shadow_report)
    return [*([warning] if warning else []), *_headline_figures(ledger_report, shadow_report)]


def _headline_figures(ledger_report: dict | None, shadow_report: dict | None) -> list[str]:
    holdout = _holdout_sentence(ledger_report)
    if holdout and not _live(ledger_report) and not _shadow_has_data(shadow_report):
        # The other figures would only say that no saving can be stated yet.
        estimated = _estimated_sentence(ledger_report)
        return [holdout, *([estimated] if estimated else [])]
    figures = _other_figures(ledger_report, shadow_report)
    return [holdout, *figures] if holdout else figures


def _other_figures(ledger_report: dict | None, shadow_report: dict | None) -> list[str]:
    coverage = _coverage_sentence(shadow_report)
    if _live(ledger_report):
        sentences = _live_headline(ledger_report, shadow_report)
        projection = _projection_sentence(shadow_report)
        if coverage and projection in sentences:
            sentences.insert(sentences.index(projection) + 1, coverage)
        return sentences
    estimated = _estimated_sentence(ledger_report)
    if _shadow_has_data(shadow_report):
        return [_shadow_headline(shadow_report), *([coverage] if coverage else []), *([estimated] if estimated else [])]
    if _ledger_has_requests(ledger_report):
        recorded = _plural(_whole(ledger_report.get("requests")), "recorded request")
        if estimated:
            return [f"Of {recorded}, none is a confirmed success with known costs and a measured frontier cost, so "
                    "no measured saving can be stated yet.", estimated]
        return [f"Of {recorded}, none has a confirmed successful outcome with known answer, routing and baseline "
                "costs, so no saving can be stated yet."]
    return ["Nothing has been measured yet: no request has been shadowed and no routed request has a confirmed "
            "successful outcome with known costs."]


def headline(ledger_report: dict | None, shadow_report: dict | None) -> str:
    """The whole headline as one plain-text string."""
    return " ".join(headline_sentences(ledger_report, shadow_report))


def _rows(pairs: list[tuple[str, str]], caption: str | None = None) -> str:
    """A two-column table of label and value. Labels and captions are fixed text;
    values are already formatted by ``count``, ``money`` or ``percent``, or escaped by
    the caller."""
    body = "".join(f'<tr><th scope="row">{label}</th><td>{value}</td></tr>' for label, value in pairs)
    head = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="figures">{head}<tbody>{body}</tbody></table>'


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


def _shadow_spend(shadow: dict) -> str:
    """Known shadow spend, marked as a lower limit when some calls have no known cost.

    The report sums only known costs, so it reads $0.00 when nothing is known. With
    calls of unknown cost the true spend is higher, and the value says so."""
    spent, unknown = money(shadow.get("shadow_spend_usd")), _whole(shadow.get("unknown_cost_shadow_calls"))
    if spent == NOT_KNOWN or not unknown:
        return spent
    calls = "1 call or check" if unknown == 1 else f"{unknown:,} calls or checks"
    return f"at least {spent} ({calls} of unknown cost not included)"


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
    if _whole(shadow.get("sent_to_frontier_for_tools")):
        pairs.append(("Sent to the frontier model untried, because the request defined tools",
                      count(shadow.get("sent_to_frontier_for_tools"))))
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
    ]
    if "sampled_requests" in shadow:  # reports from before these counts do not hold them
        coverage = count(shadow.get("sampled_requests"))
        if coverage != NOT_KNOWN and _number(shadow.get("shadow_coverage")) is not None:
            coverage = f"{coverage} ({percent(shadow.get('shadow_coverage'), rounding='down')} shadowed)"
        pairs += [
            ("Sampled requests shadow mode could have run", coverage),
            ("Skipped because shadow runs were busy", count(shadow.get("skipped_busy"))),
            ("Skipped because shadowing had stopped", count(shadow.get("skipped_stopped"))),
            ("Shadow runs that failed before they were recorded", count(shadow.get("failed_shadow_runs"))),
            ("Sampled requests that were streamed, never shadowed", count(shadow.get("streamed_not_shadowed"))),
        ]
    if "drift_check" in shadow:
        differing = count(shadow.get("accepted_differing_from_frontier"))
        if differing != NOT_KNOWN and _number(shadow.get("accepted_differing_from_frontier_share")) is not None:
            differing = (f"{differing} ({percent(shadow.get('accepted_differing_from_frontier_share'), rounding='up')}"
                         " of those compared)")
        check = shadow.get("drift_check")
        pairs += [
            ("Accepted answers that differ from the frontier model's", differing),
            ("Checked against the policy's quality limit",
             _esc(check) if isinstance(check, str) and check.strip() else NOT_KNOWN),
        ]
    pairs += [
        ("What shadow mode itself cost, on calls with a known cost", _shadow_spend(shadow)),
        ("Shadow calls and checks whose cost is not known", count(shadow.get("unknown_cost_shadow_calls"))),
        ("Shadow calls made", count(shadow.get("shadow_calls"))),
        ("Shadow calls that failed", count(shadow.get("failed_shadow_calls"))),
    ]
    if "sampling" in shadow:  # added by GET /v1/shadow; ShadowStore.report does not hold it
        sampling = shadow.get("sampling")
        pairs.append(("Shadow sampling when the report was read",
                      _esc(sampling) if isinstance(sampling, str) and sampling.strip() else NOT_KNOWN))
    saving = _number(shadow.get("projected_savings_usd"))
    warning = ('<p class="warn">On these requests the policy would have cost more than the frontier model.</p>'
               if saving is not None and saving < 0 else "")
    if _drift_warning(shadow):
        warning += f'<p class="warn">{_esc(_drift_warning(shadow))}</p>'
    coverage = _coverage_sentence(shadow)
    if coverage:
        warning += f'<p class="warn">{_esc(coverage)}</p>'
    return ("<section><h2>Shadow mode: what the policy would have done</h2>"
            "<p>Each of these requests was answered by the frontier model, as before. The policy's cheaper "
            "models then ran in the background. Their answers were never returned to the caller.</p>"
            f"{_cost_bars(shadow.get('frontier_cost_usd'), shadow.get('projected_policy_cost_usd'))}"
            f"{warning}{_rows(pairs)}"
            "<p class=\"small\">The projected policy cost is every cheaper call and check the policy made, plus "
            "the frontier model's cost wherever no cheaper answer was accepted. The cost of shadow mode is real "
            "spend on the tenant's own provider account, on top of the frontier model's bill. It is not taken off "
            "the projected saving. It sums only calls whose cost is known, so it is too low, and can read as zero, "
            "when a call failed before its usage was known or a model has no configured price. The row after it "
            "counts those calls and checks.</p></section>")


def _left_out(ledger: dict) -> str:
    """Confirmed successes in none of the measured, estimated and caller-supplied
    figures, because a cost or the baseline is not known. Without a baseline estimate
    these are the requests a cheaper model answered, so the reader must see the number."""
    outside, estimated, caller = _outside_measured(ledger), _estimated(ledger), _caller(ledger)
    if outside is None or estimated is None or caller is None or estimated + caller > outside:
        return NOT_KNOWN
    return count(outside - estimated - caller)


def _caller_saving(ledger: dict) -> str:
    """The saving on caller-supplied baselines, worded so it cannot be read as measured."""
    saving = _number(ledger.get("caller_baseline_savings_usd"))
    return NOT_KNOWN if saving is None else f"{money(saving)} (the caller's figure, not measured)"


def _estimated_saving(ledger: dict) -> str:
    """The saving on estimated baselines, worded so it cannot be read as measured."""
    requests, saving = _estimated(ledger), _number(ledger.get("estimated_baseline_savings_usd"))
    if requests == 0:
        return "no such requests"
    if saving is None:
        return NOT_KNOWN
    return f"{money(saving)} (estimated, not measured)"


def _holdout_table(ledger: dict) -> str:
    """The frontier holdout comparison, or nothing when the report has none."""
    estimate = _holdout(ledger)
    if estimate is None:
        return ""
    least = _whole(estimate.get("min_requests_per_group"))
    saving = _number(estimate.get("saving_fraction"))
    if saving is None:
        least_text = "a minimum number of" if least is None else f"{least:,}"
        shown = f"not shown until each group has {least_text} requests with known costs"
    elif saving < 0:
        shown = f"{percent(saving, rounding='down')}: routing cost more"
    else:
        shown = percent(saving, rounding="down")
    upper = _upper_limit(estimate)
    if saving is not None and upper:
        shown = f"at most {shown}"
    routed_unknown = ("Counted at the known part of their cost, routed" if "saving_is_upper_limit" in estimate
                      else "Left out because a cost is not known, routed")
    pairs = [
        ("Routed requests with known costs", count(estimate.get("routed_requests"))),
        ("Held-out requests with known costs", count(estimate.get("holdout_requests"))),
        (routed_unknown, count(estimate.get("routed_unknown_cost_requests"))),
        ("Left out because a cost is not known, held out", count(estimate.get("holdout_unknown_cost_requests"))),
        ("Mean cost per routed request, answer and routing" + (", at least" if upper else ""),
         money(estimate.get("routed_mean_cost_usd"))),
        ("Mean cost per held-out request, the frontier model", money(estimate.get("holdout_mean_cost_usd"))),
        ("Saving per request, measured from the holdout", shown),
        (f"{_confidence(estimate).capitalize()} interval for the saving" + (", an upper limit" if upper else ""),
         _interval(estimate) if saving is not None else NOT_KNOWN),
    ]
    method = estimate.get("method")
    how = (f"<p class=\"small\">How the interval is worked out: {_esc(method)}.</p>"
           if isinstance(method, str) and method.strip() else "")
    return ("<p>A random share of the requests the policy could have served skipped it and went to the frontier "
            "model. Comparing their mean cost with that of the routed requests measures the saving of routing. "
            "It is an estimate from a random sample, and it says nothing about quality.</p>"
            f"{_rows(pairs, 'Measured under routing: random frontier holdout')}{how}")


def _ledger_section(ledger: dict | None) -> str:
    if not isinstance(ledger, dict) or not _ledger_has_requests(ledger):
        return ""
    unknown = count(ledger.get("unknown_cost_requests"))
    if unknown != NOT_KNOWN:
        unknown += (f" (answer cost missing on {count(ledger.get('unknown_answer_cost_requests'))}, "
                    f"routing cost missing on {count(ledger.get('unknown_routing_cost_requests'))})")
    requests = [
        ("Requests recorded", count(ledger.get("requests"))),
        ("Confirmed successful", count(ledger.get("verified_successes"))),
        ("Confirmed failed", count(ledger.get("failures"))),
        ("Outcome still pending", count(ledger.get("pending_outcomes"))),
        ("Requests with a cost that is not known", unknown),
        ("Answered from the verified cache", count(ledger.get("cache_hits"))),
    ]
    measured = [
        ("Requests the measured saving is computed on", count(ledger.get("comparable_verified_requests"))),
        ("Verified answer cost", money(ledger.get("verified_answer_cost_usd"))),
        ("Routing cost (second opinions, escalated attempts, checks)", money(ledger.get("verified_routing_cost_usd"))),
        ("Answer and routing cost together", money(ledger.get("verified_cost_usd"))),
        ("Baseline: the frontier model's measured cost", money(ledger.get("verified_baseline_cost_usd"))),
        ("Net saving, measured", _money_with_share(ledger.get("verified_savings_usd"), ledger.get("savings_fraction"))),
    ]
    estimated = [
        ("Confirmed successful, with the frontier model's cost estimated", count(ledger.get("estimated_baseline_requests"))),
        ("Net saving on those, estimated", _estimated_saving(ledger)),
    ]
    caller = _caller(ledger)
    if caller != 0:
        estimated += [
            ("Confirmed successful, with the frontier model's cost supplied by the caller", count(caller)),
            ("Net saving on those, from the caller's baseline", _caller_saving(ledger)),
        ]
    estimated.append(("Confirmed successful but in none of these figures, because a cost or the baseline is not "
                      "known", _left_out(ledger)))
    unmeasured = "Estimated, not measured" if caller == 0 else "Estimated or caller-supplied, not measured"
    spend = [
        ("Known answer spend, all recorded requests", money(ledger.get("known_answer_cost_total_usd"))),
        ("Known routing spend, all recorded requests", money(ledger.get("known_routing_cost_total_usd"))),
        ("Known spend, answers and routing together", money(ledger.get("known_cost_total_usd"))),
        ("Known spend per confirmed success", money(ledger.get("known_cost_per_verified_success_usd"))),
    ]
    notes = []
    if _live(ledger):
        saving, outside = _number(ledger.get("verified_savings_usd")), _outside_measured(ledger)
        if saving is not None and saving < 0:
            which = "On the requests the measured saving covers" if outside else "On these requests"
            notes.append(f'<p class="warn">{which}, routing cost more than the baseline.</p>')
        if outside:
            notes.append(f"<p class=\"warn\">The measured saving covers {count(ledger.get('comparable_verified_requests'))}"
                         f" of {count(ledger.get('verified_successes'))} confirmed successes. Under live routing the "
                         "frontier model's cost is measured only for requests that escalated to it, which paid for "
                         "cheaper attempts first, so this figure is not the policy's saving. The estimated figure "
                         "below and any shadow projection cover the requests cheaper models answered; neither is a "
                         "measurement.</p>")
    else:
        notes.append("<p>No confirmed successful request has known answer and routing costs and a measured "
                     "baseline. Until one does, no measured saving is stated.</p>")
    return ("<section><h2>Ledger: requests served through the proxy</h2>"
            "<p>The measured saving is net of routing cost. It counts only requests whose outcome was confirmed "
            "successful, whose answer and routing costs are known, and whose baseline, the frontier model's cost, "
            "was measured.</p>"
            f"{''.join(notes)}{_rows(requests, 'Requests')}"
            f"{_holdout_table(ledger)}"
            f"{_rows(measured, 'Measured against the frontier model')}"
            f"{_rows(estimated, unmeasured)}"
            "<p class=\"small\">An estimated baseline is the returned answer's token counts at the frontier model's "
            "prices. The frontier model was not called, and its answer could have been longer or shorter. The "
            "estimated figure is never added to the measured one.</p>"
            f"{_rows(spend, 'Spend')}"
            "<p class=\"small\">Known spend sums only measured costs across every recorded request, whatever its "
            "outcome. It is incomplete while any cost is not known, and spend per confirmed success is not known "
            "until every request has both costs.</p></section>")


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


def _guarantee_rows(guarantee: dict, frozen: str) -> list[tuple[str, str]]:
    """The guarantee as rows. ``frozen`` is ``freeze_state`` of the policy file: a file
    edited after freezing is not shown as certified, whatever it says."""
    epsilon, delta = _number(guarantee.get("epsilon")), _number(guarantee.get("delta"))
    limit, confidence = NOT_KNOWN, NOT_KNOWN
    if epsilon is not None:
        limit = (f"{percent(epsilon)} (epsilon = {_esc(epsilon)}): the most the policy may lose, as a share of "
                 "tasks, where the frontier model succeeds and the policy does not")
    if delta is not None:
        confidence = f"{percent(1 - delta)} (delta = {_esc(delta)})"
    certified = guarantee.get("certified")
    state = "yes" if certified is True else "no" if certified is False else NOT_KNOWN
    if certified is True and frozen == FREEZE_EDITED:
        state = "not confirmed: the file was edited after freezing"
    return [
        ("Policy file", frozen),
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
        about.append(f"Tenant named in the policy file: <code>{_esc(policy.get('tenant'))}</code>.")
    if any(isinstance(stage, dict) and stage.get("check") == "agreement" for stage in stages):
        pattern = policy.get("answer_pattern")
        about.append(f"Agreement compares the answer the pattern <code>{_esc(pattern)}</code> reads from each "
                     "reply." if isinstance(pattern, str) and pattern else
                     "The policy file names no pattern for reading answers: the proxy uses its own, if one is "
                     "set, and otherwise compares whole replies, which free text rarely matches.")
    guarantee = policy.get("guarantee") if isinstance(policy.get("guarantee"), dict) else {}
    frozen = freeze_state(policy)
    edited = ('<p class="warn">This policy file was edited after it was frozen: its content does not match the '
              "digest <code>tine freeze-policy</code> wrote into it. The guarantee below is what the file says, not "
              "what was certified.</p>" if frozen == FREEZE_EDITED else "")
    parts = ["<section><h2>The policy in plain words</h2>",
             "<p class=\"small\">This describes the policy file supplied when the page was made. The reports do not "
             "name the policy behind each figure, so the page cannot confirm that every figure above came from "
             "it.</p>", edited,
             f"<p>{' '.join(about)}</p>" if about else "", plan, _rows(_guarantee_rows(guarantee, frozen))]
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
            "and the spending cap let it through. The figures describe the shadowed requests, not all traffic. "
            "Where the report counts them, sampled requests that were skipped or streamed are shown beside the "
            "projection.",
            "<strong>Shadow mode is not free.</strong> Its calls are real calls on the tenant's own provider "
            "account, and they send the tenant's prompt to the cheaper models on that account.",
        ]
    if isinstance(ledger, dict):
        items += [
            "<strong>The ledger saving is a subtotal.</strong> It is net of routing cost and counts only confirmed "
            "successful requests with known answer and routing costs and a measured baseline. Failed, pending and "
            "incompletely costed requests are left out, and so are requests whose baseline was estimated. It is "
            "not a billing amount.",
            "<strong>The measured saving uses measured baselines only.</strong> When a cheaper model answered, the "
            "frontier model was not called, so its cost for that request was not observed, and the request never "
            "counts in the measured saving. A baseline the caller supplies later through the outcome endpoint "
            "is shown on its own row, as the caller's figure, not measured. Where the proxy is set to "
            "estimate the baseline, the result is shown on its own row as estimated, not measured: the returned "
            "answer's token counts at the frontier model's prices. A frontier answer could have been longer or "
            "shorter.",
            "<strong>Under live routing the measured saving leans to the hardest requests.</strong> It covers "
            "only the requests that escalated to the frontier model, which paid for cheaper attempts first. It "
            "can show a loss while the policy saves on the requests cheaper models answered.",
            "<strong>Outcomes are reported by the caller.</strong> So are costs the proxy did not record, such "
            "as routing cost without a route policy, and baselines shown as supplied by the caller. This page "
            "does not verify them. The caller cannot replace a baseline the proxy measured, or a cost it recorded "
            "on such a request.",
        ]
        if _holdout(ledger) is not None:
            items.append(
                "<strong>The holdout saving is an estimate from a random sample.</strong> It compares the mean cost "
                "per request of routed requests with that of requests held out for the frontier model, so it assumes "
                "the two groups are alike; choosing them at random is what makes them so. Its interval is a normal "
                "approximation. It does not measure quality: whether the routed answers were as good as the frontier "
                "model's is not part of it."
                + (" Some routed requests have a cost that is not fully known; they are counted at the part that is "
                   "known, so the routed mean is a lower limit and the saving an upper limit."
                   if _upper_limit(_holdout(ledger)) else ""))
        if "estimated_baseline_requests" not in ledger:
            items.append(
                "<strong>This ledger report does not say how many baselines were estimated.</strong> It may come "
                "from a version of the ledger that counted estimated baselines in the measured saving.")
    if isinstance(policy, dict):
        items.append(
            "<strong>The quality limit describes the graded tasks the policy was certified on.</strong> It says "
            "nothing about traffic that differs from them. The quality guarantee comes from that certification, "
            "not from any comparison on this page.")
        items.append(
            "<strong>The policy file is not signed.</strong> The digest written when it was frozen shows whether "
            "the file has changed since, but anyone who can edit the file can also write a new digest.")
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
               ("Ledger estimated baselines", _get(ledger, "estimated_baseline_scope")),
               ("Ledger holdout estimate", _get(_holdout(ledger), "scope")),
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
.lead{margin:.25rem 0 .75rem;font-size:1.02rem}
table.figures{width:100%;border-collapse:collapse;margin:.75rem 0;font-variant-numeric:tabular-nums}
table.figures caption{text-align:left;font-weight:600;padding:.5rem 0 .35rem;caption-side:top}
table.figures th,table.figures td{padding:.5rem 0;border-top:1px solid var(--rule);vertical-align:top}
table.figures th{text-align:left;font-weight:400;color:var(--muted);padding-right:1rem;width:58%}
table.figures td{text-align:right;font-weight:600;overflow-wrap:anywhere}
ol,ul{margin:.5rem 0;padding-left:1.25rem}
li{margin:.4rem 0}
code{font:.9em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
.small{font-size:.875rem;color:var(--muted)}
.warn{color:var(--warn);font-weight:600}
.alarm{margin:0 0 1rem;padding:.75rem 1rem;border:2px solid var(--warn);border-radius:6px;color:var(--warn);
font-weight:600;overflow-wrap:anywhere}
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
    sentences = headline_sentences(ledger_report, shadow_report)
    # The drift warning heads the headline; it is shown above the title line, set apart.
    alarm = _drift_warning(shadow_report)
    if alarm:
        sentences = sentences[1:]
    alarmed = f'<p class="alarm" role="alert"><strong>Warning.</strong> {_esc(alarm)}</p>' if alarm else ""
    first, *rest = sentences
    lead = f'<p class="lead">{_esc(" ".join(rest))}</p>' if rest else ""
    sections = (_shadow_section(shadow_report), _ledger_section(ledger_report), _policy_section(policy),
                _limits_section(ledger_report, shadow_report, policy))
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>Savings report: {_esc(tenant)}</title><style>{STYLE}</style></head><body><main>"
        f'<header><p class="kicker">Tine savings report</p>{alarmed}'
        f"<h1>{_esc(first)}</h1>{lead}"
        f'<p class="meta">Tenant <strong>{_esc(tenant)}</strong>. Generated {_esc(_when(generated_at))}.</p>'
        f"{noted}</header>"
        f"{''.join(sections)}"
        "<footer>This page is self-contained. It loads nothing from the network and runs no script.</footer>"
        "</main></body></html>\n"
    )
