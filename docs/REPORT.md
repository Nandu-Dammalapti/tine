# Savings report page

`scripts/savings_report.py` writes one tenant's savings report as a single self-contained HTML file. It builds the same two reports the proxy serves (`GET /v1/savings` from `Ledger.report`, `GET /v1/shadow` from `ShadowStore.report`) and hands them to `tine.report_page.render`. The page adds almost no arithmetic: every cost, saving and share is one the reports computed.

## Usage

```bash
python scripts/savings_report.py --ledger LEDGER --tenant TENANT --policy POLICY.json --output report.html
python scripts/savings_report.py --report-json saved.json --tenant TENANT --note "Where the figures came from" --output report.html
```

| option | meaning |
| --- | --- |
| `--ledger` | ledger SQLite file; opened read-only |
| `--report-json` | JSON with a `shadow_report` and/or `ledger_report` object, in place of a ledger |
| `--tenant` | tenant the page is for; a saved report or policy for another tenant is refused |
| `--policy` | frozen policy JSON to describe on the page: one policy as `tine freeze-policy` writes it, or the file the proxy reads (`TINE_SHADOW_POLICIES` or `TINE_ROUTE_POLICIES`: tenant, then provider, then policy), from which the `--tenant` policy is taken |
| `--provider` | `anthropic` or `openai`: which policy to take from the proxy's file when the tenant has one for both providers |
| `--generated-at` | time to print on the page; default is now, in UTC |
| `--note` | one line shown under the headline, for example where the figures came from |
| `--output` | HTML file to write |

- `--ledger` opens the SQLite file read-only and never runs the `Ledger` or `ShadowStore` constructors, which would change the file. A ledger with no shadow table gives no shadow section.
- `--report-json` reads saved reports, such as the output of `scripts/shadow_replay.py` or saved `GET /v1/savings` and `GET /v1/shadow` responses. No model is called and nothing is sent anywhere.
- With `--ledger` and `--policy`, the script compares the policy with the digest stored on each shadow row. If any shadowed request ran with a different policy, it adds a note to the page and prints a warning.
- With `--policy`, it warns on stderr when the policy file was edited after freezing, and refuses a policy with malformed guarantee numbers ([POLICY.md](POLICY.md)). A policy for another tenant is refused with "The policy file names tenant ...".
- With `--ledger` and `--policy`, the shadow report checks accepted answers that differ from the frontier's, as a share of the shadowed requests whose answer was compared or sent to the frontier, against that policy's epsilon (the drift check in [PROXY.md](PROXY.md)). Without `--policy` there is no drift check. With `--report-json` the warning appears only if the saved report holds one.

## What is on the page

1. **Headline.** One title line, then any sentences that qualify it. When the ledger report's `holdout_estimate` has a figure, it leads as the measured saving under routing: "Measured under routing, from a random frontier holdout: routed requests cost X% less per request than requests held out for the frontier model (95% interval for the saving: L% to H%), on N routed requests and M held out. It is an estimate from a random sample, and it does not measure quality." A negative figure is worded as routed requests costing more, "so there is no saving". When `saving_is_upper_limit` is true (some routed requests have a cost that is not fully known and are counted at their known part), the sentence says "at most X% less" and "95% interval for that upper limit", gives the routed count with how many of them were counted at their known part, and a negative figure reads "at least X% more". The sentences below it then say that the per-request measured figure covers escalated and held-out requests only. Otherwise a measured saving leads, then a shadow projection, then a saving on estimated baselines. When the measured saving leaves out confirmed successes, the title says how many it covers and gives no dollar figure; the sentences below give the measured amount and say that under live routing the frontier's cost is measured only for requests that escalated to it, so the figure is not the policy's saving. They then point to the estimated figure ("Estimated, not measured") and the shadow projection ("Projected, not measured"), or say neither exists. With nothing to stand on, the headline says so instead of showing a zero.
   - When the shadow report carries `drift_warning`, the first headline sentence is that warning, in a boxed paragraph above the title line ("Warning. Cannot confirm the certification on this traffic: ...").
   - When shadow coverage is below 100%, or any sampled request streamed, a sentence follows the projection: "Shadow mode ran on X of Y sampled requests (Z%); the rest were skipped (...), so busy periods count for less than they should." and how many sampled requests were streamed. Coverage is rounded down. Under live routing it follows the "Projected, not measured" sentence.
2. **Note**, from `--note` and any policy-mismatch note.
3. **Shadow mode.** Requests shadowed; answered by cheaper models; same answer as the frontier, or "not compared" for code checks; requests with all costs known and those left out; frontier cost, projected policy cost and projected saving; after it, "Sampled requests shadow mode could have run" (with the share shadowed), "Skipped because shadow runs were busy", "Skipped because shadowing had stopped", "Shadow runs that failed before they were recorded" and "Sampled requests that were streamed, never shadowed"; when the report counts them, "Skipped while shadow mode was paused for the provider's rate limit", "Skipped over the shadow call cap per minute" and "Shadow runs left out because the provider refused a call (429 or 529)"; with a report from `GET /v1/shadow`, "Paused for a provider's rate limit when the report was read" (each provider, until when and why; a report read from the ledger has no pause state, which the proxy keeps in memory); what shadow mode itself cost, shown as "at least" when some calls or checks have unknown cost, with that count on its own row; shadow calls made and failed; "Sent to the frontier model untried, because the request defined tools" when that count is above zero; "Accepted answers that differ from the frontier model's" (count, and the share rounded up) and "Checked against the policy's quality limit", with the drift warning repeated; sampling state when the saved report has it.
4. **Ledger**, in four tables, five with a holdout: requests (recorded, confirmed, failed, pending, cost not known, cache hits); "Measured under routing: random frontier holdout", when the report has a `holdout_estimate` (requests in each group with known costs; routed requests counted at the known part of their cost, or in a report from before that change those left out; held-out requests left out; the mean cost per routed and per held-out request, the saving and its 95% interval, the method in words; with routed costs not fully known the routed mean reads "at least", the saving "at most" and the interval "an upper limit"; until each group has 30 requests the saving reads "not shown until each group has 30 requests with known costs"); measured against the frontier model (answer cost, routing cost, the frontier's measured cost, the measured net saving); estimated, not measured (successes whose frontier cost was estimated, the estimated saving on them, and successes in none of these figures); spend (known answer, routing and total spend, and known spend per confirmed success). The measured saving uses only baselines the proxy measured: baselines a caller supplied through the outcome endpoint are left out of it. When any confirmed success has one, the third table is titled "Estimated or caller-supplied, not measured" and adds how many and the saving on them, marked "the caller's figure, not measured", and the headline adds a "Supplied by the caller, not measured" sentence after the estimated one.
5. **The policy in plain words**, when `--policy` is given: each stage as a sentence, the frontier fallback, the quality limit and confidence, whether it is certified, the loss bound and the graded tasks it rests on, and the offline estimate made when it was frozen. The guarantee table starts with a "Policy file" row giving the freeze digest state, and names the "Tenant named in the policy file". When the file was edited after freezing, "Certified at this limit" reads "not confirmed: the file was edited after freezing" and a warning paragraph is shown. For a policy with agreement stages it says which pattern reads answers, or that the file names none.
6. **What this does and does not show**: the limits in plain words, including "The policy file is not signed.", and every scope text the reports carry, word for word.

## Rules the page keeps

- A missing or null value, or a value of the wrong type, is shown as "not known", never as 0.
- Every value from data is HTML-escaped, including tenant, time, note, model and check names, scope texts and sampling state.
- The page loads nothing: no script, font, image or link. Styling is one `<style>` block with light and dark colours, and the page still reads correctly where a mail client drops styles.
- No percentage is rounded in the policy's favour. Savings and shares are rounded down (53.46% shows as 53.4%); the loss bound and observed loss are rounded up. Dollar amounts are rounded to the nearest shown digit, with three significant digits below $1.
- A saving against an estimated baseline is never added to the measured saving and is always labelled "estimated, not measured". A shadow figure is always labelled a projection. The holdout saving has its own table and sentence; no estimated or projected figure is added to it.
- The holdout saving and both ends of its interval are rounded down.
- A loss is called a loss.

## Limits

- Built and tested with stand-in data and a replay of benchmark answers only. It has never been made from real traffic or seen by a customer. On 2026-11-16 a team member opened the two example pages (`artifacts/demo/savings-report-replay.html`, `savings-report-live.html`) in a desktop browser and confirmed they render correctly. Contrast was checked by computing it (every text colour passes WCAG AA in both themes); layout at phone width and in mail clients has not been checked.
- Outcomes and baseline costs come from the caller; the page does not verify them.
- The ledger report does not say which requests escalated. The page infers the escalation story from counts: when estimated baselines exist, routing is certain; otherwise the explanation is worded "if the proxy routes through a policy".
- An estimated baseline is the returned answer's token counts at the frontier's prices; a frontier answer could have been longer or shorter. The ledger report gives no estimated baseline total, so the estimated saving has no percentage.
- Shadow spend sums only known costs and is a lower limit while any call or check has unknown cost.
- A streamed request's answer cost counts in known spend when its stream carried usage ([PROXY.md](PROXY.md)). Its frontier cost is never measured, so it is never in the measured saving.
- Shadow comparisons are equality of answers, not correctness.
- The holdout saving is an estimate from a random sample with an approximate interval. It assumes the routed and held-out requests are alike, which the random choice provides, and it pools every policy and audit rate in the ledger. It does not measure quality. While some routed requests have a cost that is not fully known, it is an upper limit on the saving, and the page says so. It has been shown only with stand-in data, never with real traffic.
- The reports do not name the policy behind each figure; the digest check that catches a mismatch runs only with `--ledger`.
- The page does not show `caller_baseline_*` from `GET /v1/savings`, or `check_rejection_reasons` and `check_step_rejections` from `GET /v1/shadow`, yet.
- Saved report JSON from before the coverage and drift fields (for example the checked-in `results/2026-11-16-shadow-replay` JSON) has none of them, so neither the coverage rows nor the drift warning appear on a page made from it. That JSON also predates the change that counts calls with no extracted answer as failed: a re-run with current code reports 317 failed shadow calls where it shows 0, and would show the drift warning (58 of 1,099 accepted answers differ from the frontier's, 4.10% of the 1,414 shadowed requests, against a 2% limit) at 100% coverage. The example pages in `artifacts/demo/` are local files that git ignores, so they are not regenerated by any commit; regenerate them with the commands above and `artifacts/demo/make_live_example.py` before showing them.
- Figures rounded to the nearest tenth elsewhere (for example 53.5% in `results/2026-11-16-shadow-replay/README.md`) show one step lower here by design.

## Tests

```bash
pytest tests/test_report_page.py
```
