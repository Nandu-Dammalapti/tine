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
| `--policy` | frozen policy JSON to describe on the page |
| `--generated-at` | time to print on the page; default is now, in UTC |
| `--note` | one line shown under the headline, for example where the figures came from |
| `--output` | HTML file to write |

- `--ledger` opens the SQLite file read-only and never runs the `Ledger` or `ShadowStore` constructors, which would change the file. A ledger with no shadow table gives no shadow section.
- `--report-json` reads saved reports, such as the output of `scripts/shadow_replay.py` or saved `GET /v1/savings` and `GET /v1/shadow` responses. No model is called and nothing is sent anywhere.
- With `--ledger` and `--policy`, the script compares the policy with the digest stored on each shadow row. If any shadowed request ran with a different policy, it adds a note to the page and prints a warning.

## What is on the page

1. **Headline.** One title line, then any sentences that qualify it. A measured saving leads, then a shadow projection, then a saving on estimated baselines. When the measured saving leaves out confirmed successes, the title says how many it covers and gives no dollar figure; the sentences below give the measured amount and say that under live routing the frontier's cost is measured only for requests that escalated to it, so the figure is not the policy's saving. They then point to the estimated figure ("Estimated, not measured") and the shadow projection ("Projected, not measured"), or say neither exists. With nothing to stand on, the headline says so instead of showing a zero.
2. **Note**, from `--note` and any policy-mismatch note.
3. **Shadow mode.** Requests shadowed; answered by cheaper models; same answer as the frontier, or "not compared" for code checks; requests with all costs known and those left out; frontier cost, projected policy cost and projected saving; what shadow mode itself cost, shown as "at least" when some calls or checks have unknown cost, with that count on its own row; shadow calls made and failed; sampling state when the saved report has it.
4. **Ledger**, in four tables: requests (recorded, confirmed, failed, pending, cost not known, cache hits); measured against the frontier model (answer cost, routing cost, the frontier's measured cost, the measured net saving); estimated, not measured (successes whose frontier cost was estimated, the estimated saving on them, and successes in neither figure); spend (known answer, routing and total spend, and known spend per confirmed success).
5. **The policy in plain words**, when `--policy` is given: each stage as a sentence, the frontier fallback, the quality limit and confidence, whether it is certified, the loss bound and the graded tasks it rests on, and the offline estimate made when it was frozen.
6. **What this does and does not show**: the limits in plain words, and every scope text the reports carry, word for word.

## Rules the page keeps

- A missing or null value, or a value of the wrong type, is shown as "not known", never as 0.
- Every value from data is HTML-escaped, including tenant, time, note, model and check names, scope texts and sampling state.
- The page loads nothing: no script, font, image or link. Styling is one `<style>` block with light and dark colours, and the page still reads correctly where a mail client drops styles.
- No percentage is rounded in the policy's favour. Savings and shares are rounded down (53.46% shows as 53.4%); the loss bound and observed loss are rounded up. Dollar amounts are rounded to the nearest shown digit, with three significant digits below $1.
- A saving against an estimated baseline is never added to the measured saving and is always labelled "estimated, not measured". A shadow figure is always labelled a projection.
- A loss is called a loss.

## Limits

- Built and tested with stand-in data and a replay of benchmark answers only. It has never been made from real traffic or seen by a customer, and it has never been viewed in a browser: the sandbox this was built in blocked a headless browser. Contrast was checked by computing it (every text colour passes WCAG AA in both themes); layout at phone width was not checked by eye.
- Outcomes and baseline costs come from the caller; the page does not verify them.
- The ledger report does not say which requests escalated. The page infers the escalation story from counts: when estimated baselines exist, routing is certain; otherwise the explanation is worded "if the proxy routes through a policy".
- An estimated baseline is the returned answer's token counts at the frontier's prices; a frontier answer could have been longer or shorter. The ledger report gives no estimated baseline total, so the estimated saving has no percentage.
- Shadow spend sums only known costs and is a lower limit while any call or check has unknown cost.
- Shadow comparisons are equality of answers, not correctness.
- The reports do not name the policy behind each figure; the digest check that catches a mismatch runs only with `--ledger`.
- Figures rounded to the nearest tenth elsewhere (for example 53.5% in `results/2026-11-16-shadow-replay/README.md`) show one step lower here by design.

## Tests

```bash
pytest tests/test_report_page.py
```
