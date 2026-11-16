# Learning loop: certifying a policy from a tenant's own traffic

**This is a design. None of it is built or run.** Numbers marked "replay" come from one
replay of public benchmark answers (`results/2026-11-16-shadow-replay`), not from a
customer.

## 1. The claim

For one tenant, one task type and one policy hash, a certificate says:

> Between [start] and [end] we picked n of your eligible requests by a coin that
> ignored the request and every answer, and ran this policy beside your frontier
> model [exact id]. On K of them the policy accepted a cheaper answer that we could
> not show to be harmless. With 95% confidence (delta 0.05), the share of all eligible
> requests in that period that the frontier gets right and the policy gets wrong is at
> most U, and U is at or below your limit.

It describes a past block. It holds under drift, because the randomness is the
coin, not the traffic.

Assumptions, printed with it:

1. Right and wrong mean what the tenant's grader [name, version] says.
2. For exact-answer work, two equal extracted answers are both right or both wrong.
3. Policy, models, checks, task-type rule, n, limit and grade deadline were fixed in
   writing before the block began.
4. Eligible means single-turn, no tools, not streamed, not served from cache.

Routing tomorrow's traffic on a certificate assumes tomorrow resembles the block.
That is not guaranteed. An expiry date and a tripwire limit the exposure.

**Smaller claim, when grades are sparse.** An accepted answer that differs from the
frontier's and has no grade counts as a loss. With no grades at all, the bound is
on how often the policy's answer differs from the frontier's. Under the limit, the
full claim holds. Over it, we report the bound, do not route, and say how many
labels would close the gap.

## 2. What is recorded, and what is never stored

Recorded, in the tenant's own database:

- `shadow_runs` (exists) gains `task_type`, `block_id`, `config_sha256` and `status`.
  Every sampled id gets a row. Today `run_shadow` writes none when a run fails or
  hits the spend cap. A missing run counts as a loss.
- `audit_answers` (new): per model, the returned model id, a keyed hash of the
  extracted answer, cost, error.
- `audit_grades` (new): policy correct, frontier correct, source, grader version.
- `audit_review` (new): mismatches only, the two extracted answers, deleted at
  grading or at the deadline.
- `certificates` (new): block dates, n, losses, how many were only missing grades or
  runs, bound, limit, delta, hashes, expiry.

Never stored: any text outside the tenant's deployment; prompt or answer text in
policy files or reports; judge reasoning; user identities; credentials. Tine
receives counts, bounds and costs, and only if the tenant exports them.

To fix first: the ledger keeps `request_json` and `response_json` forever. And a
keyed hash of "B" hides nothing from the key holder.

## 3. Where grades come from

A judge model is never the grader: on SWE-bench 14% of judge-accepted patches were
wrong. Outcomes reported for served answers (`requests.success`) never enter a
certificate: people accept what they are shown, and the frontier's answer is
missing. They feed the tripwire only.

Grades come from an **audit slice**: a share of requests (`TINE_AUDIT_RATE`, new),
chosen by `sampled(request_id, rate)`, always answered by the frontier, with the
policy run in the background. It continues after routing starts; today a tenant has
a shadow policy or a route policy, never both. Grades arrive through
`PATCH /v1/evidence/{request_id}` (new).

| Work | Grade | Bias |
| --- | --- | --- |
| Exact answer | Equal answers: no loss, no grade. A mismatch is a loss unless a reference label or reviewer shows the cheap answer right or the frontier wrong. Replay: 4.1% of rows mismatch, about 12 per 300. A 4% block allows 6, so half must be cleared by a label. | The reviewer is not told which model wrote which answer, but may have seen the frontier's. |
| Code with tests | The tenant's CI runs the unserved patch against tests merged later. Tine never runs it. | Tests follow the served frontier patch, so a right cheap patch can fail, and unmerged work counts as a loss (both safe). Thin tests pass wrong patches (unsafe, not bounded). |
| Free text | None. | Not certified. Stays on the frontier. |

We expect code to be refused at first: at a 4% limit, losses and missing grades
together may not exceed 2% of a block.

## 4. The nightly job

1. Purge expired review rows and ledger text (new).
2. `tine evidence export` (new) writes the outcome and check tables
   `certify_policies` takes, with keyed ids and no text.
3. Tripwire: if losses in the open block exceed what n allows, or reported failures
   on cheap-served answers run above the slice's, write `stages: []`: all traffic
   returns to the frontier. A block may stop early for failure, never for success.
4. When a block closes (the first n sampled rows by arrival, each graded or past the
   deadline), `certify_block` (new) computes `clopper_pearson_upper(K, n, delta)`
   once and writes the `certificates` row. Over the limit: `stages: []`.
5. Choose the next ladder: at most three nested chains, ordered on rows from closed
   blocks, where "equals the frontier's answer" may stand in for a grade. Its hash
   is recorded before the next block opens, and `certify_policies` tests it there in
   fixed order. Audit rows run every stage (a change to `run_stages_async`), so each
   chain can be scored.
6. `validate_policy` (changed) refuses stages unless `guarantee.source` is
   `tenant-audit`, the hash matches and the certificate has not expired.

`freeze_policy` is not reused whole: it splits at random and needs every model
graded. "Nightly re-freeze" becomes nightly evidence and demotion; certificates
change per block. A failed policy hash is not retested unchanged.

## 5. Sample sizes

Exact one-sided Clopper-Pearson, delta 0.05.

| Limit | Zero losses needs | Block n | Losses allowed | Passes if true rate is 1% | 1.5% | 2% |
| --- | --- | --- | --- | --- | --- | --- |
| 4% | 74 | 300 | 6 | 97% | 83% | 61% |
| 2% | 149 | 1,000 | 12 | 79% | 27% | 4% |
| 2% | 149 | 1,500 | 20 | 92% | 35% | 3% |

Days = n / (eligible requests per day x audit rate) + grade deadline. At 2,000 a day
and a 10% slice, 300 rows take 1.5 days and 1,500 take 7.5. At 200 a day, 300 take
15 days and 1,500 take 75: a 2% limit is out of reach there.

Cost: shadowing every request cost 18% of the frontier bill in the replay; running
every stage costs more. With audit share a, net saving is about
(1 - a) x S - a x 0.18. For the replay's S = 53.5% and a 10% slice that is 46.4%:
arithmetic, not a measurement. Labels and CI minutes are `routing_cost_usd` too.

## 6. Drift and model replacement

- A certificate expires at twice the planned block length.
- `config_sha256` covers the model ids providers return, check configuration, judge
  prompt, answer pattern, task-type rule and grader version, not prices. Any change
  voids the certificate and that traffic returns to the frontier.
- A new cheap model enters only through a new ladder and block. A new frontier voids
  everything.
- A provider can change weights under the same id. We cannot see that.
- Delta is per certificate. A policy just over the limit, tested every two weeks,
  has a 74% chance of at least one false pass in a year.

Isolation: one database, key and evidence folder per tenant. Nothing is pooled.
A public-benchmark policy seeds the first ladder and may shadow, never route.

## 7. A two-week pilot (exact-answer work, 2,000 requests a day)

- Day 0: limit, n, deadline, grader, ladder, spend cap, retention and the models
  allowed to see prompts are signed.
- Days 1-2: frontier only. Check the answer pattern finds an answer in 95% of
  responses. Dry-run the grader. Go or no-go: is there enough gradeable traffic?
- Days 3-6: shadow under the cap, to order the ladder. Frozen and hashed on day 6.
- Days 7-9: certify block, shadow at rate 0.25. 300 rows (4%) fill in a day, 1,500
  (2%) in three. Mismatches are labelled daily.
- Days 10-11: deadline, one computation. The customer reads every loss.
- Days 12-14: if certified, route with a 10% slice. If not, the report says which
  ran short: rows, grades or quality.

Code will not route in these two weeks.

## 8. What we cannot guarantee

- Any single answer, or any future period.
- Any routed request, until a routed block closes. The first block describes a
  period when the frontier served everything.
- Absolute accuracy. The bound is against the frontier.
- That the grader is right.
- Streaming, multi-turn, tool-use, cached and free-text requests: most agent traffic.
- Positive savings. A failed pilot costs shadow spend and reviewer time.
- That prompts stay with one provider. Shadow and audit send them to every cheaper
  model the tenant allows.

## 9. Open questions

1. Should the unit be a task or session, keyed on a caller-supplied reference?
2. Can a tenant's CI grade an unserved patch at 98% coverage, in a sandbox their
   security team accepts?
3. What retention limit can the verified cache live with?
4. Should measured grader disagreement enter the bound or be printed beside it?
