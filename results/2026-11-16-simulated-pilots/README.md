# Tine pilot readiness: five simulated pilots

Date: 2026-11-16. For Ashmith, Nandan and Nirvika.

**What this report is, and is not.** The five "customers" below are simulated teams: agents playing a buyer's role. They worked against stand-in providers (httpx.MockTransport) that replayed recorded benchmark answers or synthetic text. No real or paid model calls were made. `tine serve` could not bind a port in the sandbox (EPERM), so every proxy run was in-process. The Docker image was not built. The Colab notebook was not run. These runs test onboarding, wiring, docs, security and trust. They do not show real savings or real quality. Dollar figures are benchmark prices on short prompts, so only the ratios mean anything.

Each problem the teams reported was then checked by one to three independent verifiers, who reproduced it in-process and read the docs. The severities below are the verifiers' severities, which are often lower than the teams' own.

---

## 1. Bottom line

The core works as designed. Shadow mode never changed the caller's answer or its timing. Tenant isolation held in every test. The repo's tests pass (129 and 137 in two runs). Every headline number on the demo pages recomputes from its source file. The report pages are honest: projections are labelled, a loss is called a loss, and "not known" is never shown as zero. On benchmark multiple-choice data, the proxy reproduces the policy's offline estimate: 77.7% of requests answered by cheaper models, a 53.4% projected saving and about 1% loss against the frontier.

Tine is not ready for a real pilot yet:
- **Security.** The proxy has one security hole: a check config can send the provider key and every tenant token to a webhook.
- **Easy setups mislead.** With no answer pattern, savings disappear or go negative. With mixed traffic on one token, free text gets "accepted". Certified policies use cheap models that the customer's provider does not serve.
- **The shadow report hides gaps.** It leaves out what it skipped: streamed requests, busy-hour samples and drift.
- **Billing.** Under live routing, the product cannot measure the saving its price is based on.

Four of five teams would continue to a shadow-only pilot. None would pay for routing today. Fix the five items in section 6 first, then run a shadow-only pilot on one exact-answer workload, such as ticket classification.

---

## 2. The five pilots

| Team | What they wanted | How far they got | Continue? | Pay today? |
|---|---|---|---|---|
| Helpwise (support SaaS, $90k/mo, 60% ticket labels, 40% reply drafts) | Shadow mode, then a certified limit for routing | Shadow, live routing and a self-frozen policy, all on stand-ins | Yes, shadow only | Shadow free; routing only against a frontier holdout |
| Patchwork (coding agent, $140k/mo, Anthropic, mostly streaming) | Code chain (tests, then Haiku, then Opus) in shadow | Shadow and routing, after an undocumented workaround | Yes, shadow on batch path only | Not yet |
| Northbridge (bank platform team, 14 internal teams) | Security, isolation, audit before savings | Full security and failure review | Yes, only after a security fix | Not yet |
| Quill (writing startup, $25k/mo, mostly streamed drafts) | Live routing this week | Live routing on stand-ins | No | No |
| Buyer review (VP Finance plus Head of ML) | Diligence on the two vendor report pages | All numbers rechecked; live pilot simulated | Yes, free shadow pilot with conditions | Not for routing |

### Helpwise (support SaaS)

- **What they did.** They read the docs as newcomers. They reused the shipped multiple-choice policy, which meant hand-editing its tenant. They replayed 1,414 classification tasks using raw recorded replies, plus 943 synthetic drafts. They tried four setups: one token for all traffic, separate tokens per task type, live routing, and no answer pattern. They also froze their own policy offline and tested the spending cap, the sample rate and concurrency.
- **Results.**
  - With one token for everything, shadow projected a 10.8% loss. 113 of the 943 drafts were "accepted" because their lettered steps matched the multiple-choice pattern.
  - With separate tokens, the triage tenant showed 77.7% acceptance, a 53.4% projected saving and 0.99% graded loss.
  - At bill level that is about 17%, because triage was about 31% of frontier spend.
  - Under live routing, the ledger showed −28.4% measured saving, against a true 53.4%.
  - With no answer pattern, the projected saving was 0.5%.
  - At 20 and 50 requests per second, 36% and 68% of sampled requests were silently not shadowed.
- **Decision.** Run a 2 to 4 week shadow-only pilot on classification, under its own token, at a 5 to 10% sample with a cap of about $1k. In parallel, grade historical tickets and freeze a Helpwise policy. Drafts stay on the frontier.
- **Blockers.**
  - No certification on their own intents.
  - Cheap models must live on the frontier's upstream.
  - The split-token workaround is undocumented.
  - The live-routing baseline is an estimate.
  - Skipped samples are invisible.
  - Policies are unsigned.

### Patchwork (coding agent)

- **What they did.** They replayed 274 SWE-bench tasks through an Anthropic stand-in. 73 of the 274 were streamed. They used a CI webhook stand-in and recorded judge verdicts, ran 5 shadow variants and 2 routing variants, and then tested edge cases.
- **Results.**
  - With repository context in the user turn, acceptance was 4.5% and the projected saving was −14%. The judge silently skips any request with more than 12,000 characters of user text, and 252 of the 274 prompts were over that.
  - With context moved into `system`, the projected saving was 17 to 28%.
  - Streams passed through byte-identical, but they were never shadowed or costed.
  - Tool-use turns can never pass a code check.
  - About 5% CI flakiness ended shadow mode for good.
  - With a realistic regression-only webhook, 19 of 99 accepted patches were wrong (1.0 to 1.5% of tasks lost).
- **Decision.** Do not route. Shadow at about 2% on the non-streaming path only. Put a gateway in front of the candidate models. Certify on about 300 of their own graded tasks.
- **Blockers.**
  - The 12k judge limit.
  - Streaming.
  - Agent tool turns.
  - No policy certified for the recommended check.
  - One upstream per provider.
  - The lifetime unknown-cost stop.
  - Shadow rows cannot be audited.

### Northbridge (regulated bank)

- **What they did.** They read the code end to end. They then ran tests on isolation, webhook exfiltration, failure modes, shadow timing, policy tampering, data residency and pass-through bodies. They also tried seed shopping on freeze-policy.
- **Results.**
  - Isolation held in every case they tested, and no keys or tokens were stored in the ledger.
  - A webhook check config can send the provider key and all tenant tokens to any URL, including plain http.
  - The ledger fails open: if it cannot write, the answer is still served.
  - Routed rows do not record which policy served them, and outcome history is overwritten.
  - Fields such as `store:true` and `mcp_servers` pass through unchanged.
  - Certification held up statistically: 0 of 60 seeds certified when the true loss was above epsilon.
- **Decision.** Run a closed shadow pilot on one or two low-risk tenants, only after the webhook token hole is fixed. Then rerun this harness against the fixed build.
- **Blockers.**
  - The token hole.
  - No fail-closed audit.
  - No policy trace on routed answers.
  - Self-attested "verified" savings.
  - Unsigned policies.
  - No per-tenant controls.
  - No body field policy.
  - No retention or erasure.

### Quill (writing startup)

- **What they did.** They went straight to live routing with the shipped policy, after a one-line tenant edit. They tested edge cases, ran a Quill-shaped 200-request mix, edited the policy while the proxy was running, and finally tried shadow mode.
- **Results.**
  - On multiple-choice traffic, the true saving was 52.6%, but `/v1/savings` reported −26% "verified".
  - On the Quill mix, the modelled bill went up. It was +8.9% with no policy, because cheap model names were upgraded to the frontier, and +10.4% with the policy, because free-form drafts paid five cheap calls and then the frontier.
  - Streamed requests (60% of the mix) were not routed and not costed.
  - A free-form draft listing (A), (B), (C) was served from a 9B model.
  - Emptying the policy file had no effect until restart.
  - Shadow mode would have warned them, with a projected −30%.
- **Decision.** Not live routing. Put only non-streamed classification calls behind Tine, on their own token, in shadow mode.
- **Blockers.**
  - Streaming.
  - No check for free-form text.
  - Cheap-model upgrade unless allowlisted.
  - One upstream per provider.
  - No live kill switch.
  - Live-routing savings are not measurable.

### Buyer review (finance plus ML)

- **What they did.** They recomputed every number on both report pages and reran the replays. They simulated a live pilot and probed policy tampering, prose replies and drift.
- **Results.**
  - All headline numbers tie out to their source files.
  - The checked-in replay page comes from stale JSON: current code reports 317 failed shadow calls where the page shows 0.
  - Under live routing, the three numbers disagree: 53.4% projected, −28.4% measured, and $0.146 estimated (about 40% of the true saving).
  - With prose replies and no pattern, routing cost 17% more than the frontier.
  - An edited policy still shows "Certified: yes".
  - When cheap models agree on wrong answers for about 40% of tasks, the headline saving rises to 87% and the page shows no warning.
- **Decision.** A free, time-boxed shadow pilot on one graded exact-answer workload. Conditions: the vendor ships the answer pattern, certifies on 900 or more of the buyer's own tasks, hashes the policy, and adds a drift check.
- **Would pay.** A fixed $15 to 25k for 90 days. After that, a 10 to 15% success fee on the lower 95% bound of savings measured against a random always-frontier holdout.

---

## 3. Confirmed problems, ranked

Teams are named H (Helpwise), P (Patchwork), N (Northbridge), Q (Quill) and B (Buyer). Severity is after verification. Where several teams hit the same issue, the findings are merged.

**Status after the follow-up fixes (added 2026-11-16).** The tables below are the report as written. Fixes were made on branch `claude/judge-followup` and tested with stand-in providers only. The harness has not been rerun against the fixed build.

| # | Status |
|---|---|
| 1 | Fixed in the proxy: `bearer_env` may not name `TINE_*` or a provider key, needs https, and one variable serves one tenant. Docs say check files are operator-only. Other secrets in the proxy's environment can still be named. |
| 2 | Fixed: `freeze-policy --answer-pattern`; a route policy with agreement stages and no reader is refused; shadow mode warns; `/v1/shadow` shows the reader. |
| 3 | Partly fixed: `task_type` is logged at startup, requests with tools skip agreement policies, and the docs say "one tenant token per task type" and to check acceptance per token in shadow mode. `task_type` is still not read per request. |
| 4 | Partly fixed: PROXY, DEPLOY and POLICY state the one base URL and recommend a self-hosted gateway; failed calls record their HTTP status; the stop reason names the failures. `/health` still says "policy" when stage models are not served. |
| 5 | Fixed: coverage counters, `shadow_coverage`, streamed count and a per-tenant concurrency limit, shown on the page. Other ineligible traffic is not counted. |
| 6 | Fixed: the stop needs both a count and a share; resuming is documented. There is no reset endpoint. |
| 7 | Partly fixed: a drift warning against epsilon, in `/v1/shadow` and the page headline. It also fires on the healthy replay (5.28% differing against 2%). Comparing with the certification-time rate is not built. |
| 8 | Fixed: freeze digest, `--tenant`, stricter guarantee validation, "Tenant named in the policy file", "not confirmed" for an edited file. A digest is not a signature. |
| 9 | Documented: POLICY.md "Certify on your own graded tasks"; AGENTS.md counts now say certify tasks and give totals. |
| 10 | Documented only: PROXY.md and the `savings_scope` text say the saving cannot be measured under live routing. The holdout is not built. |
| 11 | Partly fixed: a streamed row's tokens and answer cost are read from the relayed stream and written when it ends, tested with stand-in streams. Streams are still not routed, and their baseline stays unknown. |
| 12 | Fixed: `baseline_source`; only proxy-measured baselines count as verified; caller-supplied ones are reported apart and cannot overwrite a measured one. |
| 13 | Fixed: `/health` checks the stages, load errors name the variable and path, and DEPLOY.md has "Turn routing off". Policies are still read only at startup. |
| 14 | Fixed for `TINE_LEDGER_REQUIRED`, `policy_sha256`, `route_stage` and `outcome_events`. No HTTP endpoint for the history. |
| 15 | Not started. |
| 16 | Fixed: the judged stage is skipped before the cheap call, with reason `request_too_long` in shadow runs; the request shape is documented. The reason is not stored under live routing. |
| 17 | Not started (tool requests skip agreement policies only). |
| 18 to 20 | Not started. |
| 21 | Documented: JUDGE.md says the recommended check is not certified; PROXY.md and JUDGE.md state the webhook contract. |
| 22 | Fixed in EVIDENCE.md, README.md and results/README.md. |
| 23 | Open. The demo pages are local files that git ignores and were not regenerated; the checked-in replay JSON still predates the recount. |

### 3.1 Fix before a shadow pilot

| # | Problem | Hit by | Sev. | Fix |
|---|---|---|---|---|
| 1 | **Webhook `bearer_env` can name any environment variable.** A check entry with `bearer_env: TINE_OPENAI_API_KEY` sent the provider key to an attacker URL over plain http. With `TINE_TENANT_KEYS`, every tenant token was sent. One tenant's entry can also send another tenant's CI token. Exploiting it needs write access to the check config, but the docs invite per-tenant entries. | N | Medium (security; fix first) | In `validate_checks`, refuse any `bearer_env` that starts with `TINE_`. Require https whenever a token is sent. Refuse a token variable that is shared across tenants. Document that check files are operator-only config. |
| 2 | **Policies carry no answer reader, and the proxy does not warn.** With no pattern, whole reply texts are compared. Helpwise projected 0.5% where 53.4% was expected. In the buyer's prose test, routing cost 17% more than the frontier (the verifiers saw 17 to 25%, depending on the stand-in prose). Quality stayed safe, but the savings vanish or turn negative. | H, B | Medium | Add `freeze-policy --answer-pattern` to write the reader into the policy. Have the proxy refuse an agreement route policy that has no reader, and warn in shadow mode. Show the reader in use in `/v1/shadow`. |
| 3 | **One policy runs on all of a tenant's traffic, and the pattern can match non-answers.** `task_type` is not read. In H, 113 of 943 synthetic drafts were accepted on a trailing "(c)". In Q, a 9B model's draft was served live. Free-form, tool and long requests pay every cheap stage and then the frontier. The only workaround, one token per task type, is undocumented. | H, Q | Medium | Document "one tenant token per task type". Log each route policy's `task_type` at startup. Skip agreement stages when a request carries `tools`. Tell users to check acceptance per token in shadow mode before routing. |
| 4 | **Certified policies need cheap models on the frontier's own base URL.** The shipped policies use open models that api.openai.com and api.anthropic.com do not serve. Each request then makes 2 to 5 failed calls, routing cost becomes unknown, shadow mode stops after 7 requests with a cost-sounding reason, and `/health` still says "policy". It does work through one OpenAI-compatible gateway, so "impossible" was overstated. | H, P, Q | Medium | In PROXY, DEPLOY and POLICY, state that every stage model and the judge must be served at that base URL. Recommend a self-hosted gateway such as vLLM or LiteLLM, because a third-party router conflicts with AGENTS.md. Record the HTTP status on failed calls, and make the stop reason name the failures. |
| 5 | **The shadow report hides what it did not shadow.** Requests are dropped silently when 8 runs are in flight (one global limit shared across tenants) or before the first run finishes: 0 / 1.7 / 36 / 68% at 1 / 5 / 20 / 50 req/s. Streamed requests are excluded and never counted. Busy hours are under-represented. | H, P, Q (N noted the noisy-neighbour case) | Medium | Add per-tenant counters (busy, not admitted, failed) and a `shadow_coverage` field. Add a count of streamed requests. Show both next to the projection. |
| 6 | **The unknown-cost stop is a lifetime counter.** About 5% CI flakes stopped shadowing after 200 to 400 requests, and it stayed stopped after a restart. | P | Medium | Stop only when unknown-cost calls are above both N and a share of all calls. Document how to resume. |
| 7 | **No drift alarm.** When cheap models agreed on a wrong answer for about 40% of tasks, the "same answer as frontier" share fell from 96.5% to 56%. True loss was at least 35% against a 2% limit. Meanwhile the headline saving rose to 87%, the page still said "Certified: yes", and there was no warning. | B | Medium | Warn ("cannot confirm the certification on this traffic") when the share of answers that differ from the frontier is above epsilon, and put the warning in the headline. Later, record the certification-time rate and compare against it. |
| 8 | **Policy files are unsigned, and edits still show as certified.** A relabelled benchmark policy, a hand-written `{certified: true}`, a bound written as a string (which skips the bound check), and epsilon 0.5 all load. The page then prints "Tenant it was frozen for: \<edited name\>" and "Certified: yes". `freeze-policy` has no `--tenant` flag, which pushes users to hand-edit. | H, P, N, Q, B | Medium | Write a freeze digest into provenance and flag a mismatch on the page. Add `freeze-policy --tenant`. Reject a non-numeric or out-of-range epsilon, delta or bound, and require a bound on a certified policy. Change the page label to "Tenant named in the file". Say plainly that a digest is not a signature. |
| 9 | **The path to certifying on your own data is not written down.** The tools exist (`agreement_pairs.py`, then `freeze-policy`), but the outcome schema, the answers CSV and the rule that the offline reader must match the proxy's pattern are not documented. The AGENTS.md counts of 300 and 900 are certify tasks only. At the default split, a customer needs about 1,000 graded tasks for 4% and about 3,000 for 2%. | H | Medium | Add a "Certify on your own graded tasks" section to POLICY.md, covering the files, the commands, the sizes and a cost formula. Correct AGENTS.md. |

### 3.2 Fix before live routing or any fee

| # | Problem | Hit by | Sev. | Fix |
|---|---|---|---|---|
| 10 | **Under routing, the saving the price is based on cannot be measured.** Only escalated requests get a measured baseline, and those also paid for the cheap attempts, so "verified savings" can never be positive. The estimated baseline came out at 0.45x the frontier's recorded cost, because the cheap answers were shorter. A finance reader sees three numbers that disagree. | H, Q, B | Medium | State this in the `savings_scope` text and in PROXY.md. Build a random frontier holdout (`TINE_AUDIT_RATE`, LEARNING_LOOP.md §3) with a confidence interval. Write down what is billable. |
| 11 | **Streamed requests are not routed, and their cost is not recorded,** even though the SSE stream carries usage data. Every streamed row has unknown cost. | P, Q | Medium | Read usage from the relayed stream without changing the bytes, and write the cost afterwards. Document that the streamed share caps any saving. |
| 12 | **A caller-supplied baseline counts as "verified".** One PATCH turned a $0.02 estimate into a 99.96% saving. It can also overwrite a baseline the proxy measured itself. | N | Low | Add a `baseline_source` column, count only proxy-measured baselines as verified, and report caller-supplied ones separately. |
| 13 | **The policy file is read only at startup.** Emptying or corrupting it does nothing until restart. After a restart with `stages=[]`, `/health` still says "policy". A corrupt file gives an error that names neither the file nor the variable. | Q | Low | Add a "Turn routing off" section to DEPLOY.md. Make `/health` check the stages. Name the variable and path in load errors. |

### 3.3 Fix before a regulated or coding customer

| # | Problem | Hit by | Sev. | Fix |
|---|---|---|---|---|
| 14 | **Audit gaps.** The ledger fails open: the answer is served with `x-tine-ledger: unavailable`. Routed rows have no policy digest or stage. Outcome updates overwrite earlier values. | N, Q | Medium | Add `TINE_LEDGER_REQUIRED` to return 503 when the ledger cannot write. Add `policy_sha256` and `route_stage` columns. Add an append-only `outcome_events` table. |
| 15 | **Request bodies are forwarded unchanged.** `store:true`, server tools, `mcp_servers` and `anthropic-beta` all pass through. Shadow mode repeats tool calls once per stage. | N | Medium | Add per-provider field overrides, denied fields and tool types, and a beta allowlist. Document that an egress firewall cannot block tools that run on the provider's side. |
| 16 | **The judge skips any request with more than 12,000 characters of user text,** with no reason recorded. Agents usually put repository context in the user turn. The workaround (context in `system`) matches what was measured. | P | Medium | Record a reason (`request_too_long`). Skip the stage before paying for the cheap call. Document the request shape. |
| 17 | **Tool-use turns can never pass a named check, but still pay for cheap calls and CI runs.** A webhook-only check can serve a tool-use reply. | P | Medium | Reject replies that are not final text before any check step runs. Do not shadow turns where the frontier itself called a tool. |
| 18 | **Code shadow runs cannot be audited.** Check entries lose the candidate model, there are no per-step results, and the webhook gets no request ID. | P | Medium | Keep `checked_model` and per-step results. Send `request_id` and mode (shadow or route) to the webhook. |
| 19 | **No per-tenant model allowlist, rate limit or spend cap.** The judge always posts to the global Anthropic URL, so an on-prem tenant's text left the network. | N | Low | Document the per-tenant check override. Add a per-tenant allowlist as an option. |
| 20 | **No retention, erasure or request size limit.** A 25 MB prompt was accepted and stored. | N | Low | Document manual erasure. Add `TINE_MAX_BODY_BYTES`. |
| 21 | **The recommended code judge (tests, then Haiku, then Opus) is uncertified.** The order was picked after seeing results, Opus's recall of these tasks was not measured, and the judge model is newer than the frontier it is measured against. The webhook in the pilot brief was an oracle (it agreed with the grade on 819 of 819 answers). | B, P | Low | Reword JUDGE.md. State the regression-only webhook contract in PROXY.md. |

### 3.4 Evidence and docs

| # | Problem | Hit by | Sev. | Fix |
|---|---|---|---|---|
| 22 | **The evidence brief is stale.** It says $12.69 of paid spend, against $47.73 actual. It still says judge memory is unmeasured, but the probes measured 41% recall and 2.3% loss on unrecalled tasks. | B | Medium | Update EVIDENCE.md, README.md and results/README.md. |
| 23 | **The demo replay page is built from stale JSON.** It shows 0 failed calls where current code gives 317. | B | Low | Regenerate the page. Relabel "failed" calls that still reported usage. |

---

## 4. Claims that did not hold

| Claim | Team | Why not |
|---|---|---|
| `/v1/savings` shows nothing in shadow mode until `routing_cost_usd: 0` is sent | P | In shadow mode the frontier answers everything, so the saving there is $0 by construction. The shadow saving is in `/v1/shadow` and on the page headline. |
| Cheap model names are silently upgraded to the frontier | Q (H also raised it, unverified) | Documented (README:9, PROXY.md:3 and :44) and tested. The served model is in `x-tine-model` and in the ledger. A README sentence is optional. |
| PATCHing the proxy's own estimate back turns it into "measured" | Q | Intended and documented (PROXY.md:177). The estimate is not exposed over HTTP. The real gap is #12. |
| One failing cheap model makes routing cost unknown, and that is too conservative | Q | Intended and tested. A call cut off by the deadline may still be billed. |
| Mixing an OpenAI frontier with open cheap models is impossible | H | It works through one OpenAI-compatible gateway (#4). |
| Following the docs requires hand-editing a certified policy | P | The docs say to freeze your own. The real gap is the missing recipe in JUDGE.md. |
| Moving code into `system` changes what the judge reads compared with the measured judge | P | Backwards. The measured judge read the issue only, which is the split shape. |
| The code chain loses 0 tasks | P (brief) | The `run_resolved` webhook was an oracle. A realistic webhook lost 1.0 to 1.5%. |

---

## 5. Unverified low-severity notes

Nobody checked these independently. Each is small.

- No company-level roll-up. A per-tenant page shows 53%, while the bill-level figure is about 17% (H).
- Shadow projections show no uncertainty interval. A 10% sample gave 57.8% against 53.4% on the full set (H).
- The routed page has no overall net figure (P).
- PROXY.md says "just before it" for when the judge is told the tests passed, but every later judge is told too (P).
- JSON nested 100,000 levels deep returns 500 instead of 400 (N).
- Streamed and non-JSON responses, and the reasons for failures, are not stored (N).
- The allowed-model fallback retries a 400 on the frontier and records a null cost (N).
- Webhook URLs may be http, loopback or link-local (N).
- Ledger rows name the tenant but not the token (N).
- freeze-policy provenance records the winning seed but not the number of attempts (N).
- Provider error bodies, including the org ID, reach callers verbatim (N).
- The default route deadline (30 s) is long for chat (Q).
- `x-tine-route: frontier` is shown when an allowlisted cheap model answered (Q).
- `shadow_replay.py --limit` takes alphabetically first tasks, which gave 81% against 52.5% (Q).
- "Quality limit 2%" and "94.7% same answer" are easy for finance to misread as accuracy (B).
- The default ledger path is inside the repo checkout (B).

---

## 6. Before a real pilot

### Top 5 fixes (for a shadow-only pilot)

1. **Close the webhook token hole (#1).** Refuse `TINE_*` names in `bearer_env`, require https whenever a token is sent, and allow one tenant per token variable. Without this, no security reviewer will approve.
2. **Make answer reading part of the policy (#2, #3).** `freeze-policy` writes `answer_pattern`. The proxy refuses an agreement route policy that has no reader and logs each policy's `task_type` at startup. The docs say "one tenant token per task type".
3. **Make the shadow report complete (#5, #6, #7).** Report coverage (sampled but skipped, and streamed), use a share-based unknown-cost stop, and warn when the share of differing answers is above epsilon. The pilot's main output is this report, so it has to say what it left out.
4. **Document the setup a customer actually needs (#4, #9).** Cover the single base URL and a self-hosted gateway, and "certify on your own graded tasks" with honest sizes: about 1,000 tasks for 4% and about 3,000 for 2%. Also fix the stale evidence brief (#22) before sending anything to YC or a buyer.
5. **Make policies traceable (#8, #14).** Add a freeze digest, `--tenant`, stricter validation and honest page wording, and put the policy digest on every ledger row.

Before any live routing or fee, also do #10 (a random frontier holdout and a written definition of what is billable), #11 (stream costs), #12 (where each baseline came from) and the fail-closed ledger option in #14.

### What a real pilot must measure that no simulation can

- **Real reply formats.** What share of real replies the answer pattern reads, and whether shadow acceptance matches the certified estimate (77.7% here) within a few points.
- **Agreement and loss on the customer's own labels.** This needs the customer's graded tasks. Benchmark certification says nothing about their traffic.
- **The eligible share of traffic.** How much is streamed, uses tools, or is mixed task types. That share caps any saving. In the simulations it ranged from about 30% to 70% of traffic.
- **Real price ratios and answer lengths.** These decide the bill-level saving and the bias in the estimated baseline (0.45x here).
- **Shadow cost and latency.** Shadow cost was about 13 to 18% of the sampled frontier spend here. Also real provider rate limits, 4xx behaviour and usage fields.
- **Drift over weeks.** Whether agreement and acceptance stay near their certification-time values.
- **Operations.** `tine serve` and the Docker image under real load, a security review cycle, and whether the customer's finance team accepts the report as a basis for a fee.