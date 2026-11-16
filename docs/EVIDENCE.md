# Tine and Gacrux: evidence brief

Written 2026-11-16. Every number is followed by the results folder and file it comes
from. Paths are relative to [`results/`](../results/README.md). All results are on
recorded answers from public benchmarks. The only live model calls were the judge's. No
customer has used the system.

## 1. The problem

Teams often send every AI request to the most expensive model because they cannot show
that a cheaper one will get their work right. The routers we tested traded accuracy for
cost: each either sent nearly everything to the top model or lost accuracy (section 2).
Tine's routing system, Gacrux, uses a cheaper model only under a stated quality limit,
and reports zero savings when it cannot show the limit holds.

## 2. What we tested and found

"Saved" is cost against sending everything to the most accurate model in the pool (the
frontier), and includes every extra call. Means over 10 random splits unless stated.
"Certified" is defined in section 3. "Over the limit" counts splits where held-out tasks
broke the limit. The OpenRouter and judge rows use every task, with no split.

| Finding | Certified | Not certified (labelled) | Source |
| --- | --- | --- | --- |
| Our prompt-only router, LiveCodeBench, 1,055 tasks | 0.1% saved at a 2% limit, 2.1% at 4% (the best of three predictors in each case; the 4% one was over the limit in 1 of 10 splits) | Picking the right model in hindsight saves 63% on the one held-out split recorded (159 tasks) | `2026-11-11-router-rules/router-rules-summary.csv`; `2026-11-09-coding-benchmark/router-lcb-metrics.csv` |
| Avengers-Pro, official code, same tasks | Nothing. The code computes no bound | No setting keeps mean loss at or under 2% (lowest 2.6%; 2.4% with the paper's encoder). A weight picked at 2% with the repository's settings saves 18.2% and is over the limit in 6 of 10 splits (5.6% and 4 of 10 with the paper's encoder). With the paper's settings the pick saves 0% and is never over, or 6.9% and over in 3 of 10 with the paper's encoder | `2026-11-10-avengers-pro/avengers-lcb-repo-curve.csv`, `avengers8b-lcb-repo-curve.csv`, `avengers-lcb-repo-picks-eps02.csv`, `avengers8b-lcb-repo-picks.csv`, `avengers-lcb-paper-picks-eps02.csv`, `avengers8b-lcb-paper-picks.csv` |
| LLMRouter library, five routers unmodified, same tasks | -0.4% to 0.2% at 2% and 4% | Below full accuracy weight they save 80-99.8% and accuracy falls from 86% to 39-46% | `2026-11-14-xroutebench/llmrouter-lcb-summary.csv`, `llmrouter-lcb-curve.csv` |
| The same five routers on the library's own benchmark, 4,762 multiple-choice questions, 5 splits | 0-0.6% | Any lower weight: 97-98% saved, accuracy 80% against 88.9% | `2026-11-14-xroutebench/llmrouter-xroutebench-mc-summary.csv`, `llmrouter-xroutebench-mc-curve.csv` |
| OpenRouter auto-router, recorded by the benchmark's authors in late 2025 | Nothing. The record carries no limit | LiveCodeBench: 75% cheaper, 61.1% accurate against 86.4%; on 26.8% of all tasks gpt-5 is right and it is wrong. MMLU-Pro: 65% more expensive, 85.8% against 87.4% | `openrouter-vs-gpt5.csv` |
| Agreement check, MMLU-Pro, 3,000 questions | Three cheap models must agree: 32.3% at 2%, accuracy 87.6% against 87.3%, 0 of 10 over. Two must agree: 63.8% at 4%, 86.9%, 0 of 10 over. The triples use a hand-listed set of seven models; how they were picked is not recorded | Two at 2% certify only 11.1%, with 1 of 10 over | `2026-11-11-agreement-check/agreement-triples-mmlupro-summary.csv`, `agreement-pairs-mmlupro-summary.csv`, `README.md` (model list) |
| Agreement check, second benchmark (xRouteBench, 18 open models) | Multiple choice at 2%: 45.2% (two) and 53.5% (three), accuracy above the frontier's, 0 of 10 over. Math at 2%: 16.4% | Two at 4%: 73.8%, but over the limit in 2 of 10 splits, more than the 1 in 20 expected. One cheaper model alone: 0% | `2026-11-14-xroutebench/agreement-pairs-multiple-choice-summary.csv`, `agreement-triples-multiple-choice-summary.csv`, `agreement-pairs-math-summary.csv`, `agreement-singles-multiple-choice-summary.csv` |
| Code, free checks only (patch applies, files parse, existing tests pass), 274 SWE-bench tasks | 0% at 2%, 4% and 5% | These checks catch about two thirds of wrong patches | `2026-11-10-swe-static-checks/cascade-swe-274-own-grades-summary.csv`, `repo-test-rates-partial-222.csv` (per model), `README.md` (all models: 64-68%) |
| Code, free checks plus a `claude-sonnet-5-5` judge, same 274 tasks | 0% at 2%. At 4%: 0% with the default split, 22.2% when half the tasks certify (3 of 10 splits certified nothing). That split was chosen after seeing the first result | About 34% cheaper than `claude-opus-4.1` net of the judge, 55-56% accurate against 49-50%, over a 2% limit in 1 of the 20 held-out runs across the two splits | `2026-11-14-api-judge/cascade-judge-verdict-split-40-30-30-summary.csv`, `cascade-judge-verdict-split-30-50-20-summary.csv` |
| The judge itself, 1,110 patches that passed the free checks | Not a certified quantity | Rejects 79.4% of wrong patches and 9.0% of right ones, at $0.0080 per check | `2026-11-14-api-judge/judge-sonnet55-1110-own-grades.csv` |

In short: choosing a model from the prompt alone certified almost nothing, for our
router and for the LLMRouter library's. Avengers-Pro did not hold a 2% limit on this
benchmark, and OpenRouter's recorded choices lost accuracy on both benchmarks shown.
Checking a cheaper model's answer certifies large savings where answers can be compared
exactly. On code the chain looks good and is not yet proven.

## 3. How the guarantee works

The customer sets a quality limit, for example 2%. It bounds one thing: the share of
tasks the frontier model gets right and the routed answer gets wrong.

Graded tasks are split into three parts. The first picks a policy: cheaper models in
order, each behind a check, ending at the frontier. The second, used for nothing else,
certifies it. We count the policy's losses there and compute an upper bound on the true
loss rate (one-sided Clopper-Pearson, 5% error chance). Candidates are tested in a fixed
order and testing stops at the first failure, so trying many does not weaken the bound.
The third part reports savings.

Example: the frozen multiple-choice policy lost 1.05% of 1,426 certify tasks, which gives
a bound of 1.62%, under its 2% limit
(`2026-11-14-xroutebench/policy-multiple-choice-triples-eps02.json`).

"Certified" means the bound passed. If nothing passes, everything goes to the frontier
and savings are 0%.

It does not mean:

- every answer is right, or that no single task gets worse;
- the limit holds on traffic unlike the certify tasks. The proxy does not check this.
  Shadow mode warns when cheaper answers differ from the frontier's more often than the
  limit, which is a reason to grade, not a measured loss (`../docs/PROXY.md`);
- certainty. One certification in 20 can be wrong by design.

Small samples certify little. With about 80 certify tasks, zero losses still gives a
bound of 3.6% (`2026-11-14-api-judge/README.md`).

## 4. What is built

| Part | State |
| --- | --- |
| Proxy for OpenAI and Anthropic APIs, tenant tokens, verified cache | Development quality, SQLite (`../docs/PROXY.md`) |
| Ledger: answer cost and routing cost recorded separately per request | Built and tested |
| Frozen policies: one certified chain per file, with its evidence | Built and tested (`../docs/POLICY.md`) |
| Shadow mode: cheaper models run in the background, a report is produced | Built, tested with stand-in providers |
| Live routing through a policy, with checks (tests behind a customer webhook, a judge call) | Built, tested with stand-in providers |

One end-to-end check exists. Replaying 1,414 recorded multiple-choice requests through
the real proxy gave the same result in shadow mode and in live routing: 77.7% answered by
cheaper models, 53.5% lower cost, 88.9% accuracy against 87.0%, 0.99% of tasks lost. The
policy's stored estimate was 52.5%. The gap is a model shared by two stages
(`2026-11-16-shadow-replay/shadow-replay-multiple-choice-triples.json`,
`route-replay-multiple-choice-triples.json`). This shows the proxy does what the policy
says. It is not new evidence of quality.

None of this has run on real traffic. The code path was tested with stand-ins only, not
with recorded SWE-bench data. Not built: nightly retraining per customer, distillation, a
production datastore, streaming under a policy, and a way to measure the saving under live
routing (that needs a random share of requests always sent to the frontier). Paid model
calls so far total $47.73, all for judge measurements: $12.69 in
`2026-11-14-api-judge/README.md`, $31.02 in `2026-11-16-judge-probes/README.md` and $4.02
in `2026-11-17-judge-prompt/README.md`.

## 5. What we do not know yet

- **The judged code chain certifies nothing at a 2% limit.** On the 274 SWE-bench tasks
  its best certified results are 22.2% (Sonnet judge) and 22.9% (`claude-haiku-5-5`, then
  Sonnet) at a 4% limit, both with a split chosen after the first result
  (`2026-11-14-api-judge/cascade-judge-variants.csv`). The limit is the number of graded
  tasks: 2% needs about 150 certify tasks with no losses. On LiveCodeBench a
  response-size check certifies 3.6% at 2%, and was over the limit in 1 of 10 splits
  (`2026-11-10-response-check/response-lcb-summary.csv`).
- **The judge partly remembers the answers.** Its cases are public django and sympy
  issues. Asked to write the merged fix from the issue alone, `claude-sonnet-5-5`
  reproduces it for 41% of tasks. On the tasks it cannot recall, a chain judged by Haiku
  or Sonnet loses about 2.3% of the tasks the frontier solves, against 0.6-1.2% on
  recalled tasks. With `claude-opus-5-5` as the judge, Haiku screening in front of it,
  the loss on those tasks is 0.46% at about 24% savings, uncertified, on 150 tasks
  (`2026-11-16-judge-probes/README.md`). Opus's own recall was not measured. Tasks the
  judge cannot recall stand in for private code; we have no measurement on private code
  itself.
- **All results are on public benchmarks.** Single-shot patches, multiple-choice
  questions, benchmark prices.
- **Dollar amounts on the open-model benchmark are tiny.** The frontier's bill for the
  1,414 replayed requests was $0.70. What carries over is that the check certifies.
- **No customer has used it.** Token counts, costs and check quality on real traffic are
  unmeasured.
- Agreement works only where answers can be compared exactly, and is weaker on math.
- The original design, a router that reads the prompt, did not work (section 2).

## 6. Next three milestones

1. **Grade more code tasks** (the 29 django and sympy tasks left, then other SWE-bench
   repositories, about 500 in all). Proves or disproves that the judged chain certifies
   at 2% with the default split.
2. **Run the judge on code it cannot have seen.** The probes above split public tasks by
   what the judge recalls; code that was never public is the real test of whether the
   code result carries to customers.
3. **One customer's traffic in shadow mode.** Gives the first real costs, the first
   savings estimate in real dollars, and whether a policy certified on that customer's
   own graded work holds its limit.
