# Frozen policies

The experiments (`tine cascade`) repeat a choice over ten seeds to measure it. A
deployment needs the choice made once and written down. `tine freeze-policy` does
that, and `tine.policy.run_policy` executes the result on a live request.

## Freezing

```bash
python -m tine freeze-policy --input DATA/pairs.jsonl --checks RUNS/agreement.csv \
  --frontier FRONTIER --epsilon 0.02 --output RUNS/policy.json
```

Input: recorded outcomes for the candidates and measured check results, for one
tenant and one task type (for agreement checks, both files come from
`scripts/agreement_pairs.py`). It splits the tasks once (seed 42) into choose /
certify / test parts and picks and certifies a chain exactly as one seed of the
certified experiment does. The file it writes holds:

- `stages`: the cheaper models to try, in order. A stage with several models passes
  when all of them give the same answer (`"check": "agreement"`). A stage with one
  model carries the name given with `--single-check`: `none` (accepted as it is) or
  the name of the check the recorded results measured, such as `tests+judge:sonnet`.
  `freeze-policy` refuses `none` when the recorded check rejected any answer of a
  single model in the chosen chain: that chain was certified with a check, and a
  policy naming none would serve those answers unchecked.
- `frontier`: the model every request falls back to.
- `guarantee`: epsilon, delta, the Clopper-Pearson bound and the number of tasks it
  rests on. If nothing certifies, `stages` is empty and every request goes to the
  frontier.
- `estimate`: savings and accuracy on the test part, which played no part in the
  choice. Savings include the cost of every call.
- `provenance`: seed, split, task count, a digest of the task ids, and
  `content_sha256`, the freeze digest (below).
- `answer_pattern`, only with `--answer-pattern REGEX`: the regular expression the
  proxy reads a comparable answer with (the last match, its first group if it has
  one). The CLI checks it before reading any data and refuses an empty or invalid
  pattern. It must read the same answer that the recorded answers behind `--checks`
  held (for example, what you gave `scripts/agreement_pairs.py`), or live agreement
  is not the agreement that was measured. Policy files without it still load.

`--tenant NAME` names the tenant in the file (default: the tenant in `--input`).
When it differs, the outcomes' tenant is kept in `provenance.outcomes_tenant`. Do not
edit `tenant` by hand: it shows as an edit. `freeze-policy` refuses an epsilon of 0
or 1. Without the new flags the output is unchanged apart from the digest field.

### The freeze digest

`provenance.content_sha256` is the SHA-256 of the policy's canonical JSON
(`json.dumps(sort_keys=True, separators=(",", ":"), default=float)`, ASCII-escaped)
with that field left out. When a file is loaded its state is one of:

- "matches the digest written when it was frozen";
- "edited after freezing": the file still loads, but the proxy log, `GET /v1/shadow`
  and the report page flag it;
- "no digest (frozen before digests)": files from before this field, the two shipped
  `results/` policies, and hand-written files.

**A digest is not a signature.** Anyone who can edit the file can write a new
digest. It catches careless or casual edits only.

Examples: `results/2026-11-14-xroutebench/policy-multiple-choice-triples-eps02.json`
(agreement) and `results/2026-11-14-api-judge/policy-code-sonnet-judge-eps04.json`
(code: three cheap models in turn, each behind the free checks and a judge).

A policy describes the tasks it was certified on. It says nothing about traffic that
differs from them, and it must be re-frozen when the models, prices or tasks change.

**Every model a policy names must be served where the proxy will call it.** The proxy
calls each stage model, and every judge model, at the provider's one base URL with
the provider's key ([PROXY.md](PROXY.md), One base URL per provider). The two
example policies use open models that api.openai.com and api.anthropic.com do not
serve. Freeze from candidates your provider serves, or run your own OpenAI-compatible
gateway (vLLM or LiteLLM) that serves the frontier and every candidate.

## Certify on your own graded tasks

A policy certified on public benchmarks says nothing about a customer's own work.
The path for exact-answer tasks (labels, choices, numbers):

1. **Collect graded tasks of one task type.** For each task, call the frontier and
   every candidate cheaper model with the same request, keep each answer, and grade
   each answer against the right one. Keep the prompts and answers out of git.
2. **Write the outcomes file.** One row per task and model, as CSV or JSONL, with
   `tenant`, `task_id`, `prompt`, `task_type`, `model`, `success` (true or false) and
   `cost_usd` (what the call cost). Token counts, `latency_ms` and `timestamp` are
   optional. One tenant and one task type per file. `python -m tine inspect --input
   FILE` checks it.
3. **Write the answers file.** CSV with `task_id`, `model` and `answer`: the
   comparable answer read from each reply with the same regular expression the proxy
   will use. This is the rule that matters most: the offline reader and the proxy's
   reader must be the same pattern, or the agreement that was certified is not the
   agreement that runs.
4. **Build the agreement candidates.**

   ```bash
   python scripts/agreement_pairs.py --input OUTCOMES.jsonl --answers ANSWERS.csv \
     --frontier FRONTIER --output RUNS/pairs.jsonl --checks RUNS/agreement.csv \
     --group-size 2      # 3 for triples
   ```

   Every member of a group is charged, so savings are net of the second opinion.
5. **Freeze.**

   ```bash
   python -m tine freeze-policy --input RUNS/pairs.jsonl --checks RUNS/agreement.csv \
     --frontier FRONTIER --epsilon 0.04 --tenant TENANT \
     --answer-pattern 'REGEX USED IN STEP 3' --output RUNS/policy.json
   ```

6. **Shadow it before routing.** Load it as a shadow policy under its own tenant
   token and compare the acceptance share with the policy's `estimate`.

How many graded tasks. `freeze-policy` splits the tasks at random 40/30/30, so 30%
of them certify. The number needed depends on the chain's true loss rate, which is
not known in advance. From `scripts/plan_certification.py --epsilon 0.02 0.04 --rate
0.005 0.01 0.015 --split 40/30/30`, for an 80% chance that one chain tested alone
certifies:

| limit | true loss rate | certify tasks | graded tasks in total |
| --- | --- | --- | --- |
| 4% | 0.5% | 117 | about 400 |
| 4% | 1.0% | 192 | about 650 |
| 4% | 1.5% | 326 | 1,000 to 1,100 |
| 2% | 0.5% | 386 | about 1,300 |
| 2% | 1.0% | 1,091 | 3,400 to 3,650 |
| 2% | 1.5% | 4,646 | about 15,000 |

Plan on about 1,000 graded tasks for the 4% limit and about 3,000 or more for 2%. On
MMLU-Pro, about 300 certify tasks (1,000 in total) carried most of the 4% result and
the 2% result needed 900 (3,000 in total). These figures are lower limits: the
certifier tests many chains in a fixed order and stops at the first that fails
([PLANNING.md](PLANNING.md)).

What it costs to collect them: the number of graded tasks times the cost of one task
on the frontier plus every candidate, plus the cost of grading. With N tasks, a
frontier cost F per task and candidates costing c1, c2, ... per task, the model calls
cost about N × (F + c1 + c2 + ...).

For code, the outcomes come from running each patch against the repository's tests
and the checks from `scripts/judge_patches.py checks` ([JUDGE.md](JUDGE.md)); freeze
with `--single-check` naming the check the proxy will run.

## Running

```python
from tine.policy import run_policy

outcome = run_policy(policy, call)   # call(model) -> {"answer": ..., "cost_usd": ...}
```

`call` performs the request on one model and returns its comparable final answer
and what the call cost. `run_policy` tries each stage, returns the first answer whose
check passes, and otherwise calls the frontier. It reports:

- `result`, `served_by`, `stage`;
- `answer_cost_usd`: the cost of the call whose answer was returned;
- `routing_cost_usd`: the cost of every other call made (second opinions, attempts
  that were escalated). This is what AGENTS.md section 2 requires to be recorded
  separately: pass it to `Ledger.record(routing_cost_usd=...)`, and the ledger's
  savings are then net of it. Pass `0.0` when no routing call was made: the ledger
  treats a missing routing cost as unknown and leaves that request out of savings.
  It is `None` when any call or check has no known cost (a cheaper call that failed,
  a check that reported no cost): the routing cost is then not known, and passing
  `None` on keeps the request out of savings. `unknown_cost_calls` counts those
  calls, and `known_routing_cost_usd` is the part that is known. Do not record the
  known part as the routing cost.

A stage with a named check is decided by a function the caller supplies:

```python
from tine.judge import patch_judge_check

judge = patch_judge_check(client, model="claude-sonnet-5-5")       # paid: needs authorisation
def tests_then_judge(result):
    if not free_checks_pass(result):                                # the caller's own test run
        return {"passed": False, "cost_usd": 0.0}
    return judge(result)                                            # result carries repository, issue, patch

outcome = run_policy(policy, call, checks={"tests+judge:sonnet": tests_then_judge})
```

The function must do what the recorded check did (the same free checks, the same
judge model, effort and threshold); the guarantee describes that check and no other.
What a check costs is added to `routing_cost_usd`. A check that raises counts as a
failure, and a judge call that fails rejects, so the request escalates. If a policy
names a check that was not supplied, `run_policy` raises before calling any model.

A cheaper model that fails or times out is treated as having no answer and the
request moves on. A frontier failure is raised: there is nothing left to fall back to.

## What a policy file is checked for

`validate_policy` runs when the proxy starts, in `scripts/savings_report.py`, and in
`dumps` and `run_policy`. It refuses a file:

- whose version is not the number 1, whose stages are malformed, or that repeats a
  model within a stage;
- that has stages without a certified guarantee;
- whose `guarantee.epsilon` or `guarantee.delta`, when present, is not a number
  strictly between 0 and 1, or whose `guarantee.bound`, when present, is not a
  number from 0 to 1. Strings, booleans, NaN and infinity are refused;
- that says `certified: true` without numeric epsilon, delta and bound, with the
  bound at most epsilon;
- whose `answer_pattern`, when present, is empty or not a valid regular expression.

The proxy also requires `tenant` to be the tenant the policy is configured for. A
file edited after freezing still loads, flagged by the freeze digest; a file with no
digest cannot be checked. `task_type` is recorded, logged by the proxy at startup,
and not read by the runner.

Two answers agree when they are equal after collapsing whitespace. The words
`None`, `[None]` and `null` count as no answer, as they did when agreement was
measured (`scripts/agreement_pairs.py`).

A policy's `answer_pattern` is the regular expression the proxy uses to read a
comparable answer for that policy ([PROXY.md](PROXY.md)). It wins over
`TINE_ANSWER_PATTERN`. It is a record of the reader, not part of what was certified:
certification used whatever answers the answers file held.

## Not done yet

- The proxy can run a policy in shadow mode (report only) or serve answers from it,
  including code stages whose tests run behind a webhook and whose judge the proxy
  calls ([PROXY.md](PROXY.md)). Both are tested with stand-in providers only; neither
  has served real traffic.
- Running a repository's tests on a live patch happens outside Tine, behind a webhook
  the tenant hosts; Tine supplies the judge half of a `tests+judge` check.
- `run_policy` calls the members of a stage one after another, and extracting the
  comparable answer is its caller's job. The proxy calls a stage's members together
  and reads the answer itself (`TINE_ANSWER_PATTERN`, or the policy's own
  `answer_pattern`).
- Certifying a policy from a tenant's own traffic is designed in
  [LEARNING_LOOP.md](LEARNING_LOOP.md) and not built.
