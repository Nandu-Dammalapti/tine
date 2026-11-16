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
- `provenance`: seed, split, task count and a digest of the task ids.

Examples: `results/2026-11-14-xroutebench/policy-multiple-choice-triples-eps02.json`
(agreement) and `results/2026-11-14-api-judge/policy-code-sonnet-judge-eps04.json`
(code: three cheap models in turn, each behind the free checks and a judge).

A policy describes the tasks it was certified on. It says nothing about traffic that
differs from them, and it must be re-frozen when the models, prices or tasks change.

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

`validate_policy` refuses a file whose version is not the number 1, whose stages
are malformed, that repeats a model within a stage, that has stages without a
certified guarantee, or whose `bound` is above its `epsilon` when both are given.
The proxy also requires `tenant` to be the tenant the policy is configured for.
Nothing checks that the stages are the ones that were certified: a file edited
after freezing still loads. `task_type` is recorded and not read by the runner.

Two answers agree when they are equal after collapsing whitespace. The words
`None`, `[None]` and `null` count as no answer, as they did when agreement was
measured (`scripts/agreement_pairs.py`).

A policy may carry an `answer_pattern` key: the regular expression the proxy uses to
read a comparable answer for that policy ([PROXY.md](PROXY.md)). It is not written
by `freeze-policy` and is not part of what was certified.

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
