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
  savings are then net of it.

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

## Not done yet

- The proxy does not load a policy. Wiring `run_policy` into the request path, and
  running it in shadow mode beside the customer's own model, is the next step.
- Running a repository's tests on a live patch is the caller's job; `tine.judge`
  supplies only the judge half of a `tests+judge` check.
- Members of a stage are called one after another, not in parallel.
- Extracting the comparable answer from a response is the caller's job.
