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
  when all of them give the same answer (`"check": "agreement"`).
- `frontier`: the model every request falls back to.
- `guarantee`: epsilon, delta, the Clopper-Pearson bound and the number of tasks it
  rests on. If nothing certifies, `stages` is empty and every request goes to the
  frontier.
- `estimate`: savings and accuracy on the test part, which played no part in the
  choice. Savings include the cost of every call.
- `provenance`: seed, split, task count and a digest of the task ids.

Example: `results/2026-11-14-xroutebench/policy-multiple-choice-triples-eps02.json`.

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

A cheaper model that fails or times out is treated as having no answer and the
request moves on. A frontier failure is raised: there is nothing left to fall back to.

## Not done yet

- The proxy does not load a policy. Wiring `run_policy` into the request path, and
  running it in shadow mode beside the customer's own model, is the next step.
- Checks other than agreement (repository tests, a judge model) are measured in the
  experiments and are not yet stage types here.
- Members of a stage are called one after another, not in parallel.
- Extracting the comparable answer from a response is the caller's job.
