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
- `drift_baseline`, only when the checks say whether each answer equals the
  frontier's: how often the chain's accepted answer differed from the frontier's on
  the graded tasks without being a loss (The drift baseline, below). Shadow mode's
  drift check allows for that share on top of epsilon.
- `provenance`: seed, split, task count, a digest of the task ids, and
  `content_sha256`, the freeze digest (below).
- `answer_pattern`, only with `--answer-pattern REGEX`: the regular expression the
  proxy reads a comparable answer with (the last match, its first group if it has
  one). The CLI checks it before reading any data and refuses an empty or invalid
  pattern. It must read the same answer that the recorded answers behind `--checks`
  held (for example, what you gave `scripts/agreement_pairs.py`), or live agreement
  is not the agreement that was measured. Policy files without it still load.
- `answer_compare`, only with `--answer-compare {exact,math}`: how agreement compares
  two answers (Comparing answers, below). Without it the file has no such field and
  answers are compared exactly, as before. It must match the `--answer-compare` given
  to `scripts/agreement_pairs.py` when the checks were built.

`--tenant NAME` names the tenant in the file (default: the tenant in `--input`).
When it differs, the outcomes' tenant is kept in `provenance.outcomes_tenant`. Do not
edit `tenant` by hand: it shows as an edit. `freeze-policy` refuses an epsilon of 0
or 1. Without the new flags the output is unchanged apart from the digest field.

After the JSON the command prints one line to read beside the bound: the losses on
the held-out test part with their 95% upper bound, and the losses seen on the certify
part against the number the limit allowed. It warns on stderr when the held-out loss
is above epsilon, or when the certify part used every loss it was allowed. A chain
that passed has a low count on the certify part partly by selection, so quote the
held-out loss with the bound. The first live pilot's 2% policy shows why: it passed
with 4 losses of 4 allowed, lost 2.5% on its test part and 2.4% on new questions
(`results/2026-11-17-live-pilot`). The line is printed only; the policy file does not
store these counts yet.

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

### The drift baseline

On live traffic the proxy cannot see whether an answer is right. It can only see
whether an accepted cheaper answer is the same as the frontier's. A differing answer
is not always a loss: the frontier can be the one that is wrong, both can be wrong
in different ways, and one value can be written in two ways (`14.5` and `14.53`). So
a healthy policy differs from the frontier more often than it loses, and how much
more depends on the work. The policy file records it, so that shadow mode's drift
check ([PROXY.md](PROXY.md), The drift check) has something to compare with.

`scripts/agreement_pairs.py` writes a `matches_frontier` column in the checks file:
per task and candidate, whether the candidate's answer (a group's first member's)
equals the frontier's recorded answer, compared in the same `--answer-compare` mode.
A missing frontier answer counts as differing, as it does in shadow mode. If the
answers hold no frontier answer at all, the column is empty. The column changes no
candidate, no cost and no `passed` value.

`tine freeze-policy` reads the column, walks the chosen chain over the graded tasks
as it would run, and writes:

```json
"drift_baseline": {
  "parts": ["certify", "test"], "answer_compare": "exact",
  "tasks": 692, "accepted": 612, "compared": 612,
  "differing": 16, "differing_lost": 9, "differing_not_lost": 7,
  "by_part": {"choose":  {"tasks": 308, "compared": 257, "differing": 8, "differing_not_lost": 4},
              "certify": {"tasks": 495, "compared": 441, "differing": 9, "differing_not_lost": 5},
              "test":    {"tasks": 197, "compared": 171, "differing": 7, "differing_not_lost": 2}}
}
```

- The counts at the top are for the certify and test parts together. The choose
  part is where the chain was picked, so it is only listed in `by_part`.
- `differing` is accepted answers that differ from the frontier's. `differing_lost`
  is those where the frontier was right and the accepted answer wrong;
  `differing_not_lost` is the rest.
- The drift check uses four of the counts. `differing_not_lost` of `tasks` and of
  `compared` give what it allows for on top of epsilon, as a share of requests and
  as a share of accepted answers. `differing` of `tasks` and of `compared` are what
  the graded tasks themselves showed, losses included; a share above that and
  within the allowance gets a notice, not a warning.
- An answer accepted by a named check (`tests+judge`) is not compared, as in shadow
  mode. Its task stays in `tasks`, because epsilon is a share of all tasks; it is
  in `accepted` and not in `compared`.
- No block is written when nothing certified, when the checks have no
  `matches_frontier` column, when no stage is compared, or when the match of an
  accepted answer is unknown. Nothing else in the file changes: `stages`,
  `guarantee` and `estimate` are the same with and without the column.
- The block is covered by the freeze digest. No flag turns it on.

The example is the MMLU-Pro policy of `results/2026-11-17-live-pilot`, frozen again
from its recorded answers. Re-freezing the eight pilot policies this way reproduced
the `stages`, `guarantee` and `estimate` of each stored file.

**Policies frozen before this have no baseline.** They still load. The drift check
then allows epsilon plus a default 10 points of requests, or a third of the accepted
answers, and says so (`rough check, no warning`); only a large change shows at that
level. To record a baseline, run `scripts/agreement_pairs.py` and `tine
freeze-policy` again on the same recorded outcomes and answers, with the same seed
and flags. No model is called.

**Freezing again does not restart the drift check.** The new file has a new
`created` time and a new digest. Each shadow row keeps two digests: one of the
policy file it was run with, and one of how that policy runs (frontier, stages,
answer pattern and `answer_compare`). A file that runs the same way counts the
rows of the file before it, so a warning that stood before the new file still
stands. Rows stored by a proxy from before the second digest have none. Start the
new proxy once with the file those rows were run with: it marks them. Without that
they count for that very file only, and the report says how many requests the
check left out (`drift_other_policy_requests`). A file with other stages, another
frontier, pattern or comparison is another policy and starts its check from
nothing.

The baseline describes the graded tasks. It is the right allowance for live traffic
only if those tasks are a sample of that traffic and the recorded answers were read
as the proxy reads replies: the same answer pattern and the same `answer_compare`.

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
   agreement that runs. Include the frontier's answers, read the same way: step 4
   compares each candidate's answer with them for the drift baseline (above).
4. **Build the agreement candidates.**

   ```bash
   python scripts/agreement_pairs.py --input OUTCOMES.jsonl --answers ANSWERS.csv \
     --frontier FRONTIER --output RUNS/pairs.jsonl --checks RUNS/agreement.csv \
     --group-size 2      # 3 for triples
   ```

   Every member of a group is charged, so savings are net of the second opinion.
   For answers written as LaTeX (numbers in `\boxed{...}`), add
   `--answer-compare math` here and the same flag in step 5.
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
- whose `answer_pattern`, when present, is empty or not a valid regular expression;
- whose `answer_compare`, when present, is not `exact` or `math`;
- whose `drift_baseline`, when present, is not an object with whole numbers `tasks`
  (at least 1 and at most 1,000,000,000), `compared`, `differing` and
  `differing_not_lost`, with
  0 <= `differing_not_lost` <= `differing` <= `compared` <= `tasks`. A file without
  the block loads as before.

The proxy also requires `tenant` to be the tenant the policy is configured for. A
file edited after freezing still loads, flagged by the freeze digest; a file with no
digest cannot be checked. `task_type` is recorded, logged by the proxy at startup,
and not read by the runner.

### Comparing answers

By default two answers agree when they are equal after collapsing whitespace. The
words `None`, `[None]` and `null` count as no answer, as they did when agreement was
measured (`scripts/agreement_pairs.py`).

That is too strict for answers written as LaTeX. On FinQA and MATH-500 the answer is
the content of `\boxed{...}`, and `14.5\%`, `14.5%` and `14.50`, or `\dfrac{1}{2}`
and `\frac{1}{2}`, count as different answers, so the pair escalates to the
frontier and the saving is lost.

A policy with `"answer_compare": "math"` compares answers by value instead. Both
answers are rewritten by `tine.policy.canonical_answer` and then compared exactly.
The rewrite:

- unwraps `\text{...}`;
- drops `\left`, `\right`, the spaces `\!`, `\,`, `\;` and `\ `, degree signs
  (`^\circ`, `^{\circ}`), percent signs (`\%`, `%`) and escaped dollar signs (`\$`);
- writes `\dfrac` and `\tfrac` as `\frac`;
- removes all whitespace, `$` at either end and a trailing `.`;
- writes a plain decimal number in one exact form, so `14.50` is `14.5` and `1,234`
  is `1234`. Commas count only between groups of three digits.

Numbers are compared exactly. There is no tolerance: two answers agree only when they
are the same value.

It does not catch:

- the same value in two notations: `0.5` and `\frac{1}{2}`, `\frac12` and
  `\frac{1}{2}`, `9.9%` and `0.099`, `1e3` and `1000`;
- units written differently or on one side only: `5` and `5\text{ cm}`;
- reordered terms (`x+1` and `1+x`) and equivalent expressions in general.

It also joins two answers that differ only in what it drops: `50%` and `50` agree,
and the list `1, 234` reads as the number `1234`.

The same function decides agreement offline (`scripts/agreement_pairs.py
--answer-compare math`) and in the proxy, so a policy certified offline is certified
on the comparison the proxy makes. Build the checks and freeze the policy with the
same mode. Changing the field by hand changes what runs but not what was certified,
and it shows as an edit.

A policy's `answer_pattern` is the regular expression the proxy uses to read a
comparable answer for that policy ([PROXY.md](PROXY.md)). It wins over
`TINE_ANSWER_PATTERN`. It is a record of the reader, not part of what was certified:
certification used whatever answers the answers file held.

## Not done yet

- The proxy can run a policy in shadow mode (report only) or serve answers from it,
  including code stages whose tests run behind a webhook and whose judge the proxy
  calls ([PROXY.md](PROXY.md)). Shadow mode and routing through agreement policies
  have run with real current models on public benchmark questions
  (`results/2026-11-17-live-pilot`, `results/2026-11-17-live-pilot-more`), never on a
  customer's traffic. Code stages (the tests webhook and the judge call) are tested
  with stand-in providers only.
- Running a repository's tests on a live patch happens outside Tine, behind a webhook
  the tenant hosts; Tine supplies the judge half of a `tests+judge` check.
- `run_policy` calls the members of a stage one after another, and extracting the
  comparable answer is its caller's job. The proxy calls a stage's members together
  and reads the answer itself (`TINE_ANSWER_PATTERN`, or the policy's own
  `answer_pattern`).
- Certifying a policy from a tenant's own traffic is designed in
  [LEARNING_LOOP.md](LEARNING_LOOP.md) and not built.
