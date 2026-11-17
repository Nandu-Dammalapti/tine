# Measured evidence for escalation chains

Use `tine cascade --checks CHECKS.csv` with canonical graded outcomes to evaluate
an answer-checking policy. The checker must see information available at deployment,
such as public examples or customer tests. Keep the independent grading outcome
separate: using hidden benchmark grades as the check gives an optimistic oracle,
not measured checker performance.

The CSV requires `task_id`, `model`, and `passed`. Include `tenant` when sharing a
file between tenants. Each experiment evaluates one tenant. Rows for other tenants,
tasks, or models are ignored and counted in the coverage report. Without a tenant
column, the file is interpreted as belonging to the single evaluated tenant.

Accepted check labels are true/false or zero/one. An empty field is unknown.
Unknown and missing checks pass without evidence: the simulation cannot assume
they caught a wrong answer. This is not a claim that the answer is correct.
Invalid labels and duplicate tenant/task/model rows are rejected. Select a single
check run explicitly instead of pooling retries or contradictory results.
Task and model names are read literally, including IDs such as `NA` and `001`.

`cascade-summary.json` records `check_mode`, `check_coverage`, `cost_scope`, and
`certification_scope`. Coverage distinguishes known measurements from missing or
unknown results, including counts per model. Identity matching cannot establish
that checks refer to the same answer revision: retain the source outcomes and
checker-run provenance together and verify that correspondence before evaluation.

Numeric `--catch` settings simulate check outcomes. Even a certified policy in
that mode is conditional on the assumed checker process. With measured checks,
the bound applies under the stated independent-sampling assumptions. Delta applies
per seed/epsilon/check setting, not simultaneously across the whole sweep. Choosing
settings after inspecting held-out results requires fresh evaluation data.

Savings include recorded model-generation costs and all attempted models. `tine
cascade` adds no check cost of its own. For a result net of routing cost, the check's
cost must already be in the outcomes it reads: `scripts/agreement_pairs.py` charges
every member of a group, and `scripts/judge_patches.py checks` adds each judge's
recorded cost to the answers it read. Test-run compute and check latency are not
costed anywhere yet. A lower catch rate
can appear cheaper because more wrong answers are returned without escalation.
Read degradation alongside savings and overall accuracy. Keep the agreed epsilon
fixed when comparing policies.

The evaluator only reads outcome and check tables. It does not execute submitted
answers, generate tests, call providers, or launch a Colab job.

Outputs are published together after the run succeeds. A failed rerun preserves
the previous report and unrelated files in the output directory. Inputs marked
synthetic retain that label in the JSON summary and report.

## Predictor-gated chains

`tine cascade --gate logistic|knn|cluster` adds a `gated_certified` pick. A gated
chain tries a cheaper model only on tasks where the router's predicted success for
that model is at least a threshold (`--gate-threshold`, default 0.5 to 0.9), and
still escalates when the answer fails its check. A skipped model costs nothing.
The search covers every chain with every threshold and with no gate, through the
same fixed-sequence certification as `certified`.

Each seed splits tasks into train / score-calibration / choose / certify / test
parts. The predictor is fitted on the first two only, with train-only TF-IDF
features or the fixed embeddings given by `--gate-embeddings` (the NPZ format of
`tine run --embeddings`), so `certified` and `gated_certified` are compared on the same held-out
tasks. These parts are smaller than in an ungated run: do not compare the two
runs' numbers directly.

`--catch-rates RATES.csv` (columns `model`, `catch_rate`, every non-frontier model)
adds a simulated `per-model` setting where each model's wrong answers are caught at
its own rate. The rate is applied independently of the task, so it cannot show
whether a check and the predictor miss the same answers. Use per-answer `--checks`
for that. It cannot be combined with `--checks`.

## Agreement check

For tasks whose answers can be compared exactly, `scripts/agreement_pairs.py` turns
"ask two or three cheaper models and accept when they agree" into candidates for
`tine cascade --checks`. Each group of models is one candidate that costs all of its
members' calls on every task and whose check passes when their recorded answers are
identical. Certification and savings then include the second and third opinions.
See `results/2026-11-11-agreement-check`.

`scripts/agreement_datasets.py` runs this recipe over several datasets with cost-only
model rules (`--rule cheapest|all-cheaper`, `--max-tasks`, pools). See
`results/2026-11-16-agreement-more-datasets`.

Build groups for one tenant at a time. The builder rejects conflicting known grades
for agreeing answers, duplicate recorded answers, and group names that collide with
existing models or other groups. Check rows include the tenant. The answer reader
uses the latest JSON file in each model directory and verifies its bytes against
the outcomes' `source_sha256` when present; a mismatch requires a matching benchmark
run. Without source hashes, callers must verify that answers and grades came from
the same run. Input outcomes, output outcomes, and checks must use distinct paths.

With `--answers`, supply a CSV containing `task_id`, `model`, and `answer` for one
tenant. Duplicate task/model answers, duplicate headers, malformed records, and
blank identifiers are rejected. An optional `tenant` column must match the outcome
table on every row. Without that column, the caller must ensure the answers belong
to the outcome tenant. Empty answers remain missing; identifiers such as `NA` and
`001` stay literal text. The answer CSV must also have a distinct path from both
outputs and the outcome input. CSV answers carry no verified source hash, so the
caller must check that they correspond to the graded run.

## Response check

`tine cascade --response-check` adds a `response_certified` pick. A response check
(`tine/acceptance.py`) scores each cheaper model's answer from two numbers the
ledger already records, prompt tokens and response tokens, and accepts the answer
only if the score is at least a threshold (`--response-threshold`, default 0.5 to
0.95). For many models a long response goes with a wrong answer; the direction and
strength differ, so each model has its own small classifier, calibrated on held-out
tasks. It runs no tests and calls no model. An escalated answer has still been paid for.

Direct Python calls require nonempty token matrices containing known, nonnegative
whole counts. Boolean and complex token values are rejected before conversion.
When combining checks with response scores, both matrices must have exactly the
same task/model dimensions; a single check row cannot be reused implicitly across
tasks. Check results must be Boolean and scores must be real probabilities.

The search covers every chain with the setting's check alone or combined with the
response check, through the same fixed-sequence certification. `--catch 0` makes the
response check the only check, so that setting contains no simulated check. With a
simulated `--catch` or `--catch-rates` setting, the simulated check and the response
check are combined as if their misses were unrelated; per-answer `--checks` remove
that assumption.

Each seed splits tasks 20 / 10 / 15 / 40 / 15 by default (`--fit-fractions`) into
train / score-calibration / choose / certify / test parts. The classifier has four
features, so most tasks go to certification. Response tokens must be recorded for
every outcome and must mean the same thing in production as in the fitting data
(for example, whether reasoning tokens are counted).

`scripts/agreement_pairs.py` refuses data in which members of a group give the same answer and are
graded differently, because the group's grade is then not defined by the answer. `--grade-conflicts
wrong` keeps such data and counts the group as wrong on those tasks; the count is in the report. Use
it for free-form answers extracted from responses, and say so with the result. `--answers` reads
extracted answers from a CSV in place of a benchmark folder, and `--include-singles` also offers
each cheaper model alone with no check, the alternative an agreement check has to beat. When no
pair costs at most `--max-cost-share` of the frontier, the script stops and names the cheapest
pair's share. With `--include-singles` it goes on with the single models that are cheap enough, and
the report says that no pair was (`no_pair_cheap_enough`, `cheapest_pair_cost_share`). Tested with
made-up outcomes only.

### Which candidates chains are built from

`tine cascade` and `tine freeze-policy` build chains from eight shortlisted candidates.
The default (`--shortlist savings`) ranks each cheaper candidate by what it would save
tried alone if its check caught every wrong answer, which ignores how often a measured
check passes a wrong answer. `--shortlist measured` ranks on the choose part only, with
the measured check results, once per epsilon: first candidates that save something
with a single-stage degradation rate at most epsilon (most savings first), then the
other saving candidates (lowest degradation first), then the rest. Certify and test
rows never reach the shortlist. It is opt-in: on the published experiments it raised
certified savings where there are hundreds of agreement groups and lowered a few
other cells slightly (`results/2026-11-17-measured-shortlist`).
