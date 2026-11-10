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

Savings currently include recorded model-generation costs and all attempted
models, but exclude the cost and latency of running the checks. A lower catch rate
can appear cheaper because more wrong answers are returned without escalation.
Read degradation alongside savings and overall accuracy. Keep the agreed epsilon
fixed when comparing policies.

The evaluator only reads outcome and check tables. It does not execute submitted
answers, generate tests, call providers, or launch a Colab job.

Outputs are published together after the run succeeds. A failed rerun preserves
the previous report and unrelated files in the output directory. Inputs marked
synthetic retain that label in the JSON summary and report.
