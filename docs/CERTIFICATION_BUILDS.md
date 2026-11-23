# Certification builds

Build two frozen Tine policies from recorded graded outcomes for one tenant and task type.
The build runs offline on a CPU and uses the existing fixed-sequence certifier at 4% and 2%
quality-loss limits with `delta = 0.05`. The loss is the share of tasks the frontier solves
and the policy fails. See [Frozen policies](POLICY.md) for the certificate and its assumptions.

## Rebuild the Claude MMLU-Pro pilot

From the repository root, use the Tine Python environment from [Setup](../README.md#setup).
This example uses the saved Claude pilot outcomes and answers. It uses the pilot's
30% choose / 50% certify / 20% test split, seed 42, Haiku and Sonnet as agreement or
individual candidates, and Opus as the frontier. See the [recorded pilot](../results/2026-11-17-live-pilot/README.md).

```bash
python scripts/certification_build.py \
  --input artifacts/live-pilot/outcomes-calibrate.jsonl \
  --answers artifacts/live-pilot/answers-calibrate.csv \
  --frontier claude-opus-5-5 \
  --answer-pattern 'Answer:\s*\**\s*\$?\(?([A-J])\b' \
  --include-singles --grade-conflicts wrong \
  --fractions 0.3 0.5 0.2 \
  --output-dir artifacts/certification-builds/mmlupro
```

The pilot files contain recorded prompts and answers and stay under git-ignored `artifacts/`.
If they are absent in a checkout, supply your own recorded inputs or ask the data owner for
the files. The build command needs no API key or paid calls. The pilot used
`--grade-conflicts wrong` to count groups with agreeing answers and conflicting grades as wrong.

## Rebuild the other Claude pilots

Use the same command with `outcomes-<task>.jsonl`, `answers-<task>.csv` and
`artifacts/certification-builds/<task>` for each task below. Keep the pilot's frontier, seed,
split and `--include-singles --grade-conflicts wrong` settings.

| Task | Changes to the MMLU-Pro command |
| --- | --- |
| `medqa` | Use the same reader and options. |
| `arcc` | Use `[A-J1-9]` in place of `[A-J]` in the reader. |
| `finqa`, `math500` | Use `--answer-compare math` and the boxed-answer reader below. |
| `tickets` | Use `--max-cost-share 0.6` to include the recorded Haiku–Sonnet pair and both singles. |

Pass this pattern as the single-quoted `--answer-pattern` value for the math builds.

```text
\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}
```

## Build from other recorded tasks

- **Exact-answer agreement.** Supply canonical outcomes with `--input` and comparable answers
  with `--answers`, a CSV with `task_id`, `model` and `answer`. Set `--answer-pattern` to the
  same answer reader used to create the CSV. Agreement uses exact comparison by default.
  Use `--answer-compare math` for the supported numeric and LaTeX normalization.
  `--group-size 3` checks triples. `--include-singles` also offers unchecked single models.
  See [Certify on your own graded tasks](POLICY.md#certify-on-your-own-graded-tasks).
- **Prepared checks.** Supply `--input` and a measured check table with `--checks`.
  For a checked single-model stage, set `--single-check` to the name of the measured check and
  configure that same check when running the policy. Include the cost of every check in the
  outcomes used for certification. For patch judges, `scripts/judge_patches.py checks` produces
  these inputs with recorded judge costs included. Provide `--answer-pattern` for agreement
  groups and use the same `--answer-compare` that was used to prepare them.
  See [From verdicts to a chain](JUDGE.md#from-verdicts-to-a-chain).

The default split is 40% choose / 30% certify / 30% test with seed 42. Set the seed, split and
frontier before inspecting the results. `--tenant` sets the name in the frozen policy. The
certificate still describes the population represented by the source data.

## Read and use the build

The output directory contains `policy-eps04.json`, `policy-eps02.json` and `summary.json`.
Use the summary to report the certificate and its upper bound, calibration and held-out task
counts, held-out loss and savings net of recorded checks. `held_out_exceeds_limit` records
whether the independent estimate exceeded that policy's limit. `content_sha256` records the
policy's freeze digest. The build requires complete measured check coverage for each selected
agreement or named check stage. If no chain certifies at a limit, its policy has empty `stages`
and sends all requests to the frontier.

On MMLU-Pro, agreement between Haiku and Sonnet certified at both limits with a 1.84% bound.
The held-out estimate was 41.1% cheaper and had 2.54% loss across 197 questions, above the 2%
limit. Ticket triage certified at 4% with a 3.15% bound and had 5.08% loss on 197 held-out
tickets. Keep both observations alongside the statistical certificates. Recorded pilot costs,
including any batch discounts, are the basis of these offline savings estimates.
[Claude MMLU-Pro pilot](../results/2026-11-17-live-pilot/README.md#step-2-certification)

Before loading a policy for a tenant, build from representative graded tasks and verify that
the provider serves every model in the policy. Start in shadow mode on matching non-streaming
requests. Streaming requests bypass the policy, and requests with tools skip agreement
policies. See [Proxy setup](PROXY.md#shadow-mode).
