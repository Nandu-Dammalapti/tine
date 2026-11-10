# Results index

Start here. Each folder below has its own README with setup, tables, limits and the
commands that regenerate it. This page says what each experiment asked, what it
found, and how the findings fit together. Update it whenever a results folder is
added.

All results are offline, on already-graded LLMRouterBench outcomes. No model was
called. Only small summary tables are in Git; the benchmark archive, per-task
outputs, prompts and embeddings are not (see "Where the data is").

## Terms used in every folder

- **Frontier**: the model everything is compared against, `gpt-5`. It is the most
  accurate model in every pool used here.
- **Savings**: 1 - (cost of the policy / cost of sending every task to the
  frontier), on recorded per-task costs. Failed cheaper attempts are paid for.
- **Degradation**: the frontier solved the task and the policy's final answer did
  not. The degradation rate is what the guarantee bounds. Tasks the policy solves
  and the frontier does not are not credited against it.
- **epsilon**: the largest degradation rate the customer accepts (0.02 for coding,
  0.05 proposed for low-stakes work). **delta**: the chance the guarantee is wrong
  (0.05).
- **Certified**: the policy also passed a one-sided Clopper-Pearson bound <= epsilon
  on tasks used for nothing else. **Uncertified**: chosen on held-out tasks with no
  bound. Only certified numbers are a guarantee.
- **Pools**: `lcb` = LiveCodeBench, 1,055 tasks, 28 models. `lcb-swe` = LiveCodeBench
  + SWE-bench Verified, 1,555 tasks, 13 models.
- **Seeds**: each experiment repeats over 10 random splits; tables give means, and
  "runs over epsilon" counts seeds whose held-out degradation exceeded epsilon.
- **Measured vs simulated checks**: a "check" decides whether a cheaper model's
  answer is accepted. Some rows use an assumed catch rate. Each folder says which.

## What has been tried

| Date | Folder | Question | Answer |
| --- | --- | --- | --- |
| 2026-11-09 | [coding-benchmark](2026-11-09-coding-benchmark/README.md) | Can a predictor pick a cheaper model from the prompt? Can escalation chains save money? How good are public-sample checks? | Prompt-only routers certify 0% savings (TF-IDF, MiniLM and Qwen3 embeddings alike). Chains certify 38-42% with a perfect check, 4-14% at 95% catch, 0% at 90%. Measured public samples catch only 38-74% of wrong answers from the strong cheap models, so chains with real sample checks certify 0%. |
| 2026-11-10 | [gated-chains](2026-11-10-gated-chains/README.md) | Does gating a chain by the prompt-only predictor rescue weak checks? | Barely: 0% to about 1% at epsilon 0.02 with per-model sample rates; 3% to 13-18% at epsilon 0.05. Qwen3 0.6B embeddings for the gate, and an issue-only view of SWE-bench prompts, change nothing. |
| 2026-11-10 | [response-check](2026-11-10-response-check/README.md) | Does the size of the response say whether the answer is right? | Yes for coding: held-out AUC 0.76-0.99 per model. With per-model sample rates it certifies 6.2% at epsilon 0.02 and 31.8% at 0.05 on `lcb`; with no other check 3.6% and 14.3%. Weaker on multiple choice (MMLU-Pro: 3.4% and 19.4%). |
| 2026-11-10 | [avengers-pro](2026-11-10-avengers-pro/README.md) | How does the official Avengers-Pro code do on the same pool? | It is a cost-accuracy dial with no guarantee. No setting keeps mean degradation at or below 2% on `lcb`; even maximising accuracy it is less accurate than always using gpt-5 (84.4% vs 86.3%). Picking a weight on held-out tasks saves 18% at epsilon 0.02 and 34% at 0.05 but breaks the limit in about half the runs and loses 2.3-4.5 points of accuracy. On `lcb-swe` it beats gpt-5's accuracy at about 40% lower cost, because gpt-5 is weak on SWE-bench in this benchmark (15.8% against 34.6% for the best models), yet its degradation rate never drops below 3.9%. Run with both the 0.6B encoder and the paper's 8B encoder, with the same conclusions. |
| 2026-11-10 | [swe-static-checks](2026-11-10-swe-static-checks/README.md) | How many wrong SWE-bench patches do free, no-execution checks catch (patch applies, files parse)? | Half of all recorded answers fail them. They catch 53-76% of wrong answers from the cheaper models, but an answer that passes is still wrong about half the time, so chains certify 0-3% on the 500-task SWE-bench pool. gpt-5's low SWE-bench score here is mostly patches that do not apply (76% of its answers fail). On Linux, static checks catch 58% of wrong answers. A partial run of the repositories' own tests (135 django tasks, on Colab) raises that to 65%: most wrong patches break nothing and simply do not fix the issue. |

## How the findings fit together

1. The savings exist. The hindsight oracle saves 63% on `lcb` and 50% on `lcb-swe`.
2. Predicting from the prompt which model will succeed does not work on this data.
   Three feature sets (TF-IDF, MiniLM, Qwen3 0.6B and 8B) and three predictors give
   the same answer.
3. Trying a cheaper model and checking its answer does work, if the check is good.
   The check is the bottleneck: required catch rate is about 95%.
4. The official Avengers-Pro code, also prompt-only, saves money on `lcb` only by
   giving up accuracy, and cannot hold a 2% limit. Tine's router reports that it
   cannot and falls back to the frontier.
5. On real repositories (django, partial run) checks that ask "did anything break"
   catch about two thirds of wrong patches. The rest apply cleanly, break nothing and
   do not fix the issue.
6. Two cheap checks are measured so far. Public sample tests (catch 38-74% for the
   useful models) and response size (AUC 0.76-0.99). Together they are the best
   result to date, but their combination is simulated as independent, not measured.
7. At epsilon 0.02 the number of certification tasks limits the result as much as
   the check does: with about 420 tasks a policy may show at most 3 degradations.

## Open decisions

Two findings from the `lcb-swe` Avengers-Pro run question parts of the Gacrux
definition in AGENTS.md. They need a team decision; nothing in the code has changed.

1. **The routing rule.** "Cheapest model whose predicted success is at least tau,
   else the frontier" cannot leave the frontier on hard tasks, where no model's
   predicted success is high, even when the frontier is the worst choice there. On
   SWE-bench gpt-5 solves 15.8% and a model at a tenth of its cost solves 25.4%. A
   rule that compares each model with the frontier's own predicted success would
   route those tasks away. An exploratory run of that rule (not in the repository)
   certified up to 13% savings at epsilon 0.05 on `lcb-swe` where the current rule
   certifies 2-5%, with accuracy 1-2 points above gpt-5.
2. **What the guarantee bounds.** The guarantee counts only tasks the frontier
   solved and the policy did not. A policy that is cheaper and more accurate overall
   can fail it: "SWE-bench tasks to deepseek-v3.1-terminus" saves 27.7% and gains 3.1
   points of accuracy, with a 3.1% degradation rate. Bounding net accuracy loss
   instead would accept it. In the same exploratory run, the comparing rule with a
   net-loss bound certified 25-29% at epsilon 0.05 at accuracy equal to or above
   gpt-5. The trade-off: a net bound lets individual tasks regress as long as others
   improve, which a customer may or may not accept.

## Open items

- Finish the SWE-bench repository-test run (137 of 303 tasks done, django only) and
  certify chains on it: see [docs/SWE_CHECKS.md](../docs/SWE_CHECKS.md).
- A check aimed at the change itself (model-written tests for the issue, or a judge
  model). Existing tests plus static checks catch about two thirds of wrong patches
  on django; chains need about 95%.
- Per-answer sample-check results for `lcb` exist only on the machine that ran
  `scripts/lcb_sample_checks.py` (Linux, isolated). With them, `tine cascade
  --checks ... --response-check` measures the combined check.
- Avengers-Pro on a pool with more kinds of task.
- No result here uses customer traffic. Token counts, costs and check quality all
  need measuring in shadow mode.

## Where the data is

Nothing below is in Git.

| What | Where | How to get it |
| --- | --- | --- |
| LLMRouterBench archive (1.3 GB, 7.7 GB extracted) | `artifacts/bench/` on the machine that ran the experiment | `python scripts/fetch_benchmark.py --output-dir BENCH --revision b95870f9e60c589e82f7972c91ffa8de9ca89f85` |
| Canonical outcomes (`lcb.jsonl`, `lcb-swe.jsonl`, ...) | `artifacts/data/` | Commands in each folder's "Reproduce" section |
| Full run outputs (per-seed CSVs, reports) | `artifacts/runs/` | Rerun the commands |
| Qwen3 0.6B and 8B full-prompt embeddings, `tine run` outputs with them, Colab SWE-bench test results (`swe/`) | Nandu's Google Drive, `MyDrive/tine` (shared with Ashmith on 2026-11-10) | Download from the shared folder |
| Qwen3 0.6B embeddings with the issue-only view | `artifacts/embeddings/` on Nandu's Mac | `scripts/embed.py --prompt-view issue` |
| Official Avengers-Pro code | `artifacts/third_party/AvengersPro` | See [docs/AVENGERS_PRO.md](../docs/AVENGERS_PRO.md) |

## Conventions for new results

- One folder per experiment, named `YYYY-MM-DD-topic`, with a README that states the
  question, setup, results, limits, next steps and exact reproduce commands.
- Say for every number whether it is certified, and whether its check is measured
  or simulated.
- Commit summary tables only. Never commit prompts, per-task outputs, model answers
  or the benchmark files.
- Add a row to the table above.
