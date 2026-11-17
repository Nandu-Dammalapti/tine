# Results index

**Where we stand in one page:** [SCOREBOARD.md](SCOREBOARD.md) compares Gacrux with
OpenRouter, Avengers-Pro and always using gpt-5.

A two-page summary with sources for every number: [docs/EVIDENCE.md](../docs/EVIDENCE.md).

Start here. Each folder below has its own README with setup, tables, limits and the
commands that regenerate it. This page says what each experiment asked, what it
found, and how the findings fit together. Update it whenever a results folder is
added.

All results but four are offline, on recorded benchmark answers (LLMRouterBench, and
xRouteBench from 2026-11-14). The exceptions are four paid live runs (2026-11-17), where
current Claude models answered requests through the real proxy: the two live pilots on
public benchmark questions ($23.65 and $60.00), in shadow mode and routed live; the
rate-limit tests and a shadow run on 1,500 more benchmark questions ($26.43); and a pilot
on a public set of one company's support tickets ($14.74), in shadow mode and routed
live. None of them ran on a customer's traffic. Otherwise models were called only to judge
or to read: a Claude Code session and open models as judges (judge-pilot), pinned
Claude models through the paid API ($47.73: api-judge $12.69, judge-probes $31.02,
judge-prompt $4.02; $172.55 in all with the four live runs),
and frozen encoders for embeddings and hidden states. Only small summary tables are in
Git; the benchmark archives, per-task outputs, prompts and embeddings are not (see
"Where the data is").

## Terms used in every folder

- **Frontier**: the model everything is compared against. `gpt-5` on the
  LiveCodeBench, `lcb-swe` and MMLU-Pro pools, where it is the most accurate model;
  `claude-opus-4.1` on the SWE-bench-only pool (gpt-5 is weak there in this
  benchmark); `cogito-v2-1-671b` on xRouteBench. Each folder names its frontier.
- **Savings**: 1 - (cost of the policy / cost of sending every task to the
  frontier), on recorded per-task costs. Failed cheaper attempts are paid for.
- **Degradation**: the frontier solved the task and the policy's final answer did
  not. The degradation rate is what the guarantee bounds. Tasks the policy solves
  and the frontier does not are not credited against it.
- **epsilon**: the largest degradation rate the customer accepts. The team's working
  limit since 2026-11-11 is 0.04, reported first, with 0.02 alongside; folders before
  that date used 0.02 and 0.05. **delta**: the chance the guarantee is wrong (0.05).
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
| 2026-11-10 | [swe-static-checks](2026-11-10-swe-static-checks/README.md) | How many wrong SWE-bench patches do free, no-execution checks catch (patch applies, files parse)? | Half of all recorded answers fail them. They catch 53-76% of wrong answers from the cheaper models, but an answer that passes is still wrong about half the time, so chains certify 0-3% on the 500-task SWE-bench pool. gpt-5's low SWE-bench score here is mostly patches that do not apply (76% of its answers fail). On Linux, static checks catch 58% of wrong answers. A run of the repositories' own tests on Colab (277 of 306 django and sympy tasks, 274 valid) raises that to 64-68%: most wrong patches break nothing and simply do not fix the issue. Chains on those 274 tasks with the free checks alone certify 0% at every limit. |
| 2026-11-11 | [judge-pilot](2026-11-11-judge-pilot/README.md) | Can a judge that reads only the issue and the patch reject the wrong patches that pass every free check? | In a 150-case pilot with a Claude Code session as judge: it rejects 95% of them by our own test run's grades (82% by the benchmark's, which under-credits some correct patches) and rejects about 2% of right patches. With the free checks that is about a 98% catch rate on django. A ceiling, not a baseline: the judge is unpinned and may know these public issues. A small local model (qwen2.5-coder:14b) on the same cases is at chance: it accepts nearly everything or rejects nearly everything depending on the instructions. |
| 2026-11-11 | [agreement-check](2026-11-11-agreement-check/README.md) | For exact-answer tasks, does "accept when two or three cheaper models agree" certify savings after paying for every call? | Yes, on MMLU-Pro (3,000 tasks), from measured data only: three models that agree certify 32% at epsilon 0.02 with accuracy at gpt-5's level; two that agree certify 64% at epsilon 0.04. |
| 2026-11-11 | [router-rules](2026-11-11-router-rules/README.md) | Do a relative routing rule and a net-accuracy guarantee (both opt-in) unblock the prompt-only router? What does a 4% limit change? | On `lcb` neither matters (under 5% at any limit). On `lcb-swe` together they take the router from 4-5% to 18-24% at epsilon 0.04-0.05 with accuracy above gpt-5's, but that pool's gain rests on gpt-5's formatting failures on SWE-bench. Raising epsilon from 0.02 to 0.04 alone moves the router by 2-4 points. |
| 2026-11-11 | [hidden-states](2026-11-11-hidden-states/README.md) | Does a frozen LLM's internal state work in front of the kNN index where topic embeddings failed? | It predicts difficulty clearly better (kNN AUC 0.81 against 0.70-0.74). That still does not make a prompt-only router save much (2-8% at epsilon 0.04), but as a gate in front of chains it certifies 15% where topic-embedding gates certify 8-11%. |
| 2026-11-13 | [trained-verifier](2026-11-13-trained-verifier/README.md) | Can a small classifier trained on graded patches verify a cheap model's patch? | On 861 accepted patches from 204 tasks, scored on held-out tasks: patch size and wording give AUC about 0.6, no better than a 32B open model. Agreement between cheap models' patches gives 0.76, the best signal found that needs no strong model: it catches 42% of wrong patches while rejecting 10% of right ones. Hidden states of a frozen 7B model give 0.69 and do not add to agreement. The session judge is at 0.99, so training on 861 patches does not replace a strong judge. |
| 2026-11-14 | [xroutebench](2026-11-14-xroutebench/README.md) | Does the agreement check hold on a second benchmark, and how do the LLMRouter library's routers do under a quality limit? | Agreement repeats on xRouteBench (18 open models): on 4,762 multiple-choice questions two cheap models that agree save 45% at the 2% limit and 74% at 4% (over the limit in 2 of 10 runs at 4%); three save 53% at 2%. Math: 16-20%. Switching to one cheaper model certifies 0%. The library's kNN, SVM, MLP, matrix-factorisation and Elo routers, run unmodified on LiveCodeBench, either send everything to gpt-5 or drop accuracy from 86% to 39-46%; certified savings 0%. |
| 2026-11-14 | [api-judge](2026-11-14-api-judge/README.md) | Does a pinned strong model verify patches, what does a check cost, and does a chain with it save money net of the judge? (Paid runs: $12.69 in total, of which $8.85 for the Sonnet run on all 1,110 cases.) | `claude-sonnet-5-5` on 1,110 accepted patches: rejects 79% of wrong ones and 9% of right ones (AUC 0.92; 89% and 6% on the 150 pilot cases, which were optimistic) at $0.008 per check, 2.4% of a frontier answer. Chains with free checks plus the judge on 274 tasks: about a third cheaper than `claude-opus-4.1` net of the judge, 55% accurate against 49-50%, within the limit in 29 of 30 held-out runs. Certified: 0% with the default split (too few tasks), 22% at a 4% limit and 30% at 5% when half the tasks certify. Nothing at 2% yet. |
| 2026-11-16 | [shadow-replay](2026-11-16-shadow-replay/README.md) | Do the proxy's shadow mode and live routing do what a frozen policy says? | Yes, both: replaying 1,414 recorded multiple-choice requests through the real proxy reproduces the policy's stored estimate (77.7% answered by cheaper models, 88.9% accuracy against 87.0%, 0.99% degradation; savings 53.5% against 52.5%, the difference being a model shared by two stages). The savings report page shows the same replay as 53.4%: the figure is 53.46% and the page rounds savings down. The replay caught an accounting bug before the result. No model called. |
| 2026-11-16 | [simulated-pilots](2026-11-16-simulated-pilots/README.md) | What would five realistic customer teams hit if they tried Tine from the docs? (Simulated teams, stand-in providers, no model called.) | The core holds: shadow mode never changed an answer, tenant isolation held, every page number recomputes. Not pilot-ready: a webhook check can send the provider key to any URL; policies carry no answer reader; the shadow report hides what it skipped; edited policies still show as certified; live-routing savings cannot be measured. 39 findings survived independent checking (23 distinct problems after merging duplicates); 8 claims did not hold; top 5 fixes listed. A follow-up fixed or documented #1-10, #12-14, #16, #21 and #22, several in part; its README has a status table. |
| 2026-11-16 | [agreement-more-datasets](2026-11-16-agreement-more-datasets/README.md) | Does the agreement check hold on the other exact-answer datasets in the LLMRouterBench archive? | No. On 16 more datasets and a second split of MMLU-Pro it certifies almost nothing at 2%, and at 4% only arcc (38.9%, triples, 7 of 10 runs) and math500 (12.1%, pairs) pass 10%. One cheaper model alone certifies 0% everywhere. Main cause at 4%: the cheap 8B models are too weak next to the frontier. Size explains the four small sets and the 2% column (MMLU-Pro cut to 1,000 tasks keeps about 40% at 4% and a third of its 2% result, over six random cuts). Free-form answers rarely match as strings. |
| 2026-11-16 | [swe-agreement-and-judge](2026-11-16-swe-agreement-and-judge/README.md) | On SWE-bench code chains, can agreement between cheap models' patches cut what the judge costs or certify more? | No reliable gain. Identical patches are too rare (a pair agrees on 0.5% of tasks). Similar patches (overlap >= 0.8) agree on 11% of tasks and are right 84% of the time. Certified savings show no reliable gain. Uncertified, as `tine cascade` charges it, pairs save the same or less than the plain chain (Haiku 39.3% vs 39.3% and 35.2% vs 37.9%); that charge pays a model twice when it sits in two groups, and with it paid once pairs are within 1.4 points either way (40.2% vs 39.3% and 38.7% vs 37.9% with Haiku; 33.6% vs 33.6% and 32.9% vs 34.3% with Sonnet). Agreement alone saves 3-7%. Post hoc: requiring agreement and the judge certifies in every run at 4% with Sonnet but saves only 3-6%. |
| 2026-11-16 | [judge-probes](2026-11-16-judge-probes/README.md) | Does the code judge rely on memory of these public fixes, can a patch talk it into accepting, and does context or a stronger judge help? (Paid: $31.02.) | The judges remember: Sonnet writes the merged fix from the issue alone for 41% of tasks. On the other tasks Sonnet still rejects 78% of wrong patches, but the code chain loses about 2.3% of the frontier's solved tasks, not 0.7%, so plan Haiku- or Sonnet-judged chains at a 4-5% limit. `claude-opus-5-5` on those tasks rejects 89% of wrong patches and 9% of right ones; with Haiku screening in front of it the chain saves 23.7%, is 6 points more accurate than the frontier and loses 0.46% (uncertified, 150 tasks). Injected approvals and removing the project's name did not move the judges; code context did not help. |
| 2026-11-16 | [judge-designs](2026-11-16-judge-designs/README.md) | Offline: does a tiered judge, a judge that picks among several cheap patches, or recalibration beat the plain chain? | No. A Haiku-then-Sonnet tier matches Sonnet's catch rate at 43-55% of its cost but judge calls are only 2.5% of frontier spend, so chains barely change. Selecting among patches adds little; its apparent gain comes from reaching more models, which longer chains also do. Recalibration halves calibration error without helping decisions. Reading 60 errors: 21 of 30 false rejects name a real flaw the hidden tests do not check, so the measured false-reject rate overstates the judge's mistakes. |
| 2026-11-17 | [measured-shortlist](2026-11-17-measured-shortlist/README.md) | Does building chains from candidates ranked by their measured checks, not by savings under a perfect check, certify more? | Mixed. Opt-in `--shortlist measured` raises certified savings where there are hundreds of agreement groups (MMLU-Pro test_1000 wide pool triples at 4%: 15.0% to 30.5%; pool-wide triples: 5.9% to 23.0%) and xRouteBench pairs at 2% (45.2% to 54.3%), lowers 7 of 44 certified cells by 0.2-2.2 points, and changes nothing on code. The default is unchanged and reproduces every published number. |
| 2026-11-17 | [judge-prompt](2026-11-17-judge-prompt/README.md) | Does a stricter judge prompt, written from the error reading, catch more wrong patches? (Paid: $4.02.) | No. Haiku and Sonnet catch slightly fewer wrong patches with it, within run-to-run noise. Prompt, context and hardening have all failed to move the judge; only a stronger model has. The original prompt stays. |
| 2026-11-17 | [live-pilot](2026-11-17-live-pilot/README.md) | Does Tine work end to end with real current models, as one customer would run it? (Paid: $23.65; plan written before any call.) | Yes, on exact-answer traffic at the 4% limit. Haiku+Sonnet agreement, else Opus 5.5, certified at 4% on 1,000 MMLU-Pro questions. Its 2% certificate passed on the last allowed loss with 495 certify tasks and did not hold up (2.3% on 2,305 questions outside the certify part, 95% upper bound 2.8%; 2.4% on 1,800 new questions); the 1,800 include the 1,500 of [concurrency-and-long-shadow](2026-11-17-concurrency-and-long-shadow/README.md). Do not quote 2% for this policy. On 300 new questions in shadow mode it lost 2.0% and would have saved 47.8%; routed live on 200 more it scored 91.5% against Opus's 91.7-92.2% at about 52% lower cost. Every pre-registered check passed. Free text on the same token cost 31% more; `/v1/savings` showed nothing under live routing. |
| 2026-11-17 | [live-pilot-more](2026-11-17-live-pilot-more/README.md) | Does the live pilot hold on four more kinds of traffic (MedQA, ARC-Challenge, FinQA, MATH-500)? (Paid: $60.00 from a credits key; plan written first.) | Yes at 4%, not at 2%. Shadow loss 0.5-2.5% on new questions. Measured saving under live routing with a random frontier holdout: MedQA 56% (49-64%), ARC 97% (Haiku alone certified), FinQA 18% (5-32%). Comparing answers by value raised MATH-500 from 44.5% to 60.5%. Shadow calls sharing rate limits with real traffic caused refused requests; `tine serve` ran as a real server. |
| 2026-11-17 | [concurrency-and-long-shadow](2026-11-17-concurrency-and-long-shadow/README.md) | Where is this account's rate limit, does shadow mode's pause keep real requests from being refused there, and does the first pilot's policy hold on 1,500 more questions? (Paid: $26.43 from a credits key; plan committed after the rate-limit part had run.) | The 4% limit holds on 1,500 new MMLU-Pro questions in shadow mode: loss 2.47% (37), 95% upper bound 3.2%. 2% is not confirmed: the bound is above 2%, and the policy's 2% certificate did not hold up (see live-pilot). The key allows 25 requests in flight, and a refusal names no wait. The planned pause test did not meet its pass marks: no real request was refused with either code version, so it did not exercise the pause. In a test added after the plan, the provider refused 9 of 69 real requests without the pause and none of 60 with it; 55 of those 60 were not shadowed. Five blocks of 300 lost 1.3-3.3%, and the drift check warned in 2 of them while the loss in each was under the limit. |
| 2026-11-17 | [ticket-pilot](2026-11-17-ticket-pilot/README.md) | Does the live pilot hold on real support tickets from one company, labelled by its staff? (Paid: $14.74 from a credits key. Public data labelled in 2020; the triage prompt is ours and the company is not a customer. Plan committed about two minutes after the first calls.) | Only just, at 4%. Sonnet alone, with no check, certified at 4% (bound 3.15%); nothing certified at 2%. On 300 new tickets in shadow mode the loss was exactly 4.0% (12), with a 95% upper bound of 6.4%, so the run does not show the loss on new tickets is under 4%. On the held-out part of the history it was 5.1% (10 of 197). Measured saving under live routing with a random 30% frontier holdout: 49.4% (47.9-50.9%). Opus matches the staff label on only 76-82% of tickets, so the labels are noisy. |

## How the findings fit together

1. The savings exist. The hindsight oracle saves 63% on `lcb` and 50% on `lcb-swe`.
   The 63% is one held-out split of 159 tasks, not the whole pool. A review of the
   evidence brief recomputed it as 59.7% on average over ten splits and 57.5% on all
   1,055 tasks; those two figures are not in a results file yet.
2. Predicting from the prompt which model will succeed does not work on this data.
   Five feature sets (TF-IDF, MiniLM, Qwen3 0.6B and 8B embeddings, hidden states of
   a frozen 7B model) and three predictors give the same answer. Hidden states
   predict difficulty better and help as a gate in front of a chain, not as a router.
3. Trying a cheaper model and checking its answer does work, if the check is good.
   The check is the bottleneck: required catch rate is about 95%.
4. The official Avengers-Pro code, also prompt-only, saves money on `lcb` only by
   giving up accuracy, and cannot hold a 2% limit. Tine's router reports that it
   cannot and falls back to the frontier.
5. On real repositories (django and sympy, 277 of 306 tasks run) checks that ask "did anything break"
   catch about two thirds of wrong patches. The rest apply cleanly, break nothing and
   do not fix the issue.
6. Public sample tests (catch 38-74% for the useful models) and response size (AUC
   0.76-0.99) are weak checks. Together they certify 6% at epsilon 0.02 on `lcb`,
   with their combination simulated as independent, not measured.
7. A judge that reads the issue and the patch (Gacrux stage 4, applied after the
   answer) works only when it is a strong model. Open models of 14B and 32B and
   classifiers trained on graded patches do not work as a judge. `claude-sonnet-5-5`
   rejects 79% of wrong patches and 9% of right ones at $0.008 per check (the
   150-case pilot overstated this), and a chain with it certifies 22% at a 4% limit
   when half the tasks certify, nothing at 2%.
8. Where answers can be compared exactly, agreement between cheaper models is a
   strong check that needs no judge: the first large certified savings at epsilon
   0.02 in this repository (MMLU-Pro), repeated on xRouteBench multiple choice. It
   did not repeat on 16 other LLMRouterBench datasets. It needs cheap models close to
   the frontier in accuracy, about 300 certify tasks for the 4% limit and about 900
   for 2% (about 1,000 and 3,000 graded tasks in all at the default 40/30/30 split),
   and answers that compare as strings. Every member's call is paid for.
   On code, agreement between cheap models' patches gives no reliable gain.
9. At epsilon 0.02 the number of certification tasks limits the result as much as
   the check does: with about 420 tasks a policy may show at most 3 degradations.

## Open decisions

Two findings from the `lcb-swe` Avengers-Pro run question parts of the Gacrux
definition in AGENTS.md. Both now exist as opt-in options (`tine run --rule relative`,
`--loss net`) and are measured in [router-rules](2026-11-11-router-rules/README.md).
Whether either becomes the default is still a team decision.

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

- Finish the SWE-bench repository-test run (277 of 306 tasks done as of 2026-11-14; the free checks alone certify 0%) and
  certify chains on it: see [docs/SWE_CHECKS.md](../docs/SWE_CHECKS.md).
- Code: a strong judge works and pays for itself ([api-judge](2026-11-14-api-judge/README.md)); open
  models and trained classifiers do not. What is missing is more graded tasks, so that a chain
  with the judge certifies at a 2% limit, and a test on code the judge cannot have seen.
- Per-answer sample-check results for `lcb` exist only on the machine that ran
  `scripts/lcb_sample_checks.py` (Linux, isolated). With them, `tine cascade
  --checks ... --response-check` measures the combined check.
- Avengers-Pro on a pool with more kinds of task.
- No result here uses customer traffic. Token counts, costs and check quality all
  need measuring in shadow mode.
- `candidate_chains` in `tine/cascade.py` shortlists the top eight candidates by
  savings under a perfect check and ignores the measured check results. With many
  agreement groups it may miss a reliable group. Found in the review of
  [agreement-more-datasets](2026-11-16-agreement-more-datasets/README.md). Not
  changed, because it would change every earlier result.
- `chain_inputs` in `scripts/judge_patches.py` adds judge cost to the frontier's own
  answers, which a chain never judges. This raises the always-frontier baseline and
  makes the published api-judge savings up to 0.44 points higher (Sonnet uncertified
  34.0% against 33.6%; certified 22.2% against 21.9%). Not changed.
- The shared importer refuses the `hle` dataset because five records have negative
  token counts. `scripts/agreement_datasets.py` works around it by leaving those
  tasks out of a copy.
- The container image (`Dockerfile`, `docker-compose.yml`, `docs/DEPLOY.md`) was built
  and run once from an earlier Dockerfile (40 paid requests through a route policy,
  2026-11-17). The files as they are now have not been built.
- The savings report page (`docs/REPORT.md`) has been checked in a desktop browser
  only (2026-11-16, the two example pages), not in a mail client or on a phone.
- The proxy does not read answers the way a policy's agreement check was certified.
  A policy can now carry its pattern (`freeze-policy --answer-pattern`), and a route
  policy with no reader is refused, but the proxy still compares the raw match with
  no case, comma or trailing-zero handling. A narrower pattern can accept answers the
  certified check would reject. Stated in [docs/PROXY.md](../docs/PROXY.md).
- Policy files carry a freeze digest, so an edit after freezing is flagged, but they
  are not signed: whoever edits a file can write a new digest. `task_type` is logged
  at startup and not read per request; the docs ask for one tenant token per task
  type. Stated in [docs/PROXY.md](../docs/PROXY.md) and [docs/POLICY.md](../docs/POLICY.md).
- Under live routing the saving is measured only with the random frontier holdout
  (`TINE_AUDIT_RATE`, off by default), which gives an estimate with a 95% interval from a
  random sample and says nothing about quality. It is built and tested with stand-ins, and
  has run with real current models on public benchmark questions
  ([live-pilot-more](2026-11-17-live-pilot-more/README.md)) and on a public set of one
  company's support tickets ([ticket-pilot](2026-11-17-ticket-pilot/README.md)), never on a
  customer's traffic ([docs/PROXY.md](../docs/PROXY.md)).
- The certification planning figures in [docs/PLANNING.md](../docs/PLANNING.md) are
  for one chain tested alone. Nothing yet models the many-chain sequence that
  `tine cascade` and `tine freeze-policy` run today.

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
