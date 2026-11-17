# Tine: handoff note

Written 2026-11-17 for Nandan, Ashmith and Nirvika, and for any engineer or coding agent
who picks the project up after a pause. Every statement comes from a file in this branch,
and the file is named beside it. Paths starting with a date are folders under
[`results/`](../results/README.md). Where this note and a results file disagree, the
results file is right.

## 1. Where things stand

Tine is a proxy that sends a request to cheaper models first and uses their answer only
when a check passes, under a quality limit certified on graded tasks (`AGENTS.md`,
section 12). On exact-answer work the 4% limit has held with real Claude models on five
public benchmarks, and the one 2% certificate did not hold up
(`2026-11-17-live-pilot/README.md`, `2026-11-17-live-pilot-more/README.md`). The proxy,
ledger, frozen policies, shadow mode, live routing, a frontier holdout and a drift check
are built at development quality on SQLite (`docs/PROXY.md`). All evidence is public
data, the pilot work spans two days (2026-11-16 and 2026-11-17), and every paid call,
$172.55 in all, went to one provider (`results/README.md`). No customer has used any of
it.

## 2. What is built

Status is quoted from `AGENTS.md` section 12 unless another file is named.

| Component | Files | Status as the docs state it |
| --- | --- | --- |
| Offline toolkit: outcome data, calibration, prompt routers, baselines | `tine/data.py`, `calibration.py`, `routing.py`, `pipeline.py` | Built and tested |
| Chains with checks, certified by fixed-sequence Clopper-Pearson | `tine/cascade.py`, `tine cascade` | Built and tested |
| Agreement check from recorded answers | `scripts/agreement_pairs.py`, `scripts/agreement_datasets.py` | Built and tested |
| Judge for code patches | `tine/judge.py`, `scripts/judge_patches.py` | Built and tested; measured with paid calls ($47.73) |
| Frozen policies, with a freeze digest | `tine/policy.py`, `tine freeze-policy`, `docs/POLICY.md` | Built and tested. The digest is not a signature. No customer has certified on its own tasks |
| Answer comparison by value (`answer_compare: "math"`) | `tine/policy.py`, `tine/shadow.py`, `tine/proxy.py` | Built; a policy certified at 4% with it on FinQA and MATH-500, nothing at 2% |
| Proxy with tenant tokens, ledger, verified cache | `tine/proxy.py`, `tine/ledger.py`, `docs/PROXY.md` | Development quality; SQLite |
| Shadow mode and live routing through a policy | `tine/shadow.py`, `tine/proxy.py` | Run with real current models on public benchmark questions; never on a customer's traffic |
| Named checks (tests behind a webhook, a judge call) | `tine/runtime_checks.py` | Tested with stand-ins only |
| Random frontier holdout (`TINE_AUDIT_RATE`) | `tine/proxy.py`, `tine/ledger.py` | Run with real models on public data; an estimate from a random sample that does not measure quality |
| Drift check in shadow mode | `tine/shadow.py`, `tine/policy.py`, `2026-11-17-drift-check` | Built and tested; replayed on recorded runs only; the notice has had no review |
| Shadow pause on provider rate limits | `tine/shadow.py`, `tine/proxy.py` | Met a real provider's limit once, in a test added after the plan |
| Cost of streamed requests | `tine/proxy.py`, `tine/ledger.py` | Tested with stand-in streams. Streams are not shadowed, routed or cached |
| Live pilot harness | `scripts/live_pilot.py` | Run with paid calls on five public benchmarks and one public ticket set |
| Container packaging | `Dockerfile`, `docker-compose.yml`, `deploy/`, `docs/DEPLOY.md` | The current files have not been built or run by a daemon (section 5) |
| Load test tool | `scripts/load_test.py`, `tests/test_load_test.py` | In the branch; no results folder yet (section 5) |
| Learning loop: certify a tenant's policy from its own traffic | `docs/LEARNING_LOOP.md` | A design. Not built |

Not built: nightly per-tenant retraining, distillation, a production datastore, streaming
under a policy, a measurement of the loss on live traffic, a record of past drift
warnings (`AGENTS.md`, section 12).

## 3. What is proven, and what is not

Terms: epsilon is the largest share of tasks the customer accepts losing, where the
frontier is right and the routed answer is wrong. Delta is 0.05 throughout. "Certified"
means a one-sided Clopper-Pearson bound on the certify part is at or under epsilon
(`results/README.md`). The team reports epsilon 0.04 first and keeps 0.02 beside it.

### The 4% claims hold

All rows use Haiku 5.5 and Sonnet 5.5 as the cheaper models and `claude-opus-5-5` as the
frontier, through the real proxy.

| Traffic | Policy certified at 4% (bound) | Loss on new requests in shadow mode | Source |
| --- | --- | --- | --- |
| MMLU-Pro | Haiku and Sonnet agree, else Opus; 1,000 graded questions | 2.0% of 300; then 2.47% of 1,500 (37 lost), 95% upper bound 3.2% | `2026-11-17-live-pilot`, `2026-11-17-concurrency-and-long-shadow` |
| MedQA | Haiku and Sonnet agree (2.6%) | 0.5% of 200 | `2026-11-17-live-pilot-more` |
| ARC-Challenge | Haiku alone, no check (3.6%) | 2.5% of 200 | same |
| FinQA, exact | Haiku and Sonnet agree (2.6%) | 1.0% of 200 | same |
| MATH-500, exact | Haiku and Sonnet agree (2.8%) | 1.0% of 100 | same |
| IT support tickets | Sonnet alone, no check (3.15%) | 4.0% of 300 (12), upper bound 6.4% | `2026-11-17-ticket-pilot` |

- The MMLU-Pro certify part was 495 questions. The four other benchmarks had certify
  sets of about 400, or 165 for MATH-500.
- ARC's held-out loss was 3.25%, close to the limit.
- The ticket row is at its limit. See below.

### The 2% pass was not confirmed

- The MMLU-Pro policy also passed at 2%, on the last allowed loss: 4 of 495 certify
  questions, bound 1.84%.
- On 1,500 new questions it lost 37 (2.47%). The bound of 3.2% is above 2%.
- Outside the certify part the loss was 2.3% on 2,305 questions (upper bound 2.8%), and
  2.4% on the 1,800 new ones.
- 2% is not ruled out either: the lower bound on the 1,500 is 1.8%.
- Do not quote 2% for this policy. Nothing certified at 2% on the other five kinds of
  traffic (`2026-11-17-concurrency-and-long-shadow/README.md`).

### Measured saving under live routing

Each run sent a random 30% of requests to the frontier (`TINE_AUDIT_RATE=0.3`). The
saving is 1 - mean(routed cost) / mean(held-out cost), with a 95% interval. It compares
cost only and says nothing about quality.

| Traffic | Routed / held out | Measured saving (95% interval) | Source |
| --- | --- | --- | --- |
| MedQA | 105 / 45 | 56.3% (48.8-63.8%) | `2026-11-17-live-pilot-more/live-routing.csv` |
| ARC-Challenge | 104 / 46 | 97.3% (97.0-97.6%) | same |
| FinQA, exact | 96 / 37 | 18.3% (4.9-31.6%) | same |
| MATH-500, exact | 56 / 14 | Not shown: fewer than 30 held out | same |
| IT support tickets | 216 / 84 | 49.4% (47.9-50.9%) | `2026-11-17-ticket-pilot/live-routing.csv` |

- MMLU-Pro has no holdout figure. Its routed run of 200 questions cost about 52% less
  than Opus, at 91.5% accuracy against 91.7-92.2%, before the holdout existed.
- The saving follows how close the cheaper models are to Opus. On FinQA Haiku gets 64%
  right against Opus's 79%.

### The ticket pilot is at its limit

- 2,229 public tickets from one IT support company, labelled by its staff in 2020. The
  triage prompt is ours. The company is not a customer.
- Sonnet alone certified at 4% with 9 losses on 495 certify tickets. It lost 5.1% (10 of
  197) on the held-out part and exactly 4.0% (12 of 300) on new tickets.
- The README says this evidence does not support routing these tickets through Sonnet
  alone at a 4% limit.
- A safer policy exists on paper. Haiku and Sonnet agreeing, else Opus, passed at 4%
  with a bound of 2.11%. On the 197 held-out tickets it lost 2.0% and saved 32.0%
  (`certification.csv`). The search preferred Sonnet alone because it saved more.
- That pair policy was never run in shadow mode or live. No results file says it was
  written to a policy file.
- Opus matches the staff label on only 76-82% of tickets, so the labels are noisy.

### Offline results on recorded answers

- Agreement between cheap models certifies 32.3% saved at 2% and 63.8% at 4% on 3,000
  MMLU-Pro questions, and 45-53% at 2% on xRouteBench multiple choice
  (`docs/EVIDENCE.md`, section 2).
- It did not repeat on 16 other datasets (`2026-11-16-agreement-more-datasets`).
- Prompt-only routers, ours and others', certify 0-5% on coding (`AGENTS.md`, section 12).

### Not proven

- **Any result on a customer's traffic.** There is no customer.
- **Anything beyond public data.** The models may have seen every benchmark and the
  tickets.
- **Anything over time.** The paid live runs all happened on 2026-11-17, and the
  recorded-answer work in the days before it. Nothing shows drift over weeks.
- **Any provider but one.** Every paid run called Claude models through the Anthropic
  API.
- **A 2% limit**, on any live traffic.
- **The loss on live traffic.** The drift check counts differing answers and grades
  nothing. Knowing the loss still means grading a sample.
- **Code on private repositories.** No measurement exists.
- **The named checks with real services.** The tests webhook and the judge call ran with
  stand-ins only.

## 4. Known problems and open decisions

| Item | What the files say | Source |
| --- | --- | --- |
| Drift check: the notice tier | Added after the reviews. It has had no review of its own. The constants were chosen with the recorded runs in view | `2026-11-17-drift-check/README.md`, Limits |
| Drift check: weak near the limit | A steady loss of twice the limit warned in at least 80% of runs on 5 of 15 policy and limit pairs at 1,000 requests. On FinQA, at most 9% at any length | same, section 3 |
| Drift check: old policy files | A file frozen before the change has no baseline and gets a default allowance of 10 points. Freeze it again | same |
| The 30-second first shadow pause | The provider's refusal carried no `retry-after`, so the proxy paused for its own 30 seconds. With the pause, 4 of 60 requests were shadowed. The pause was never seen to end | `2026-11-17-concurrency-and-long-shadow/README.md` |
| Shadow needs spare capacity | The test key allows 25 requests in flight across all models. While real traffic nearly fills the account, shadow mode measures almost nothing | same, Limits |
| Ticket prompt examples repeat | 30 of the 300 shadow tickets (10%) repeat one of the 35 examples in the prompt. 97 of the 300 repeat some earlier text; on new text the loss was 4.9% | `2026-11-17-ticket-pilot/README.md`, Step 3 |
| Streams | Passed through to the frontier. Never shadowed, routed or cached. The streamed share of traffic caps any saving | `docs/PROXY.md`, `docs/DEPLOY.md` |
| SQLite | One file on one volume, one process, no replication. Prompts and responses are stored in plain text | `docs/DEPLOY.md`, What is development quality today |
| Spend caps | Shadow spend has a per-tenant cap (`TINE_SHADOW_MAX_COST_USD`). It counts known costs only. The docs describe no cap on a tenant's real spend and no rate limit | `docs/PROXY.md`, The spending cap |
| Retention | The docs describe no retention setting. The learning-loop design lists retention as something to agree on day 0 | `docs/PROXY.md`, `docs/LEARNING_LOOP.md` section 7 |
| OpenAI | No results file records a call to a real OpenAI account. Streamed cost reading was never run against a real provider's stream | `results/README.md`, `docs/PROXY.md` |
| Who pays for shadow calls | The plan says shadow mode and the report are free. Shadow calls are real calls on the customer's provider account. Shadowing every request cost 18% of the frontier bill in the replay | `AGENTS.md` sections 1-2, `docs/DEPLOY.md`, `docs/LEARNING_LOOP.md` section 5 |
| Tokens and policies | Tenant tokens are static. Policies are not signed. `task_type` is not read per request, so use one token per task type | `docs/DEPLOY.md`, `results/README.md` |
| Router rule and net-loss guarantee | Both exist as opt-in options. Whether either becomes the default is a team decision | `results/README.md`, Open decisions |

Three docs lag the results. `docs/EVIDENCE.md` still opens with "The only live model
calls were the judge's", though its section 4 reports the pilots. `README.md`'s Current
state gives $131.38 and says nothing has served real traffic. `docs/DEPLOY.md` says
"There is no load test" while `scripts/load_test.py` is in the branch.

## 5. Not yet run

- **The container files as they now stand have not been built by a Docker daemon.** An
  earlier Dockerfile was built once on 2026-11-17 and its container served 40 paid
  requests through a route policy. The `Dockerfile` and `docker-compose.yml` were changed
  afterwards. `docs/DEPLOY.md` says to expect to fix something on the first build.
- **The one command to run is `python3 scripts/container_selftest.py`.** It needs Docker
  running and nothing else. No model is called and nothing is paid for. It builds the
  image, tests a running container against a stand-in provider, and prints PASS or FAIL
  per check. The script itself has only run against a stand-in `docker` command.
- **The load test has no results in the repository.** `python scripts/load_test.py matrix
  --output FOLDER` runs the named scenarios against a stand-in provider on the same
  machine. A run was in progress when this note was written, and its results are not in
  the repo yet. Until they are, capacity is unknown. A run on a busy laptop sets
  `busy_machine` and is not to be reported as a measurement (`scripts/load_test.py`).

## 6. What to build next, in order

No doc in the branch gives a size estimate for items 1 to 4. The only schedule in the
repo is the two-week pilot in `docs/LEARNING_LOOP.md` section 7.

1. **Streaming.** Shadow streamed requests first, then route them by buffering the
   cheaper answer and replaying it. The design for this is not in the repo. Today a
   stream is relayed byte for byte and only its cost is read. Two of the five simulated
   teams sent mostly streamed traffic (`2026-11-16-simulated-pilots/README.md`).
2. **A production datastore.** Postgres is planned and not built. SQLite limits the
   service to one process on one volume (`docs/DEPLOY.md`).
3. **Billing from the holdout.** The price is 20% of verified savings (`AGENTS.md`,
   section 1). Under live routing the saving is measured only by the frontier holdout,
   and `docs/PROXY.md` calls that figure an estimate, not a billing amount. It pools every
   policy digest and rate in the ledger.
4. **Monitoring and re-certification.** There are no metrics or structured logs, and a
   drift warning that goes away leaves no trace. `docs/LEARNING_LOOP.md` designs
   certificates that expire and are renewed from the tenant's own traffic. It is not
   built.

The step that matters most is not code: **one design partner in shadow mode on
exact-answer traffic.** `docs/EVIDENCE.md` lists it as a milestone, and the simulated
pilots recommend a shadow-only pilot on one exact-answer workload such as ticket
classification. `docs/LEARNING_LOOP.md` section 7 sketches it as two weeks at 2,000
requests a day: 300 certify rows for a 4% limit fill in about a day at that volume, and
a 2% limit is out of reach at 200 requests a day. It would give the first real costs and
show whether a policy certified on the customer's own graded work holds its limit.

Two research milestones from `docs/EVIDENCE.md` section 6 are still open: grade more code
tasks (about 500 in all), and run the judge on code it cannot have seen.

## 7. How to work in this repo

- **Read first:** `AGENTS.md`, then `results/README.md`, which is kept current.
- **Set up and check:**

  ```bash
  bash scripts/setup.sh
  source .tine-dev/venv/bin/activate
  bash scripts/check.sh
  ```

  The check runs the dependency check, Ruff, the smoke script and the test suite
  (`scripts/check.sh`). Passing tests do not establish savings or a guarantee
  (`README.md`).
- **Paid API calls need explicit authorisation and a spending cap.** Keys come from the
  environment and are never written to the repository or printed. Model downloads, Colab
  compute and bulk dataset downloads need authorisation too (`AGENTS.md`).
- **Run paid runs on mains power with the lid open.** A laptop sleep of 106 seconds hit
  two runs (`2026-11-17-concurrency-and-long-shadow/README.md`).
- **Model-written code runs only on a disposable Linux machine (Colab),** never on a team
  laptop.
- **Where data lives.** `artifacts/` is git-ignored and holds benchmark files, per-task
  outputs, prompts, replies, ledgers and policies from the runs. `results/` holds small
  summary tables and a README per experiment, nothing else. Never commit prompts,
  model answers, per-task outputs, benchmark files, customer data or credentials. Some
  inputs exist only on the machine that ran them or on Nandu's Google Drive
  (`results/README.md`, Where the
  data is).
- **New results:** one folder named `YYYY-MM-DD-topic` with a README, and a row in
  `results/README.md`. Say for every number whether it is certified. Say when a split,
  threshold or model list was chosen after seeing a result. Quote the held-out loss
  beside the bound.
- **Several tools share this repository,** so work in a separate worktree or clone and
  never edit another session's checkout. This line comes from the team's instruction for
  this note. `AGENTS.md` itself says cloud tasks are already isolated and should use the
  existing checkout unless a worktree is asked for.
