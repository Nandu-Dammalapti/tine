# A live pilot on one company's real IT support tickets, 2026-11-17

**Paid run.** `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`, called
through the Anthropic API (direct calls and the Message Batches API) and then through
Tine's proxy. The run used the team's separate credits key.

- **Spend:** **$14.74**, from the API's own token counts. The cap was $15.
- **The plan:** [`PLAN.md`](PLAN.md). Its commit is about two minutes later than
  the first calls (see Deviations).
- **The tickets:** Zenodo record 7648117, "Classification of IT Support Tickets",
  by Leonardo Santiago Benitez Pereira, licensed CC BY 4.0. 2,229 tickets from one
  IT support company, each given one of 7 categories by its own staff.
- **Data here:** the summary tables are `models.csv`, `accuracy-by-category.csv`,
  `certification.csv`, `shadow.csv` and `live-routing.csv`. Ticket text, prompts,
  replies and per-ticket rows stay local.

## Bottom line

- **This is the first run on one company's real traffic with its own labels.**
  Every earlier live result is on benchmark questions.
- **A policy certified at 4%: Sonnet alone, with no check.** The bound is 3.15%.
  Nothing certified at 2%.
- **The 4% result held with no margin.** Read the bound beside what was measured
  outside the certify part:
  - Held-out part of the history: 5.1% (10 of 197), above the limit.
  - 300 new tickets in shadow mode: 4.0% (12), exactly at the limit. One more
    would have failed the pass mark. The one-sided 95% upper bound is 6.4%, so
    this run does not show the loss on new tickets is under 4%.
  - Every ticket outside the certify part, 505 from the history and the 300 new
    ones: 3.98% (32 of 805; 95% interval 2.7% to 5.6%).
- **The certify part's 1.8% is not the loss to expect.** It is the count the
  policy passed on, and it is lower than every other part.
- **New text does worse than repeated text.**
  - The 203 new tickets whose text was not seen before: 4.9%.
  - The 97 whose text repeats an earlier ticket: 2.1%.
- **The saving is about half, and it was measured.** Live routing with a random 30%
  holdout measured 49.4% (95% interval 47.9-50.9%). Shadow mode had projected 49.9%.
  Every routed ticket was answered by Sonnet, which costs about half of Opus here.
- **The labels are noisy.** Opus matches the staff label on 76-82% of tickets. A
  "loss" means Opus matched the staff and Sonnet did not. On the 300 new tickets
  there were 12 of those and 7 the other way.
- **A safer policy was not run live.** Haiku and Sonnet agreeing, else Opus, lost
  1.3% over the history and also passed at 4%, with a bound of 2.1%. It saved 38%
  over the history against Sonnet's 49%, so the search preferred Sonnet alone.

## The customer and the setup

The company's triage step sends every new ticket to `claude-opus-5-5` to pick a
category. The prompt lists the 7 categories as options A-G, shows 35 past tickets
with their category, then the new ticket, and asks for one line, `Answer: X`
(`scripts/ticket_pilot_prep.py`). The 35 examples are used nowhere else.

- **Prompts are long and replies short.** A prompt is about 2,300 input tokens
  (2,246 to 2,848). The mean reply is 49 to 92 tokens.
- **"Right" means the category the company's staff gave.** The record's own files
  report 79% for the author's trained classifier on these labels.
- **Tickets repeat.** 830 of the 2,229 share their exact text with another ticket,
  once names are tags.
- **Answer reader:** `Answer:\s*\**\s*\$?\(?([A-J])\b`, written into the policy.
  Answers are compared exactly.

Prices, $ per million input / output tokens: Haiku 0.10 / 0.50, Sonnet 2 / 10,
Opus 4 / 20. Batch calls are priced at half of that.

## Results

### Step 1: the graded history (1,000 tickets, all three models)

`models.csv`, `accuracy-by-category.csv`. A probe of 20 tickets went by direct
calls ($0.30). The other 980 went through the Message Batches API ($7.71): three
batches of 980, all succeeded, the last one collected 11 minutes after submission.

| model | accuracy against the staff labels | reader matched | mean reply tokens | mean cost per ticket at direct prices |
| --- | --- | --- | --- | --- |
| Haiku 5.5 | 79.6% | 100% | 91.8 | $0.00028 |
| Sonnet 5.5 | 80.3% | 100% | 56.8 | $0.0052 |
| Opus 5.5 | 82.0% | 100% | 48.9 | $0.0102 |

- **The three models are close.** 2.4 points separate Haiku from Opus.
- **Sonnet and Opus gave the same answer on 94.5% of tickets.** Haiku and Sonnet
  agreed on 92.8%.
- **Cost against Opus:** Sonnet 50.8%, Haiku 2.7%, and the two together 53.5%.
- **No reply was cut off.** The longest was 488 tokens, with `max_tokens` at 512.

By category:

| category | tickets | Haiku | Sonnet | Opus |
| --- | --- | --- | --- | --- |
| Support general | 355 | 63.9% | 64.2% | 67.6% |
| Fileservice | 315 | 96.2% | 96.8% | 96.5% |
| Software | 99 | 76.8% | 77.8% | 75.8% |
| O365 | 89 | 77.5% | 78.7% | 80.9% |
| Active Directory | 56 | 64.3% | 66.1% | 76.8% |
| Computer-Services | 48 | 97.9% | 100% | 100% |
| EOL | 38 | 100% | 100% | 100% |
| all | 1,000 | 79.6% | 80.3% | 82.0% |

"Support general" is the weakest category for every model. Opus's largest lead
over Sonnet is on "Active Directory", 76.8% against 66.1% on 56 tickets.

### Step 2: certification

`certification.csv`. `scripts/agreement_pairs.py` built three candidates: the
Haiku+Sonnet pair, and each model alone. `freeze-policy` used a 0.3 / 0.5 / 0.2
split (308 / 495 / 197 tickets) and delta 0.05. Freezing again from the recorded
files gives the same policy.

| chain | at the 4% limit | loss on the 495 certify tickets | bound | held-out 197: loss / saving | all 1,000: loss / saving |
| --- | --- | --- | --- | --- | --- |
| Sonnet alone | **certified, chosen** | 1.8% (9) | 3.15% | 5.1% / 49.0% | 2.9% / 49.2% |
| Haiku + Sonnet agree, else Opus | passed, not chosen | 1.0% (5) | 2.11% | 2.0% / 32.0% | 1.3% / 38.3% |
| Haiku alone | not tested: lost 5.2% on the choose part | 3.0% (15) | - | 7.1% / 97.3% | 4.5% / 97.3% |

- **Why Sonnet alone won.** Both chains passed. The search keeps the one that saved
  more on the choose part.
- **Nothing certified at 2%.** Only the pair was tested, and its bound of 2.11% is
  just over.
- **The held-out loss is above the limit.** Sonnet alone lost 10 of the 197
  held-out tickets, 5.1% (95% upper bound 8.5%). The guarantee is the bound on the
  certify part. The held-out part is an estimate on a small sample, and here it
  points the other way.
- **The certify part is the lowest of the three.** Sonnet alone lost 3.2% on the
  choose part (10 of 308), 1.8% on the certify part (9 of 495) and 5.1% on the
  held-out part. The certify part against the rest of the history is 9 of 495
  against 20 of 505 (Fisher p = 0.058). A chain that passed has a low count on its
  certify part partly by selection, so 1.8% is not the loss to expect. Over all
  1,000 tickets the loss is 2.9% (29), with a bound of 3.93%: just inside 4%.
- **Not a violation.** A true loss of exactly 4% would give 32 or more losses in
  805 tickets about half the time (probability 0.54). The data fit a loss at the
  limit. They do not show a margin under it.
- **The pair sends 7.2% of tickets to Opus** and answers the rest when the two
  cheaper models agree.

### Step 3: shadow mode through the real proxy (300 new tickets)

`shadow.csv`. Sample rate 1, 3 requests in flight, `max_tokens` 512.

| tickets | requests | loss (limit 4%) | one-sided 95% upper bound | gain | accuracy: policy / Opus |
| --- | --- | --- | --- | --- | --- |
| all | 300 | 4.0% (12) | 6.4% | 2.3% (7) | 74.7% / 76.3% |
| text repeats an earlier ticket | 97 | 2.1% (2) | 6.3% | 0% (0) | 91.8% / 93.8% |
| new text | 203 | 4.9% (10) | 8.2% | 3.4% (7) | 66.5% / 68.0% |

- **Exactly at the limit.** 12 of 300 is 4.0%, and the pass mark was "at most 4%".
  A 13th lost ticket would have made it 4.3%.
- **Repeats flatter the result.** A third of the tickets repeat the exact text of
  an earlier ticket. 30 of them repeat one of the 35 examples in the prompt itself.
  On the rest the loss is 4.9% and both models are right about two times in three.
- **How "repeats" is counted.** The prep script compared each ticket with the
  examples and the 1,200 tickets first set aside for the history. Against the 1,000
  that were asked, the split is 96 and 204, with the same 2 and 10 losses.
- **Projected saving: 49.9%.** The certified estimate was 49.0%.
- **Coverage 100%.** No provider errors, no retries, nothing skipped.
- **The drift check warned.** 32 of the 300 answers (10.7%) differ from Opus's,
  more than the 4% limit. Of the 32, 12 were losses, 7 were gains, and in 13
  neither answer matched the staff label. The check counts all 32, so it runs above
  the loss. Here the loss itself is at the limit, so the warning is fair.

The pre-registered checks:

| check | pass mark | result | |
| --- | --- | --- | --- |
| shadow loss on new tickets | at most 4% | 4.0% (12 of 300) | pass, with nothing to spare |
| share answered by cheaper models | within 7 points of the certified estimate | 100% against 100% | pass |
| reader matched real replies | at least 98% for each model | 100% for each, in every step | pass |
| live: accuracy and measured saving | reported, no pass mark | below | - |

The second check cannot fail for this policy: Sonnet alone accepts every answer.

### Step 4: live routing with a random frontier holdout (300 more tickets)

`live-routing.csv`. `TINE_AUDIT_RATE=0.3`: 84 of the 300 requests (28%) skipped
the policy and went to Opus. 3 requests in flight, `max_tokens` 512.

| | routed (216) | held out (84) |
| --- | --- | --- |
| answered by | Sonnet, all 216 | Opus |
| accuracy against the staff labels | 81.9% | 76.2% |
| mean cost per request | $0.00524 | $0.01036 |
| caller latency, median / p95 | 1.38s / 2.44s | 1.89s / 3.01s |

- **Measured saving: 49.4%** (95% interval 47.9-50.9%). The interval is narrow
  because every request costs nearly the same.
- **The accuracy columns come from different tickets.** The gap of 5.8 points has
  a rough 95% margin of 10 points either way. It does not show Sonnet is better.
  The paired comparison is the shadow loss above.
- **The holdout does not measure quality.** It compares cost only.
- **No provider errors and no retries.**

### Opus against the staff labels

| set | Opus matches the staff label |
| --- | --- |
| history, 1,000 tickets | 82.0% |
| shadow, 300 new tickets | 76.3% |
| live, 84 held-out tickets | 76.2% |

Both models scored lower on the new tickets than on the history: Opus 82.0%
against 76.3% (Fisher p = 0.037), and Sonnet 80.3% against 74.7% (p = 0.043). The
history was asked through batches and the new tickets through the proxy. The cause
was not looked into.

Roughly a fifth to a quarter of tickets are "wrong" for Opus too. So the loss
against the frontier here measures agreement with one team's labelling as much as
it measures quality. Some losses will be tickets where two categories were both
reasonable. That was not checked.

## Deviations from the plan

1. **Smaller sets: a history of 1,000, not 1,200, and 300 live tickets, not 400.**
   A ticket prompt was about 2,300 tokens and cost more than the plan projected.
   The history cost $0.0080 a ticket against a planned $0.0060, and shadow $0.0158
   against $0.0117. So the held-out group is 84, not about 120.
2. **`max_tokens` 512 for the batches, the shadow run and the live run.** The plan
   said the first pilot's settings, which use 4,096. The probe ran at the harness
   default of 4,096 (no file records its setting), and its longest reply was 180
   tokens. At 4,096 the harness's worst-case count for the three batches would have
   been $66.67, over the cap. At 512 it was $13.11.
3. **`--max-cost-share 0.6` for `agreement_pairs.py`.** On these prompts the
   Haiku+Sonnet pair costs 53.5% of Opus and Sonnet alone 50.8%. With the default
   0.5 the script stops with "No pair of models is cheap enough". The value 0.6
   is from the run notes; the recorded candidates need at least 0.535.
4. **The plan's commit is later than the first calls.** The commit (`2341a23`) is
   at 10:10:33 UTC. The probe's 60 calls are stamped 10:08:35 to 10:09:00, and the
   batches were submitted at 10:10:05. The cut to 1,000 and 300 had been made by
   then, and the plan still says 1,200 and 400. Git cannot show that the plan was
   written before the probe.
5. **Code version:** `2341a23` for the shadow and live runs, from the version
   file. The proxy code under `tine/` is the same as in the long shadow run
   (`6dcd48d`).

6. **The laptop slept for 106 seconds during the shadow run.** The system's power
   log has a lid-closed sleep on battery from 10:27:57 to 10:29:43 UTC. The run's
   ledger has no finished request between 10:27:58 and 10:29:58.
   - Three provider calls took 118 to 127 seconds by the wall clock. All returned
     200.
   - For those three requests the ledger's latency is 1.4 to 23.7 seconds. The
     proxy times a request with a clock that stops while the machine sleeps.
   - No request failed and no count changed.
7. **The runs shared the account with the long shadow run.** The shadow run (10:22
   to 10:30 UTC) and the live run (10:30 to 10:33) ran beside blocks 4 and 5 of
   `../2026-11-17-concurrency-and-long-shadow`, each at 3 requests in flight. No
   call in any of them got a 429.

## Limits

- **One company, one task.** The labels are from 2020 and names are tags. This
  company is not a customer of ours.
- **Public data.** The models may have seen these tickets.
- **Noisy labels.** See above. The author's own classifier reaches 79%.
- **Thin margin.** The shadow loss is at the limit and its upper bound is 6.4%.
  This evidence does not support routing these tickets through Sonnet alone at a
  4% limit.
- **Sonnet alone has no check.** Nothing looks at an answer before it is returned.
  The policy is "use the cheaper model".
- **The pair policy is untested on new tickets.** Its figures are from the history
  only. It was not run in shadow mode or live.
- **Repeats.** 97 of the 300 shadow tickets and 106 of the 300 live ones repeat an
  earlier text. A company with fewer templated tickets would look more like the
  "new text" row.
- **Small samples, one hour.** 300 tickets per step. Nothing here shows drift.
- **The 35 examples were drawn at random,** not chosen by the company.
- **The history was paid at batch prices.** That halves every model's cost alike,
  so the cost shares and savings are unchanged.

## Spend (credits key)

| step | spent |
| --- | --- |
| probe, 20 tickets × 3 models, direct | $0.30 |
| history, 980 tickets × 3 models, batches | $7.71 |
| shadow, 300 tickets | $4.73 |
| live routing, 300 tickets | $2.00 |
| **total** | **$14.74** |

## Reproduce

The commands as typed were not saved. These give the same steps; the caps are
examples. The tickets are the four CSV files of Zenodo record 7648117.

```bash
python -I scripts/ticket_pilot_prep.py --data-dir artifacts/tickets-zenodo-7648117 --output RUNS/sets
head -n 1000 RUNS/sets/tickets-calibrate.jsonl > RUNS/sets/tickets-history.jsonl
head -n 20 RUNS/sets/tickets-calibrate.jsonl > RUNS/sets/tickets-probe.jsonl
head -n 300 RUNS/sets/tickets-live.jsonl > RUNS/sets/tickets-live300.jsonl
# paid from here: needs authorisation and ANTHROPIC_API_KEY in the environment
python scripts/live_pilot.py ask --questions RUNS/sets/tickets-probe.jsonl --output RUNS/asked.jsonl \
  --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5 --max-cost-usd 0.5
python scripts/live_pilot.py ask --batch --max-tokens 512 --questions RUNS/sets/tickets-history.jsonl \
  --output RUNS/asked.jsonl --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5 --max-cost-usd 14
python scripts/live_pilot.py export --asked RUNS/asked.jsonl --outcomes RUNS/outcomes.jsonl --answers RUNS/answers.csv
python scripts/agreement_pairs.py --input RUNS/outcomes.jsonl --answers RUNS/answers.csv --frontier claude-opus-5-5 \
  --output RUNS/pairs.jsonl --checks RUNS/checks.csv --include-singles --grade-conflicts wrong --max-cost-share 0.6
python -m tine freeze-policy --input RUNS/pairs.jsonl --frontier claude-opus-5-5 --checks RUNS/checks.csv \
  --epsilon 0.04 --fractions 0.3 0.5 0.2 --answer-pattern 'Answer:\s*\**\s*\$?\(?([A-J])\b' --output RUNS/policy.json
python scripts/live_pilot.py proxy --policy RUNS/policy.json --questions RUNS/sets/tickets-shadow.jsonl \
  --mode shadow --max-tokens 512 --max-cost-usd 5 --concurrency 3 --ledger RUNS/shadow.db \
  --output RUNS/proxy-shadow.json --summary RUNS/summary-shadow.json
python scripts/live_pilot.py proxy --policy RUNS/policy.json --questions RUNS/sets/tickets-live300.jsonl \
  --mode route --audit-rate 0.3 --max-tokens 512 --max-cost-usd 2.5 --concurrency 3 --ledger RUNS/route.db \
  --output RUNS/proxy-route.json --summary RUNS/summary-route.json
```

The split by repeated text uses the `text_seen_before` field that the prep script
writes on each shadow and live ticket.
