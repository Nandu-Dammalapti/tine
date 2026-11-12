# Hidden-state representations for the index model, 2026-11-11

Offline results on already-graded LLMRouterBench outcomes. The representation was
computed once on a Colab GPU by reading each prompt with a frozen model; nothing was
trained and no answer was generated. Small summary tables only. Design:
[docs/INDEX_MODEL.md](../../docs/INDEX_MODEL.md).

## Question

Every topic embedding tried in front of the router failed. arXiv:2610.19198 reports
that a frozen LLM's internal state works where topic embeddings do not. Does it on
our data? Representation: last-token hidden state at layer 21 of
Qwen2.5-7B-Instruct (3,584-d), prompt wrapped as a chat turn, issue-only view for
SWE-bench, 2,048-token cut (`scripts/hidden_states.py`).

## Result 1: it does carry more signal about difficulty

`predictive-auc-lcb.csv`. Five-fold cross-validated AUC for "will this model solve
this task" on the 1,055 `lcb` tasks, mean over eight models (gpt-5 and seven cheaper ones):

| representation | 20 nearest neighbours | logistic regression |
| --- | --- | --- |
| Qwen3-Embedding-0.6B (topic) | 0.705 | 0.758 |
| Qwen3-Embedding-8B (topic) | 0.736 | 0.795 |
| hidden state, layer 21 | **0.809** | **0.818** |

The paper's claim holds here for retrieval: with the same 20-neighbour vote, the
hidden state is 7 to 10 points better than a topic embedding, for every one of the
eight models. The kNN design was not the problem; the representation was a large
part of it.

## Result 2: the prompt-only router still certifies little

`router-hidden-vs-previous.csv`. `tine run`, 10 seeds, epsilon 0.04, best predictor
per cell. "Previous" is TF-IDF on `lcb` and Qwen3-0.6B issue-view on `lcb-swe`.

| pool | rule | previous | hidden state |
| --- | --- | --- | --- |
| `lcb` | absolute | 2.1% | 1.7% |
| `lcb` | relative | 2.9% | 4.8% |
| `lcb-swe` | absolute | 4.3% | 5.1% |
| `lcb-swe` | relative | 8.6% | 8.0% |

Better prediction did not turn into savings for a router that must choose before
any model answers. An AUC of 0.8 says a task is probably hard or easy; a 4% limit
needs to know, almost without error, that a specific cheaper model will solve a task
gpt-5 solves. Difficulty is shared across models, so knowing it does not say which
cheaper model is safe.

## Result 3: it helps as a gate in front of chains

`gated-lcb-hidden-knn-summary.csv`. `tine cascade --gate knn`, `lcb`, 10 seeds,
sample checks simulated at each model's measured catch rate. The gate decides which
tasks a cheaper model is tried on at all; the check still verifies the answer.

| epsilon | no gate | TF-IDF gate | Qwen3-8B gate | hidden-state gate |
| --- | --- | --- | --- | --- |
| 0.04 | 0% | 8.2% | 10.6% | **15.3%** |
| 0.02 | 0% | 1.1% | 1.8% | 5.1% (2 of 10 runs over) |

At epsilon 0.04 each gated cell had 1 of 10 runs over the limit.

## Reading

- The index model's job is not to pick the winner. It is to decide where a cheaper
  attempt is worth making, and the verifier decides whether to keep the answer. In
  that role the hidden state roughly doubles what a topic-embedding gate certifies.
- The next step named in the paper's line of work, and in docs/INDEX_MODEL.md, is to
  learn the representation from outcomes. These results say that is worth doing for
  the gate, not for a prompt-only router.
- A 7B model read once per request is not free: it is routing cost and must be netted
  from savings (AGENTS.md section 2). Reading 21 of 28 layers of a 7B model on about
  700 prompt tokens is small next to a frontier call, but it needs a GPU on the
  customer's server.

## Limits

- One layer, one model, one prompt format. Whether the paper wraps prompts as a chat
  turn was not confirmed; other layers were not tried.
- Coding tasks only, from benchmarks: no repeated requests.
- The sample checks in result 3 are simulated at measured rates.
- The paper's own numbers were not reproduced; only its idea was tested.

## Reproduce

```bash
python scripts/hidden_states.py --input DATA/lcb-swe.jsonl --prompt-view issue \
  --output EMB/lcb-swe-qwen25-7b-l21-issue-2048.npz        # GPU; see the Colab notebook
python -m tine run --input DATA/lcb.jsonl --frontier gpt-5 --epsilon 0.04 --rule relative \
  --embeddings EMB/lcb-swe-qwen25-7b-l21-issue-2048.npz --output RUNS/hidden/lcb-relative-0.04-0
python -m tine cascade --input DATA/lcb.jsonl --frontier gpt-5 --gate knn \
  --gate-embeddings EMB/lcb-swe-qwen25-7b-l21-issue-2048.npz --catch 0.9 0.95 \
  --catch-rates results/2026-11-10-gated-chains/lcb-catch-rates.csv --epsilon 0.04 0.02 \
  --output RUNS/hidden-gate-knn
```
