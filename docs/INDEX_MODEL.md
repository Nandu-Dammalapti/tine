# Index model

The index model is the part of Gacrux that looks at a request **before** any model
runs: it turns the request into a vector, finds similar past requests for the same
company, and uses how each model did on them. It is the classifier and retrieval
stages of AGENTS.md section 3. The **verifier** is the other half: it looks at a
cheaper model's answer afterwards (agreement, tests, a model that reads the answer;
see `docs/CHECKS.md` and `docs/JUDGE.md`). The two are separate and can be used
together: the index model decides what to try, the verifier decides whether to keep it.

## Where it stands

The retrieval design (20 nearest past requests, `tine/routing.py`) is built. On the
benchmarks it certifies about 0% savings, with every representation tried: TF-IDF,
MiniLM, Qwen3-Embedding 0.6B and 8B, full prompt and issue-only
(`results/README.md`). Two reasons, and only the first can be tested on a benchmark:

1. **The representation.** Sentence embeddings place prompts by topic. Two prompts
   about the same subject sit together whether they are trivial or hard. Neighbours
   found that way say little about which model will succeed.
2. **The data.** A benchmark is unrelated one-off tasks. A company's traffic repeats.
   Retrieval was designed for the second.

## The experiment: hidden states in place of embeddings

"Embedding Geometry Shapes LLM Routing" (arXiv:2610.19198) reports that keeping the
routing method and replacing the sentence embedding with one internal state of a
frozen LLM makes kNN routing generalise to unseen tasks. The state is the last
token's hidden state at layer 21 of Qwen2.5-7B-Instruct (3,584 numbers): the model's
own state at the point where it is about to answer. The paper's numbers have not
been checked by us; this experiment is the check, on our data.

`scripts/hidden_states.py` produces that representation in the NPZ format of
`scripts/embed.py`, so nothing else changes:

```bash
python scripts/hidden_states.py --input DATA/lcb-swe.jsonl --prompt-view issue \
  --output EMB/lcb-swe-qwen25-7b-l21-issue-2048.npz          # GPU
python -m tine run --input DATA/lcb-swe.jsonl --frontier gpt-5 --epsilon 0.04 \
  --embeddings EMB/lcb-swe-qwen25-7b-l21-issue-2048.npz --output RUNS/router-hidden
```

Choices made, all recorded in the output's metadata:

- Model `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, frozen, half precision on GPU.
- Layer 21 of 28. Layers above it are dropped before the forward pass.
- The prompt is wrapped as a user turn followed by the start of the assistant's
  reply (`--no-chat-template` turns this off). Whether the paper does the same has
  not been confirmed.
- Prompts are cut to 2,048 tokens from the start. With `--prompt-view issue`,
  SWE-bench prompts are reduced to their issue text first, so the cut rarely matters.
- Vectors are L2-normalised on save, as for every other representation here.

On Colab: `notebooks/tine_colab_hidden_states.ipynb`, GPU runtime (L4 or A100).

What would count as success: at the working limit (epsilon 0.04) the kNN router
certifies clearly more than the 2-4% it gets with sentence embeddings, on both pools,
and sits above the random-router line. If it does not, the signal is not in a frozen
model's state either, at least for these tasks.

## After that

1. **Learn the representation from outcomes.** Train so that two requests are close
   when the same models handled them well, whatever their topic. Each company's
   ledger supplies the training pairs; this is what nightly retraining would update.
   A retrained encoder moves every vector, so the nightly job must re-embed and
   rebuild the index, not insert into the old one.
2. **Hard rules before any learned routing**: context length, tool support, modality,
   data residency.
3. **Index size is not a concern yet.** One company's history at a million requests
   is a few gigabytes of raw vectors. A plain exact search is enough until measured
   otherwise.

## Reading list (not yet verified beyond existence unless noted)

- arXiv:2610.19198, Embedding Geometry Shapes LLM Routing. Existence and the
  layer-21 claim confirmed from the abstract page, 2026-11-11.
- arXiv:2603.04445, Dynamic Model Routing and Cascading for Efficient LLM Inference:
  A Survey (Moslem and Kelleher, February 2026). Existence confirmed.
- RouterDC (contrastive router) and IRT-Router (item response theory): named by a
  teammate's source; not checked.
