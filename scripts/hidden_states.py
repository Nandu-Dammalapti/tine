#!/usr/bin/env python3
"""Represent each prompt by a frozen language model's hidden state, for routing.

Sentence embeddings place prompts by topic. Routing needs to know how hard a prompt
is for each model, and every topic embedding tried so far failed at that
(results/README.md). "Embedding Geometry Shapes LLM Routing" (arXiv:2610.19198)
reports that keeping the routing method and replacing the embedding with one
internal state of a frozen LLM, the last token's hidden state at layer 21 of
Qwen2.5-7B-Instruct, makes kNN routing generalise to unseen tasks. This script
produces that representation in the same NPZ format as scripts/embed.py, so
``tine run --embeddings`` and ``tine cascade --gate-embeddings`` take it unchanged.

The model is never trained and never generates text: it reads the prompt once, up
to the chosen layer, and the vector at the final position is kept. Needs a GPU for
a 7B model. Resumable, like embed.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embed  # noqa: E402
from tine.data import read_outcomes  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LAYER = 21


class HiddenStateEncoder:
    """Last-token hidden state after ``layer`` transformer blocks of a frozen model.

    With ``chat_template`` the prompt is wrapped as a user turn followed by the
    marker that starts the assistant's reply, so the final position is where the
    model is about to answer. Prompts are cut to ``max_tokens`` tokens before
    wrapping, keeping the start. Layers above ``layer`` are dropped before the
    forward pass: they cannot affect the state that is read.
    """

    def __init__(self, model_name: str, revision: str, *, layer: int, max_tokens: int,
                 device: str = "cuda", chat_template: bool = True, cache_dir: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, cache_dir=cache_dir,
                                                       trust_remote_code=False)
        self.tokenizer.padding_side = "left"  # so the last real token is always at position -1
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir,
                                               torch_dtype=dtype, trust_remote_code=False).to(device).eval()
        blocks = self.model.layers
        if not 1 <= layer <= len(blocks):
            raise ValueError(f"layer must be between 1 and {len(blocks)} for this model")
        self.model.layers = torch.nn.ModuleList(blocks[:layer])
        self.layer, self.max_tokens, self.device, self.chat_template = layer, max_tokens, device, chat_template

    def _text(self, prompt: str) -> str:
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_tokens:
            prompt = self.tokenizer.decode(ids[: self.max_tokens])
        if not self.chat_template:
            return prompt
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)

    def encode(self, prompts, batch_size: int = 4, **_ignored):
        vectors = []
        for start in range(0, len(prompts), batch_size):
            batch = self.tokenizer([self._text(prompt) for prompt in prompts[start:start + batch_size]],
                                   return_tensors="pt", padding=True, add_special_tokens=not self.chat_template)
            batch = {name: value.to(self.device) for name, value in batch.items()}
            with self.torch.no_grad():
                states = self.model(**batch, output_hidden_states=True).hidden_states
            # hidden_states[-1] is the output of the last kept block, before the final norm.
            vectors.append(states[-1][:, -1, :].float().cpu().numpy())
        import numpy as np
        return np.concatenate(vectors, axis=0)


def main(argv=None, encoder=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=Path, help="canonical outcomes (prompts are read from it)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="full Hugging Face commit SHA")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="read the state after this many transformer blocks")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="prompt tokens kept, from the start")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="feed the bare prompt instead of a user turn awaiting a reply")
    parser.add_argument("--prompt-view", choices=sorted(embed.PROMPT_VIEWS), default="full",
                        help="issue: read only a prompt's <issue> block when it has one")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.layer < 1:
        raise SystemExit("--layer must be at least 1")
    tasks = embed.prepare_tasks(read_outcomes(args.input), args.limit)
    if encoder is None:
        encoder = HiddenStateEncoder(args.model, args.revision, layer=args.layer,
                                     max_tokens=args.max_seq_length, device=args.device,
                                     chat_template=not args.no_chat_template, cache_dir=args.cache_dir)
    return embed.embed_tasks(
        tasks, args.output, model_name=args.model, revision=args.revision, batch_size=args.batch_size,
        device=args.device, max_seq_length=args.max_seq_length, encoder=encoder,
        prompt_view=args.prompt_view,
        extra_metadata={"representation": "last_token_hidden_state", "layer": args.layer,
                        "chat_template": not args.no_chat_template})


if __name__ == "__main__":
    main()
