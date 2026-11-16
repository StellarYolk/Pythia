#!/usr/bin/env python
"""Prompt selection for the generator.

The brief requires an identical prompt across strategies, so the prompt is a shared
hyper-parameter.  This script compares a small set of candidate prompts on *no-context*
questions (parametric knowledge only) and reports EM / F1 so we can pick one before the
expensive full runs.  Selection uses TRAIN split questions only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import exact_match, get_logger, load_config, set_seed, token_f1  # noqa: E402

LOG = get_logger("prompt_select")

VARIANTS = {
    "v1_minimal": dict(
        system="You are a helpful assistant that answers questions. Answer with the shortest possible span: just the answer, no explanation.",
        user="Question: {q}\nAnswer:",
        stop=["\n"],
    ),
    "v2_terse": dict(
        system="You are a precise question answering system. Respond with only the answer entity or span. Do not write a sentence.",
        user="Q: {q}\nA:",
        stop=["\n"],
    ),
    "v3_rag": dict(
        system="You are a question answering system. Use the context when it is relevant, otherwise rely on your own knowledge. Reply with only the answer (a name, date, number or short phrase). No explanation, no full sentence.",
        user="{ctx}Question: {q}\nAnswer:",
        stop=["\n"],
    ),
    "v4_nostop": dict(
        system="You are a helpful assistant that answers questions. Answer with the shortest possible span: just the answer, no explanation.",
        user="Question: {q}\nAnswer:",
        stop=["\n\n", "Question:"],
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/tables/prompt_selection.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], init_cuda=False)

    from data.datasets_io import load_all

    data = load_all(cfg)
    rng = np.random.default_rng(cfg["seed"])
    samples = []
    for ds in ("nq", "triviaqa"):
        rows = data[ds]["train"]
        idx = rng.choice(len(rows), size=args.n, replace=False)
        for i in idx:
            samples.append({"dataset": ds, "question": rows[int(i)]["question"], "answers": rows[int(i)]["answers"]})
    LOG.info("prompt selection on %d no-context train questions", len(samples))

    from generation.engine import VLLMEngine
    from generation.prompts import build_chat_prompt

    engine = VLLMEngine(cfg)

    rows = []
    for name, v in VARIANTS.items():
        prompts = []
        for s in samples:
            user = v["user"].format(q=s["question"], ctx="")
            prompts.append(build_chat_prompt(engine.tokenizer, v["system"], user))
        outs = engine.generate(prompts, stop=v["stop"])
        em = np.mean([exact_match(o["text"], s["answers"]) for o, s in zip(outs, samples)])
        f1 = np.mean([token_f1(o["text"], s["answers"]) for o, s in zip(outs, samples)])
        avg_len = np.mean([o["completion_tokens"] for o in outs])
        rows.append({"variant": name, "em": em, "f1": f1, "avg_completion_tokens": avg_len,
                     "example": outs[0]["text"][:120]})
        LOG.info("%-12s EM=%.4f F1=%.4f avg_tok=%.1f ex=%r", name, em, f1, avg_len, outs[0]["text"][:100])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    with open(out.with_suffix(".variants.json"), "w") as fh:
        json.dump(VARIANTS, fh, indent=2)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
