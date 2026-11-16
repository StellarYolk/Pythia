#!/usr/bin/env python
"""Smoke-test the vLLM engine: model loads, generation works, throughput measured."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_logger, load_config, set_seed  # noqa: E402

LOG = get_logger("smoke")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--prompts", default="smoke")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["generation"]["model"] = args.model
    set_seed(cfg["seed"])

    from generation.engine import VLLMEngine

    t0 = time.time()
    engine = VLLMEngine(cfg)
    LOG.info("engine ready in %.1f s (model=%s)", time.time() - t0, engine.model_name)

    qa = [
        ("Context:\nBarack Obama\nBarack Hussein Obama II is an American politician who served as the 44th president of the United States.\n\nQuestion: Who was the 44th president of the United States?\nAnswer:", "Barack Obama"),
        ("Question: What is the capital of France?\nAnswer:", "Paris"),
    ]
    prompts = [p for p, _ in qa]
    outs = engine.generate(prompts)
    for (p, gold), o in zip(qa, outs):
        LOG.info("pred=%r gold=%r tokens=%d mean_lp=%.3f ent=%.3f", o["text"], gold, o["completion_tokens"], o["mean_logprob"], o["first_token_entropy"])

    # throughput probe with realistic prompt lengths
    long_ctx = "Context:\n" + ("Wikipedia passage. " * 120) + "\n\n"
    prompts2 = [long_ctx + f"Question: what is fact number {i}?\nAnswer:" for i in range(args.n)]
    engine.generate(prompts2[:2])
    t0 = time.time()
    outs2 = engine.generate(prompts2)
    dt = time.time() - t0
    n_prompt = sum(o["prompt_tokens"] for o in outs2)
    n_comp = sum(o["completion_tokens"] for o in outs2)
    LOG.info("throughput: %d prompts in %.2fs -> %.0f prompt-tok/s, %.0f comp-tok/s", len(prompts2), dt, n_prompt / dt, n_comp / dt)
    LOG.info("per-query latency (amortised): %.1f ms", dt / len(prompts2) * 1000)


if __name__ == "__main__":
    main()
