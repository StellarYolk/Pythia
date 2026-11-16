"""vLLM generation engine (deterministic, temperature 0) with token/latency accounting."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

from common import get_logger

# vLLM spawns its engine-core worker; with the default "fork" start method that fails as
# soon as CUDA has been touched in the parent process ("Cannot re-initialize CUDA in
# forked subprocess").  Must be set before vllm is imported.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOG = get_logger("generation.engine")


class VLLMEngine:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        g = cfg["generation"]
        self.model_name = g["model"]
        self.tokenizer = None
        self.llm = None
        self._load_error = None
        self._init_engine()

    # ------------------------------------------------------------------
    def _init_engine(self) -> None:
        g = self.cfg["generation"]
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            LOG.info("loading vLLM engine: %s", self.model_name)
            self.llm = LLM(
                model=self.model_name,
                tokenizer=g.get("tokenizer") or self.model_name,
                dtype=g.get("dtype", "auto"),
                quantization=g.get("quantization"),
                max_model_len=g["max_model_len"],
                gpu_memory_utilization=g["gpu_memory_utilization"],
                max_num_seqs=g["max_num_seqs"],
                enforce_eager=g.get("enforce_eager", False),
                seed=g.get("seed", 0),
                trust_remote_code=g.get("trust_remote_code", False),
                disable_log_stats=True,
            )
            self.tokenizer = self.llm.get_tokenizer()
        except Exception as exc:  # pragma: no cover - fallback path
            LOG.error("vLLM init failed for %s: %s", self.model_name, exc)
            self._load_error = exc
            fallback = g.get("fallback_model")
            if not fallback:
                raise
            LOG.warning("falling back to %s", fallback)
            self.model_name = fallback
            self.llm = LLM(
                model=fallback,
                dtype="auto",
                max_model_len=g["max_model_len"],
                gpu_memory_utilization=g["gpu_memory_utilization"],
                max_num_seqs=g["max_num_seqs"],
                seed=g.get("seed", 0),
                disable_log_stats=True,
            )
            self.tokenizer = self.llm.get_tokenizer()

    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompts: Sequence[str], stop: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate for a list of chat-formatted prompts.

        Returns one dict per prompt with: text, prompt_tokens, completion_tokens,
        mean logprob, first-token entropy and first-token margin.
        """
        from vllm import SamplingParams

        g = self.cfg["generation"]
        sp = SamplingParams(
            temperature=g["temperature"],
            top_p=g["top_p"],
            max_tokens=g["max_tokens"],
            seed=g.get("seed", 0),
            logprobs=5,
            stop=stop if stop is not None else (g.get("stop") or None),
        )
        outs = self.llm.generate(list(prompts), sp)
        results = []
        for o in outs:
            comp = o.outputs[0]
            lp = comp.logprobs or []
            mean_lp, entropies, first_margin = _logprob_stats(lp)
            results.append(
                {
                    "text": comp.text.strip(),
                    "prompt_tokens": len(o.prompt_token_ids),
                    "completion_tokens": len(comp.token_ids),
                    "mean_logprob": mean_lp,
                    "first_token_entropy": entropies[0] if entropies else float("nan"),
                    "mean_entropy": sum(entropies) / len(entropies) if entropies else float("nan"),
                    "first_token_margin": first_margin,
                }
            )
        return results


def _logprob_stats(lp_list):
    """Mean logprob of the generated tokens, per-token entropy and first-token margin."""
    mean_lp = float("nan")
    entropies = []
    first_margin = float("nan")
    lps = []
    for step in lp_list:
        if not step:
            continue
        vals = []
        for tid, lp in step.items():
            vals.append((float(lp.logprob), tid))
        vals.sort(reverse=True)
        lps.append(vals[0][0])
        probs = [math.exp(v) for v, _ in vals]
        total = sum(probs)
        ent = -sum(p * math.log(max(p / total, 1e-12)) for p in probs)
        entropies.append(ent)
        if len(vals) > 1 and first_margin != first_margin:
            first_margin = vals[0][0] - vals[1][0]
    if lps:
        mean_lp = sum(lps) / len(lps)
    return mean_lp, entropies, first_margin
