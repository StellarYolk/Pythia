"""Runs the four strategies over a set of queries and logs everything.

One record per (query, strategy).  All logs are written to parquet so that labelling,
evaluation and analysis never have to re-run the pipeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from common import get_logger, hash_text, normalize_answer
from generation.prompts import build_chat_prompt, build_context, build_user_message

LOG = get_logger("generation.runner")

STRATEGIES = ["none", "bm25", "dense", "hybrid"]
# cost order (cheapest first); verified against measured latency in Phase 2
COST_ORDER = ["none", "bm25", "dense", "hybrid"]


def context_contains_answer(context: str, golds: Sequence[str]) -> int:
    """Does the retrieved context contain a gold answer string? (answer recall proxy)"""
    if not context:
        return 0
    ctx = " " + normalize_answer(context) + " "
    for g in golds:
        ng = normalize_answer(g)
        if ng and f" {ng} " in ctx:
            return 1
    return 0


class StrategyRunner:
    def __init__(self, cfg: Dict[str, Any], engine, retrievers: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.engine = engine
        self.retrievers = retrievers

    # ------------------------------------------------------------------
    def build_prompt(self, question: str, passages: Optional[Sequence[Dict[str, Any]]]):
        g = self.cfg["generation"]
        if passages:
            ctx = build_context(passages, g["max_context_tokens"], self.engine.tokenizer)
        else:
            ctx = {"context": "", "passages": [], "n_passages": 0, "approx_ctx_tokens": 0}
        user = build_user_message(question, ctx["context"] or None)
        prompt = build_chat_prompt(self.engine.tokenizer, self.cfg["prompt"]["system"], user)
        return prompt, ctx

    # ------------------------------------------------------------------
    def run_strategy(
        self,
        queries: List[Dict[str, Any]],
        strategy: str,
        k: int = 5,
        dataset: str = "",
        subset: str = "",
        retrieval_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Run one strategy over all queries."""
        questions = [q["question"] for q in queries]
        passages_per_q: List[List[Dict[str, Any]]] = [[] for _ in queries]
        retr_latency = [0.0] * len(queries)
        n_calls = [0] * len(queries)

        if strategy != "none":
            if retrieval_cache is not None and all(q["qid"] in retrieval_cache for q in queries):
                LOG.info("%s: using cached retrieval for %d queries", strategy, len(queries))
                for i, q in enumerate(queries):
                    c = retrieval_cache[q["qid"]]
                    passages_per_q[i] = c["passages"]
                    retr_latency[i] = c["latency_ms"]
                    n_calls[i] = c["n_calls"]
            else:
                t0 = time.perf_counter()
                res = self.retrievers[strategy].retrieve(questions, k=k)
                batch_ms = (time.perf_counter() - t0) * 1000.0
                for i, r in enumerate(res):
                    passages_per_q[i] = r.passages
                    retr_latency[i] = r.latency_ms
                    n_calls[i] = r.n_calls
                LOG.info(
                    "%s: retrieval for %d queries in %.1f s (%.1f ms/query)",
                    strategy,
                    len(queries),
                    batch_ms / 1000,
                    batch_ms / max(1, len(queries)),
                )

        prompts = []
        ctx_infos = []
        for q, ps in zip(queries, passages_per_q):
            prompt, ctx = self.build_prompt(q["question"], ps)
            prompts.append(prompt)
            ctx_infos.append(ctx)

        t0 = time.perf_counter()
        outs = self.engine.generate(prompts)
        gen_batch_ms = (time.perf_counter() - t0) * 1000.0

        rows = []
        for i, q in enumerate(queries):
            o = outs[i]
            golds = q["answers"]
            gold_titles = set(t.lower() for t in q.get("gold_titles", []) or [])
            retrieved = [p["pid"] for p in passages_per_q[i]]
            retrieved_titles = [p["title"] for p in passages_per_q[i]]
            hit = int(bool(gold_titles & set(t.lower() for t in retrieved_titles))) if gold_titles else -1
            ctx = ctx_infos[i]
            gen_latency = o.get("latency_ms")
            if gen_latency is None:
                gen_latency = gen_batch_ms / max(1, len(queries))
            row = {
                "qid": q["qid"],
                "dataset": dataset or q.get("dataset", ""),
                "subset": subset or q.get("subset", ""),
                "strategy": strategy,
                "k": k if strategy != "none" else 0,
                "question": q["question"],
                "answers": json.dumps(golds),
                "pred": o["text"],
                "retrieved_ids": json.dumps(retrieved),
                "retrieved_titles": json.dumps(retrieved_titles),
                "retrieved_scores": json.dumps([float(p["score"]) for p in passages_per_q[i]]),
                "n_gold_articles": len(gold_titles),
                "hit": hit,
                "ctx_has_answer": context_contains_answer(ctx["context"], golds),
                "n_passages": ctx["n_passages"],
                "prompt_tokens": o["prompt_tokens"],
                "completion_tokens": o["completion_tokens"],
                "ctx_tokens": ctx["approx_ctx_tokens"],
                "retrieval_latency_ms": retr_latency[i],
                "generation_latency_ms": gen_latency,
                "total_latency_ms": retr_latency[i] + gen_latency,
                "n_retrieval_calls": n_calls[i],
                "mean_logprob": o["mean_logprob"],
                "first_token_entropy": o["first_token_entropy"],
                "mean_entropy": o["mean_entropy"],
                "first_token_margin": o["first_token_margin"],
                "prompt_hash": hash_text(prompts[i]),
                "model": self.engine.model_name,
            }
            row.update(_score_row(row["pred"], golds))
            rows.append(row)

        df = pd.DataFrame(rows)
        LOG.info(
            "%s: EM=%.4f F1=%.4f hit=%.4f ctx_has_answer=%.4f mean_prompt_tok=%.0f mean_latency=%.0f ms",
            strategy,
            df["em"].mean(),
            df["f1"].mean(),
            df["hit"].mean(),
            df["ctx_has_answer"].mean(),
            df["prompt_tokens"].mean(),
            df["total_latency_ms"].mean(),
        )
        return df


def _score_row(pred: str, golds: Sequence[str]) -> Dict[str, float]:
    from common import exact_match, token_f1

    return {"em": exact_match(pred, golds), "f1": token_f1(pred, golds)}


def save_log(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_log(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def logs_complete(path: str | Path, qids: Iterable[str]) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["qid"])
    except Exception:
        return False
    return set(df["qid"]) >= set(qids)
