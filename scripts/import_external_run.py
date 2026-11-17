#!/usr/bin/env python
"""Import an external Pythia run (real logs) into this checkout.

The importer is a *provenance-preserving bridge*, not a pipeline stage.  It is
used when the expensive part (corpus, retrieval indices, generation) was run in a
different checkout and only the per-query logs were copied over.  It never writes
to the source tree.

Expected source layout (the original Pythia checkout):

    data/splits/{train,dev,test}.jsonl
    results/labels/{train,dev,test}/all_strategy_logs.parquet

The importer maps ``train -> label``, ``dev -> tune`` and ``test -> eval``,
rebuilds ``data/processed/queries/<dataset>__<subset>.jsonl`` and rewrites one
parquet per ``(dataset, subset, strategy)`` in the schema written by
``generation/runner.py``.  Fields the external run did not record are derived or
marked explicitly (see the provenance JSON it writes).

Usage:
    python scripts/import_external_run.py \
        --source /path/to/external/Pythia \
        --root . \
        --run imported_qwen1.5b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import get_logger, read_jsonl, write_json  # noqa: E402

LOG = get_logger("import_external_run")

DATASETS = ("nq", "triviaqa", "hotpotqa")
STRATEGIES = ("none", "bm25", "dense", "hybrid")
SPLIT_MAP = {"train": "label", "dev": "tune", "test": "eval"}


# --------------------------------------------------------------------------------------
# small coercion helpers (parquet stores lists as numpy arrays / JSON strings)
# --------------------------------------------------------------------------------------
def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = eval(value, {"__builtins__": {}}, {})  # noqa: S307 - trusted local logs
            except Exception:  # noqa: BLE001
                return []
        return parsed if isinstance(parsed, list) else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def as_dicts(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in as_list(value):
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------------------
def convert_queries(rows: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    out = []
    for q in rows:
        out.append(
            {
                "qid": q["qid"],
                "dataset": q["dataset"],
                "subset": subset,
                "question": q.get("question", ""),
                "answers": list(q.get("answers") or []),
                "gold_passage_ids": [str(x) for x in as_list(q.get("gold_passage_ids"))],
                "meta": q.get("meta") or {},
            }
        )
    return out


def convert_logs(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Rewrite an external log frame into generation/runner.py's schema.

    Derived / unavailable fields:
      * ``hit``: 1 when a gold passage id is among the retrieved passage ids
        (the external checkout stored pids, not titles), 0 otherwise, -1 for
        the no-retrieval strategy.
      * ``ctx_has_answer``: a proxy equal to ``hit`` for retrieval strategies and
        0 for ``none``; the external logs did not keep the context text.
      * ``ctx_tokens``: -1 (not recorded).
      * ``first_token_entropy`` / ``mean_entropy``: the external ``entropy``
        column; ``first_token_margin``: 0.0 (not recorded).
    """
    rows: List[Dict[str, Any]] = []
    for r in df.itertuples():
        strategy = str(getattr(r, "strategy"))
        gold = {str(x) for x in as_list(getattr(r, "gold_passage_ids"))}
        recs = as_dicts(getattr(r, "retrieved"))
        pids = [str(x.get("pid", "")) for x in recs if x.get("pid") is not None]
        scores = [float(x.get("score", 0.0)) for x in recs]
        hit = -1 if strategy == "none" else int(bool(gold & set(pids)))
        entropy = float(getattr(r, "entropy", 0.0) or 0.0)
        prompt_tokens = int(float(getattr(r, "n_prompt_tokens", 0) or 0))
        completion_tokens = int(float(getattr(r, "n_gen_tokens", 0) or 0))
        rows.append(
            {
                "qid": str(getattr(r, "qid")),
                "dataset": str(getattr(r, "dataset")),
                "subset": subset,
                "strategy": strategy,
                "k": int(float(getattr(r, "k", 5) or 5)) if strategy != "none" else 0,
                "question": str(getattr(r, "question")),
                "answers": json.dumps(as_list(getattr(r, "answers"))),
                "pred": str(getattr(r, "pred", "")),
                "retrieved_ids": json.dumps(pids),
                "retrieved_titles": json.dumps([]),
                "retrieved_scores": json.dumps(scores),
                "n_gold_articles": len(gold),
                "hit": hit,
                "ctx_has_answer": int(hit == 1),
                "n_passages": len(pids),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ctx_tokens": -1,
                "retrieval_latency_ms": float(getattr(r, "retrieval_latency_s", 0.0) or 0.0) * 1000.0,
                "generation_latency_ms": float(getattr(r, "generation_latency_s", 0.0) or 0.0) * 1000.0,
                "total_latency_ms": float(getattr(r, "total_latency_s", 0.0) or 0.0) * 1000.0,
                "n_retrieval_calls": int(float(getattr(r, "retrieval_calls", 0) or 0)),
                "mean_logprob": float(getattr(r, "mean_logprob", 0.0) or 0.0),
                "first_token_entropy": entropy,
                "mean_entropy": entropy,
                "first_token_margin": 0.0,
                "prompt_hash": str(getattr(r, "prompt_template_hash", "")),
                "model": str(getattr(r, "generator", "")),
                "em": float(getattr(r, "em", 0.0) or 0.0),
                "f1": float(getattr(r, "f1", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="external Pythia checkout with real logs")
    ap.add_argument("--root", default=".", help="destination repository root")
    ap.add_argument("--run", default="imported_qwen1.5b", help="destination run name")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    out_queries = root / "data" / "processed" / "queries"
    out_logs = root / "results" / "logs" / args.run
    if (out_logs.exists() and any(out_logs.iterdir())) and not args.overwrite:
        raise SystemExit(f"{out_logs} already exists - pass --overwrite to replace it")
    out_queries.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    provenance: Dict[str, Any] = {
        "run": args.run,
        "kind": "imported_external_run",
        "source": str(src),
        "split_map": SPLIT_MAP,
        "derived_fields": {
            "hit": "pid-level: gold_passage_ids intersect retrieved[].pid (-1 for none)",
            "ctx_has_answer": "proxy: hit for retrieval strategies, 0 for none (context text was not stored)",
            "ctx_tokens": "-1 (not recorded)",
            "first_token_entropy": "external entropy column",
            "mean_entropy": "external entropy column",
            "first_token_margin": "0.0 (not recorded)",
            "retrieved_titles": "[] (external logs store pids and scores only)",
        },
        "source_files": {},
        "splits": {},
    }

    for ext_split, subset in SPLIT_MAP.items():
        qpath = src / "data" / "splits" / f"{ext_split}.jsonl"
        lpath = src / "results" / "labels" / ext_split / "all_strategy_logs.parquet"
        if not qpath.exists() or not lpath.exists():
            raise SystemExit(f"missing source artefact: {qpath} or {lpath}")
        provenance["source_files"][str(qpath.relative_to(src))] = sha256_file(qpath)
        provenance["source_files"][str(lpath.relative_to(src))] = sha256_file(lpath)

        queries = read_jsonl(qpath)
        raw = pd.read_parquet(lpath)
        provenance["splits"][ext_split] = {
            "destination_subset": subset,
            "queries": len(queries),
            "log_rows": int(len(raw)),
            "generators": sorted({str(x) for x in raw.get("generator", pd.Series(dtype=str)).unique()}),
            "config_sha256": sorted({str(x) for x in raw.get("config_sha256", pd.Series(dtype=str)).unique()}),
            "prompt_template_hash": sorted({str(x) for x in raw.get("prompt_template_hash", pd.Series(dtype=str)).unique()}),
            "k": sorted({int(x) for x in raw.get("k", pd.Series(dtype=int)).unique()}),
        }

        for ds in DATASETS:
            qrows = convert_queries([q for q in queries if q["dataset"] == ds], subset)
            if not qrows:
                continue
            with open(out_queries / f"{ds}__{subset}.jsonl", "w") as fh:
                for q in qrows:
                    fh.write(json.dumps(q) + "\n")

            logs = convert_logs(raw[raw["dataset"] == ds], subset)
            qid_set = {q["qid"] for q in qrows}
            missing = qid_set - set(logs["qid"])
            extra = set(logs["qid"]) - qid_set
            if missing or extra:
                raise SystemExit(f"{ds}/{subset}: query/log mismatch ({len(missing)} missing, {len(extra)} extra)")
            for strategy in STRATEGIES:
                part = logs[logs["strategy"] == strategy]
                if part.empty:
                    raise SystemExit(f"{ds}/{subset}/{strategy}: no rows in the external logs")
                part.to_parquet(out_logs / f"{ds}__{subset}__{strategy}__k5.parquet", index=False)
            bm25 = logs[logs.strategy == "bm25"]
            LOG.info(
                "%s/%s: %d queries, hit(bm25)=%.3f, ctx_has_answer(bm25)=%.3f",
                ds,
                subset,
                len(qrows),
                float((bm25["hit"] == 1).mean()),
                float(bm25["ctx_has_answer"].mean()),
            )

    provenance["caveats"] = [
        "This is an imported secondary run, not the pre-registered 7B experiment in PLAN.md.",
        "Retrieval used the external checkout's curated corpus, not the 1M-passage DPR sample.",
        "hit is passage-id level; the original runner's hit is title level.",
        "ctx_has_answer is a proxy derived from hit because the context text was not stored.",
    ]
    write_json(root / "results" / f"imported_run_{args.run}.json", provenance)
    LOG.info("wrote queries to %s and logs to %s", out_queries, out_logs)
    LOG.info("next: python scripts/05_build_labels.py --config configs/imported.yaml --run %s", args.run)


if __name__ == "__main__":
    main()
