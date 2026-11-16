#!/usr/bin/env python
"""Phase 0/1a - prepare data: query splits, gold mapping, corpus subsample.

Usage:
    python scripts/01_prepare_data.py --config configs/default.yaml [--pilot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_logger, load_config, set_seed, write_json, write_jsonl  # noqa: E402
from data.corpus import build_corpus  # noqa: E402
from data.datasets_io import build_query_splits, load_all  # noqa: E402
from data.gold import hotpot_gold_titles, map_positive_titles, map_positives  # noqa: E402

LOG = get_logger("prepare_data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pilot", action="store_true", help="use the pilot config (500 q/dataset)")
    ap.add_argument("--skip-corpus", action="store_true")
    ap.add_argument("--skip-gold", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.pilot:
        cfg["sizes"]["label_pool"] = cfg["sizes"]["pilot"]
        cfg["sizes"]["dev_from_train"] = max(1, cfg["sizes"]["pilot"] // 5)
        cfg["sizes"]["eval_dev"] = cfg["sizes"]["pilot"]
    set_seed(cfg["seed"])

    root = Path(cfg["paths"]["root"])
    proc = root / cfg["paths"]["data_processed"]
    proc.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- queries
    LOG.info("loading datasets ...")
    data = load_all(cfg)
    oversample = cfg["sizes"].get("oversample", {})
    splits = build_query_splits(cfg, data, oversample=oversample)

    all_queries: List[Dict[str, Any]] = []
    for ds, subs in splits.items():
        for sub, rows in subs.items():
            all_queries.extend(rows)
    LOG.info("%d query candidates", len(all_queries))

    # ---------------------------------------------------------------- gold articles
    gold_titles: Dict[str, List[str]] = {}
    dev_gold_path = proc / "gold_dev_titles.json"
    dev_gold = {}
    if dev_gold_path.exists():
        dev_gold = json.loads(dev_gold_path.read_text())
        LOG.info("loaded dev gold articles for %s", {k: len(v) for k, v in dev_gold.items()})
    else:
        LOG.warning("no %s - the official dev splits will have no gold articles. Run scripts/01b_extract_dev_gold.py", dev_gold_path)

    if not args.skip_gold:
        cache_dir = root / cfg["paths"]["data_raw"] / "tevatron"
        for ds in ("nq", "triviaqa"):
            for sub, src_split in (("label", "train"), ("tune", "train"), ("eval", "dev")):
                qs = [q for q in all_queries if q["dataset"] == ds and q["subset"] == sub]
                if not qs:
                    continue
                if sub == "eval":
                    # the DPR test files carry no passages -> use the dataset's own annotations
                    from data.gold import loose_normalize

                    table = dev_gold.get(ds, {})
                    t = {}
                    for q in qs:
                        titles = table.get(loose_normalize(q["question"]))
                        if titles:
                            t[q["qid"]] = list(titles)
                else:
                    t = map_positive_titles(ds, src_split, qs, cache_dir)
                gold_titles.update(t)
                LOG.info("%s/%s: %d/%d queries with gold articles", ds, sub, len(t), len(qs))
        for sub in ("label", "tune", "eval"):
            qs = [q for q in all_queries if q["dataset"] == "hotpotqa" and q["subset"] == sub]
            if qs:
                gold_titles.update({q["qid"]: list(q["meta"].get("gold_titles", [])) for q in qs})
                LOG.info("hotpotqa/%s: %d queries with gold articles", sub, len(qs))

        # ------------------------------------------------ fallback: answer -> article title
        missing = [q for q in all_queries if not gold_titles.get(q["qid"])]
        if missing:
            from data.answer_gold import build_answer_title_index, titles_for_answers

            answers = []
            for q in missing:
                answers.extend(q.get("answers", []) or [])
            idx_path = proc / "answer_title_index.json"
            if idx_path.exists():
                index = json.loads(idx_path.read_text())
                LOG.info("loaded answer->title index (%d entries)", len(index))
            else:
                index = build_answer_title_index(root / cfg["corpus"]["dpr_file"], answers, idx_path)
            n_fixed = 0
            for q in missing:
                titles = titles_for_answers(q.get("answers", []) or [], index)
                if titles:
                    gold_titles[q["qid"]] = titles
                    n_fixed += 1
            LOG.info("answer->title fallback resolved %d/%d queries", n_fixed, len(missing))
            for ds in ("nq", "triviaqa", "hotpotqa"):
                for sub in ("label", "tune", "eval"):
                    sub_qs = [q for q in all_queries if q["dataset"] == ds and q["subset"] == sub]
                    if sub_qs:
                        have = sum(1 for q in sub_qs if gold_titles.get(q["qid"]))
                        LOG.info("  %s/%s: %d/%d with gold articles", ds, sub, have, len(sub_qs))

    # ---------------------------------------------------------------- corpus
    corpus_path = proc / "corpus.jsonl"
    if not args.skip_corpus:
        all_titles: Set[str] = set()
        for ts in gold_titles.values():
            all_titles.update(ts)
        LOG.info("building corpus: %d distinct gold articles", len(all_titles))
        meta = build_corpus(
            dpr_path=root / cfg["corpus"]["dpr_file"],
            out_path=corpus_path,
            gold_titles=all_titles,
            n_distractors=cfg["corpus"]["n_distractors"],
            seed=cfg["seed"],
            max_passage_words=cfg["corpus"]["max_passage_words"],
        )
        write_json(proc / "corpus_meta.json", meta)

    # ---------------------------------------------------------------- write queries
    # Drop the candidates without a gold article (DPR annotations cover only part of the
    # NQ-open train split) and truncate back to the requested sizes.
    targets = {
        "label": cfg["sizes"]["label_pool"],
        "tune": cfg["sizes"]["dev_from_train"],
        "eval": cfg["sizes"]["eval_dev"],
    }
    kept: List[Dict[str, Any]] = []
    dropped = 0
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for q in all_queries:
        titles = gold_titles.get(q["qid"], [])
        q["gold_titles"] = titles
        q["n_gold_articles"] = len(titles)
        q["has_gold"] = bool(titles)
        by_key.setdefault(f"{q['dataset']}__{q['subset']}", []).append(q)
    for key, rows in by_key.items():
        ds, sub = key.split("__")
        want = targets[sub]
        usable = [q for q in rows if q["has_gold"]]
        dropped += len(rows) - len(usable)
        kept.extend(usable[:want])
        LOG.info("%-22s usable=%d kept=%d (target %d)", key, len(usable), min(len(usable), want), want)

    coverage = {"total": len(kept), "with_gold": len(kept), "without_gold": 0, "dropped_no_gold": dropped, "by_dataset": {}}
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for q in kept:
        d = coverage["by_dataset"].setdefault(q["dataset"], {"total": 0, "with_gold": 0, "by_subset": {}})
        d["total"] += 1
        sd = d["by_subset"].setdefault(q["subset"], {"total": 0, "with_gold": 0})
        sd["total"] += 1
        d["with_gold"] += 1
        sd["with_gold"] += 1
        groups.setdefault(f"{q['dataset']}__{q['subset']}", []).append(q)

    for name, rows in groups.items():
        write_jsonl(proc / "queries" / f"{name}.jsonl", rows)
        LOG.info("wrote %s (%d queries)", name, len(rows))

    write_json(root / cfg["paths"]["results"] / "corpus_coverage.json", coverage)
    LOG.info("gold coverage: %d/%d kept queries (%d candidates dropped)", len(kept), len(kept), dropped)
    for ds, d in coverage["by_dataset"].items():
        LOG.info("  %-10s %d queries", ds, d["total"])


if __name__ == "__main__":
    main()
