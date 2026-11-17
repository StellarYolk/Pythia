#!/usr/bin/env python
"""Assemble docs/REPORT.md from the artefacts in results/ and figures/.

The report is generated, never hand-edited: every number in it comes from a CSV/JSON
written by stages 05-08.  Missing stages are reported honestly instead of being skipped
silently.

Usage:
    python scripts/09_report.py --config configs/default.yaml --run main
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import env_info, get_logger, git_sha, load_config, write_json  # noqa: E402

LOG = get_logger("report")

DATASETS = ["nq", "triviaqa", "hotpotqa"]


def md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, floatfmt: str = "{:.4f}", max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_not available_"
    df = df.copy()
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    df = df.head(max_rows)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def missing(label: str, stage: str) -> str:
    return f"_not available — run `{stage}`._ ({label})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", default="main")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Pythia: cost-aware adaptive retrieval routing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    res = root / "results"
    tabs = res / "tables"
    docs = root / cfg["paths"]["docs"]
    docs.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else docs / "REPORT.md"

    summary = read_csv(tabs / f"eval_summary_{args.run}.csv")
    per_ds = read_csv(tabs / f"eval_per_dataset_{args.run}.csv")
    succ = read_csv(tabs / f"success_criterion_{args.run}.csv")
    sig = read_csv(tabs / f"eval_significance_{args.run}.csv")
    budgets = read_csv(tabs / f"strategy_baselines_{args.run}.csv")
    label_grid = read_csv(res / "label_grid.csv")
    train_sum = read_csv(tabs / f"router_training_summary_{args.run}.csv")
    if train_sum is None:
        train_sum = read_csv(res / f"router_training_summary_{args.run}.csv")
    tier_sum = read_csv(res / f"router_training_summary_{args.run}_tiers.csv")
    err = read_csv(tabs / f"error_costs_{args.run}.csv")
    fail_counts = read_csv(tabs / f"failure_counts_{args.run}.csv")
    bd = read_csv(tabs / f"breakdowns_{args.run}.csv")
    lodo = read_csv(tabs / f"lodo_transfer_{args.run}.csv")
    analysis = read_json_file(tabs / f"analysis_summary_{args.run}.json")
    oracle = read_json_file(res / f"oracle_summary_{args.run}.json")
    eval_meta = read_json_file(res / f"eval_meta_{args.run}.json")

    L: List[str] = []
    L.append(f"# {args.title}\n")
    L.append(
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`results/` at git `{git_sha(root)}`; run = `{args.run}`._\n"
    )
    L.append(
        "> **Status.** This document is produced by `scripts/09_report.py`. Numbers are "
        "filled in only where the corresponding stage artefacts exist; missing stages are "
        "marked explicitly.\n"
    )

    # ---------------------------------------------------------------- summary
    L.append("## 1. Summary\n")
    if analysis:
        L.append(
            f"The primary router is **`{analysis['primary_system']}`**. On {analysis['n']} "
            f"official-dev queries it reaches **EM {analysis['em']:.4f}** / F1 {analysis['f1']:.4f} "
            f"at a retrieval-call rate of **{analysis['retrieval_call_rate']:.3f}** "
            f"(oracle over the four strategies: EM {analysis['oracle_em']:.4f}).\n"
        )
    else:
        L.append(missing("analysis summary", "scripts/08_analysis.py") + "\n")
    if eval_meta:
        L.append(
            f"Pre-registered success criterion: EM within **{eval_meta['success_margin_em']}** "
            f"absolute of the best always-retrieve baseline ({eval_meta['best_always']}) while "
            f"cutting retrieval calls by **{eval_meta['success_min_call_reduction'] * 100:.0f}%** "
            f"relative. Systems passing: **{eval_meta.get('n_success', 0)}** / "
            f"{len(succ) if succ is not None else 0}.\n"
        )

    # ---------------------------------------------------------------- setup
    L.append("## 2. Setup\n")
    L.append(
        f"- Generator: `{cfg['generation']['model']}` (fallback `{cfg['generation']['fallback_model']}`), "
        f"temperature {cfg['generation']['temperature']}, max {cfg['generation']['max_tokens']} new tokens.\n"
        f"- Retriever: BM25 + dense `{cfg['retrieval']['dense_model']}` (k={cfg['retrieval']['k']}), "
        f"hybrid fusion `{cfg['retrieval']['hybrid']['fusion']}`.\n"
        f"- Router models: {', '.join(cfg['router']['models'])}; seeds {cfg['router']['seeds']}; "
        f"feature tiers {cfg['router']['feature_tiers']}.\n"
        f"- Label rule: metric `{cfg['labels']['primary_metric']}`, epsilon {cfg['labels']['epsilon']}, "
        f"cost order {cfg['labels']['cost_order']}.\n"
    )
    sizes = cfg.get("sizes", {})
    L.append(
        f"- Sample sizes per dataset: label pool {sizes.get('label_pool')}, tune "
        f"{sizes.get('dev_from_train')}, eval {sizes.get('eval_dev')} (official dev split).\n"
    )

    # ---------------------------------------------------------------- main table
    L.append("## 3. Main results (official dev)\n")
    if summary is not None:
        keep = ["never_retrieve", "always_bm25", "always_dense", "always_hybrid", "oracle",
                "complexity_classifier"]
        cand = summary[summary["system"].isin(keep)]
        # best confidence threshold
        conf = summary[summary["system"].str.startswith("conf_thresh_")].sort_values("em", ascending=False)
        if len(conf):
            cand = pd.concat([cand, conf.head(1)], ignore_index=True)
        # best Pythia per variant (pooled / per_dataset / lodo), primary tiers
        for variant in ("pooled", "per_dataset", "lodo"):
            sub = summary[summary["system"].str.startswith(f"pythia_{variant}_")]
            if len(sub):
                cand = pd.concat([cand, sub.sort_values("em", ascending=False).head(1)], ignore_index=True)
        cols = ["system", "n", "em", "f1", "retrieval_call_rate", "mean_calls", "mean_latency_ms", "p95_latency_ms", "total_tokens"]
        L.append(md_table(cand.sort_values("em", ascending=False), cols) + "\n")
        L.append(f"Full table: `results/tables/eval_summary_{args.run}.csv` ({len(summary)} systems).\n")
    else:
        L.append(missing("evaluation summary", "scripts/07_run_eval.py") + "\n")

    if budgets is not None:
        L.append("### Per-strategy reference points\n")
        L.append(md_table(budgets, ["dataset", "subset", "strategy", "n", "em", "f1", "hit", "ctx_has_answer",
                                   "mean_latency_ms", "p95_latency_ms", "retrieval_calls"], max_rows=30) + "\n")

    # ---------------------------------------------------------------- success
    L.append("## 4. Pre-registered success criterion\n")
    if succ is not None:
        L.append(
            "A system passes if `em_gap >= -margin` **and** `call_reduction >= "
            f"{cfg['eval']['success_min_call_reduction']}`.\n"
        )
        L.append(md_table(succ, ["system", "variant", "model", "tiers", "seed", "lambda", "em",
                                 "best_always_em", "em_gap", "call_reduction", "latency_reduction", "success"],
                          max_rows=20) + "\n")
        L.append(f"Passing systems: **{int(succ['success'].sum())}** of {len(succ)}.\n")
    else:
        L.append(missing("success criterion", "scripts/07_run_eval.py") + "\n")

    # ---------------------------------------------------------------- significance
    L.append("## 5. Statistical comparison\n")
    if sig is not None:
        L.append(
            f"Paired bootstrap ({cfg['eval']['bootstrap_samples']} resamples) on per-query EM "
            f"difference vs the best always-retrieve baseline; Wilcoxon signed-rank on F1; "
            f"exact McNemar on EM outcomes.\n"
        )
        L.append(md_table(sig.sort_values("em_diff", ascending=False),
                          ["system", "n", "em_diff", "em_diff_lo", "em_diff_hi", "p_bootstrap",
                           "p_wilcoxon_f1", "p_mcnemar_em"], max_rows=20) + "\n")
    else:
        L.append(missing("significance table", "scripts/07_run_eval.py") + "\n")

    # ---------------------------------------------------------------- per dataset
    L.append("## 6. Per-dataset results\n")
    if per_ds is not None and summary is not None:
        systems = ["always_hybrid", "oracle"]
        for variant in ("pooled", "per_dataset", "lodo"):
            sub = summary[summary["system"].str.startswith(f"pythia_{variant}_")]
            if len(sub):
                systems.append(sub.sort_values("em", ascending=False)["system"].iloc[0])
        sel = per_ds[per_ds["system"].isin(systems)]
        L.append(md_table(sel, ["system", "dataset", "n", "em", "f1", "retrieval_call_rate",
                                "mean_latency_ms", "p95_latency_ms"], max_rows=30) + "\n")
    else:
        L.append(missing("per-dataset table", "scripts/07_run_eval.py") + "\n")

    # ---------------------------------------------------------------- LODO
    L.append("## 7. Cross-dataset transfer (leave-one-dataset-out)\n")
    if lodo is not None and len(lodo):
        L.append(md_table(lodo, ["system", "dataset", "n", "em", "f1", "retrieval_call_rate",
                                 "mean_latency_ms"], max_rows=30) + "\n")
    else:
        L.append(missing("LODO table (train routers with `--variants all`)", "scripts/06_train_router.py") + "\n")

    # ---------------------------------------------------------------- labels / oracle
    L.append("## 8. Labels and oracle\n")
    if oracle:
        rows = []
        for ds, d in oracle.items():
            rows.append({"dataset": ds, **d})
        L.append(md_table(pd.DataFrame(rows), ["dataset", "n", "oracle_em", "oracle_f1", "mean_label_calls",
                                               "mean_label_latency_ms", "retrieval_hurts_rate"], max_rows=10) + "\n")
    elif label_grid is not None:
        L.append(md_table(label_grid.head(20)) + "\n")
    else:
        L.append(missing("label/oracle statistics", "scripts/05_build_labels.py") + "\n")

    # ---------------------------------------------------------------- ablations
    L.append("## 9. Router ablations\n")
    if train_sum is not None:
        L.append(md_table(train_sum, ["variant", "heldout", "model", "seed", "tiers", "n_train", "n_tune",
                                      "tune_acc", "temperature", "seconds"], max_rows=40) + "\n")
    else:
        L.append(missing("router training summary", "scripts/06_train_router.py") + "\n")
    if tier_sum is not None:
        L.append("### Feature-tier ablation (pooled, logreg/gbt)\n")
        L.append(md_table(tier_sum, ["model", "seed", "tiers", "tune_acc", "temperature"], max_rows=40) + "\n")
    if label_grid is not None:
        L.append("### Label metric × epsilon grid\n")
        L.append(md_table(label_grid, ["metric", "epsilon", "dataset", "n", "oracle_em", "mean_calls",
                                        "retrieval_hurts"], max_rows=30) + "\n")

    # ---------------------------------------------------------------- errors / failures
    L.append("## 10. Error analysis\n")
    if err is not None:
        L.append("### Cost of routing errors\n")
        L.append(md_table(err, ["category", "n", "share", "lost_em_points", "mean_calls", "mean_latency_ms", "em"]) + "\n")
    if fail_counts is not None:
        L.append("### Failure taxonomy\n")
        piv = fail_counts.pivot_table(index="failure", columns="dataset", values="n", fill_value=0)
        piv = piv.reindex([c for c in [
            "router_skipped_long_tail_fact", "retrieval_returned_distractors",
            "multi_hop_needs_iterative_retrieval", "retrieval_hurt_over_retrieval",
            "answer_normalisation_artefact", "label_noise_em_strictness",
            "generator_hallucination", "ambiguous_or_wrong_gold",
        ] if c in piv.index])
        L.append(md_table(piv.reset_index(), max_rows=20) + "\n")
        L.append(
            f"Rule-based first pass over all failures; `results/tables/failures_{args.run}.csv` "
            f"contains a deterministic sample of up to {cfg['eval'].get('failure_sample_per_dataset', 50)} "
            "failures per dataset for manual verification (see `docs/failures.md`).\n"
        )
    if err is None and fail_counts is None:
        L.append(missing("error/failure tables", "scripts/08_analysis.py") + "\n")

    if bd is not None:
        L.append("### Breakdowns\n")
        L.append(
            "Long-format table with EM/F1/call rate by dataset, wh-word, answer type, question-length "
            f"quartile, hop type and gold-retrieval hit: `results/tables/breakdowns_{args.run}.csv`.\n"
        )

    # ---------------------------------------------------------------- figures
    L.append("## 11. Figures\n")
    fig_dir = root / cfg["paths"]["figures"]
    figs = sorted(fig_dir.glob(f"*_{args.run}.png"))
    if figs:
        for p in figs:
            rel = p.relative_to(root)
            L.append(f"![{p.stem}](../{rel})\n")
    else:
        L.append(missing("figures", "scripts/08_analysis.py") + "\n")

    # ---------------------------------------------------------------- limitations
    L.append("## 12. Limitations, deviations and reproduction\n")
    L.append(
        "- Retrieval is a single shot at k passages; multi-hop questions are not handled with "
        "iterative retrieval, so HotpotQA is expected to under-perform.\n"
        "- EM is strict and sensitive to answer formatting; F1 and the label grid are reported "
        "as secondary evidence.\n"
        "- The failure taxonomy is a rule-based first pass; `docs/failures.md` records the "
        "manual verification sample.\n"
        "- The router's cost knob uses measured mean latency per strategy; per-query latency "
        "variance is not modelled.\n"
        "- All deviations from `PLAN.md` are recorded in `docs/DEVIATIONS.md` and summarised in "
        "the report rather than silently changed.\n"
    )
    L.append(
        "Reproduce with:\n\n```bash\nmake data indices pilot label router eval analysis report\n```\n"
    )
    L.append("## Appendix: artefact index\n")
    for p in sorted(tabs.glob(f"*_{args.run}.csv")):
        L.append(f"- `{p.relative_to(root)}`\n")
    for name in ("cost_order.json", f"oracle_summary_{args.run}.json", f"eval_meta_{args.run}.json"):
        p = res / name
        if p.exists():
            L.append(f"- `{p.relative_to(root)}`\n")

    out.write_text("\n".join(L) + "\n")
    LOG.info("wrote %s", out)

    # run manifest (deliverable in PLAN.md §10)
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "run": args.run,
        "git_sha": git_sha(root),
        "env": env_info(),
        "config": cfg,
        "tables": sorted(str(p.relative_to(root)) for p in tabs.glob(f"*_{args.run}.csv")),
        "figures": sorted(str(p.relative_to(root)) for p in fig_dir.glob(f"*_{args.run}.png")),
        "report": str(out.relative_to(root)),
    }
    write_json(res / f"run_manifest_{args.run}.json", manifest)


if __name__ == "__main__":
    main()
