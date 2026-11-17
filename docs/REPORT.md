# Pythia: cost-aware adaptive retrieval routing

_This is a placeholder. The report is generated, never hand-edited._

```bash
python scripts/09_report.py --config configs/default.yaml --run main
```

The script reads `results/tables/*.csv`, `results/*.json`, `results/logs/` and
`figures/*.png` and writes this file. Missing stages are marked explicitly; it will not
invent numbers.

Run the full pipeline first:

```bash
make data indices gen label router tiers eval analysis report
```

For the current code status and the list of experiments that still need the GPU/data
machine, see `docs/STATUS.md`.
