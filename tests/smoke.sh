#!/usr/bin/env bash
# End-to-end smoke test on synthetic logs (no GPU, no corpus, no network).
#
# Stages exercised: 05 build labels -> 06 train routers -> 07 evaluate ->
# 08 analysis/figures -> 09 report.  Retrieval/generation (01-04) are simulated by
# tests/make_synthetic.py because they need the 21M-passage DPR corpus and a GPU.
#
# Usage:  bash tests/smoke.sh [destination] [python]
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/tmp/opencode/pythia-smoke}"
PY="${2:-python3}"

# Optional interpreter-specific site-packages (useful on boxes where the system
# python is bare but a venv with the scientific stack exists).  Ignored elsewhere.
if [ -z "${PYTHONPATH:-}" ] && [ -d "${PYTHIA_SITE_PACKAGES:-/nonexistent}" ]; then
    export PYTHONPATH="$PYTHIA_SITE_PACKAGES"
fi

echo "== smoke test: $SRC -> $DEST (python: $PY) =="
rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$SRC" \
    --exclude=.git --exclude=results --exclude=figures --exclude=data \
    --exclude=session-ses_e1c1.md -cf - . | tar -C "$DEST" -xf -
cd "$DEST"

run() {
    echo "--- $*"
    "$@"
}

run "$PY" tests/make_synthetic.py --root . --n-label 240 --n-tune 90 --n-eval 150
run "$PY" scripts/05_build_labels.py --config configs/smoke.yaml
run "$PY" scripts/06_train_router.py --config configs/smoke.yaml --models logreg --tiers lex --seeds 7
run "$PY" scripts/07_run_eval.py --config configs/smoke.yaml --sig-top-k 8
run "$PY" scripts/08_analysis.py --config configs/smoke.yaml
run "$PY" scripts/09_report.py --config configs/smoke.yaml

echo "== artefacts =="
ls -1 results/tables | head -30
ls -1 figures | head -20
test -s docs/REPORT.md && echo "[ok] docs/REPORT.md"
echo "== smoke test passed =="
