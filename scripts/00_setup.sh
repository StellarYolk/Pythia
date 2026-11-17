#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 0 setup: external data root, symlinks, gitignore, resource guards.
# Idempotent - safe to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${PYTHIA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${PYTHIA_DATA_ROOT:-$HOME/pythia-data}"
VENV="${PYTHIA_VENV:-$HOME/venvs/pythia}"

echo "== Pythia setup =="
echo "repo      : $REPO"
echo "data root : $DATA_ROOT"
echo "venv      : $VENV"

mkdir -p "$DATA_ROOT"/{raw,processed,indices,logs}

cd "$REPO"
for sub in raw processed indices; do
    if [ ! -L "data/$sub" ]; then
        [ -e "data/$sub" ] && mv "data/$sub" "$DATA_ROOT/$sub" || mkdir -p "$DATA_ROOT/$sub"
        rm -rf "data/$sub"
        ln -sfn "$DATA_ROOT/$sub" "data/$sub"
    fi
done
echo "[ok] data/ symlinked to $DATA_ROOT"

# keep every log out of /tmp (tmpfs is wiped on reboot)
echo "[ok] logs -> $DATA_ROOT/logs"

# git hygiene
git rev-parse --git-dir >/dev/null 2>&1 && git config core.excludesfile "$REPO/.gitignore" || true

mkdir -p results/logs results/tables figures docs
echo "[ok] directory skeleton present"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[warn] venv missing at $VENV - run scripts/00_install.sh" >&2
fi

echo
echo "== free space =="
df -h "$REPO" | tail -1
du -sh "$DATA_ROOT" 2>/dev/null || true
echo
echo "Setup done. Next: make doctor"
