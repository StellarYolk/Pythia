#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Resource-limited launcher for every heavy Pythia stage.
#
# Rationale (see docs/DEVIATIONS.md): the original box is a hybrid-graphics
# laptop whose display shares system RAM, so saturating CPU/RAM can wedge the
# display driver.  Every stage is therefore pinned to 8 cores, niced, and
# capped at 12 GB when a systemd user session is available.  Override with
# ``PYTHIA_CORES`` / ``PYTHIA_MEM_LIMIT``; set ``PYTHIA_VENV`` to use a venv.
# ---------------------------------------------------------------------------
set -uo pipefail

NAME="${1:-stage}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PYTHIA_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${PYTHIA_DATA_ROOT:-$HOME/pythia-data}"
VENV="${PYTHIA_VENV:-$HOME/venvs/pythia}"
LOGDIR="${PYTHIA_LOGDIR:-$DATA_ROOT/logs}"
MEM_LIMIT="${PYTHIA_MEM_LIMIT:-12G}"
CORES="${PYTHIA_CORES:-0-7}"

mkdir -p "$LOGDIR"
cd "$REPO"

if [ -x "$VENV/bin/python" ]; then
    export PATH="$VENV/bin:$PATH"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"      # NVIDIA only - never touch the AMD iGPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

echo "[run_stage] $NAME | repo=$REPO | data=$DATA_ROOT | cores=$CORES | mem<=$MEM_LIMIT" >&2

if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope --quiet true >/dev/null 2>&1; then
    exec systemd-run --user --scope --quiet \
        -p "MemoryMax=$MEM_LIMIT" \
        -p "MemorySwapMax=2G" \
        -p "CPUQuota=800%" \
        -p "IOWeight=100" \
        taskset -c "$CORES" \
        ionice -c 3 \
        nice -n 15 \
        "$@"
fi

# No user systemd session (containers, CI, SSH without lingering): keep the CPU
# affinity and niceness, skip the cgroup memory cap.
exec taskset -c "$CORES" ionice -c 3 nice -n 15 "$@"
