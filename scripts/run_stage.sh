#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Resource-limited launcher for every Pythia stage.
#
# Rationale (see docs/DEVIATIONS.md): this box is a hybrid-graphics laptop with
# 29 GB RAM whose *display* runs on a weak AMD iGPU that allocates from system
# RAM.  Saturating the CPU/RAM can wedge the display driver, so every stage is
# pinned to 8 of 32 cores, niced, and hard-capped at 12 GB inside a cgroup.
# All GPU work is restricted to the NVIDIA device (CUDA_VISIBLE_DEVICES=0).
# ---------------------------------------------------------------------------
set -uo pipefail

NAME="${1:-stage}"
shift || true

REPO="/home/parth/college/NLP/lab/pythia"
VENV="/home/parth/venvs/pythia"
LOGDIR="/home/parth/pythia-data/logs"
MEM_LIMIT="${PYTHIA_MEM_LIMIT:-12G}"
CORES="${PYTHIA_CORES:-0-7}"

mkdir -p "$LOGDIR"
cd "$REPO"

export PATH="$VENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/home/parth/.cache/huggingface}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0          # NVIDIA only - never touch the AMD iGPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

exec systemd-run --user --scope --quiet \
    -p "MemoryMax=$MEM_LIMIT" \
    -p "MemorySwapMax=2G" \
    -p "CPUQuota=800%" \
    -p "IOWeight=100" \
    /usr/bin/taskset -c "$CORES" \
    /usr/bin/ionice -c 3 \
    /usr/bin/nice -n 15 \
    "$@"
