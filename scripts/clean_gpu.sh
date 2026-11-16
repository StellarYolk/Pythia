#!/usr/bin/env bash
# Kill stale vLLM engine processes that keep VRAM allocated (they survive a
# crashed run and block the next one).
set -uo pipefail

echo "== GPU before =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
for p in $PIDS; do
    if [ -n "$p" ]; then
        echo "killing GPU pid $p"
        kill -9 "$p" 2>/dev/null || true
    fi
done
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 3
echo "== GPU after =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
