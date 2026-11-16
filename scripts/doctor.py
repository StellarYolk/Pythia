#!/usr/bin/env python
"""Preflight checks before a heavy stage.

Refuses to start when the machine is short on RAM (the display runs on an AMD
iGPU that allocates from system RAM) or when another heavy stage is running.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

MIN_FREE_RAM_GB = 8.0
MIN_FREE_DISK_GB = 20.0
HEAVY_PATTERNS = [
    re.compile(r"scripts/0[1-8]_"),
    re.compile(r"scripts/(build|run)"),
]


def meminfo() -> dict:
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":")
            info[k.strip()] = int(v.strip().split()[0]) // 1024  # MB
    return info


def main() -> int:
    ok = True
    m = meminfo()
    free = m["MemAvailable"] / 1024
    total = m["MemTotal"] / 1024
    swap_used = (m["SwapTotal"] - m["SwapFree"]) / 1024
    print(f"RAM: total={total:.1f} GB available={free:.1f} GB  swap used={swap_used:.1f} GB")
    if free < MIN_FREE_RAM_GB:
        print(f"  [FAIL] only {free:.1f} GB available (need >= {MIN_FREE_RAM_GB} GB). Close apps or wait.")
        ok = False
    else:
        print(f"  [ok] {free:.1f} GB available (>= {MIN_FREE_RAM_GB} GB)")

    du = shutil_free = None
    st = os.statvfs("/home/parth")
    disk_free = st.f_bavail * st.f_frsize / 1e9
    print(f"Disk: free={disk_free:.1f} GB")
    if disk_free < MIN_FREE_DISK_GB:
        print(f"  [FAIL] only {disk_free:.1f} GB free (need >= {MIN_FREE_DISK_GB} GB)")
        ok = False
    else:
        print(f"  [ok] {disk_free:.1f} GB free")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"GPU: {out}")
        used = int(out.split(",")[2].split()[0])
        if used > 1024:
            print("  [WARN] >1 GB VRAM in use - a stale vLLM engine may still be running (make clean-gpu)")
    except Exception as exc:
        print(f"GPU: unavailable ({exc})")
        ok = False

    # stale engine processes
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        stale = [l for l in ps.splitlines() if "EngineCore" in l or "smoke_vllm" in l]
        heavy = []
        for l in ps.splitlines():
            if "python" in l and any(p.search(l) for p in HEAVY_PATTERNS):
                heavy.append(l.strip())
        if stale:
            print(f"  [WARN] {len(stale)} stale vLLM engine process(es):")
            for l in stale[:5]:
                print("        " + l[:120])
            print("         run: make clean-gpu")
        if heavy:
            print(f"  [WARN] {len(heavy)} Pythia stage process(es) already running:")
            for l in heavy[:5]:
                print("        " + l[:120])
            print("         only one stage at a time is supported")
    except Exception:
        pass

    print("PREFLIGHT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
