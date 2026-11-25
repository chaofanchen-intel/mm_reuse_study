"""检查服务所在 GPU 上是否有别的进程（DESIGN 10.2）。5880 是多人共用的机器，别人的任务可能在测量中途占用同一张卡。

foreign_procs(manifest) → [{"gpu": 索引, "pid": ..., "mib": ...}]：在服务的 GPU 上、但不属于服务进程组的计算进程。
服务以 setsid 启动，manifest["pid"] 就是进程组 ID。查询失败时返回 None（记录里写 None，不当作"没有"）。
"""
from __future__ import annotations

import os
import subprocess


def _query(args: list[str]) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=10, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def foreign_procs(manifest: dict) -> list[dict] | None:
    gpus = {g.strip() for g in str(manifest.get("gpus", "")).split(",") if g.strip()}
    pgid = manifest.get("pid")
    if not gpus or pgid is None:
        return None
    try:
        bus = {b: idx for idx, b in _query(["--query-gpu=index,pci.bus_id"])}
        apps = _query(["--query-compute-apps=pid,gpu_bus_id,used_memory"])
    except Exception:
        return None
    found = []
    for pid, b, mib in apps:
        idx = bus.get(b)
        if idx not in gpus:
            continue
        try:
            if os.getpgid(int(pid)) == int(pgid):
                continue
        except ProcessLookupError:
            continue
        found.append({"gpu": idx, "pid": int(pid), "mib": float(mib) if mib.replace(".", "", 1).isdigit() else None})
    return found
