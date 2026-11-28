"""汇总 bench/stream.py 的记录（DESIGN 16）。

python -m bench.stream_summary results/g3-stream/5880/*/*/records.jsonl > results/g3-stream/5880/summary.md

表 1（timing）：每个 框架变体 × 负载 × 臂 × 提问序号：缓冲帧数、TTFT 中位数与范围、正确率、提问时 warmup 是否在途。
表 2（audit）：单次提问的编码帧数、前缀命中/查询 token、TTFT。
表 3（timing，会话级）：每会话的编码帧数与编码器时间（计数器差），warmup 次数与取消次数。
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict


def variant(path: str, meta: dict) -> str:
    fw = meta.get("framework", "?")
    env = (meta.get("manifest") or {}).get("study_env") or {}
    if fw == "mstar":
        mb = env.get("MSTAR_ENCODER_CACHE_MB", "0")
        if env.get("MSTAR_MEDIA_KEYS") == "1":
            return f"M* M1+M3({env.get('MSTAR_GROUP_IMAGES', '?')})+M4" + ("+C1" if env.get("MSTAR_C1") == "1" else "") + ("+C2" if env.get("MSTAR_C2") == "1" else "") + ("+C4" if env.get("MSTAR_C4") == "1" else "")
        return f"M* {'M1 ' + mb + 'MB' if mb not in ('', '0') else '基线'}"
    return "vLLM-Omni"


def enc_items(obs: dict | None) -> int | None:
    if not obs:
        return None
    nc = obs.get("node_calls") or {}
    if "vision_items_encoded" in nc:
        return int(nc["vision_items_encoded"])
    return obs.get("encoder_calls")


def enc_ms(obs: dict | None) -> float | None:
    if not obs:
        return None
    ms = [v for k, v in (obs.get("node_ms") or {}).items() if k.startswith(("vision_encoder", "vit_encoder"))]
    return sum(ms) if ms else obs.get("encoder_ms")


def fmt(x, nd=0):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(paths: list[str]) -> None:
    timing = defaultdict(list)     # (var, load, arm, qi) -> [(frames, ttft, correct, inflight)]
    audit = defaultdict(list)      # (var, load, arm, qi) -> [(frames, enc, hit, query, ttft)]
    sessions = defaultdict(list)   # (var, load, arm) -> [(enc_items, enc_ms, warmups, cancelled, frames_sent)]
    for p in paths:
        meta = {}
        for line in open(p):
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
                continue
            var = variant(p, meta)
            key = (var, r["load"], r["arm"])
            for q in r["queries"]:
                ttft = None if q["t_first_out"] is None else (q["t_first_out"] - q["t_start"]) * 1000
                k = (*key, q["query_index"])
                if r["mode"] == "timing":
                    timing[k].append((q["n_frames"], ttft, q["correct"], q["warmup_inflight_at_query"]))
                else:
                    o = q.get("observation") or {}
                    hit, qry = o.get("prefix_hit_tokens"), o.get("prefix_query_tokens")
                    pr = o.get("per_request_prefix") or {}
                    if pr:   # 去掉计数 flush 请求（最小的 prompt）
                        big = max(pr.values(), key=lambda v: v[0])
                        qry, hit = big
                    audit[k].append((q["n_frames"], enc_items(o), hit, qry, ttft))
            if r["mode"] == "timing" and r.get("session_totals"):
                t = r["session_totals"]
                sessions[key].append((enc_items(t), enc_ms(t), r["warmup_count"], r["warmup_cancelled"], r["frames_sent"]))

    print("## 提问到首 token（timing 模式，真实节奏）\n")
    print("| 变体 | 负载 | 臂 | 提问 | 缓冲帧数 | n | TTFT 中位 ms | 范围 | 正确 | 提问时 warmup 在途 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(timing):
        v = timing[k]
        t = [x[1] for x in v if x[1] is not None]
        frames = sorted({x[0] for x in v})
        print(f"| {k[0]} | {k[1]} | {k[2]} | q{k[3]} | {'/'.join(map(str, frames))} | {len(v)} | "
              f"{fmt(statistics.median(t)) if t else '-'} | {fmt(min(t)) if t else '-'}–{fmt(max(t)) if t else '-'} | "
              f"{sum(x[2] for x in v)}/{len(v)} | {sum(bool(x[3]) for x in v)} |")

    print("\n## 单次提问的工作量（audit 模式：提问前等 warmup 完成、计数器稳定）\n")
    print("| 变体 | 负载 | 臂 | 提问 | 帧数 | 编码帧数 | 前缀命中 / prompt token | TTFT ms |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(audit):
        for frames, enc, hit, qry, ttft in audit[k]:
            print(f"| {k[0]} | {k[1]} | {k[2]} | q{k[3]} | {frames} | {fmt(enc)} | {fmt(hit)} / {fmt(qry)} | {fmt(ttft)} |")

    print("\n## 每会话的编码工作（timing 模式，会话首尾计数器差，含 warmup）\n")
    print("| 变体 | 负载 | 臂 | 会话数 | 发送帧数 | 编码帧数 中位 | 编码器 ms 中位 | warmup 次数 中位 | 被取消 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(sessions):
        v = sessions[k]
        med = lambda i: statistics.median([x[i] for x in v if x[i] is not None]) if any(x[i] is not None for x in v) else None
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {fmt(med(4))} | {fmt(med(0))} | {fmt(med(1))} | "
              f"{fmt(med(2))} | {sum(x[3] for x in v)} |")


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit("用法：python -m bench.stream_summary RECORDS.jsonl..."))
