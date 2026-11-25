"""汇总 runner 的 JSONL（DESIGN 10.2）：按 框架 × 臂 × 序列 × 分辨率 × 观测到的缓存状态 分组。

用法：python3 -m bench.summarize results/g1/5880/*/*/records.jsonl [--by scenario,res,prefix_coverage,encoder]

缓存状态（列 state）：
  cov=<full|partial|none>  图像被前缀命中覆盖的程度
  enc=<run|cache|skip>      编码器运行了 / 编码器缓存命中 / 没有被调度（只有 vLLM-Omni 会出现 skip）
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def q(values, p):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def enc_state(r):
    if r.get("encoder_ran"):
        return "run"
    if r.get("encoder_cache_hit"):
        return "cache"
    return "skip" if r.get("num_images") else "-"


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("_meta") or r.get("warmup"):
                continue
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-n", type=int, default=1)
    args = ap.parse_args()
    rows = load(args.paths)
    groups = defaultdict(list)
    for r in rows:
        key = (r["framework"], r["arm"], r["scenario"], r["gen_params"]["res"],
               f"cov={r.get('prefix_coverage')} enc={enc_state(r)}")
        groups[key].append(r)
    print("| 框架 | 臂 | 序列 | 分辨率 | 状态 | n | TTFT 中位 ms | TTFT IQR | 编码器 ms 中位 | 前缀命中中位 | 输出 token 中位 | 正确率 | 警告 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        g = groups[key]
        if len(g) < args.min_n:
            continue
        ttft = [r["ttft_ms"] for r in g if r.get("ttft_ms") is not None]
        enc = [r["encoder_ms"] for r in g if r.get("encoder_ms") is not None]
        hit = [r["prefix_hit_tokens"] for r in g if r.get("prefix_hit_tokens") is not None]
        out = [r["output_tokens"] for r in g if r.get("output_tokens") is not None]
        corr = [r["output_quality"]["correct"] for r in g if r["output_quality"].get("correct") is not None]
        warn = sum(1 for r in g if r.get("attribution_warning") or r.get("error"))
        def fmt(x):
            return "-" if x is None or x != x else f"{x:.1f}"
        med = lambda v: statistics.median(v) if v else None
        corr_s = f"{sum(corr) / len(corr) * 100:.0f}% ({len(corr)})" if corr else "-"
        cells = [key[0], key[1], key[2], str(key[3]), key[4], str(len(g)),
                 fmt(med(ttft)), f"{fmt(q(ttft, .25))}–{fmt(q(ttft, .75))}" if ttft else "-",
                 fmt(med(enc)), fmt(med(hit)), fmt(med(out)), corr_s, str(warn)]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
