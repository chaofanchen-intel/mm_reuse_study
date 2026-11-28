"""汇总 bench/upload.py 的记录：变体 × 臂 × 分辨率 × 第几问：TTFT 中位数、编码器运行比例、前缀命中中位数、正确率。

python -m bench.upload_summary RECORDS.jsonl... > summary.md
"""
import json
import statistics
import sys
from collections import defaultdict


def variant(meta: dict) -> str:
    fw = meta.get("framework", "?")
    env = (meta.get("manifest") or {}).get("study_env") or {}
    if fw == "mstar":
        mb = env.get("MSTAR_ENCODER_CACHE_MB", "0")
        return "M* M1" if mb not in ("", "0") else "M* 基线"
    return "vLLM-Omni"


def main(paths):
    g = defaultdict(list)
    for p in paths:
        meta = {}
        for line in open(p):
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
                continue
            for x in r["rows"]:
                o = x["observation"]
                nc = o.get("node_calls") or {}
                enc = nc.get("vision_items_encoded", o.get("encoder_calls"))
                key = (variant(meta), r["arm"], r["res"], "预热" if x["kind"] == "warmup" else f"第{x['index'] + 1}问")
                g[key].append((x.get("ttft_ms") or x.get("e2e_ms"), enc, o.get("prefix_hit_tokens"), x.get("correct"),
                               bool(r.get("gpu_foreign_procs"))))
    print("| 变体 | 臂 | 分辨率 | 请求 | n | TTFT（预热为 E2E）中位 ms | 编码器运行 | 前缀命中中位 | 正确 | 共卡 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g):
        v = g[k]
        t = [x[0] for x in v if x[0] is not None]
        enc = [x[1] for x in v if x[1] is not None]
        hit = [x[2] for x in v if x[2] is not None]
        cor = [x[3] for x in v if x[3] is not None]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(v)} | {statistics.median(t):.1f} | "
              f"{sum(1 for e in enc if e) }/{len(enc)} | {statistics.median(hit) if hit else '-'} | "
              f"{sum(cor)}/{len(cor) if k[3] != '预热' else 0} | {sum(x[4] for x in v)} |")


if __name__ == "__main__":
    main(sys.argv[1:])
