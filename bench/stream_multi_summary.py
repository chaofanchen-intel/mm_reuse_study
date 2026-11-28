"""汇总 bench/stream_multi.py 的记录：变体 × 同时会话数：TTFT、逐 token 间隔（ITL）、回答耗时、正确率。

python -m bench.stream_multi_summary RECORDS.jsonl... > summary.md
"""
import json
import os
import statistics
import sys
from collections import defaultdict


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def variant(meta):
    if meta.get("framework") != "mstar":
        return "vLLM-Omni（2 卡）"
    cfg = os.path.basename((meta.get("manifest") or {}).get("config") or "")
    return "M*（3 卡）编码器在 rank 2" if "enc2" in cfg else "M*（3 卡）编码器在 rank 0"


def main(paths):
    g = defaultdict(lambda: {"ttft": [], "itl": [], "dec": [], "ok": 0, "n": 0, "err": 0, "foreign": 0})
    for p in paths:
        meta = {}
        for line in open(p):
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
                continue
            x = g[(variant(meta), r["concurrent"])]
            x["foreign"] += bool(r.get("gpu_foreign_procs"))
            for q in r["queries"]:
                x["n"] += 1
                x["err"] += bool(q.get("error"))
                # 主色写在回答开头一句里（前 80 字符）；记录里的 correct 只看了前 40 字符
                x["ok"] += bool(q.get("expected_color") and q["expected_color"] in (q.get("text") or "").lower()[:80])
                if q.get("ttft_ms") is not None:
                    x["ttft"].append(q["ttft_ms"])
                x["itl"] += q.get("itl_ms") or []
                tt = q.get("token_times") or []
                if len(tt) > 1:
                    x["dec"].append((tt[-1] - tt[0]) * 1000 / (len(tt) - 1))
    print("| 变体 | 同时会话 | 提问 n | TTFT 中位 / p90 ms | ITL 中位 / p90 / p99 / 最大 ms | 每 token 平均 ms（中位） | 正确 | 错误 | 共卡 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g):
        x = g[k]
        f = lambda v: "-" if v is None else f"{v:.1f}"
        print(f"| {k[0]} | {k[1]} | {x['n']} | {f(statistics.median(x['ttft']) if x['ttft'] else None)} / {f(pct(x['ttft'], 0.9))} | "
              f"{f(pct(x['itl'], 0.5))} / {f(pct(x['itl'], 0.9))} / {f(pct(x['itl'], 0.99))} / {f(max(x['itl']) if x['itl'] else None)} | "
              f"{f(statistics.median(x['dec']) if x['dec'] else None)} | {x['ok']}/{x['n']} | {x['err']} | {x['foreign']} |")


if __name__ == "__main__":
    main(sys.argv[1:])
