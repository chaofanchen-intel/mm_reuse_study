"""G0 审计（M* BAGEL）：不改代码，用 /metrics 计数器确认每个请求实际执行了哪些节点、命中了多少前缀。

用法：
    envs/mstar/bin/python bench/g0_audit_mstar.py --base http://localhost:18301 --out results/g0/mstar

服务需以 --enable-metrics 启动。每个请求前后各抓一次 /metrics，取计数器差值。
输出 JSONL（每请求一行，字段对应 bench/metrics.py 的 RequestRecord 子集）与一个汇总表。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

SETTLE_S = 2.0   # 计数器经由状态消息异步更新（约 1 s 一次），每次请求后等待再抓取

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
WANTED = ("mstar_step_seconds_count", "mstar_step_seconds_sum", "mstar_prefix_cache_query_tokens_total",
          "mstar_prefix_cache_hit_tokens_total", "mstar_requests_total")


def scrape(base: str) -> dict:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if not m or m.group(1) not in WANTED:
            continue
        labels = dict(LABEL.findall(m.group(2) or ""))
        out[(m.group(1), tuple(sorted(labels.items())))] = float(m.group(3))
    return out


def delta(before: dict, after: dict) -> dict:
    """返回 {指标名: {标签描述: 增量}}，只保留增量非零的项。"""
    res = defaultdict(dict)
    for key, val in after.items():
        d = val - before.get(key, 0.0)
        if abs(d) > 1e-9:
            name, labels = key
            desc = ",".join(f"{k}={v}" for k, v in labels if k in ("component", "graph_walk", "outcome"))
            res[name][desc or "-"] = d
    return {k: dict(v) for k, v in res.items()}


def png_b64(color, size=512) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(base: str, path: str, body: dict, timeout=600) -> tuple[float, dict]:
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode())
    return time.monotonic() - t0, payload


def run_case(base, label, path, body, log, prompt_layout):
    time.sleep(SETTLE_S)
    before = scrape(base)
    elapsed, payload = post(base, path, body)
    time.sleep(SETTLE_S)
    after = scrape(base)
    rec = {"label": label, "path": path, "prompt_layout": prompt_layout,
           "elapsed_ms": elapsed * 1000, "delta": delta(before, after),
           "error": payload.get("error")}
    log.append(rec)
    print(f"{label:<34} {elapsed*1000:9.0f} ms  " +
          json.dumps({k: v for k, v in rec["delta"].items() if k != "mstar_requests_total"}, ensure_ascii=False))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:18301")
    ap.add_argument("--out", default="results/g0/mstar")
    ap.add_argument("--skip-edits", action="store_true", help="跳过 i2i（显存不足时单独跑）")
    ap.add_argument("--only-edits", action="store_true", help="只跑 i2i")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_a, img_b = png_b64((200, 40, 40)), png_b64((40, 60, 200))
    log: list[dict] = []

    if args.only_edits:
        return_edits_only(args, out, img_a, log)
        return
    # 0. warmup（丢弃）
    run_case(args.base, "warmup-text", "/v1/chat/completions",
             {"model": "bagel", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}, [], "text")

    questions = ["What color is this image? One word.",
                 "Is this image mostly red? Yes or no.",
                 "Describe the image in three words.",
                 "What is the main object? One word."]
    # 场景 A：图像在前（同一图 A，多问题）
    for i, q in enumerate(questions):
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a}"}},
                   {"type": "text", "text": q}]
        run_case(args.base, f"A-image-first-q{i}", "/v1/chat/completions",
                 {"model": "bagel", "messages": [{"role": "user", "content": content}], "max_tokens": 8},
                 log, "image-then-text")
    # 场景 B：文本在前（同一图 A，同样的问题）
    for i, q in enumerate(questions[:2]):
        content = [{"type": "text", "text": q},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a}"}}]
        run_case(args.base, f"B-text-first-q{i}", "/v1/chat/completions",
                 {"model": "bagel", "messages": [{"role": "user", "content": content}], "max_tokens": 8},
                 log, "text-then-image")
    # 新图 B（图像在前）
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b}"}},
               {"type": "text", "text": questions[0]}]
    run_case(args.base, "A-new-image-B", "/v1/chat/completions",
             {"model": "bagel", "messages": [{"role": "user", "content": content}], "max_tokens": 8},
             log, "image-then-text")
    # 文生图（无输入图像）
    run_case(args.base, "t2i-512", "/v1/images/generations",
             {"model": "bagel", "prompt": "a red cube on a table", "size": "512x512", "n": 1}, log, None)
    if not args.skip_edits:
        edits(args, img_a, log)
    _write(out, log)


def edits(args, img_a, log):
    """/v1/images/edits 只接受 multipart（JSON 会得到 400）。"""
    import requests
    raw = base64.b64decode(img_a)
    for i, prompt in enumerate(["make the cube blue", "make the cube green"]):
        time.sleep(SETTLE_S)
        before = scrape(args.base)
        t0 = time.monotonic()
        r = requests.post(f"{args.base}/v1/images/edits", timeout=900,
                          data={"model": "bagel", "prompt": prompt},
                          files={"image": ("ref.png", raw, "image/png")})
        elapsed = time.monotonic() - t0
        time.sleep(SETTLE_S)
        after = scrape(args.base)
        payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        rec = {"label": f"i2i-ref-A-{i}", "path": "/v1/images/edits", "prompt_layout": "image-then-text",
               "elapsed_ms": elapsed * 1000, "delta": delta(before, after),
               "error": payload.get("error") or (None if r.ok else f"HTTP {r.status_code}")}
        log.append(rec)
        print(f"{rec['label']:<34} {elapsed*1000:9.0f} ms  status={r.status_code}  " +
              json.dumps({k: v for k, v in rec["delta"].items() if k != "mstar_requests_total"}, ensure_ascii=False))


def _write(out, log):
    with open(out / "requests.jsonl", "w", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out / 'requests.jsonl'}")


def return_edits_only(args, out, img_a, log):
    edits(args, img_a, log)
    _write(out, log)


if __name__ == "__main__":
    main()
