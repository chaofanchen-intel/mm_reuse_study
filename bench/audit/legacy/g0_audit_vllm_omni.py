"""G0 审计（vLLM-Omni BAGEL）：与 bench/g0_audit_mstar.py 同样的用例，用计数器和 ViT 调用日志确认每个请求的实际执行。

用法（服务需先以 scripts/audit_vo/sitecustomize.py 的 shim 启动，见 docs/ 或 results/g0/5880/vllm-omni/NOTES.md）：
    envs/mstar/bin/python bench/g0_audit_vllm_omni.py --base http://localhost:18401 \
        --vit-log results/g0/5880/vllm-omni/launches/vit_calls.log --out results/g0/5880/vllm-omni/img2img-path-prefix-on

计数来源：
- prefix：/metrics 的 vllm:prefix_cache_queries_total、vllm:prefix_cache_hits_total（阶段 0）。
- ViT：shim 日志中 encode_vit 行数的增量（每次 ViT 前向一行）。
- 计数器异步更新，每次请求前后各等待 SETTLE_S 秒。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.request
from pathlib import Path

import requests

from g0_audit_mstar import png_b64, post

SETTLE_S = 2.0
MODEL = "models/cache/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b"
LINE = re.compile(r'^([a-zA-Z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$')
WANTED = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
          "vllm:e2e_request_latency_seconds_count")


def scrape(base: str) -> dict:
    text = urllib.request.urlopen(f"{base}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and m.group(1) in WANTED:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def count_vit(vit_log: Path) -> tuple[int, int]:
    """返回 (encode_vit 行数, embed_multimodal 行数)。日志不存在时为 (0, 0)。"""
    if not vit_log.exists():
        return 0, 0
    vit = emb = 0
    for line in vit_log.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] == "encode_vit":
            vit += 1
        elif len(parts) >= 3 and parts[2] == "embed_multimodal":
            emb += 1
    return vit, emb


def answer_text(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(payload)[:200]


def run_chat(args, log, label, img_b64, question, layout):
    # 图像理解必须显式 modalities=["text"]；不带时 BAGEL 走图像路径（img2img + DiT 阶段）。
    body = {"model": MODEL, "max_tokens": 8, "modalities": ["text"], "temperature": 0,
            "messages": [{"role": "user", "content": (
        [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}, {"type": "text", "text": question}]
        if layout == "image-then-text" else
        [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    )}]}
    time.sleep(SETTLE_S)
    before, vb = scrape(args.base), count_vit(args.vit_log)
    elapsed, payload = post(args.base, "/v1/chat/completions", body, timeout=900)
    time.sleep(SETTLE_S)
    after, va = scrape(args.base), count_vit(args.vit_log)
    rec = {"label": label, "path": "/v1/chat/completions", "prompt_layout": layout,
           "elapsed_ms": elapsed * 1000,
           "prefix_queries": after["vllm:prefix_cache_queries_total"] - before.get("vllm:prefix_cache_queries_total", 0.0),
           "prefix_hits": after["vllm:prefix_cache_hits_total"] - before.get("vllm:prefix_cache_hits_total", 0.0),
           "vit_calls": va[0] - vb[0], "embed_multimodal_calls": va[1] - vb[1],
           "answer": answer_text(payload), "error": payload.get("error")}
    log.append(rec)
    print(f"{label:<26} {rec['elapsed_ms']:8.0f} ms  prefix {rec['prefix_hits']:.0f}/{rec['prefix_queries']:.0f}  "
          f"vit {rec['vit_calls']}  emb {rec['embed_multimodal_calls']}  ans={rec['answer']!r}", flush=True)
    return rec


def run_images(args, log, label, prompt):
    body = {"model": MODEL, "prompt": prompt, "size": "512x512", "n": 1}
    time.sleep(SETTLE_S)
    before, vb = scrape(args.base), count_vit(args.vit_log)
    t0 = time.monotonic()
    r = requests.post(f"{args.base}/v1/images/generations", json=body, timeout=900)
    elapsed = time.monotonic() - t0
    time.sleep(SETTLE_S)
    after, va = scrape(args.base), count_vit(args.vit_log)
    rec = {"label": label, "path": "/v1/images/generations", "prompt_layout": None,
           "elapsed_ms": elapsed * 1000, "status": r.status_code,
           "prefix_queries": after["vllm:prefix_cache_queries_total"] - before.get("vllm:prefix_cache_queries_total", 0.0),
           "prefix_hits": after["vllm:prefix_cache_hits_total"] - before.get("vllm:prefix_cache_hits_total", 0.0),
           "vit_calls": va[0] - vb[0], "embed_multimodal_calls": va[1] - vb[1],
           "error": None if r.ok else r.text[:200]}
    log.append(rec)
    print(f"{label:<26} {rec['elapsed_ms']:8.0f} ms  status={r.status_code}  vit {rec['vit_calls']}  emb {rec['embed_multimodal_calls']}", flush=True)


def run_edits(args, log, img_raw: bytes):
    for i, prompt in enumerate(["make the cube blue", "make the cube green"]):
        time.sleep(SETTLE_S)
        before, vb = scrape(args.base), count_vit(args.vit_log)
        t0 = time.monotonic()
        r = requests.post(f"{args.base}/v1/images/edits", timeout=900,
                          data={"model": MODEL, "prompt": prompt, "size": "512x512"},
                          files={"image": ("ref.png", img_raw, "image/png")})
        elapsed = time.monotonic() - t0
        time.sleep(SETTLE_S)
        after, va = scrape(args.base), count_vit(args.vit_log)
        rec = {"label": f"i2i-ref-A-{i}", "path": "/v1/images/edits", "prompt_layout": "image-then-text",
               "elapsed_ms": elapsed * 1000, "status": r.status_code,
               "prefix_queries": after["vllm:prefix_cache_queries_total"] - before.get("vllm:prefix_cache_queries_total", 0.0),
               "prefix_hits": after["vllm:prefix_cache_hits_total"] - before.get("vllm:prefix_cache_hits_total", 0.0),
               "vit_calls": va[0] - vb[0], "embed_multimodal_calls": va[1] - vb[1],
               "error": None if r.ok else r.text[:200]}
        log.append(rec)
        print(f"{rec['label']:<26} {rec['elapsed_ms']:8.0f} ms  status={r.status_code}  vit {rec['vit_calls']}  emb {rec['embed_multimodal_calls']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:18401")
    ap.add_argument("--vit-log", type=Path, default=Path("results/g0/5880/vllm-omni/launches/vit_calls.log"))
    ap.add_argument("--out", type=Path, default=Path("results/g0/5880/vllm-omni/img2img-path-prefix-on"))
    ap.add_argument("--only-edits", action="store_true")
    ap.add_argument("--skip-edits", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    img_a, img_b = png_b64((200, 40, 40)), png_b64((40, 60, 200))
    raw_a = base64.b64decode(img_a)

    if not args.only_edits:
        questions = ["What color is this image? One word.",
                     "Is this image mostly red? Yes or no.",
                     "Describe the image in three words.",
                     "What is the main object? One word."]
        for i, q in enumerate(questions):
            run_chat(args, log, f"A-image-first-q{i}", img_a, q, "image-then-text")
        for i, q in enumerate(questions[:2]):
            run_chat(args, log, f"B-text-first-q{i}", img_a, q, "text-then-image")
        run_chat(args, log, "A-new-image-B", img_b, questions[0], "image-then-text")
        run_images(args, log, "t2i-512", "a red cube on a table")
    if not args.skip_edits:
        run_edits(args, log, raw_a)

    with open(args.out / "requests.jsonl", "w", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {args.out / 'requests.jsonl'}")


if __name__ == "__main__":
    main()
