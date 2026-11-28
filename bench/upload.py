"""BAGEL"先上传图像、稍后提问"（DESIGN 17）：单图版的到达即算。

每个会话一张新图（640x480 以外的尺寸由 --res 给，方形）：
- Q 臂：提问时才发 图像 + 问题；
- W 臂：上传时（t=0）发一个只含图像的 max_tokens=1 请求（预热编码器与前缀），--gap 秒后开始提问。
每个会话 --questions 个不同问题，依次提问。每个请求前后读计数器（audit），记录编码器是否运行、前缀命中、TTFT。

python -m bench.upload --server-dir <服务目录> --arm W --sessions 8 --questions 3 --res 512 --out rec.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from .adapters import SETTLE_S, make_adapter
from .client import Client, image_part, text_part
from .gpu_guard import foreign_procs
from .sequences import color_correct, make_image

SYSTEM = "You are a helpful assistant. Answer briefly."
QUESTIONS = ["What is the main color of this image? Answer with one word.",
             "Which single color dominates the picture? One word.",
             "Name the color that fills most of the image, in one word.",
             "What color is the background? One word."]


def observe(adapter, before):
    time.sleep(SETTLE_S)
    o = asdict(adapter.observe(before, adapter.snapshot(), True))
    o.pop("raw", None)
    return o


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server-dir", required=True)
    p.add_argument("--arm", choices=["Q", "W"], required=True)
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--questions", type=int, default=3)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--gap", type=float, default=2.0, help="W 臂：上传到第一次提问的间隔（秒）")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--warmup-text", default="", help="W 臂：预热请求在图像后附带的文本（空 = 只含图像）")
    p.add_argument("--session-tag", default="")
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    manifest = json.load(open(os.path.join(a.server_dir, "manifest.json")))
    adapter = make_adapter(manifest, a.model)
    client = Client(manifest["base_url"], a.model or getattr(adapter, "model", None) or "bagel")
    extra = {**adapter.chat_extra(True), "temperature": 0.0}
    tag = a.session_tag or f"{a.arm}-{int(time.time())}"
    with open(a.out, "a") as f:
        f.write(json.dumps({"_meta": {"args": vars(a), "manifest": manifest, "framework": adapter.framework,
                                      "settle_s": SETTLE_S, "started": time.time()}}) + "\n")
        for s in range(a.sessions):
            image_id = f"upload-{tag}-{s}"
            png, color = make_image(image_id, a.res)
            img = image_part(png)
            rows = []
            if a.arm == "W":
                snap = adapter.snapshot()
                content = [img] + ([text_part(a.warmup_text)] if a.warmup_text else [])
                r = client.chat_stream([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                                       max_tokens=1, extra=extra)
                rows.append({"kind": "warmup", "e2e_ms": r.e2e_ms, "error": r.error,
                             "observation": observe(adapter, snap)})
                time.sleep(max(0.0, a.gap - SETTLE_S))
            for k in range(a.questions):
                q = QUESTIONS[k % len(QUESTIONS)]
                snap = adapter.snapshot()
                r = client.chat_stream([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": [img, text_part(q)]}],
                                       max_tokens=a.max_tokens, extra=extra)
                rows.append({"kind": "question", "index": k, "ttft_ms": r.ttft_ms, "e2e_ms": r.e2e_ms,
                             "text": r.text, "correct": color_correct(r.text, color), "error": r.error,
                             "observation": observe(adapter, snap)})
            rec = {"session": image_id, "arm": a.arm, "res": a.res, "framework": adapter.framework,
                   "rows": rows, "gpu_foreign_procs": foreign_procs(manifest)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            for x in rows:
                o = x["observation"]
                nc = o.get("node_calls") or {}
                enc = nc.get("vision_items_encoded", o.get("encoder_calls"))
                print(f"{image_id} {x['kind']:8s} {x.get('index', '')} ttft {x.get('ttft_ms') and round(x['ttft_ms'], 1)} "
                      f"e2e {x['e2e_ms'] and round(x['e2e_ms'], 1)} enc {enc} pfx {o.get('prefix_hit_tokens')}/"
                      f"{o.get('prefix_query_tokens')} {'ok' if x.get('correct') else ('-' if x['kind'] == 'warmup' else 'X')} "
                      f"{x.get('error') or ''}", flush=True)


if __name__ == "__main__":
    main()
