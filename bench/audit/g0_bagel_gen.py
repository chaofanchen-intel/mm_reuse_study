"""G0 审计：BAGEL 的 t2i 与图像编辑（DESIGN 3.3、5.5）。image→text 用 bench/runner.py。

每个用例：读计数器 → 请求 → 等待 SETTLE_S → 读计数器，记录编码器调用、前缀命中、节点耗时。
编辑用同一参考图做两次（第二次检查编码器是否被复用），参考图每次运行唯一（不会被以前的运行命中）。

用法：
    envs/mstar/bin/python -m bench.audit.g0_bagel_gen --server-dir <serve 的 --out 目录> --out <records.jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import time

from bench.adapters import SETTLE_S, make_adapter
from bench.client import Client
from bench.sequences import make_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"), help="进入参考图 id，保证每次运行唯一")
    ap.add_argument("--skip-edits", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.server_dir, "manifest.json")))
    adapter = make_adapter(manifest)
    client = Client(manifest["base_url"], adapter.model)
    res = int(args.size.split("x")[0])
    ref_png, ref_color = make_image(f"g0-gen|{args.tag}|ref", res)
    cases = [("t2i", None, "a red cube on a table")]
    if not args.skip_edits:
        cases += [("i2i-ref-0", ref_png, "make the image blue"), ("i2i-ref-1", ref_png, "make the image green")]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        snap = adapter.snapshot()
        for label, png, prompt in cases:
            if png is None:
                ms, status, err = client.images_generate(prompt, size=args.size)
            else:
                ms, status, err = client.images_edit(png, prompt, size=args.size)
            time.sleep(SETTLE_S)
            after = adapter.snapshot()
            obs = adapter.observe(snap, after, has_media=png is not None)
            snap = after
            row = {"label": label, "framework": manifest["framework"], "machine": manifest.get("machine"),
                   "e2e_ms": ms, "status": status, "error": err, "size": args.size, "ref_color": ref_color,
                   "encoder_calls": obs.encoder_calls, "encoder_cache_hit": obs.encoder_cache_hit,
                   "encoder_ms": obs.encoder_ms, "prefix_hit_tokens": obs.prefix_hit_tokens,
                   "prefix_query_tokens": obs.prefix_query_tokens, "node_calls": obs.node_calls,
                   "node_ms": obs.node_ms, "mm_keys": obs.mm_keys, "requests_done": obs.requests_done,
                   "server_manifest": {k: manifest.get(k) for k in ("name", "config", "config_sha256", "gpus",
                                                                    "shim_mode", "framework_commit")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{label:<10} {ms:9.0f} ms  status={status}  enc {obs.encoder_calls}"
                  f"{'' if obs.encoder_cache_hit is None else ('/hit' if obs.encoder_cache_hit else '/miss')}  "
                  f"nodes={obs.node_calls}{'  ERR ' + err[:80] if err else ''}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
