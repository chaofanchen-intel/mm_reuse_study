"""序列 × 臂 → RequestRecord JSONL（DESIGN 5.1、10、13）。

用法（服务由 scripts/serve_*.sh 启动，--server-dir 指向它的 --out 目录）：
    envs/mstar/bin/python -m bench.runner --server-dir results/g1/5880/mstar/pfx1/server \
        --arm enc0-pfx1 --sequences s1,s2,s3,s4 --res 512,980 --repeats 8 \
        --out results/g1/5880/mstar/pfx1/records.jsonl

- 服务状态：server-dir 下的 prompts_seen.txt 记录本服务上所有经 runner 发出的 prompt 哈希。
  文件不存在而服务已完成过请求时拒绝运行（--allow-reused-server 跳过），避免"冷/新"判断被以前的请求污染。
- 每个请求：流式发送（TTFT），等待 SETTLE_S 秒后读计数器，窗口内完成请求数不为 1 时在记录里标出。
- 图像与臂绑定（图像 id 含 --arm），同一服务上跑多个臂不会互相命中。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

from bench.adapters import SETTLE_S, make_adapter
from bench.client import Client, image_part, text_part
from bench.gpu_guard import foreign_procs
from bench.metrics import RequestRecord, Timeline
from bench.sequences import SEQUENCES, color_correct, make_image, warmup_requests


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def render(messages: list[dict], res: int) -> tuple[list[dict], str]:
    """把序列里的图像占位换成 data URL；返回 (OpenAI 消息, prompt 键)。prompt 键用图像内容哈希代替字节。"""
    out, key_parts = [], []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(m)
            key_parts.append([m["role"], m["content"]])
            continue
        parts, kparts = [], []
        for p in m["content"]:
            if p["type"] == "image":
                png = make_image(p["image_id"], res)[0]
                parts.append(image_part(png))
                kparts.append(["image", _sha(png)])
            else:
                parts.append(text_part(p["text"]))
                kparts.append(["text", p["text"]])
        out.append({"role": m["role"], "content": parts})
        key_parts.append([m["role"], kparts])
    return out, _sha(json.dumps(key_parts, ensure_ascii=False))


class Ledger:
    def __init__(self, server_dir: str):
        self.path = os.path.join(server_dir, "prompts_seen.txt")
        self.seen = set()
        if os.path.exists(self.path):
            self.seen = set(open(self.path).read().split())

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def check_and_add(self, key: str) -> bool:
        before = key in self.seen
        if not before:
            self.seen.add(key)
            with open(self.path, "a") as f:
                f.write(key + "\n")
        return before


def requests_completed(adapter) -> float:
    snap = adapter.snapshot()["metrics"]
    names = ("mstar_requests_total", "vllm:e2e_request_latency_seconds_count")
    return sum(v for (n, _l), v in snap.items() if n in names)


_ERRORS: list[bool] = []


def run_one(args, client, adapter, ledger, req, arm, warmup, index, manifest, out_f, prev_snap):
    messages, key = render(req.messages, req.res)
    seen = ledger.check_and_add(key)
    extra = adapter.chat_extra(prefix_cache=args.prefix_cache == "on")
    before = prev_snap if prev_snap is not None else adapter.snapshot()
    res = client.chat_stream(messages, max_tokens=args.max_tokens, extra=extra)
    foreign = foreign_procs(manifest)          # 在等待计数器的时间里检查共卡（不占请求时间）
    time.sleep(SETTLE_S)
    after = adapter.snapshot()
    obs = adapter.observe(before, after, has_media=bool(req.image_ids))
    correct = color_correct(res.text, req.expect) if req.expect else None
    warns = []
    if obs.requests_done is not None and abs(obs.requests_done - 1) > 1e-9:
        warns.append(f"窗口内完成请求数 {obs.requests_done}，归因不可靠")
    if foreign:
        warns.append("共卡：" + ",".join(f"GPU{f['gpu']}:pid{f['pid']}" for f in foreign))
    warn = "；".join(warns) or None
    tl = Timeline(t_in_first=0.0, t_in_last=0.0, t_query=0.0,
                  t_first_out=None if res.t_first_out is None else res.t_first_out - res.t_query,
                  t_done=None if res.t_done is None else res.t_done - res.t_query)
    rec = RequestRecord(
        scenario=req.seq, arm=arm, machine=manifest.get("machine", ""), repeat=req.repeat,
        request_index=index, cache_state="cold" if req.new_image else "warm", warmup=warmup,
        prompt_layout="image-then-text",
        framework=manifest["framework"], framework_commit=manifest.get("framework_commit") or "",
        model_id=adapter.model or "", config_path=manifest.get("config"), config_hash=manifest.get("config_sha256"),
        env={"gpus": manifest.get("gpus"), "gpu_info": manifest.get("gpu_info"), "host": manifest.get("host")},
        gen_params={"res": req.res, "max_tokens": args.max_tokens, "temperature": 0.0, "step": req.step,
                    "prefix_cache_request": args.prefix_cache, **extra},
        timeline=tl, ttft_ms=res.ttft_ms, e2e_ms=res.e2e_ms,
        stage_ms=obs.node_ms,
        cache_hits={"prefix_tokens": obs.prefix_hit_tokens, "encoder_cache_hit": obs.encoder_cache_hit},
        node_calls=obs.node_calls,
        prefix_hit_tokens=obs.prefix_hit_tokens, prefix_coverage=obs.prefix_coverage,
        encoder_ran=obs.encoder_ran, encoder_cache_hit=obs.encoder_cache_hit,
        system_prompt_hash=_sha(req.system_prompt)[:16], prompt_seen_before=seen,
        output_tokens=res.output_tokens,
        server_manifest={k: manifest.get(k) for k in ("name", "config", "config_sha256", "gpus", "shim_mode",
                                                      "framework_commit", "started_at")},
        output_digest=_sha(res.text)[:16],
        output_quality={"answer": res.text, "expected": req.expect, "correct": correct},
        error=res.error,
    )
    row = rec.to_json()
    row.update({
        "prefix_query_tokens": obs.prefix_query_tokens, "encoder_calls": obs.encoder_calls,
        "encoder_ms": obs.encoder_ms, "mm_keys": obs.mm_keys, "requests_done": obs.requests_done,
        "attribution_warning": warn, "output_tokens_source": res.output_tokens_source,
        "gpu_foreign_procs": foreign,
        "question": req.question, "num_images": len(req.image_ids), "step": req.step,
    })
    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_f.flush()
    flag = "" if correct is None else (" ok" if correct else " WRONG")
    print(f"{index:4d} {'W' if warmup else ' '} {req.seq} r{req.repeat} s{req.step} {req.res:4d}  "
          f"ttft {res.ttft_ms or float('nan'):7.1f}  e2e {res.e2e_ms or float('nan'):7.1f}  "
          f"enc {obs.encoder_calls}{'' if obs.encoder_cache_hit is None else ('/hit' if obs.encoder_cache_hit else '/miss')}  "
          f"pfx {obs.prefix_hit_tokens}/{obs.prefix_query_tokens} {obs.prefix_coverage}  "
          f"{res.text[:30]!r}{flag}{'  ' + warn if warn else ''}{'  ERR ' + res.error[:80] if res.error else ''}",
          flush=True)
    _ERRORS.append(bool(res.error))
    if len(_ERRORS) >= args.max_consecutive_errors and all(_ERRORS[-args.max_consecutive_errors:]):
        sys.exit(f"连续 {args.max_consecutive_errors} 个请求出错，中止。最后的错误：{res.error}")
    return after, res.text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--arm", required=True, help="臂名，如 enc0-pfx1；也进入图像 id")
    ap.add_argument("--prefix-cache", choices=("on", "off"), default="on",
                    help="请求级前缀缓存（M* 有效；vLLM-Omni 要用部署配置，off 会报错）")
    ap.add_argument("--sequences", default="s1,s2,s3,s4")
    ap.add_argument("--res", default="512,980")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3, help="每档分辨率的 warmup 图像请求数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="覆盖服务的模型名")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-reused-server", action="store_true")
    ap.add_argument("--max-consecutive-errors", type=int, default=3, help="连续出错这么多次就中止（默认 3）")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.server_dir, "manifest.json")))
    adapter = make_adapter(manifest, args.model)
    client = Client(manifest["base_url"], adapter.model)
    ledger = Ledger(args.server_dir)
    done = requests_completed(adapter)
    if not ledger.exists() and done > 0 and not args.allow_reused_server:
        sys.exit(f"服务已完成 {done:.0f} 个请求但没有 prompts_seen.txt：不是新服务。重启服务或加 --allow-reused-server")

    seqs = [s for s in args.sequences.split(",") if s]
    resolutions = [int(r) for r in args.res.split(",") if r]
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tag = f"{args.arm}|seed{args.seed}"
    index = 0
    with open(args.out, "a", encoding="utf-8") as out_f:
        meta = {"_meta": True, "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "requests_before": done, "ledger_existed": ledger.exists(), "settle_s": SETTLE_S}
        out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        snap = None
        for res in resolutions:
            for req in warmup_requests(tag, res, args.warmup):
                snap, _ = run_one(args, client, adapter, ledger, req, args.arm, True, index, manifest, out_f, snap)
                index += 1
        for rep in range(args.repeats):
            order = [(s, r) for s in seqs for r in resolutions]
            rng.shuffle(order)
            for seq, res in order:
                gen = SEQUENCES[seq](tag, rep, res)
                try:
                    req = next(gen)
                    while True:
                        snap, answer = run_one(args, client, adapter, ledger, req, args.arm, False, index,
                                               manifest, out_f, snap)
                        index += 1
                        req = gen.send(answer)
                except StopIteration:
                    pass
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
