"""多个摄像头会话同时运行（DESIGN 18 的 E2）：一个会话回答时，别的会话的帧仍在到达、被 warmup 编码与 prefill。
看回答的 TTFT 与逐 token 间隔（ITL）受多大干扰——编码器与 Thinker 同卡（vLLM-Omni；M* rank 0）对比编码器独占一卡（M* rank 2）。

会话 s 晚 s*stagger 秒开始，提问时刻因此错开。每个提问要求较长的回答（--max-tokens，默认 64），记录每个文本增量的到达时间。
复用 bench/stream.py 的 Session（缓冲、warmup 线程、取消），只替换提问函数；不读计数器（timing）。

python -m bench.stream_multi --server-dir <服务目录> --arm W --sessions 3 --stagger 2.7 --out rec.jsonl
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import threading
import time
from urllib.parse import urlsplit

from .adapters import make_adapter
from .gpu_guard import foreign_procs
from .stream import Session

LONG_Q = "Describe the most recent image in two sentences, starting with its main color."


def chat_timed(base: str, body: dict, t0: float) -> dict:
    data = json.dumps(body).encode()
    u = urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    out = {"t_start": time.perf_counter() - t0, "token_times": [], "text": "", "error": None}
    pieces = []
    try:
        conn.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status != 200:
            out["error"] = r.read()[:300].decode(errors="replace")
        else:
            while True:
                raw = r.readline()
                if not raw:
                    break
                raw = raw.strip()
                if not raw.startswith(b"data:"):
                    continue
                d = raw[5:].strip()
                if d == b"[DONE]":
                    break
                chunk = json.loads(d)
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        out["token_times"].append(time.perf_counter() - t0)
                        pieces.append(delta)
    except (OSError, http.client.HTTPException, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        conn.close()
    out["t_done"] = time.perf_counter() - t0
    out["text"] = "".join(pieces)
    return out


class TimedSession(Session):
    def query(self, k: int, sess) -> dict:
        with self.lock:
            self.query_active = True
            inflight = self.warm_inflight is not None
            if inflight:
                self.warm_cancel.set()
            ids = list(self.buffer)
            warmed = self.warmed_version == self.version
        body = {"model": self.model, "max_tokens": self.a.max_tokens, "stream": True, **self.extra,
                "messages": self.messages(ids, LONG_Q)}
        r = chat_timed(self.base, body, self.t0)
        with self.lock:
            self.query_active = False
            self.lock.notify_all()
        tt = r["token_times"]
        color = self.frames[ids[-1]][1] if ids else None
        return {"query_index": k, "n_frames": len(ids), "warmup_inflight_at_query": inflight,
                "buffer_fully_warmed": warmed, "expected_color": color,
                "correct": bool(color and color in r["text"].lower()[:80]),
                "ttft_ms": None if not tt else (tt[0] - r["t_start"]) * 1000,
                "itl_ms": [(b - a) * 1000 for a, b in zip(tt, tt[1:])],
                "n_chunks": len(tt), **r}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server-dir", required=True)
    p.add_argument("--arm", choices=["Q", "W"], default="W")
    p.add_argument("--load", choices=["grow", "slide"], default="grow")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--queries", default="8,16,24")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--scene-len", type=int, default=6)
    p.add_argument("--res", default="640x480")
    p.add_argument("--sessions", type=int, default=3, help="同时运行的会话数")
    p.add_argument("--stagger", type=float, default=2.7, help="会话之间的开始间隔（秒）")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--session-tag", default="")
    p.add_argument("--model", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.queries = [float(x) for x in a.queries.split(",")]
    a.res = tuple(int(x) for x in a.res.split("x"))
    a.mode, a.counters = "timing", False
    manifest = json.load(open(os.path.join(a.server_dir, "manifest.json")))
    adapter = make_adapter(manifest, a.model)
    model = a.model or getattr(adapter, "model", None) or "qwen3_omni"
    tag = a.session_tag or f"multi-{a.arm}-{int(time.time())}"
    results: list[dict | None] = [None] * a.sessions

    def one(s: int) -> None:
        time.sleep(s * a.stagger)
        rec = TimedSession(a, adapter, model, f"{tag}-s{s}").run()
        rec.update(slot=s, concurrent=a.sessions, arm=a.arm, load=a.load, framework=adapter.framework)
        results[s] = rec

    threads = [threading.Thread(target=one, args=(s,)) for s in range(a.sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(a.out, "a") as f:
        f.write(json.dumps({"_meta": {"args": vars(a), "manifest": manifest, "framework": adapter.framework,
                                      "started": time.time()}}) + "\n")
        for rec in results:
            rec["gpu_foreign_procs"] = foreign_procs(manifest)
            f.write(json.dumps(rec) + "\n")
            for q in rec["queries"]:
                itl = sorted(q["itl_ms"])
                med = itl[len(itl) // 2] if itl else None
                print(f"{rec['session']} q{q['query_index']} frames {q['n_frames']:3d} ttft "
                      f"{q['ttft_ms'] and round(q['ttft_ms'], 1)} itl_med {med and round(med, 1)} "
                      f"itl_max {itl and round(itl[-1], 1)} chunks {q['n_chunks']} {'ok' if q['correct'] else 'X'} "
                      f"{q['text'][:40]!r} {q['error'] or ''}", flush=True)
            print(f"{rec['session']} warmups {rec['warmup_count']} cancelled {rec['warmup_cancelled']}", flush=True)


if __name__ == "__main__":
    main()
