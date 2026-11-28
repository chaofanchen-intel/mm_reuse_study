"""摄像头流 + 随时提问（DESIGN 16）。两个框架用同一个客户端、同一帧序列，走 OpenAI 聊天接口。

时间线：帧按 fps 到达，进入缓冲（策略 grow / slide）；提问在给定时刻发出，包含缓冲里的全部帧 + 问题。
臂：
- Q：只在提问时发请求。
- W：客户端 warmup，逻辑同 vLLM-Omni 的 WS 端点（video_stream_base.py:466-533）：缓冲变化且空闲时，
     发一个 max_tokens=1 的前缀请求（系统提示 + 只含帧的 user 消息）；串行、合并；提问开始时取消在途 warmup，
     提问结束后再继续。
模式：
- timing：帧不停，warmup 不等计数器；会话开始与结束各读一次计数器（结束前等 SETTLE_S）。
- audit：每次提问前停止送帧、等 warmup 完成并等计数器稳定，提问后再等，得到单个提问的计数器差。

用法：
  python -m bench.stream --server-dir results/<exp>/5880/<fw>/<arm>/server --arm W --load grow \
      --fps 2 --queries 10,25,40 --sessions 3 --out results/.../records.jsonl
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import http.client
import socket
import threading
from urllib.parse import urlsplit
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import requests
from PIL import Image

from .adapters import SETTLE_S, make_adapter
from .gpu_guard import foreign_procs

SYSTEM = "You are a helpful home assistant watching a live camera feed. Answer briefly."
PALETTE = {"red": (200, 30, 30), "green": (30, 160, 50), "blue": (30, 60, 200), "yellow": (220, 200, 30),
           "purple": (130, 40, 160), "orange": (230, 120, 20), "white": (235, 235, 235), "black": (20, 20, 20)}
QUESTIONS = ["What is the main color of the most recent image? Answer with one word.",
             "Which color appeared in the most recent image? One word.",
             "Name the dominant color of the latest frame in one word."]


def make_frame(session: str, i: int, scene_len: int, w: int = 640, h: int = 480) -> tuple[bytes, str]:
    """第 i 帧：每 scene_len 帧换一个主色（场景），帧之间噪声不同（内容不重复，缓存不会因为字节相同而命中）。"""
    scene = int.from_bytes(hashlib.sha256(f"{session}/scene{i // scene_len}".encode()).digest()[:4], "little")
    color = list(PALETTE)[scene % len(PALETTE)]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{session}/{i}".encode()).digest()[:8], "little"))
    arr = np.clip(np.array(PALETTE[color], np.int16) + rng.integers(-12, 13, (h, w, 3), dtype=np.int16), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue(), color


def c4_ref_part(jpeg: bytes) -> dict:
    """C4：服务端已存过的帧，只发它的引用（几十字节的 data URL，内容是帧字节的 xxh3-128）。"""
    import xxhash
    ref = b"MSTAR-FRAME:" + xxhash.xxh3_128_hexdigest(jpeg).encode()
    return {"type": "image_url", "image_url": {"url": "data:image/x-mstar-frame;base64," + base64.b64encode(ref).decode()}}


def frame_part(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


@dataclass
class Call:
    kind: str                      # warmup | query
    t_start: float                 # 相对会话开始的秒
    n_frames: int
    frame_ids: list[int]
    t_first_out: float | None = None
    t_done: float | None = None
    cancelled: bool = False
    status: int | None = None
    error: str | None = None
    text: str = ""
    request_bytes: int = 0
    c4_warm: int = 0               # C4：随请求发出的 c4_warm
    c4_refs: int = 0               # C4：以引用发出的帧数
    c4_retry: str | None = None    # C4：首发失败、以完整数据重发时，首发的错误


class Cancel:
    """取消在途请求：置位并立刻关闭它的套接字（服务端看到断开即中止）。
    只置标志不够：max_tokens=1 的 warmup 在 prefill 结束前不发任何数据，读循环看不到标志（2026-11-26 G3 发现）。"""

    def __init__(self):
        self.flag = threading.Event()
        self.conn: http.client.HTTPConnection | None = None
        self.lock = threading.Lock()

    def clear(self) -> None:
        with self.lock:
            self.flag.clear()
            self.conn = None

    def set(self) -> None:
        with self.lock:
            self.flag.set()
            if self.conn is not None and self.conn.sock is not None:
                try:
                    self.conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def is_set(self) -> bool:
        return self.flag.is_set()


def chat_once(sess, base: str, body: dict, cancel: Cancel | None, t0: float, call: Call) -> Call:
    """流式请求（http.client，每次新连接，便于从别的线程关闭）。sess 参数保留兼容，不使用。"""
    data = json.dumps(body).encode()
    call.request_bytes = len(data)
    u = urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    pieces = []
    try:
        if cancel is not None:
            with cancel.lock:
                if cancel.flag.is_set():
                    call.cancelled = True
                    raise _Cancelled
                cancel.conn = conn
        conn.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        call.status = r.status
        if r.status != 200:
            call.error = r.read()[:300].decode(errors="replace")
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
                if chunk.get("error"):
                    call.error = json.dumps(chunk["error"])[:300]
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        if call.t_first_out is None:
                            call.t_first_out = time.perf_counter() - t0
                        pieces.append(delta)
    except _Cancelled:
        pass
    except (OSError, http.client.HTTPException, ValueError) as e:
        if cancel is not None and cancel.is_set():
            call.cancelled = True
        else:
            call.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        conn.close()
        if cancel is not None:
            with cancel.lock:
                cancel.conn = None
    call.t_done = time.perf_counter() - t0
    call.text = "".join(pieces)
    if cancel is not None and cancel.is_set() and call.t_first_out is None:
        call.cancelled = True
    return call


class _Cancelled(Exception):
    pass


class Session:
    def __init__(self, args, adapter, model: str, name: str):
        self.a, self.adapter, self.model, self.name = args, adapter, model, name
        self.base = adapter.base
        self.t0 = 0.0
        self.lock = threading.Condition()
        self.buffer: list[int] = []          # 帧序号
        self.frames: dict[int, tuple[bytes, str]] = {}
        self.version = 0                      # 缓冲每变一次 +1
        self.warmed_version = -1
        self.query_active = False
        self.stop = False
        self.warm_cancel = Cancel()
        self.warm_inflight: Call | None = None
        self.calls: list[Call] = []
        self.extra = {"modalities": ["text"], "temperature": 0.0}
        # C4（M* 会话，MSTAR_C4=1）：已送达的帧只发引用；c4_warm = 与上一个成功请求的帧序列公共前缀 - 1
        # （- 1：那个请求的最后一帧可能止于半页，不整段在缓存里）
        self.c4 = getattr(args, "c4", False)
        self.c4_sent: set[int] = set()
        self.c4_done: list[list[int]] = []    # 最近两个成功请求的帧序列
        # 提示按哪个算：latest = 最近一个（默认）；older = 较早的那个（G3 C4 轮所用，当时在补引用键的错误）
        self.c4_hint = getattr(args, "c4_hint", "latest")

    # —— 缓冲 ——
    def add_frame(self, i: int) -> None:
        self.frames[i] = make_frame(self.name, i, self.a.scene_len, *self.a.res)
        with self.lock:
            if len(self.buffer) >= self.a.max_frames:
                drop = self.a.max_frames // 2 if self.a.load == "slide" else 0
                if self.a.load == "grow":
                    return            # grow：满了不再加
                del self.buffer[:drop]
            self.buffer.append(i)
            self.version += 1
            self.lock.notify_all()

    def messages(self, ids: list[int], question: str | None, refs: bool = False) -> list[dict]:
        content = [c4_ref_part(self.frames[i][0]) if refs and i in self.c4_sent else frame_part(self.frames[i][0])
                   for i in ids]
        if question is not None:
            content.append({"type": "text", "text": question})
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    # —— warmup 线程（W 臂）——
    def warmup_loop(self) -> None:
        sess = requests.Session()
        while True:
            with self.lock:
                while not self.stop and (self.query_active or not self.buffer or self.version == self.warmed_version):
                    self.lock.wait(0.5)
                if self.stop:
                    return
                ids, ver = list(self.buffer), self.version
                self.warm_cancel.clear()
                call = Call("warmup", time.perf_counter() - self.t0, len(ids), ids)
                self.warm_inflight = call
            self.send(sess, ids, None, 1, self.warm_cancel, call)
            with self.lock:
                self.warm_inflight = None
                self.calls.append(call)
                # 失败也标记为已尝试，避免热循环（同 vLLM-Omni）；被取消的不算完成
                if not call.cancelled:
                    self.warmed_version = ver
                self.lock.notify_all()

    def send(self, sess, ids: list[int], question: str | None, max_tokens: int, cancel, call: Call) -> Call:
        """发一个请求（warmup 或提问）。C4：已送达的帧发引用并带 c4_warm；失败则以完整数据、不带提示重发一次。"""
        extra = dict(self.extra)
        if self.c4:
            with self.lock:
                done = list(self.c4_done[0 if self.c4_hint == "older" else -1]) if self.c4_done else []
            common = 0
            while common < min(len(ids), len(done)) and ids[common] == done[common]:
                common += 1
            call.c4_warm = max(0, common - 1)
            call.c4_refs = sum(i in self.c4_sent for i in ids)
            if call.c4_warm:
                extra["c4_warm"] = call.c4_warm   # M* 聊天端点把未知的顶层字段当作 model_kwargs
        body = {"model": self.model, "messages": self.messages(ids, question, refs=self.c4), "max_tokens": max_tokens,
                "stream": True, **extra}
        chat_once(sess, self.base, body, cancel, self.t0, call)
        if self.c4 and (call.error or (call.status or 200) != 200) and not call.cancelled:
            call.c4_retry, call.error, call.status, call.text = call.error, None, None, ""
            body = {"model": self.model, "messages": self.messages(ids, question), "max_tokens": max_tokens,
                    "stream": True, **self.extra}
            chat_once(sess, self.base, body, cancel, self.t0, call)
        if self.c4 and not call.cancelled and not call.error and call.status == 200:
            with self.lock:
                self.c4_sent.update(ids)
                self.c4_done = (self.c4_done + [list(ids)])[-2:]
        return call

    def flush(self, sess: requests.Session) -> None:
        """vLLM 在请求紧接着请求时，后面请求的前缀计数要等下一个请求到来才发布（2026-11-26 实测）。
        读计数器之前先发一个极小的纯文本请求把积压的统计推出来；它自身约十几个查询 token、0 命中。"""
        body = {"model": self.model, "messages": [{"role": "user", "content": f"flush {time.time_ns()}"}],
                "max_tokens": 1, "stream": True, **self.extra}
        chat_once(sess, self.base, body, None, self.t0, Call("flush", 0.0, 0, []))
        time.sleep(SETTLE_S)

    # —— 提问 ——
    def query(self, k: int, sess: requests.Session) -> dict:
        audit = self.a.mode == "audit"
        with self.lock:
            if audit:   # 等 warmup 自然完成
                while self.warm_inflight is not None or (self.a.arm == "W" and self.version != self.warmed_version):
                    self.lock.wait(0.5)
            self.query_active = True
            inflight = self.warm_inflight is not None
            if inflight:
                self.warm_cancel.set()
            ids = list(self.buffer)
            warmed = self.warmed_version == self.version
        snap = None
        if audit:
            self.flush(sess)
            snap = self.adapter.snapshot()
        q = QUESTIONS[k % len(QUESTIONS)]
        call = self.send(sess, ids, q, self.a.max_tokens, None, Call("query", time.perf_counter() - self.t0, len(ids), ids))
        obs = None
        if audit:
            time.sleep(SETTLE_S)
            self.flush(sess)
            obs = asdict(self.adapter.observe(snap, self.adapter.snapshot(), True))
            obs.pop("raw", None)
        with self.lock:
            self.query_active = False
            self.calls.append(call)
            self.lock.notify_all()
        color = self.frames[ids[-1]][1] if ids else None
        return {"query_index": k, "warmup_inflight_at_query": inflight, "buffer_fully_warmed": warmed,
                "expected_color": color, "correct": bool(color and color in call.text.lower()),
                "observation": obs, **asdict(call)}

    def run(self) -> dict:
        a = self.a
        snap0 = None
        if a.counters:
            self.flush(requests.Session())
            snap0 = self.adapter.snapshot()
        self.t0 = time.perf_counter()
        wt = threading.Thread(target=self.warmup_loop, daemon=True) if a.arm == "W" else None
        if wt:
            wt.start()
        qsess = requests.Session()
        queries_out = []
        frame_times = []
        q_times = sorted(a.queries)
        end_t = q_times[-1]
        i = 0
        qi = 0
        qthread: threading.Thread | None = None
        while True:
            now = time.perf_counter() - self.t0
            next_frame_t = i / a.fps
            next_q_t = q_times[qi] if qi < len(q_times) else None
            if next_q_t is not None and next_q_t <= next_frame_t:
                # 提问时刻（audit 模式下同步执行，送帧暂停；timing 模式下帧照常到达）
                time.sleep(max(0.0, next_q_t - now))
                if qthread is not None:
                    qthread.join()
                k = qi
                if a.mode == "audit":
                    queries_out.append(self.query(k, qsess))
                else:
                    qthread = threading.Thread(target=lambda kk=k: queries_out.append(self.query(kk, qsess)))
                    qthread.start()
                qi += 1
                if qi >= len(q_times):
                    break
                continue
            if next_frame_t > end_t:
                break
            time.sleep(max(0.0, next_frame_t - now))
            frame_times.append(time.perf_counter() - self.t0)
            self.add_frame(i)
            i += 1
        if qthread is not None:
            qthread.join()
        with self.lock:
            self.stop = True
            self.lock.notify_all()
        if wt:
            self.warm_cancel.set()
            wt.join()
        totals = None
        if a.counters:
            self.flush(qsess)
            totals = asdict(self.adapter.observe(snap0, self.adapter.snapshot(), True))
            totals.pop("raw", None)
        warm = [c for c in self.calls if c.kind == "warmup"]
        return {"session": self.name, "frames_sent": i, "frame_times": frame_times,
                "queries": sorted(queries_out, key=lambda r: r["query_index"]),
                "warmups": [asdict(c) | {"frame_ids": None} for c in warm],
                "warmup_count": len(warm), "warmup_cancelled": sum(c.cancelled for c in warm),
                "session_totals": totals}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server-dir", required=True)
    p.add_argument("--arm", choices=["Q", "W"], required=True)
    p.add_argument("--load", choices=["grow", "slide"], default="grow")
    p.add_argument("--mode", choices=["timing", "audit"], default="timing")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--queries", default="10,20,30", help="提问时刻（秒，相对会话开始）")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--scene-len", type=int, default=6, help="每多少帧换一个主色")
    p.add_argument("--res", default="640x480")
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--session-tag", default="", help="会话名前缀；不同臂用不同前缀，帧内容互不相同")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--model", default=None)
    p.add_argument("--no-counters", dest="counters", action="store_false")
    p.add_argument("--c4", action="store_true", help="M* C4 会话模式（服务需 MSTAR_C4=1）")
    p.add_argument("--c4-hint", choices=["latest", "older"], default="latest",
                   help="c4_warm 按最近两个成功请求中较早的（older）还是最近的（latest）算")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.queries = [float(x) for x in a.queries.split(",")]
    a.res = tuple(int(x) for x in a.res.split("x"))
    manifest = json.load(open(os.path.join(a.server_dir, "manifest.json")))
    adapter = make_adapter(manifest, a.model)
    model = a.model or getattr(adapter, "model", None) or "qwen3_omni"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps({"_meta": {"args": {k: v for k, v in vars(a).items()}, "manifest": manifest,
                                      "framework": adapter.framework, "settle_s": SETTLE_S,
                                      "started": time.time()}}) + "\n")
        for s in range(a.sessions):
            name = f"{a.session_tag or a.arm}-{a.load}-s{s}-{int(time.time())}"
            rec = Session(a, adapter, model, name).run()
            rec.update(arm=a.arm, load=a.load, mode=a.mode, framework=adapter.framework,
                       gpu_foreign_procs=foreign_procs(manifest))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            for q in rec["queries"]:
                ttft = None if q["t_first_out"] is None else (q["t_first_out"] - q["t_start"]) * 1000
                print(f"{name} q{q['query_index']} frames {q['n_frames']:3d} warmed {q['buffer_fully_warmed']!s:5} "
                      f"inflight {q['warmup_inflight_at_query']!s:5} ttft {ttft if ttft is None else round(ttft, 1)} "
                      f"{'ok' if q['correct'] else 'X'} {q['text'][:30]!r} {q['error'] or ''}", flush=True)
            print(f"{name} warmups {rec['warmup_count']} cancelled {rec['warmup_cancelled']}", flush=True)


if __name__ == "__main__":
    main()
