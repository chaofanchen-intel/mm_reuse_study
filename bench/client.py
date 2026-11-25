"""统一的 OpenAI 兼容客户端（DESIGN 6、13）。两个框架用同一份实现，保持连接，流式读取以得到 TTFT。

时间都用 time.perf_counter()（单调时钟），单位秒；记录里换成相对 t_query 的毫秒。
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field

import requests


@dataclass
class ChatResult:
    text: str = ""
    t_query: float = 0.0
    t_first_out: float | None = None     # 第一个非空文本增量到达
    t_done: float | None = None
    output_tokens: int | None = None     # 服务端 usage；没有时为 None
    output_tokens_source: str = "none"   # "usage" | "chunks" | "none"
    num_chunks: int = 0
    status: int | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def ttft_ms(self) -> float | None:
        return None if self.t_first_out is None else (self.t_first_out - self.t_query) * 1000

    @property
    def e2e_ms(self) -> float | None:
        return None if self.t_done is None else (self.t_done - self.t_query) * 1000


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class Client:
    def __init__(self, base_url: str, model: str, timeout: float = 900.0):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def chat_stream(self, messages: list[dict], *, max_tokens: int, temperature: float = 0.0,
                    extra: dict | None = None, include_usage: bool = True) -> ChatResult:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True}
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        if extra:
            body.update(extra)
        res = ChatResult()
        pieces: list[str] = []
        res.t_query = time.perf_counter()
        try:
            with self.session.post(f"{self.base}/v1/chat/completions", json=body, stream=True,
                                   timeout=self.timeout) as r:
                res.status = r.status_code
                if r.status_code != 200:
                    res.error = r.text[:500]
                    res.t_done = time.perf_counter()
                    return res
                for raw in r.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        res.error = json.dumps(chunk["error"])[:500]
                    if chunk.get("usage"):
                        res.usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            if res.t_first_out is None:
                                res.t_first_out = time.perf_counter()
                            pieces.append(delta)
                            res.num_chunks += 1
        except requests.RequestException as e:
            res.error = f"{type(e).__name__}: {e}"[:500]
        res.t_done = time.perf_counter()
        res.text = "".join(pieces)
        if not res.text and res.error is None:
            res.error = "空输出：流里没有可解析的文本增量（检查流式格式）"
        if res.usage.get("completion_tokens") is not None:
            res.output_tokens, res.output_tokens_source = int(res.usage["completion_tokens"]), "usage"
        elif res.num_chunks:
            res.output_tokens, res.output_tokens_source = res.num_chunks, "chunks"
        return res

    def images_generate(self, prompt: str, size: str = "512x512", extra: dict | None = None) -> tuple[float, int, str | None]:
        body = {"model": self.model, "prompt": prompt, "size": size, "n": 1, **(extra or {})}
        t0 = time.perf_counter()
        r = self.session.post(f"{self.base}/v1/images/generations", json=body, timeout=self.timeout)
        return (time.perf_counter() - t0) * 1000, r.status_code, None if r.ok else r.text[:500]

    def images_edit(self, png: bytes, prompt: str, size: str | None = None) -> tuple[float, int, str | None]:
        """/v1/images/edits 只接受 multipart（两个框架都是）。"""
        data = {"model": self.model, "prompt": prompt}
        if size:
            data["size"] = size
        t0 = time.perf_counter()
        r = self.session.post(f"{self.base}/v1/images/edits", data=data,
                              files={"image": ("ref.png", png, "image/png")}, timeout=self.timeout)
        return (time.perf_counter() - t0) * 1000, r.status_code, None if r.ok else r.text[:500]
