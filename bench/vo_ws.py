"""vLLM-Omni 自带的流式视频端点（WS /v1/video/chat/stream）的客户端（DESIGN 16.5 的 V1 与语音提问）。

与 bench/stream.py 同一帧序列（make_frame），但由服务端做 warmup（video_stream_base.py）。
每个提问记录：发出提问 → response.start、首个文本增量、首个音频块、文本结束、音频结束。

python -m bench.vo_ws --server-dir <服务目录> --modalities text,audio --queries 8,16,24 --session-tag x --out rec.jsonl
  --question-audio q.wav：语音提问（PCM16 16 kHz 单声道 WAV；以 audio_data 随 video.query 发出，text 为空）
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import time
import wave

import websockets

from .stream import QUESTIONS, SYSTEM, make_frame


def read_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000, \
            f"{path}: 需要 PCM16 16 kHz 单声道"
        return w.readframes(w.getnframes())


async def session(a, name: str) -> dict:
    t0 = time.perf_counter()
    rel = lambda: time.perf_counter() - t0
    events: list[tuple[float, str, dict]] = []
    stages = [{"temperature": 0.0, "max_tokens": a.max_tokens}]
    if "audio" in a.modalities:
        talker = {"max_tokens": a.talker_max_tokens}
        if a.talker_greedy:   # 音频可复现，用于核对 warmup 后与冷请求一致（默认 temperature 0.9 采样）
            talker |= {"temperature": 0.0, "top_k": 1, "seed": 42}
        stages += [talker, {}]
    cfg = {"type": "session.config", "model": a.model, "modalities": a.modalities, "system_prompt": SYSTEM,
           "max_frames": a.max_frames, "num_frames": min(128, a.max_frames), "enable_frame_filter": False,
           "use_audio_in_video": False, "sampling_params_list": stages}
    question_audio = base64.b64encode(read_pcm16(a.question_audio)).decode() if a.question_audio else None
    colors: dict[int, str] = {}
    async with websockets.connect(a.url, max_size=None) as ws:
        await ws.send(json.dumps(cfg))
        done = asyncio.Event()
        current: dict | None = None
        queries: list[dict] = []
        need = {"text"} | ({"audio"} if "audio" in a.modalities else set())

        async def receiver():
            nonlocal current
            async for raw in ws:
                m = json.loads(raw)
                t, typ = rel(), m.get("type")
                events.append((t, typ, {}))
                if current is None:
                    continue
                if typ == "response.start":
                    current.setdefault("t_start_ack", t)
                elif typ == "response.text.delta" and m.get("delta"):
                    current.setdefault("t_first_text", t)
                    current["text"] = current.get("text", "") + m["delta"]
                elif typ == "response.output_audio.delta":
                    current.setdefault("t_first_audio", t)
                    try:   # 每块是一个 WAV：取 PCM，整段做哈希（核对 warmup 后的音频与冷请求一致）
                        with wave.open(io.BytesIO(base64.b64decode(m.get("data", ""))), "rb") as w:
                            pcm = w.readframes(w.getnframes())
                        current.setdefault("_pcm", hashlib.sha256())
                        current["_pcm"].update(pcm)
                        current["audio_samples"] = current.get("audio_samples", 0) + len(pcm) // 2
                    except Exception as e:   # noqa: BLE001
                        current["audio_parse_error"] = str(e)[:100]
                    current["audio_chunks"] = current.get("audio_chunks", 0) + 1
                    current["audio_bytes"] = current.get("audio_bytes", 0) + len(m.get("data", ""))
                elif typ == "response.text.done":
                    current["t_text_done"] = t
                    current.setdefault("text", m.get("text", ""))
                    current["done"] = current.get("done", set()) | {"text"}
                elif typ == "response.output_audio.done":
                    current["t_audio_done"] = t
                    current["done"] = current.get("done", set()) | {"audio"}
                elif typ == "error":
                    current["error"] = m.get("message")
                    current["done"] = need
                if current.get("done", set()) >= need:
                    current["done"] = sorted(current["done"])
                    done.set()

        recv = asyncio.create_task(receiver())
        q_times = sorted(a.queries)
        i = qi = 0
        while qi < len(q_times):
            nf, nq = i / a.fps, q_times[qi]
            if nq <= nf:
                await asyncio.sleep(max(0.0, nq - rel()))
                last = max(colors) if colors else None
                text = "" if question_audio else QUESTIONS[qi % len(QUESTIONS)]
                current = {"query_index": qi, "t_query": rel(), "frames_sent": i,
                           "expected_color": colors.get(last), "spoken": bool(question_audio)}
                done.clear()
                msg = {"type": "video.query", "text": text}
                if question_audio:
                    msg["audio_data"] = question_audio
                await ws.send(json.dumps(msg))
                try:
                    await asyncio.wait_for(done.wait(), timeout=300)
                except asyncio.TimeoutError:
                    current["error"] = "timeout"
                c = current
                c["correct"] = bool(c.get("expected_color") and c["expected_color"] in c.get("text", "").lower())
                if "_pcm" in c:
                    c["audio_sha256"] = c.pop("_pcm").hexdigest()
                c.pop("done", None)
                queries.append(c)
                current = None
                qi += 1
                continue
            await asyncio.sleep(max(0.0, nf - rel()))
            jpg, color = make_frame(name, i, a.scene_len, *a.res)
            colors[i] = color
            await ws.send(json.dumps({"type": "video.frame", "data": base64.b64encode(jpg).decode(),
                                      "frame_id": str(i), "pts_ms": int(i * 1000 / a.fps)}))
            i += 1
        await ws.send(json.dumps({"type": "video.done"}))
        await asyncio.sleep(0.5)
        recv.cancel()
    for q in queries:
        for k in ("t_start_ack", "t_first_text", "t_first_audio", "t_text_done", "t_audio_done"):
            if k in q:
                q[k.replace("t_", "ms_", 1)] = round((q[k] - q["t_query"]) * 1000, 1)
    return {"session": name, "frames_sent": i, "queries": queries,
            "event_counts": {k: sum(1 for e in events if e[1] == k) for k in {e[1] for e in events}}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server-dir", required=True)
    p.add_argument("--modalities", default="text")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--queries", default="8,16,24")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--scene-len", type=int, default=6)
    p.add_argument("--res", default="640x480")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--talker-max-tokens", type=int, default=512)
    p.add_argument("--question-audio", default=None)
    p.add_argument("--talker-greedy", action="store_true")
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--session-tag", default="ws")
    p.add_argument("--fixed-name", action="store_true", help="会话名就是 session-tag（帧内容可在不同服务上复现）")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.modalities = a.modalities.split(",")
    a.queries = [float(x) for x in a.queries.split(",")]
    a.res = tuple(int(x) for x in a.res.split("x"))
    manifest = json.load(open(os.path.join(a.server_dir, "manifest.json")))
    a.url = manifest["base_url"].replace("http://", "ws://") + "/v1/video/chat/stream"
    cmd = manifest.get("command", [])
    a.model = cmd[cmd.index("--served-model-name") + 1] if "--served-model-name" in cmd else None
    with open(a.out, "a") as f:
        f.write(json.dumps({"_meta": {"args": {k: v for k, v in vars(a).items()}, "manifest": manifest,
                                      "started": time.time()}}) + "\n")
        for s in range(a.sessions):
            name = a.session_tag if a.fixed_name else f"{a.session_tag}-s{s}-{int(time.time())}"
            rec = asyncio.run(session(a, name))
            rec["modalities"] = a.modalities
            f.write(json.dumps(rec) + "\n")
            f.flush()
            for q in rec["queries"]:
                print(f"{name} q{q['query_index']} frames {q['frames_sent']:3d} first_text {q.get('ms_first_text')} "
                      f"first_audio {q.get('ms_first_audio')} {'ok' if q['correct'] else 'X'} "
                      f"{q.get('text', '')[:40]!r} audio {q.get('audio_samples')} {str(q.get('audio_sha256'))[:10]} {q.get('error') or ''}",
                      flush=True)


if __name__ == "__main__":
    main()
