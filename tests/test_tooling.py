"""测量工具的离线测试（不需要 GPU）：序列、prompt 渲染、两个适配器的覆盖判定、runner 端到端（假服务）。

运行：MM_SETTLE_S=0 envs/mstar/bin/python -m pytest -q tests/test_tooling.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("MM_SETTLE_S", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import sequences as S  # noqa: E402
from bench.adapters.mstar import MStarAdapter  # noqa: E402
from bench.adapters.vllm_omni import VllmOmniAdapter  # noqa: E402
from bench.runner import render  # noqa: E402


# ---------- 序列与渲染 ----------

def test_images_unique_across_arms_and_sequences():
    ids = [f"armA|s1|r0|512|A", f"armB|s1|r0|512|A", f"armA|s2|r0|512|A", f"armA|s1|r1|512|A"]
    digests = {S.make_image(i, 512)[0] for i in ids}
    assert len(digests) == len(ids)


def test_image_deterministic_and_color_check():
    a1, c1 = S.make_image("x", 64)
    a2, c2 = S.make_image("x", 64)
    assert a1 == a2 and c1 == c2 and c1 in S.PALETTE
    assert S.color_correct("The image is Red.", "red")
    assert S.color_correct("violet", "purple")
    assert not S.color_correct("blue", "red")


def test_s1_first_request_new_rest_warm():
    reqs = list(S.s1_same_image_questions("t", 0, 64))
    assert [r.new_image for r in reqs] == [True, False, False, False, False]
    assert len({r.question for r in reqs}) == 5 and len({tuple(r.image_ids) for r in reqs}) == 1


def test_s2_varies_only_system_prompt():
    reqs = list(S.s2_same_image_system_prompts("t", 0, 64))
    assert len({r.system_prompt for r in reqs}) == 5
    assert {r.question for r in reqs} == {S.Q_COLOR}


def test_s4_feeds_answers_back():
    gen = S.s4_multi_turn("t", 0, 64)
    r = next(gen)
    steps = [r]
    for ans in ["Red", "Warm", "Apple"]:
        r = gen.send(ans)
        steps.append(r)
    assert [m["role"] for m in steps[-1].messages] == ["system", "user", "assistant", "user", "assistant", "user",
                                                       "assistant", "user"]
    assert steps[-1].messages[2]["content"] == "Red"
    assert steps[-1].expect == steps[0].expect


def test_render_replaces_images_and_key_is_stable():
    req = next(S.s1_same_image_questions("t", 0, 64))
    m1, k1 = render(req.messages, 64)
    m2, k2 = render(req.messages, 64)
    assert k1 == k2
    url = m1[1]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert m1[1]["content"][1] == {"type": "text", "text": req.question}


# ---------- 适配器的覆盖判定 ----------

def _k(name, **labels):
    return (name, tuple(sorted(labels.items())))


def test_mstar_coverage_rules():
    ad = MStarAdapter({"base_url": "http://x", "command": ["mstar", "serve", "bagel"]})
    base = {"metrics": {}}

    def after(llm_vit, hit, query=1400):
        m = {_k("mstar_step_seconds_count", component="vit_encoder", graph_walk="prefill_vit"): 1,
             _k("mstar_step_seconds_sum", component="vit_encoder", graph_walk="prefill_vit"): 0.05,
             _k("mstar_prefix_cache_query_tokens_total", component="LLM"): query,
             _k("mstar_requests_total", outcome="success"): 1}
        if hit:
            m[_k("mstar_prefix_cache_hit_tokens_total", component="LLM")] = hit
        if llm_vit:
            m[_k("mstar_step_seconds_count", component="LLM", graph_walk="prefill_vit")] = 1
        return {"metrics": m}

    full = ad.observe(base, after(llm_vit=False, hit=1399), True)
    assert full.prefix_coverage == "full" and full.encoder_ran and full.encoder_cache_hit is None
    assert abs(full.encoder_ms - 50.0) < 1e-6 and full.requests_done == 1
    assert ad.observe(base, after(llm_vit=True, hit=700), True).prefix_coverage == "partial"
    none = ad.observe(base, after(llm_vit=True, hit=0), True)
    assert none.prefix_coverage == "none" and none.prefix_hit_tokens == 0
    assert ad.chat_extra(False) == {"prefix_cache": False} and ad.chat_extra(True) == {}


def test_vllm_omni_coverage_rules(tmp_path):
    log = tmp_path / "shim.log"
    manifest = {"base_url": "http://x", "log": str(tmp_path / "server.log"), "shim_mode": "count",
                "command": ["vllm", "serve", "p", "--served-model-name", "bagel"]}
    ad = VllmOmniAdapter(manifest)
    assert ad.model == "bagel" and ad.chat_extra() == {"modalities": ["text"]}

    def window(lines, hit, query=1383):
        start = log.stat().st_size if log.exists() else 0
        with open(log, "a") as f:
            for ln in lines:
                f.write(ln + "\n")
        m = {_k("vllm:prefix_cache_queries_total", stage="0"): query,
             _k("vllm:e2e_request_latency_seconds_count", stage="0"): 1}
        if hit:
            m[_k("vllm:prefix_cache_hits_total", stage="0")] = hit
        return {"metrics": {}, "log_offset": start}, {"metrics": m, "log_offset": log.stat().st_size}

    b, a = window(["1\t9\tecm_check\thit=False input_id=0 req=r mm_hash=h",
                   "1\t9\tembed_multimodal\tkeys=pixel_values",
                   "1\t9\tencode_vit\tshape=(3, 518, 518) ms=27.5"], hit=0)
    o = ad.observe(b, a, True)
    assert o.prefix_coverage == "none" and o.encoder_ran and o.encoder_cache_hit is False
    assert o.encoder_ms == 27.5 and o.mm_keys == ["pixel_values"]
    b, a = window(["1\t9\tecm_check\thit=True input_id=0 req=r mm_hash=h"], hit=1344)
    o = ad.observe(b, a, True)
    assert o.prefix_coverage == "partial" and not o.encoder_ran and o.encoder_cache_hit is True
    b, a = window([], hit=1383)
    o = ad.observe(b, a, True)
    assert o.prefix_coverage == "full" and not o.encoder_ran and o.encoder_cache_hit is None


# ---------- runner 端到端（假的 M* 服务） ----------

class _FakeMStar(BaseHTTPRequestHandler):
    """流式聊天 + M* 风格的 /metrics。见过的图像：LLM prefill_vit 不运行、前缀全命中（模拟 M* 的 R1′ 缺口）。"""
    counters: dict = {}
    seen_images: set = set()
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _inc(self, key, v=1.0):
        self.counters[key] = self.counters.get(key, 0.0) + v

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        lines = []
        for (name, labels), v in self.counters.items():
            lab = ",".join(f'{k}="{val}"' for k, val in labels)
            lines.append(f"{name}{{{lab}}} {v}")
        body = ("\n".join(lines) + "\n").encode()
        self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] is True
        images = [p["image_url"]["url"] for m in body["messages"] if isinstance(m["content"], list)
                  for p in m["content"] if p["type"] == "image_url"]
        with self.lock:
            self._inc(_k("mstar_step_seconds_count", component="vit_encoder", graph_walk="prefill_vit"), len(images))
            q = 1400 if images else 20
            self._inc(_k("mstar_prefix_cache_query_tokens_total", component="LLM"), q)
            new = [u for u in images if u not in self.seen_images]
            if images and not new and body.get("prefix_cache", True):
                self._inc(_k("mstar_prefix_cache_hit_tokens_total", component="LLM"), q - 1)
            elif images:
                self._inc(_k("mstar_step_seconds_count", component="LLM", graph_walk="prefill_vit"))
            self.seen_images.update(images)
            self._inc(_k("mstar_requests_total", outcome="success"))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for piece in ["Re", "d"]:
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 2}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def test_runner_end_to_end(tmp_path):
    _FakeMStar.counters, _FakeMStar.seen_images = {}, set()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMStar)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "manifest.json").write_text(json.dumps({
        "name": "fake", "framework": "mstar", "machine": "test", "base_url": f"http://127.0.0.1:{port}",
        "command": ["mstar", "serve", "bagel"], "gpus": "0", "log": str(server_dir / "server.log")}))
    out = tmp_path / "records.jsonl"
    env = dict(os.environ, MM_SETTLE_S="0")
    r = subprocess.run([sys.executable, "-m", "bench.runner", "--server-dir", str(server_dir), "--arm", "enc0-pfx1",
                        "--sequences", "s1,s2,s3,s4", "--res", "64", "--repeats", "1", "--warmup", "1",
                        "--out", str(out)], cwd=ROOT, env=env, capture_output=True, text=True)
    srv.shutdown()
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["_meta"] and rows[0]["requests_before"] == 0
    recs = [x for x in rows[1:] if not x["warmup"]]
    assert len(recs) == 5 + 5 + 5 + 4
    s1 = [x for x in recs if x["scenario"] == "s1"]
    assert s1[0]["prefix_coverage"] == "none" and all(x["prefix_coverage"] == "full" for x in s1[1:])
    assert all(x["encoder_ran"] for x in s1)               # 假服务模拟 M*：编码器总是运行
    assert all(x["ttft_ms"] is not None and x["output_tokens"] == 2 for x in recs)
    assert all(x["requests_done"] == 1 and x["attribution_warning"] is None for x in recs)
    assert all(x["prompt_seen_before"] is False for x in recs)
    assert all(x["output_quality"]["correct"] in (True, False, None) for x in recs)
    # 账本：每个发出的 prompt 一行，且不重复
    seen = (server_dir / "prompts_seen.txt").read_text().split()
    assert len(seen) == len(set(seen)) == len(rows) - 1


def test_runner_refuses_reused_server(tmp_path):
    _FakeMStar.counters = {_k("mstar_requests_total", outcome="success"): 3.0}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMStar)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "manifest.json").write_text(json.dumps({
        "name": "fake", "framework": "mstar", "base_url": f"http://127.0.0.1:{srv.server_address[1]}",
        "command": ["mstar", "serve", "bagel"], "log": str(server_dir / "server.log")}))
    r = subprocess.run([sys.executable, "-m", "bench.runner", "--server-dir", str(server_dir), "--arm", "a",
                        "--sequences", "s3", "--res", "64", "--repeats", "1", "--out", str(tmp_path / "o.jsonl")],
                       cwd=ROOT, env=dict(os.environ, MM_SETTLE_S="0"), capture_output=True, text=True)
    srv.shutdown()
    _FakeMStar.counters = {}
    assert r.returncode != 0 and "不是新服务" in (r.stderr + r.stdout)


# ---------- 摄像头流客户端（DESIGN 16） ----------

class _FakeStream(_FakeMStar):
    """每张图像 5 ms 的"编码"；回答为最后一帧的主色（解码 JPEG 求均值，取最近的调色板颜色）。"""
    calls: list = []

    def do_POST(self):
        import base64, io
        import numpy as np
        from PIL import Image
        from bench.stream import PALETTE
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        imgs = [p["image_url"]["url"] for m in body["messages"] if isinstance(m["content"], list)
                for p in m["content"] if p["type"] == "image_url"]
        with self.lock:
            self.calls.append((body["max_tokens"], len(imgs)))
            self._inc(_k("mstar_step_seconds_count", component="vision_encoder", graph_walk="prefill_vision"), len(imgs))
        time.sleep(0.005 * len(imgs))
        answer = "none"
        if imgs:
            arr = np.asarray(Image.open(io.BytesIO(base64.b64decode(imgs[-1].split(",", 1)[1])))).reshape(-1, 3).mean(0)
            answer = min(PALETTE, key=lambda c: sum((a - b) ** 2 for a, b in zip(PALETTE[c], arr)))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': answer}}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def _stream_run(tmp_path, arm, mode, load="grow", max_frames=50):
    _FakeStream.counters, _FakeStream.calls = {}, []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeStream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    server_dir = tmp_path / f"server-{arm}-{mode}-{load}"
    server_dir.mkdir()
    (server_dir / "manifest.json").write_text(json.dumps({
        "name": "fake", "framework": "mstar", "base_url": f"http://127.0.0.1:{srv.server_address[1]}",
        "command": ["mstar", "serve", "qwen3_omni"], "gpus": "", "log": str(server_dir / "server.log")}))
    out = tmp_path / f"{arm}-{mode}-{load}.jsonl"
    r = subprocess.run([sys.executable, "-m", "bench.stream", "--server-dir", str(server_dir), "--arm", arm,
                        "--mode", mode, "--load", load, "--fps", "20", "--queries", "0.6,1.2",
                        "--max-frames", str(max_frames), "--res", "64x48", "--out", str(out)],
                       cwd=ROOT, env=dict(os.environ, MM_SETTLE_S="0"), capture_output=True, text=True)
    srv.shutdown()
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    return rows[1], list(_FakeStream.calls)


def test_stream_query_only(tmp_path):
    rec, calls = _stream_run(tmp_path, "Q", "timing")
    assert rec["warmup_count"] == 0 and all(mt != 1 for mt, n in calls if n)  # 0 图的是计数 flush
    assert [q["n_frames"] for q in rec["queries"]] == [12, 24]
    assert all(q["correct"] for q in rec["queries"]), rec["queries"]
    assert rec["session_totals"]["node_calls"]["vision_encoder:prefill_vision"] == 36


def test_stream_warmup_audit(tmp_path):
    rec, calls = _stream_run(tmp_path, "W", "audit")
    assert rec["warmup_count"] >= 1 and any(mt == 1 for mt, _ in calls)
    for q in rec["queries"]:
        assert q["buffer_fully_warmed"] and not q["warmup_inflight_at_query"] and q["correct"]
        assert q["observation"]["node_calls"]["vision_encoder:prefill_vision"] == q["n_frames"]


def test_stream_slide_drops_half(tmp_path):
    rec, _ = _stream_run(tmp_path, "Q", "timing", load="slide", max_frames=10)
    assert all(q["n_frames"] <= 10 for q in rec["queries"]) and all(q["correct"] for q in rec["queries"])


class _FakeSlowWarm(_FakeStream):
    """warmup（max_tokens=1、带图）在 2 s 内不发任何字节，模拟长 prefill。"""

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        raw = self.rfile.read(n)
        body = json.loads(raw)
        if body["max_tokens"] == 1 and "image_url" in raw.decode():
            time.sleep(2.0)
        self.rfile = __import__("io").BytesIO(raw)
        self.headers.replace_header("Content-Length", str(n))
        super().do_POST()


def test_stream_cancel_is_prompt(tmp_path):
    _FakeSlowWarm.counters, _FakeSlowWarm.calls = {}, []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSlowWarm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "manifest.json").write_text(json.dumps({
        "name": "fake", "framework": "mstar", "base_url": f"http://127.0.0.1:{srv.server_address[1]}",
        "command": ["mstar", "serve", "qwen3_omni"], "gpus": "", "log": str(server_dir / "server.log")}))
    out = tmp_path / "o.jsonl"
    r = subprocess.run([sys.executable, "-m", "bench.stream", "--server-dir", str(server_dir), "--arm", "W",
                        "--fps", "20", "--queries", "0.6", "--res", "64x48", "--no-counters", "--out", str(out)],
                       cwd=ROOT, env=dict(os.environ, MM_SETTLE_S="0"), capture_output=True, text=True)
    srv.shutdown()
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text().splitlines()[1])
    q = rec["queries"][0]
    assert q["warmup_inflight_at_query"]
    w = [x for x in rec["warmups"] if x["cancelled"]]
    assert w and all(x["t_done"] - q["t_start"] < 0.3 for x in w), (w, q["t_start"])
