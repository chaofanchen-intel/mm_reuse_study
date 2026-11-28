"""vLLM-Omni 适配器。

计数来源：
- /metrics：vllm:prefix_cache_queries_total / hits_total（按 stage 标签求和；BAGEL 只有阶段 0 有前缀缓存）、
  vllm:e2e_request_latency_seconds_count（完成的请求数）
- shim 日志（scripts/audit_vo/sitecustomize.py，路径写在 manifest 的 log 目录下 shim.log）：
  encode_vit（编码器前向；time 模式带 ms）、ecm_check（编码器缓存检查结果）、embed_multimodal（多模态 key）

前缀覆盖程度（观测判定）：请求带图像时
- 窗口内既没有 ecm_check 也没有 encode_vit → full（调度器只为未被已缓存 token 覆盖的图像调度编码器）
- 否则前缀命中 > 0 → partial；命中 0 → none
"""
from __future__ import annotations

import os

from . import Observation, counter_delta, scrape


class VllmOmniAdapter:
    framework = "vllm-omni"

    def __init__(self, manifest: dict, model: str | None = None):
        self.manifest = manifest
        self.base = manifest["base_url"]
        cmd = manifest.get("command", [])
        self.model = model or (cmd[cmd.index("--served-model-name") + 1] if "--served-model-name" in cmd else None)
        self.shim_log = os.path.join(os.path.dirname(manifest["log"]), "shim.log") \
            if manifest.get("shim_mode", "off") != "off" else None

    def chat_extra(self, prefix_cache: bool = True) -> dict:
        """BAGEL 的 image→text 必须显式 modalities=["text"]，否则走 img2img + DiT（DESIGN 3.5）。
        前缀缓存只能在部署配置里关（阶段 0 的 enable_prefix_caching），不能按请求关。"""
        if not prefix_cache:
            raise ValueError("vLLM-Omni 的前缀缓存要用部署配置关闭（configs/bagel_vllm_omni_audit_noprefix.yaml）")
        return {"modalities": ["text"]}

    def _log_size(self) -> int:
        if not self.shim_log or not os.path.exists(self.shim_log):
            return 0
        return os.path.getsize(self.shim_log)

    def snapshot(self) -> dict:
        return {"metrics": scrape(self.base, ("vllm:prefix_cache_", "vllm:e2e_request_latency_seconds_count")),
                "log_offset": self._log_size()}

    def _log_lines(self, start: int, end: int) -> list[list[str]]:
        if not self.shim_log or end <= start:
            return []
        with open(self.shim_log, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")
        return [ln.split("\t") for ln in chunk.splitlines() if ln.count("\t") >= 3]

    def observe(self, before: dict, after: dict, has_media: bool) -> Observation:
        d = counter_delta(before["metrics"], after["metrics"])
        obs = Observation()
        for (name, _labels), v in d.items():
            if name == "vllm:prefix_cache_queries_total":
                obs.prefix_query_tokens = int((obs.prefix_query_tokens or 0) + v)
            elif name == "vllm:prefix_cache_hits_total":
                obs.prefix_hit_tokens = int((obs.prefix_hit_tokens or 0) + v)
            elif name == "vllm:e2e_request_latency_seconds_count":
                obs.requests_done = (obs.requests_done or 0) + v
        if obs.prefix_query_tokens is not None and obs.prefix_hit_tokens is None:
            obs.prefix_hit_tokens = 0
        lines = self._log_lines(before["log_offset"], after["log_offset"])
        checks, ms = [], []
        enc_items: dict[str, int] = {}     # pid → 编码项数（TP>1 时每个 rank 都记录，取一个 rank）
        enc_ms: dict[str, float] = {}
        lookups: dict[str, tuple[int, int]] = {}   # 请求 id → (prompt, hit)，取最后一次
        for _ts, pid, kind, detail in (ln[:4] for ln in lines):
            if kind == "encode_vit":
                obs.encoder_calls += 1
                for tok in detail.split():
                    if tok.startswith("ms="):
                        ms.append(float(tok[3:]))
            elif kind == "mm_encoder":
                kv = dict(t.split("=", 1) for t in detail.split() if "=" in t)
                enc_items[pid] = enc_items.get(pid, 0) + int(kv.get("items", 0))
                if "ms" in kv:
                    enc_ms[pid] = enc_ms.get(pid, 0.0) + float(kv["ms"])
            elif kind == "prefix_lookup":
                kv = dict(t.split("=", 1) for t in detail.split() if "=" in t)
                lookups[kv["req"]] = (int(kv["prompt"]), int(kv["hit"]))
            elif kind == "ecm_check":
                checks.append("hit=True" in detail)
            elif kind == "embed_multimodal":
                obs.mm_keys.append(detail.removeprefix("keys="))
        if enc_items and not obs.encoder_calls:   # 非 BAGEL 模型（如 Qwen3-Omni）：用模型无关的挂点
            pid = min(enc_items)
            obs.encoder_calls = enc_items[pid]
            if pid in enc_ms:
                ms = [enc_ms[pid]]
        if lookups:   # 按请求的命中（优先于聚合计数器，后者在请求紧接请求时会延迟发布）
            obs.per_request_prefix = lookups
            obs.prefix_query_tokens = sum(p for p, _ in lookups.values())
            obs.prefix_hit_tokens = sum(h for _, h in lookups.values())
        obs.encoder_ran = obs.encoder_calls > 0
        obs.encoder_ms = sum(ms) if ms else None
        obs.encoder_cache_hit = (any(checks) if checks else None)
        obs.node_calls = {"encode_vit": obs.encoder_calls, "ecm_check": len(checks)}
        if has_media:
            if not checks and not obs.encoder_ran:
                obs.prefix_coverage = "full"
            elif (obs.prefix_hit_tokens or 0) > 0:
                obs.prefix_coverage = "partial"
            else:
                obs.prefix_coverage = "none"
        obs.raw = {f"{n}{dict(l)}": v for (n, l), v in d.items()}
        obs.raw["shim_lines"] = len(lines)
        return obs
