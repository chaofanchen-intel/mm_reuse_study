"""M* 适配器。

计数来源：/metrics（服务需 --enable-metrics）
- mstar_step_seconds_count / _sum {component, graph_walk}：每个节点的 walk 次数与耗时（是否带设备同步待确认，DESIGN 15）
- mstar_prefix_cache_query_tokens_total / _hit_tokens_total {component}
- mstar_requests_total {outcome}

前缀覆盖程度（观测判定）：请求带图像时，看 LLM 消费图像的 walk（BAGEL：LLM 的 prefill_vit）是否运行：
- 没有运行 → full（图像的 KV 全部来自前缀缓存）
- 运行且前缀命中 > 0 → partial
- 运行且命中 0 → none
注意：M* 的编码器节点不看前缀缓存，所以 full 时编码器仍会运行（DESIGN 2 的 R1′）。
"""
from __future__ import annotations

from . import Observation, counter_delta, scrape

# 模型 → (编码器节点, LLM 中消费媒体的 walk)
MODEL_LAYOUT = {
    "bagel": {"encoders": {"vit_encoder", "vae_encoder"}, "media_walks": {("LLM", "prefill_vit")}},
}


class MStarAdapter:
    framework = "mstar"

    def __init__(self, manifest: dict, model: str | None = None):
        self.manifest = manifest
        self.base = manifest["base_url"]
        cmd = manifest.get("command", [])
        self.model = model or (cmd[cmd.index("serve") + 1] if "serve" in cmd else "bagel")
        layout = MODEL_LAYOUT.get(self.model, {"encoders": set(), "media_walks": set()})
        self.encoders = layout["encoders"]
        self.media_walks = layout["media_walks"]

    def chat_extra(self, prefix_cache: bool = True) -> dict:
        """请求级关闭前缀缓存：prefix_cache=False 随 model_kwargs 传递（serving.mdx）。"""
        return {} if prefix_cache else {"prefix_cache": False}

    def snapshot(self) -> dict:
        return {"metrics": scrape(self.base, ("mstar_",))}

    def observe(self, before: dict, after: dict, has_media: bool) -> Observation:
        d = counter_delta(before["metrics"], after["metrics"])
        obs = Observation()
        for (name, labels), v in d.items():
            lab = dict(labels)
            if name in ("mstar_step_seconds_count", "mstar_step_seconds_sum"):
                key = f"{lab.get('component')}:{lab.get('graph_walk')}"
                if name.endswith("_count"):
                    obs.node_calls[key] = v
                else:
                    obs.node_ms[key] = v * 1000
            elif name == "mstar_prefix_cache_query_tokens_total":
                obs.prefix_query_tokens = int((obs.prefix_query_tokens or 0) + v)
            elif name == "mstar_prefix_cache_hit_tokens_total":
                obs.prefix_hit_tokens = int((obs.prefix_hit_tokens or 0) + v)
            elif name == "mstar_requests_total":
                obs.requests_done = (obs.requests_done or 0) + v
        if obs.prefix_query_tokens is not None and obs.prefix_hit_tokens is None:
            obs.prefix_hit_tokens = 0
        enc_keys = [k for k in obs.node_calls if k.split(":")[0] in self.encoders]
        obs.encoder_calls = int(sum(obs.node_calls[k] for k in enc_keys))
        obs.encoder_ran = obs.encoder_calls > 0
        enc_ms = [obs.node_ms[k] for k in enc_keys if k in obs.node_ms]
        obs.encoder_ms = sum(enc_ms) if enc_ms else None
        if has_media:
            media_ran = any(obs.node_calls.get(f"{c}:{w}", 0) > 0 for c, w in self.media_walks)
            if not media_ran:
                obs.prefix_coverage = "full"
            elif (obs.prefix_hit_tokens or 0) > 0:
                obs.prefix_coverage = "partial"
            else:
                obs.prefix_coverage = "none"
        obs.raw = {f"{n}{dict(l)}": v for (n, l), v in d.items()}
        return obs
