"""框架适配器：把各自的 /metrics（以及 vLLM-Omni 的 shim 日志）变成同一组观测量（DESIGN 5.1、10.1）。

用法：
    ad = make_adapter(manifest)               # manifest 来自 serve_*.sh 写的 manifest.json
    before = ad.snapshot()
    ... 发请求 ...
    time.sleep(SETTLE_S)                      # 计数器异步更新
    after = ad.snapshot()
    obs = ad.observe(before, after, has_media=True)
"""
from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass, field

SETTLE_S = float(os.environ.get("MM_SETTLE_S", "2.0"))   # 计数器异步更新的等待；测试可用环境变量缩短

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def scrape(base_url: str, prefixes: tuple[str, ...]) -> dict[tuple[str, tuple], float]:
    text = urllib.request.urlopen(f"{base_url}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m or not m.group(1).startswith(prefixes):
            continue
        labels = tuple(sorted(_LABEL.findall(m.group(2) or "")))
        out[(m.group(1), labels)] = float(m.group(3))
    return out


def counter_delta(before: dict, after: dict) -> dict[tuple[str, tuple], float]:
    return {k: v - before.get(k, 0.0) for k, v in after.items() if abs(v - before.get(k, 0.0)) > 1e-9}


@dataclass
class Observation:
    encoder_calls: int = 0                    # 本请求窗口内图像/视频编码器的前向次数
    encoder_ran: bool = False
    encoder_cache_hit: bool | None = None     # None：框架没有编码器缓存，或本请求没有检查
    encoder_ms: float | None = None           # 带同步的编码器耗时（vLLM-Omni time 模式；M* 为计数器的 sum）
    prefix_query_tokens: int | None = None
    prefix_hit_tokens: int | None = None
    prefix_coverage: str | None = None        # full / partial / none（按观测判定，见各适配器）
    node_calls: dict[str, float] = field(default_factory=dict)
    node_ms: dict[str, float] = field(default_factory=dict)
    requests_done: float | None = None        # 窗口内完成的请求数（应为 1，否则归因不可靠）
    mm_keys: list[str] = field(default_factory=list)   # vLLM-Omni：多模态 key，用于核对任务路径
    per_request_prefix: dict = field(default_factory=dict)  # vLLM-Omni shim：请求 id → (prompt token, 命中 token)
    raw: dict = field(default_factory=dict)


def make_adapter(manifest: dict, model: str | None = None):
    fw = manifest["framework"]
    if fw == "mstar":
        from .mstar import MStarAdapter
        return MStarAdapter(manifest, model)
    if fw == "vllm-omni":
        from .vllm_omni import VllmOmniAdapter
        return VllmOmniAdapter(manifest, model)
    raise ValueError(f"未知框架 {fw}")
