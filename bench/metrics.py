"""结果记录格式（每次请求一行 JSON，写入 results/<实验名>/<机器>/*.jsonl）。

字段在实现适配器时再细化；这里只固定必须有的字段，保证两台机器的结果可以合并比较。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RequestRecord:
    scenario: str
    framework: str
    arm: str
    machine: str                   # "5880" | "B70"
    framework_commit: str          # external/<name> 的 commit，便于追溯
    repeat: int
    request_index: int
    ttft_ms: float | None = None   # query / 提交后到第一个 token
    e2e_ms: float | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)  # 节点或阶段耗时
    cache_hits: dict[str, int] = field(default_factory=dict)
    output_digest: str | None = None   # 输出哈希，用于一致性检查（缓存开/关应相同）

    def to_json(self) -> dict:
        return asdict(self)
