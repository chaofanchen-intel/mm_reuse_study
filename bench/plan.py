"""实验矩阵的数据结构。每个实验 = 场景 × 臂（arm）× 重复次数。

臂（arm）是一种运行配置，例如 "vllm-omni: encoder cache 开" 或 "M*: 节点缓存关"。
场景（scenario）描述输入时序与变化的部分，例如 "同一图像 + N 个不同 prompt"。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Arm:
    framework: str          # "mstar" | "vllm-omni"
    name: str               # 例如 "encoder-cache-on"
    config: dict = field(default_factory=dict)  # 传给适配器的参数（以后确定）


@dataclass(frozen=True)
class Scenario:
    key: str                # 例如 "bagen-same-image-n-prompts"
    model: str              # "BAGEL-7B-MoT" | "Qwen3-Omni"
    hypothesis: str         # 对应 docs/DESIGN.md 中的 H1..H4
    machine: str            # "5880" (CUDA) | "B70" (XPU)
    description: str = ""


SCENARIOS: list[Scenario] = [
    Scenario("phase0-profile-bagel", "BAGEL-7B-MoT", "阶段0", "B70",
             "测 ViT / VAE / LLM prefill / 扩散 / 解码各自耗时占比"),
    Scenario("h1-same-image-n-prompts", "BAGEL-7B-MoT", "H1", "B70",
             "同一参考图，N 个不同 prompt，顺序提交"),
    Scenario("h2-upload-then-type", "BAGEL-7B-MoT", "H2（近似）", "B70",
             "先上传图片，延迟 Δ 后提交 prompt；Δ = 0 / 1 / 3 s"),
    Scenario("h4-video-fixed-text-varies", "Qwen3-Omni", "H4", "5880",
             "视频固定，文本变化；比较视频在前 / 文本在前"),
    Scenario("h2-multirate-stream", "Qwen3-Omni", "H2", "5880",
             "视频 / 音频 / 文本以不同速率到达，query 随机"),
]


def dry_run() -> None:
    """打印矩阵（不执行任何推理）。"""
    for s in SCENARIOS:
        print(f"[{s.hypothesis}] {s.key}  模型={s.model}  机器={s.machine}")
        print(f"    {s.description}")


if __name__ == "__main__":
    dry_run()
