"""实验矩阵（与 docs/DESIGN.md v5 第 5、8、11 节对应）。

Scenario = 负载（模型、任务、输入顺序、到达 trace）。
Arm      = 开关组合（encoder 缓存、KV 前缀缓存）× 框架。
注意：两个开关不独立（DESIGN 5.1）。归因按每请求记录的覆盖程度与编码器是否运行分组，臂只描述配置。
矩阵展开时，标记为 implemented=False 的臂会被跳过并报告，不静默丢弃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass(frozen=True)
class Arm:
    framework: str                 # "mstar" | "vllm-omni"
    encoder_cache: bool            # 编码器输出复用（R1）
    prefix_cache: bool             # KV 前缀复用（R2）
    implemented: bool = True       # 该框架上该开关是否已可控
    note: str = ""

    @property
    def name(self) -> str:
        return f"enc{int(self.encoder_cache)}-pfx{int(self.prefix_cache)}"


@dataclass(frozen=True)
class Scenario:
    key: str
    model: str
    machine: str                   # "5880" | "B70" | "5880+B70"（两台机器各跑一次，结果分开）
    hypothesis: str                # G0 / H1 / H3 / H2 / H4 / RQ5
    task: str                      # "i2t" | "i2i" | "t2i" | "video-qa" | "multirate"
    prompt_order: str | None       # "image-then-text" | "media-then-text" | None
    output_modality: str           # 固定记录
    description: str = ""
    variants: tuple[str, ...] = field(default_factory=tuple)  # 如 Δ 取值、视频长度
    gate: str = ""                 # 进入条件，见第 11 节


SCENARIOS: list[Scenario] = [
    # --- G0 审计（不改框架代码） ---
    Scenario("g0-i2t-bagel", "BAGEL-7B-MoT", "5880+B70", "G0", "i2t", "image-then-text", "text",
             "审计：S1–S4 各跑一轮，记录编码器是否运行、编码器缓存命中、前缀命中与覆盖程度（新启动的服务）"),
    # --- BAGEL image→text（主任务，DESIGN 5.1） ---
    Scenario("s1-same-image-questions-bagel", "BAGEL-7B-MoT", "5880+B70", "R2/R1'", "i2t", "image-then-text", "text",
             "S1：同图 A + N 个不同问题，系统提示固定（R2 收益；M* 被浪费的 ViT）",
             variants=("res=512", "res=980"), gate="G0 完成"),
    Scenario("s2-same-image-system-prompts-bagel", "BAGEL-7B-MoT", "5880+B70", "R1", "i2t", "image-then-text", "text",
             "S2：同图 A + 同一问题，N 个不同系统提示（前缀在图像前分叉，R1 的独立收益；替代原场景 B）",
             variants=("res=512", "res=980"), gate="G0 完成"),
    Scenario("s3-new-images-bagel", "BAGEL-7B-MoT", "5880+B70", "基线", "i2t", "image-then-text", "text",
             "S3：每次新图（编码器每次都跑）",
             variants=("res=512", "res=980"), gate="G0 完成"),
    Scenario("s4-multi-turn-bagel", "BAGEL-7B-MoT", "5880+B70", "R1'/R2", "i2t", "image-then-text", "text",
             "S4：多轮对话，第 1 轮带图，后续轮带历史（前缀完全覆盖图像）",
             variants=("turns=4",), gate="G0 完成"),
    Scenario("concurrency-bagel", "BAGEL-7B-MoT", "5880", "吞吐", "i2t", "image-then-text", "text",
             "S1 与 S3 在并发 4、8 下的吞吐与 TTFT 分布（DESIGN 5.3）",
             variants=("c=4", "c=8"), gate="image→text 的闸门通过"),
    # --- 阶段 0 画像（缓存关，计时 shim 开） ---
    Scenario("phase0-i2t-bagel", "BAGEL-7B-MoT", "5880+B70", "阶段0", "i2t", None, "text",
             "画像：ViT、LLM prefill、decode 各自耗时", variants=("res=512", "res=980")),
    Scenario("phase0-i2i-bagel", "BAGEL-7B-MoT", "5880+B70", "阶段0", "i2i", None, "image",
             "画像：ViT、VAE encode、LLM prefill、扩散、VAE decode（只画像，默认不做 R1，DESIGN 5.5）"),
    Scenario("phase0-t2i-bagel", "BAGEL-7B-MoT", "5880+B70", "阶段0", "t2i", None, "image",
             "画像：无输入图像（对照）"),
    # --- Qwen3-Omni（DESIGN 5.7） ---
    Scenario("g0-video-qwen", "Qwen3-Omni", "5880", "G0", "video-qa", "media-then-text", "text",
             "审计：视频帧的编码器缓存、跳过被覆盖的编码器、前缀缓存；视频编码耗时"),
    Scenario("s1-same-video-questions-qwen", "Qwen3-Omni", "5880", "R2/R1'", "video-qa", "media-then-text", "text",
             "同一视频 + N 个不同问题", variants=("N=5",), gate="g0-video-qwen 完成"),
    Scenario("s2-same-video-system-prompts-qwen", "Qwen3-Omni", "5880", "R1", "video-qa", "media-then-text", "text",
             "同一视频 + N 个不同系统提示", variants=("N=5",), gate="g0-video-qwen 完成"),
    Scenario("h2-multirate-qwen", "Qwen3-Omni", "5880", "L1", "multirate", None, "text",
             "固定到达 trace 的多速率输入，固定 query 时刻（双重计时）",
             gate="S1/S2 之后仍需要区分到达即计算的收益"),
    Scenario("rq5-audio-output-qwen", "Qwen3-Omni", "5880", "RQ5", "video-qa", "media-then-text", "audio",
             "音频输出时增量预填关闭的 TTFT 代价",
             gate="G0 显示 vLLM-Omni 的音频输出 TTFT 有可观差距"),
]


def arms_for(framework: str) -> list[Arm]:
    """2×2 开关组合。implemented 标记当前能否独立控制该开关（DESIGN 5.1 的对照表）。"""
    arms = []
    for enc, pfx in product([False, True], repeat=2):
        if framework == "mstar":
            impl = not enc
            note = "M* 没有编码器缓存，需先实现 I2（DESIGN 9）" if enc else ""
        else:
            impl = enc
            note = ("vLLM-Omni 没有关闭编码器缓存的官方选项；测量用的强制 miss 尚未实现与验证"
                    "（DESIGN 15 第 2 条）") if not enc else ""
        arms.append(Arm(framework, enc, pfx, implemented=impl, note=note))
    return arms


def expand(scenarios=SCENARIOS, frameworks=("mstar", "vllm-omni")):
    """返回 (可运行的 (场景, 臂) 列表, 被跳过的 (场景, 臂) 列表)。"""
    runnable, skipped = [], []
    for s in scenarios:
        for fw in frameworks:
            for arm in arms_for(fw):
                (runnable if arm.implemented else skipped).append((s, arm))
    return runnable, skipped


def dry_run() -> None:
    runnable, skipped = expand()
    print(f"场景 {len(SCENARIOS)} 个；可运行 (场景,臂) {len(runnable)}；跳过 {len(skipped)}")
    for s in SCENARIOS:
        print(f"[{s.hypothesis}] {s.key}  模型={s.model}  机器={s.machine}  输出={s.output_modality}")
        print(f"    {s.description}")
        if s.gate:
            print(f"    闸门：{s.gate}")
    print("跳过的臂（需先实现）：")
    for s, arm in skipped:
        print(f"    {s.key} × {arm.framework} {arm.name}：{arm.note}")


if __name__ == "__main__":
    dry_run()
