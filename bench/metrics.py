"""结果记录格式（每次请求一行 JSON，写入 results/<实验名>/<机器>/<臂>.jsonl）。

与 docs/DESIGN.md（v5）第 6、10 节对应。字段在适配器里填写；缺失的值用 None，不要用 0 代替。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Timeline:
    """客户端单调时钟，单位秒（相对本请求的 t_in_first；无输入时为 None）。"""
    t_in_first: float | None = None   # 首个输入到达客户端
    t_in_last: float | None = None    # 最后一个输入到达
    t_query: float | None = None      # 提交查询（发出请求）
    t_first_out: float | None = None  # 首个输出（文本首 token；图像为首个可用输出）
    t_done: float | None = None       # 完成


@dataclass
class RequestRecord:
    # --- 身份 ---
    scenario: str                 # plan.py 中的 Scenario.key
    arm: str                      # 开关组合，如 "enc0-pfx0"、"enc1-pfx1"
    machine: str                  # "5880" | "B70"
    repeat: int
    request_index: int            # 本臂内的顺序编号（用于随机化/交错后追溯）
    cache_state: str              # "cold" | "warm"（开始时缓存是否已有前序请求）
    warmup: bool = False          # warmup 请求，报告时丢弃
    prompt_layout: str | None = None   # 例如 "image-then-text"，由 tokenizer 输出确认

    # --- 来源与环境 ---
    framework: str = ""           # "mstar" | "vllm-omni"
    framework_commit: str = ""    # external/<name> 的 commit
    model_id: str = ""            # 如 "ByteDance-Seed/BAGEL-7B-MoT"
    model_revision: str | None = None  # HF snapshot 哈希
    processor_hash: str | None = None  # 处理器配置哈希
    dtype: str | None = None
    quantization: str | None = None
    config_path: str | None = None     # 服务配置文件
    config_hash: str | None = None
    env: dict = field(default_factory=dict)  # torch/cuda/xpu 版本、驱动、GPU 型号与编号

    # --- 生成参数 ---
    gen_params: dict = field(default_factory=dict)  # steps、cfg、seed、size、采样参数

    # --- 时间 ---
    timeline: Timeline = field(default_factory=Timeline)
    ttft_ms: float | None = None          # 文本：query→首 token
    time_to_image_ms: float | None = None # 图像：query→首图
    e2e_ms: float | None = None           # query→完成
    precompute_ms: float | None = None    # 输入到 query 之间框架已完成的工作（若可观测）
    stage_ms: dict[str, float] = field(default_factory=dict)  # 各节点/walk 耗时（同步计时）

    # --- 缓存与计数 ---
    cache_hits: dict[str, int] = field(default_factory=dict)   # encoder / prefix / cfg_img
    cache_overhead_ms: float | None = None
    cache_memory_mb: float | None = None
    node_calls: dict[str, int] = field(default_factory=dict)   # 每请求各编码器节点的调用次数
    image_tokens: list[tuple[int, int]] = field(default_factory=list)  # 每张图像的 token 区间 [start, end)
    prefix_hit_tokens: int | None = None
    prefix_coverage: str | None = None    # "full" | "partial" | "none"（图像被前缀命中覆盖的程度）
    encoder_ran: bool | None = None       # 本请求是否运行了图像/视频编码器
    encoder_cache_hit: bool | None = None # 编码器输出是否来自缓存（M* 目前恒为 None）
    gpu_ms: float | None = None           # 各节点同步耗时之和（只在画像轮有值）

    # --- 请求与服务状态 ---
    system_prompt_hash: str | None = None
    prompt_seen_before: bool | None = None   # 同一服务上是否出现过完全相同的 prompt
    output_tokens: int | None = None          # 实际输出长度（端到端只在固定长度下比较）
    server_manifest: dict = field(default_factory=dict)  # 启动命令、配置哈希、GPU、commit、shim 模式

    # --- 输出 ---
    output_digest: str | None = None      # 文本：token 序列哈希；图像：原始字节哈希
    output_quality: dict = field(default_factory=dict)  # 噪声底比较用（PSNR、LPIPS 等）
    error: str | None = None

    def derived(self) -> dict:
        """从时间线派生两个口径（第 6 节）。缺输入时 in_first 口径为 None。"""
        tl = self.timeline
        out = {}
        if tl.t_query is not None and tl.t_first_out is not None:
            out["query_to_first_out_ms"] = (tl.t_first_out - tl.t_query) * 1000
        if tl.t_in_first is not None and tl.t_first_out is not None:
            out["in_first_to_first_out_ms"] = (tl.t_first_out - tl.t_in_first) * 1000
        if tl.t_in_first is not None and tl.t_query is not None:
            out["precompute_window_ms"] = (tl.t_query - tl.t_in_first) * 1000
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d["derived"] = self.derived()
        return d
