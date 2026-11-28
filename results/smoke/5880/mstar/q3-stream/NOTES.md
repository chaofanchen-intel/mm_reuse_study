# M* Qwen3-Omni 摄像头流 smoke（2026-11-26，5880，GPU 0,1,4）

配置：`configs/qwen3omni_mstar_thinker_tp2.yaml`（Thinker TP2 + 编码器在 rank 0，Talker/Code2Wav 在 rank 2；prefix_cache_tp）。
环境：`MSTAR_SPEECH_NODES_EAGER=1`（M0：5880 共享内存 100 KB，Talker 的 Triton 核需要 144 KB，捕获尺寸时启动失败；文本输出不用这两个节点，跳过它们的 CUDA graph）；`MSTAR_MAX_MEDIA_FILES=64`（默认每请求 16 个媒体文件）。
客户端：`bench/stream.py`，640x480 JPEG，2 fps，audit 模式，提问在 5/10/15 s（10/20/30 帧）。

| 目录 | 内容 |
|---|---|
| `server-failed-talker-smem/` | 启动失败（M0 之前） |
| `server-cache0-limit16/`、`records-cache0-limit16.jsonl` | 16 个媒体文件上限，20/30 帧的提问被拒 |
| `server-cache0/`、`records-cache0.jsonl` | 基线（未改 M*，M1 关） |
| `server-cache1024-nohit/`、`server-cache1024-debug/` | M1 第一版：每个视觉 walk 只有一张图，"至少编码一项"使其永不命中（无效） |
| `server-cache1024/`、`records-cache1024.jsonl` | M1（引擎跳过被缓存服务的行），1024 MB |

## 结果（audit，单次提问）

| 臂 | 帧数 | TTFT ms 基线 | TTFT ms M1 | 编码帧数 基线 → M1 |
|---|---|---|---|---|
| Q | 10 / 20 / 30 | 1432 / 1781 / 2455 | 1480 / 1633 / 1938 | 10/20/30 → 10/10/10 |
| W | 10 / 20 / 30 | 539 / 1024 / 1643 | 274 / 532 / 748 | 10/20/30 → 0/0/0 |

vLLM-Omni 同场景（`../../vllm-omni/q3-stream/`）：Q 620–700 ms，W 92–138 ms，且不随帧数增长。

## 发现

1. **M* 对被前缀缓存覆盖的帧仍重跑视觉编码器**（基线 W：前缀命中 8960/9169，仍编码 30 帧、946 ms）。M1 消除后，W 的 TTFT 降 49–55%。
2. **M* 每张图像是独立的一对 walk（视觉编码 → Thinker prefill_vision），串行执行。** Q 臂 Thinker 的 prefill_vision 每张图约 91 ms（10 张 910 ms），而一次性 prefill 3000 token 远小于此（vLLM-Omni Q 的整体 TTFT 620 ms）。
3. **M1 之后剩余的线性项在主机侧**：10 → 30 帧（全部缓存）时，预处理 +98 ms（约 5 ms/帧：解码、sha256、写上传文件），引擎 +255 ms（约 13 ms/帧），而 GPU 步数不变（Thinker prefill_vision 1 次）。被缓存服务的 walk 仍逐个经过调度。
4. 这是图调度的两面：按媒体拆成独立子图，使逐项复用（M1）只需约 60 行引擎改动；但每个子图有固定的调度开销，且媒体之间的 prefill 不合并。
