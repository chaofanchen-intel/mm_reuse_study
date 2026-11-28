# M\* C2：编码提前、与 Thinker prefill 重叠（5880，2026-11-28）

设计：DESIGN 18.4 的 C2。开关 `MSTAR_C2=1`（需要 M1；本轮同时开 C1、M3(16)、M3b、M4），配置 `configs/qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml`。
探针同 `../c1-profile/`（`eq_frames.py`、`cached_frames_prof.py 8,48 17,41`）。

## 做法（用图的分区并发）

- 模型多一个分区 `VisionAhead`，计划是每个图像组一个"只编码"的 walk（`vision_ahead`），出边指向空目的地。
- `vision_ahead` 是第二个 ViT 节点（独立权重副本约 1.3 GB），与 `vision_encoder` 同在 rank 2 的进程里，两者共享 M1 编码输出缓存（`EncoderOutputCache.share`）。
- 请求进入时，Thinker 分区与 VisionAhead 分区同时开始。VisionAhead 依次编码各组、写入缓存；Thinker 自己的图像 walk 到某组时，编码器直接命中缓存。
- 已在缓存里的组，ahead walk 立即结束，不把缓存拷回 GPU；C1 已覆盖的图同样跳过（ahead 只读登记，不删）。

第一版只让 ahead 编码第 1 组之后的组（第 0 组留给 Thinker 的 walk）。结果卡 2 的 worker 先调度了 ahead 的第 1 组，Thinker 的第 0 组排在后面，首组 prefill 晚开始 0.39 s（`server-c2-skip0/`）。改为 ahead 也编码第 0 组，谁先到谁编码、另一个命中（`server-c2/`）。

## 正确性

`eq-c2.json`、`eq-c2b.json`：8 个提问与参照 `../q3-m3/eq-g3pp.json` 逐字相同。

## 48 帧冷请求（服务端 `request_stats.log`）

| 配置（放置） | 请求总计 ms | worker ms | Thinker 3 组 prefill ms | 编码在关键路径上 |
|---|---|---|---|---|
| M4（编码器在 rank 0，`../m4-profile/server-m4memo-g16/`） | 2753 | 2377 | 1010 | 3 组全部 |
| + C1（rank 0，`../c1-profile/server-c1b/`） | 2586 | 2218 | 1032 | 3 组全部 |
| + C1（rank 2，`server-enc2-c1only/`） | 3169 | 2765 | 1492 | 3 组全部 |
| + C1 + C2（rank 2，`server-c2/`） | 2682 | 2182 | 1483 | 只有第 0 组（0.37 s） |

时间线（`server-c2/c1.log`）：ahead 编码第 0 组 0.37 s → Thinker 的第 0 组命中缓存，prefill 开始；ahead 第 1、2 组在 Thinker prefill 第 0、1 组时编完，Thinker 到第 1、2 组都命中。

客户端：48 帧冷 3.22 → 2.73 s；41 帧冷 2.63 → 2.26 s（同为 rank 2 放置）。全缓存与增长 +1 帧的请求不变（C2 只影响要编码的请求）。

## 结论

- 同一放置下，C2 让 48 帧冷请求少 0.49 s（15%，worker 时间少 21%）：三组编码只剩第一组在关键路径上，其余被 Thinker prefill 盖住。这是"按图里真实依赖重排"的直接收益，vLLM-Omni 的编码器与 Thinker 在同一步里，不能这样重叠。
- 但编码器放在 rank 2 时，Thinker 的 prefill 本身慢约 50%（3 组 1.49 s 对 1.01–1.03 s）。卡间传输时间两种放置相同（每组约 4 ms），原因未查清。于是 C2 后的绝对值（2.68 s）只与编码器在 rank 0 的 C1（2.59 s）持平。
- 编码器在 rank 0 时用不了 C2：两个 ViT 节点在同一 worker 进程里按步轮流执行，且与 Thinker rank 0 同卡；放到不同进程又不能共享缓存。要两者兼得，需要跨进程的编码输出缓存，或查清 rank 2 放置下 Thinker 变慢的原因。
- 改动规模：模型约 40 行（分区、节点、计划）、编码器约 30 行、缓存共享约 25 行、配置 1 行。没有改引擎和调度器——这一项是图的形式"天然适合"的直接证据：并行只是多声明一个分区。
