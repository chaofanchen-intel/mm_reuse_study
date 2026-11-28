# M*：全部已缓存时的请求还剩什么（M4 与大分组，5880，2026-11-28）

起点：G3 的 W 臂，M1 + M3（N=3）+ M3b 之后，提问 TTFT 仍是 0.36–0.83 s（vLLM-Omni 0.1–0.15 s）。
探针：`scripts/probes/cached_frames_prof.py`（同一组帧连发：冷一次、全缓存三次；可选"增长"：n 帧后 n+1、n+2、n+3 帧）；
`--log-stats`（`request_stats.log` 每请求时间线）；`MSTAR_PREPROCESS_LOG`（data worker 分段计时）；`encoder_cache.log`（每个视觉 walk 的键/查找耗时）。
640x480，`max_tokens=1`，只有帧（无问题文本）。

## 1. 起点的拆分（`server-m1m3pp/`、`server-m1m3pp-b/`）

48 帧、全部已缓存，一次请求 830–900 ms：

| 段 | ms | 内容 |
|---|---|---|
| data worker：解码 JPEG | 102–123 | 48 个文件解码成 fp32 张量 |
| data worker：process_prompt | 120–128 | 其中图像循环 66 ms（M3b 全部命中，仍要 fp32→uint8 转换与哈希） |
| data worker：存张量 | 88–154 | **每请求写 531 MB**：解码图像 169 MB（没有节点读）+ fp32 像素 338 MB |
| 引擎：16 个视觉 walk | 408 | 每个约 25 ms，**依次执行**；其中键哈希 8 ms（读 21 MB 像素）；输出 225 MB 再送给 TP rank 1 |
| 引擎：GPU 实际工作 | 57 | 最后一组的 Thinker prefill + 1 步 decode |

全部已缓存，GPU 只忙 57 ms；其余都是"为不需要做的工作搬数据、排调度"。

## 2. 改动与效果

| 配置 | 8 帧全缓存 | 48 帧全缓存 | 48 帧冷 | 输出与基线 |
|---|---|---|---|---|
| M1 + M3(3) + M3b（起点） | 186–193 | 828–936 | 4057–4337 | — |
| + M4 键随图发送、bf16 像素、不存解码图（`server-m4/`） | 149–152 | 550–586 | 3611 | 逐字相同 ×2 |
| + M4 文件记忆（同一文件字节跳过解码与处理，`server-m4memo/`） | 123–133 | 339–431 | 3613 | 逐字相同 ×2 |
| + 每步预算 16384、每组 ≤ 16 张（`server-m4memo-g16/`） | 112–127 | **276–319** | **2809** | 逐字相同 ×2 |

逐字相同：`scripts/probes/eq_frames.py`，8 个提问（3/8/8/20 张图，temperature 0，48 token），对照 `../q3-m3/eq-g3pp.json`。

M4 之后（16 张一组）的 48 帧全缓存请求：data worker 约 70 ms（解码 4、prompt 24、键 5、存 35 ms，177 MB），引擎 3 个视觉 walk + 最后的 prefill。

增长（同一会话每次多一帧，前缀在组内切开，KV 日志 "matched 99 of its 101 pages"）：17–19 帧 272–300 ms，41–43 帧 349–386 ms。
41 帧时：预处理 77 ms，视觉编码 1 张新图 + 10 张缓存 44 ms，Thinker prefill 未覆盖的尾部 95 ms。

## 3. 大分组为什么之前失败（M3 的 N ≥ 6）

Thinker 每步默认预算 1024 token（`ThinkerSubmodule.MAX_BATCH_TOKENS`）。3 张一组约 920 token，不分块；6 张约 1830 token，被分块 prefill。
分块 + 前缀缓存 + TP2 的路径上，第一次请求的页没有进入索引（第二次请求"matched 0 of its 19 pages"），之后一次部分租约让 rank 1 报"stream here already holds tokens"，TP 分歧退出。
预算放大到 16384（`configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml`）后分组行不再分块，N = 16 正常，N = 64（48 帧一组）时视觉编码一次 5.8 万 patch，GPU 0 OOM（`server-m4memo-g64-oom48/`，8 与 20 张一组正常）。
分块路径本身的问题没有修（上游行为；单图不超过 1024 token 时不触发）。代价：大 prefill 步会挡住同一步里其他请求的 decode，多会话时要另外看。

## 4. 结论

- M* 的剩余差距主要不在 GPU，而在"已缓存的东西仍被搬运、仍被逐个调度"。M4 这几项都是 vLLM 已有的（处理器缓存、前端算 mm_hash 随请求发、只传需要的特征），属于功能对齐，不是图调度的特性。
- 图调度特有的部分是 walk 的粒度：每个 walk 一次往返（约 12 ms，即使整段被跳过）。把 walk 变粗（16 张一组）比任何数据层面的优化都有效（340–430 → 276–319 ms，冷请求 3.6 → 2.8 s），但受两个上游限制：分块 prefill 与前缀缓存、TP 的组合，以及视觉编码器的显存规划只按单次调用的上限。
