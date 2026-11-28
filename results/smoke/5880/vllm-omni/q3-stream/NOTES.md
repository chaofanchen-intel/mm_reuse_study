# vLLM-Omni Qwen3-Omni 摄像头流 smoke（2026-11-26，5880，GPU 2,3）

配置：`configs/qwen3omni_vllm_omni_thinker.yaml`（thinker-only，TP2，CUDA graph 开，max_num_seqs 4）。客户端 `bench/stream.py`，640x480 JPEG，2 fps，temperature 0。

启动日志：编码器缓存 62720 token（与 CPU 推导一致）；KV 9.32 GiB/卡；每卡占用 32.6 GiB（权重 + 非 torch）。

| 目录/文件 | 内容 | 状态 |
|---|---|---|
| `server-run1/`、`records-before-flush-fix.jsonl`、`records.jsonl` | 第一轮 Q/W audit | 计数有误：W 臂的单次提问计数混入了此前两个 warmup（见下） |
| `server/`、`records-run2.jsonl` | 加 `prefix_lookup` shim 后的 W audit（grow）与 Q/W timing（slide，max_frames 20） | 有效 |

发现：
1. **vLLM 的聚合前缀计数器在请求紧接请求时延迟发布**：连续三个请求（中间无空闲）后，只有第一个的计数出现；后两个的计数要等下一个请求到来才发布（复现脚本见会话记录，结果：`b2b 3,8,9 +5s delta 934` → 下一个请求后 `8238`）。等待 2 s 不能解决。按请求的命中改用 shim 的 `prefix_lookup`（`KVCacheManager.get_computed_blocks`）。G1 的 runner 每个请求之间有 ≥ 2 s 空闲且请求不重叠，不受此影响（待 G1 数据核对）。
2. W（warmup）臂：提问时编码器 0 次，前缀命中 = prompt − 最后不满一块的部分（3008/3064、6016/6080、9024/9100）。TTFT 约 100–140 ms；Q 臂 620–820 ms。
3. Q 臂在同一会话内仍有前缀命中（上一次提问的帧），只编码新帧；slide 负载丢帧后前缀失效，TTFT 回到 730–820 ms。
