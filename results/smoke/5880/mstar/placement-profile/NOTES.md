# 编码器放 rank 2 时 Thinker prefill 变慢的原因（5880，2026-11-28）

现象（`../c2-profile/NOTES.md`）：编码器在 rank 2 时，48 帧冷请求的 Thinker 3 组 prefill 共 1.49 s，编码器在 rank 0 时 1.01 s；卡间传输时间相同。

诊断：`MSTAR_FWD_LOG`（engine `_forward` 前后各同步一次，记"待完成的已排队工作 / CPU 发射 / GPU 尾部"与是否回放 CUDA graph）。脚本 `scripts/probes/placement_fwd.sh NAME CONFIG`。

| 放置 | 16 张一组（约 4900 token）的 prefill | 每组 GPU 时间 |
|---|---|---|
| rank 0（`enc0-allgraphs/fwd.log`） | eager（4096–16384 token 的图捕获失败：显存被视觉编码器占用） | 330–380 ms |
| rank 2（`enc2-allgraphs/fwd.log`） | 回放 8192 token 的 CUDA graph（显存够，全部捕获） | 447–536 ms |

原因：大 prefill 的图按桶补齐（约 4900 → 8192 token）回放，比 eager 慢约 40%。不是放置本身。另：新服务第一次走 eager 大 prefill 有约 1.6 s 的一次性开销（`*-nowarm/`），计时前要先预热（`PROF_WARM=1`）。

修复：`..._enc2.yaml`、`..._enc2_c2.yaml` 加 `prefill_token_buckets: {Thinker: {prefill_vision: [128 … 2048]}}`，超过 2048 token 的步 eager。rank 0 的配置不加：加了以后省下的显存被分给 KV，rank 0 上视觉编码 16 张一组 OOM（`enc0-nowarm/`）。

修复后（预热后，客户端计时；服务端 `request_stats.log` 同向）：

| 配置 | 48 帧冷 | 41 帧冷 | 服务端 worker | Thinker 3 组 prefill |
|---|---|---|---|---|
| C1，编码器 rank 0（`enc0/`） | 2.64 s | 2.52 s | 2.24 s | 1.02 s |
| C1，编码器 rank 2（`enc2/`） | 2.56 s | 2.23 s | 2.19 s | 1.01 s |
| C1 + C2，编码器 rank 2（`enc2c2/`） | **1.90 s** | **1.77 s** | **1.53 s** | 1.02 s |

全缓存与增长 +1 帧不变（220–360 ms）。最终配置逐字对照两次与参照相同（`eq-c2final/`）。

影响：E2（`results/e2-multi`）的 enc2 变体用的是修复前的配置，它的 prefill 也回放了补齐的大图；E2 的 TTFT 对比因此对 enc2 偏不利，结论（每 token 时间）不受影响。
