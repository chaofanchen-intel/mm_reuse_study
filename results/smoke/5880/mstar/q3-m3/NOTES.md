# M* M3：减少每帧的调度与预处理开销（5880，2026-11-26）

起点（G3 的 M1）：全部被缓存的 32 帧请求，GPU 步时间约 55 ms，但引擎时间 514 ms、预处理 200 ms。
`--log-stats-file`（`server-m1-stats/request_stats.log`）：conductor 接收 → 首 token 525 ms，worker 活跃仅 60 ms。
320x240（数据量 1/4）时引擎时间 467 ms：开销主要不是数据量，而是每张图一对 walk（视觉编码 → Thinker prefill_vision）逐个经过调度，约 13 ms/图。

## 改动（patches/README.md）

- M3：相邻图像最多 N 张合成一个 walk（`MSTAR_GROUP_IMAGES=N`）。
- M3b：图像预处理缓存（`MSTAR_PREPROCESS_CACHE_MB`）。

## 正确性

`eq-*.json`：8 个提问（3/8/8/20 张图，temperature 0，48 token），与不分组（`eq-ungrouped.json`）**逐字相同**：N=3、N=3+M3b（两次）。
N=6 与不限（`server-m1m3g6-failed/`、`server-m1m3-unbounded-failed/`）：第二次请求前缀命中 0，随后 "TP ranks diverged" 崩溃。原因未查明（推测与长行的分块 prefill 或前缀链有关），因此 N 固定为 3。

## 效果

全部被缓存的请求（`/tmp/m3prof.py`，640x480）：

| | 8 帧 引擎 / 预处理 ms | 32 帧 引擎 / 预处理 ms |
|---|---|---|
| M1 | 150 / 45 | 514 / 178 |
| M1 + M3（N=3） | 120 / 50 | 253 / 207 |
| M1 + M3 + M3b | 115 / 58 | 333 / 200（图像循环 94 → 47 ms；其余在文件读取/解码、摘要、写 7 MiB/帧的 fp32 像素） |

摄像头流 audit（`records-m1m3g3pp.jsonl`，与 G3 的 M1 audit 对比，TTFT ms）：

| | M1（G3） | M1 + M3 + M3b |
|---|---|---|
| W grow 16/32/48 | 410 / 735 / 970 | 356 / 549 / 827 |
| Q grow 16/32/48 | 2358 / 2677 / 3100 | 1350 / 1571 / 1889 |
| W slide 40/31/47/38 | 875 / 659 / 884 / 701 | 714 / 628 / 757 / 725 |
| Q slide 40/31/47/38 | 5334 / 3699 / 2875 / 4414 | 3047 / 2085 / 1942 / 2471 |

结论：分组（改变子图粒度）主要省 Q 臂的冷 prefill（每张图独立 walk 的 Thinker prefill 约 91 ms/图），−40%；W 臂 −13~25%。剩余差距（W 0.36–0.83 s 对 vLLM-Omni 0.1–0.15 s）在预处理与像素传输（每请求都重发全部帧的 fp32 像素，即使编码器输出已缓存）以及 N=3 分组仍有的每组调度。
