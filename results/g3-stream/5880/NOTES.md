# G3：Qwen3-Omni 摄像头流 + 随时提问（5880，2026-11-26）

设计：DESIGN 第 16 节。脚本：`scripts/run_g3_stream_cuda.sh`。客户端：`bench/stream.py`（640x480 JPEG，2 fps，temperature 0，max_tokens 16）。汇总：`summary.md`（第一轮全部臂）、`summary-wfix.md`（W 臂 timing 重跑）。

| 变体 | 服务 | GPU |
|---|---|---|
| vLLM-Omni | `configs/qwen3omni_vllm_omni_thinker.yaml`（thinker-only TP2，CUDA graph，max_num_seqs 4） | 2 |
| M* 基线 | `configs/qwen3omni_mstar_thinker_tp2.yaml` + M0（`MSTAR_SPEECH_NODES_EAGER=1`）+ 媒体上限 64；不开 M1 | 3 |
| M* M1 | 同上 + M1 编码器输出缓存 2048 MB（锁页主存） | 3 |

负载：grow（缓冲只增，≤ 50 帧，提问在 8/16/24 s）；slide（满 50 帧后丢最旧一半，提问在 20/28/36/44 s）。每种组合 audit 1 会话 + timing 4 会话。

## 目录与有效性

| 目录 | 内容 | 有效性 |
|---|---|---|
| `vllm-omni/stream/`、`mstar/stream-cache0/`、`mstar/stream-cache2048/` | 第一轮：audit 与 timing（Q、W） | Q 臂与 audit 有效。**W 臂 timing 无效**：客户端取消 warmup 只在读到下一行时生效，而 `max_tokens=1` 的 warmup 在 prefill 结束前不发数据，取消要等 warmup 做完（M* 的 warmup 长于帧间隔，提问总被挡）。已修（`Cancel` 直接关闭套接字，测试 `test_stream_cancel_is_prompt`） |
| `mstar/stream-cache2048-failed-gpu-oom/` | M1 第一版把缓存放在 GPU，rank 0 OOM（M* 的显存规划不知道这块缓存） | 无效 |
| `*-wfix/` | 修复取消后，W 臂 timing 重跑（`RUN_TAG=-wfix ARMS_TIMING="W W" SKIP_AUDIT=1`） | 有效 |

所有记录的 `gpu_foreign_procs` 为空，只有无效的 OOM 轮有一个他人/未知进程（1.6 GB）。

## 结果

### 提问到首 token，timing（真实节奏），中位数 ms

| 负载 / 缓冲帧数 | vLLM-Omni Q | vLLM-Omni W | M* 基线 Q | M* 基线 W | M* M1 Q | M* M1 W |
|---|---|---|---|---|---|---|
| grow 16 | 915 | 105 | 2041 | 900 | 2191 | 345 |
| grow 32 | 999 | 124 | 2829 | 1782 | 2576 | 866 |
| grow 48 | 1046 | 152 | 3546 | 2819 | 2980 | 1178 |
| slide 40（首次满） | 2336 | 130 | 5137 | 2298 | 5357 | 1116 |
| slide 31（丢帧后） | 1233 | 120 | 3947 | 3900 | 3699 | 1684 |
| slide 47 | 1064 | 140 | 3553 | 2934 | 2956 | 1226 |
| slide 38（丢帧后） | 1404 | 134 | 4863 | 2571 | 4418 | 1001 |

Q 与 W 的数据分别来自第一轮与 `-wfix` 轮。正确率：除一次 audit（期望 red，答 Orange）外全部正确。

### 单次提问的工作量（audit：提问前 warmup 已完成）

| 负载 / 帧数 | vLLM-Omni W：编码帧 / TTFT | M* 基线 W | M* M1 W |
|---|---|---|---|
| grow 16 / 32 / 48 | 0 / 132、152、168 ms | 16、32、48 帧 / 809、1492、2224 ms | 0 / 410、735、970 ms |
| slide 40 / 31 / 47 / 38 | 0 / 135–165 ms | 全部帧 / 1463–2151 ms | 0 / 659–884 ms |

前缀命中两边都覆盖到最后一帧（M* 14464/14605，vLLM-Omni 14464/14536）。

### 每会话的编码工作（timing，W 臂，`-wfix`）

| | 发送帧数 | 编码帧数 | 编码器 GPU ms |
|---|---|---|---|
| vLLM-Omni grow / slide | 48 / 88 | 48 / 88 | （count 模式未计时） |
| M* 基线 grow / slide | 48 / 88 | 442 / 694 | 13186 / 21042 |
| M* M1 grow / slide | 48 / 88 | 48 / 88 | 1877 / 3263 |

## 结论

1. **到达即算（W）对两边都是最大的一项收益**：vLLM-Omni 提问 TTFT 从 0.9–2.3 s 降到 0.1–0.15 s（6–18 倍）；M*（M1）从 2.2–5.4 s 降到 0.35–1.7 s。vLLM-Omni 已内置此能力（WS 端点），本实验用客户端以相同逻辑在两边实现。
2. **M* 的缺口是"被前缀覆盖的帧仍重新编码"**：基线 W 每会话编码 442 帧（只来了 48 帧），编码器 GPU 时间 13.2 s；M1 降到 48 帧、1.9 s（−86%），提问 TTFT 降 52–68%。这是 G1 上"前缀命中仍跑 ViT"的放大版：缓冲越长越严重。
3. **滑动窗口检验编码器输出复用**：丢帧后前缀失效（命中 0），vLLM-Omni Q 臂只编码新帧（31 帧中 16 帧），TTFT 1233 ms 对比冷启 2336 ms；M* M1 Q 同样只编码 16 帧。W 臂在丢帧后由下一个 warmup 重新 prefill，提问不受影响。
4. **M1 之后 M* 仍比 vLLM-Omni 慢一个数量级（W：0.35–1.7 s 对 0.1–0.15 s）**，原因不在编码器：
   - 每张图像是独立的一对 walk（视觉编码 → Thinker prefill_vision），即使全部由缓存服务也逐个调度：约 13 ms/帧主机侧 + 约 5 ms/帧预处理（smoke 测得）。
   - M* 的 warmup 因此长于帧间隔（0.5 s），提问到达时总有 warmup 在途；取消后提问仍要等当前步结束。
   - Q 臂的 Thinker prefill 每张图约 91 ms（独立 walk、不合并），vLLM-Omni 一次 prefill 全部未命中的 token。
5. **两种架构的对照**：
   - 图调度（M*）：按媒体拆成独立子图，使"逐项复用、命中即跳过"的改动小（M1 约 150 行，引擎已有"缓存服务的行"）；代价是每个子图的固定调度开销与 prefill 不合并，在多媒体项（几十帧）时成为主导。
   - 阶段流水线（vLLM-Omni）：编码器在调度器内按未计算 token 决定是否运行，编码器缓存按 mm_hash 全局共享，覆盖即跳过是内置的；不足在音频（提问时才编码）与音频输出时关闭 warmup（DESIGN 16.2）。

## M4 轮（2026-11-28，`mstar/stream-m4/`，汇总 `summary-m4.md`）

M\* 配置：M1 + M3（每组 ≤ 16 张）+ M3b + M4，`configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml`（Thinker 每步预算 16384），3 卡；`PHASES=mstar4`。画像与改动见 `../../smoke/5880/mstar/m4-profile/NOTES.md`。
vLLM-Omni 与 M\* M1 的数据沿用上面各轮（同一客户端）。全部回答正确，共卡进程 0。

| 负载 / 缓冲帧数 | vLLM-Omni Q | vLLM-Omni W | M\* M1 Q | M\* M1 W | **M\* M4 Q** | **M\* M4 W** |
|---|---|---|---|---|---|---|
| grow 16 | 915 | 105 | 2191 | 345 | 835 | 166 |
| grow 32 | 999 | 122 | 2576 | 932 | 994 | 194 |
| grow 48 | 1046 | 152 | 2980 | 1299 | 1119 | 280 |
| slide 40（首次满） | 2336 | 128 | 5357 | 1116 | 2023 | 243 |
| slide 31（丢帧后） | 1233 | 119 | 3699 | 1805 | 1161 | 215 |
| slide 47 | 1064 | 140 | 2956 | 1239 | 1088 | 275 |
| slide 38（丢帧后） | 1404 | 133 | 4418 | 964 | 1366 | 235 |

（TTFT 中位数 ms，timing；M\* M1 W 取 `-wfix` 轮。）

- Q 臂：M\* 与 vLLM-Omni 持平，首次满窗（40 帧冷）快 13%。原因是 16 张一组的 walk 把 Thinker prefill 合并了（上一轮每张图一个 prefill，约 91 ms/图）。
- W 臂：M\* 从 0.35–1.8 s 降到 0.17–0.28 s，仍是 vLLM-Omni 的 1.6–1.9 倍。提问时 warmup 不再在途（warmup 短于帧间隔）。
- 剩余差距（audit：48 帧 294 对 168 ms）在每请求仍重发全部帧（data worker 约 70 ms）与 3 个已缓存 walk 的往返。

## 下一步

- ~~M3、M4~~（见上）；~~V1、语音提问~~（`../../smoke/5880/vllm-omni/v1-audio/NOTES.md`）。
- 多会话并发与编码器放置：E2（`../../e2-multi/5880/`，DESIGN 18）。

## C1 轮（2026-11-28，`mstar/stream-c1/`，`summary-c1.md`）

M4 段配置 + `MSTAR_C1=1`（前缀已覆盖的图不编码、不送像素；开头文本单独一步，见 `../../smoke/5880/mstar/c1-profile/NOTES.md`）。脚本段 `mstar5`。汇总：`python -m bench.stream_summary vllm-omni/stream-wfix/records.jsonl mstar/stream-m4/records.jsonl mstar/stream-c1/records.jsonl`。

| 提问 TTFT 中位 ms（W 臂） | grow q0/q1/q2（16/32/48 帧） | slide q0–q3 |
|---|---|---|
| vLLM-Omni | 105 / 124 / 152 | 130 / 120 / 140 / 134 |
| M\* M4 | 166 / 194 / 280 | 243 / 215 / 275 / 235 |
| M\* M4 + C1 | 158 / 195 / 231 | 214 / 182 / 227 / 201 |

W 臂少 0–50 ms（帧越多越多），与 vLLM-Omni 的差距缩小约三分之一；Q 臂（不预热）不变，在预期内（没有被覆盖的图）。56 个提问全部正确。

## C2 轮（2026-11-28，`mstar/stream-c2/`，`summary-c2.md`）

C1 段 + `MSTAR_C2=1`，配置 `qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml`（编码器与 vision_ahead 在 rank 2；图像 prefill 只捕获到 2048 token，见 `../../smoke/5880/mstar/placement-profile/NOTES.md`）。脚本段 `mstar6`。

| 提问 TTFT 中位 ms | grow q0/q1/q2 | slide q0/q1/q2/q3 |
|---|---|---|
| Q 臂（不预热）vLLM-Omni | 915 / 999 / 1046 | 2336 / 1233 / 1064 / 1404 |
| Q 臂 M\* M4 | 835 / 994 / 1119 | 2023 / 1161 / 1088 / 1366 |
| Q 臂 M\* M4+C1+C2 | 851 / 960 / 1081 | **1697** / 1154 / 962 / 1305 |
| W 臂 vLLM-Omni | 105 / 124 / 152 | 130 / 120 / 140 / 134 |
| W 臂 M\* M4+C1+C2 | 156 / 183 / 232 | 205 / 184 / 237 / 206 |

Q 臂：M\* 在 7 个提问点中 6 个快于 vLLM-Omni；第一个满窗（40 帧全新）1697 对 2336 ms（−27%），比 M4 再快 16%。grow 的 Q 臂每次只有 16 帧是新的（前缀覆盖其余），C2 能重叠的编码少，变化在噪声内。W 臂与 C1 轮相同（C2 只影响要编码的请求）。56 个提问全部正确。

## C4 轮（2026-11-28，`mstar/stream-c4/`，`summary-c4.md`）

C2 段 + `MSTAR_C4=1`（`MSTAR_C4_KEEP=4`），客户端 `--c4`：已送达的帧只发引用（`MSTAR-FRAME:<xxh3-128>`，约 60 字节），请求顶层 `c4_warm` = 与最近两个成功请求中较早那个的公共帧前缀 − 1；这些帧并成一组由 C1 跳过，除最后 4 张外不带像素。脚本段 `mstar7`。第一次运行在会话中途被打断（70 个提问做完 42 个，保留在 `mstar/stream-c4-cut/`），本节为完整重跑。
汇总：`python -m bench.stream_summary vllm-omni/stream-wfix/records.jsonl mstar/stream-c2/records.jsonl mstar/stream-c4/records.jsonl`。

| 提问 TTFT 中位 ms | grow q0/q1/q2（16/32/48 帧） | slide q0/q1/q2/q3（40/31/47/38 帧） |
|---|---|---|
| W 臂 vLLM-Omni | 105 / 124 / 152 | 130 / 120 / 140 / 134 |
| W 臂 M\* M4+C1+C2 | 156 / 183 / 232 | 205 / 184 / 237 / 206 |
| W 臂 M\* M4+C1+C2+C4 | 126 / **112** / **120** | **123** / **116** / **118** / **130** |
| Q 臂 vLLM-Omni | 915 / 999 / 1046 | 2336 / 1233 / 1064 / 1404 |
| Q 臂 M\* M4+C1+C2 | 851 / 960 / 1081 | 1697 / 1154 / 962 / 1305 |
| Q 臂 M\* M4+C1+C2+C4 | 878 / 984 / 1075 | 1694 / 1182 / 1008 / 1262 |

- W 臂：M\* 的提问 TTFT 不再随帧数增长（112–130 ms，16 帧到 48 帧几乎相同），7 个提问点中 6 个快于 vLLM-Omni（唯一慢的是 16 帧：126 对 105）。比 C2 轮少 30–119 ms。原因：48 个已预热帧合成一个 walk（C1 整组跳过），而不是 3 个 16 帧组各走一次；请求体 2.6–7 KB（C2 轮约 4.8 MB）。
- Q 臂：在噪声内不变，符合预期（提问前没有预热，帧大多是新的，只有前一次提问发过的帧走引用）。
- 688 个请求（预热 + 提问）中没有一次因提示不成立而重发；70 个提问全部正确。
- 本轮的提示规则（取较早的请求、保留最后 4 张图的像素）后来查明只是在补引用键的错误（已修复）：按最近一个请求提示、全部不带像素时 283 个请求 0 次重发，TTFT 相同（`../../smoke/5880/mstar/c4-profile/NOTES.md`）。现在默认是后者；脚本段 `mstar7` 固定用本轮的规则，以便复现。
