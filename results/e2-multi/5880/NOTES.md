# E2：多个摄像头会话同时运行，编码器放置（5880，2026-11-28）

设计：DESIGN 18.5。脚本 `scripts/run_e2_multi_cuda.sh`（默认 CONC=3、REPS=2），客户端 `bench/stream_multi.py`，汇总 `summary.md`（`bench/stream_multi_summary.py`）。
每轮：先 1 个会话（无干扰参照），再 3 个会话同时运行（开始时间错开 2.7 s）。W 臂，grow，2 fps，640x480，提问在 8/16/24 s，回答最多 64 token（"Describe the most recent image in two sentences, starting with its main color."），temperature 0。共卡进程 0。

| 变体 | 服务 | GPU |
|---|---|---|
| vLLM-Omni | `configs/qwen3omni_vllm_omni_thinker.yaml`（thinker-only TP2；编码器在 Thinker 的 model runner 里） | 2 |
| M\* enc0 | M1+M3(16)+M3b+M4，`..._bigstep.yaml`：编码器在 rank 0（与 Thinker 同卡） | 3 |
| M\* enc2 | 同上，`..._bigstep_enc2.yaml`：编码器在 rank 2（与空闲的 Talker/Code2Wav 同卡） | 3 |

两个 M\* 变体卡数相同，只差一行 `node_groups`。

## 结果

| 变体 | 同时会话 | TTFT 中位 / p90 ms | 每 token 平均 ms（中位） | ITL p90 / p99 ms |
|---|---|---|---|---|
| vLLM-Omni | 1 | 105 / 141 | 8.6 | 9.3 / 10.5 |
| vLLM-Omni | 3 | 108 / 143 | 13.3 | 15.5 / 112 |
| M\* enc0 | 1 | 189 / 288 | 6.5 | 6.9 / 9.3 |
| M\* enc0 | 3 | 315 / 618 | 25.3 | 83 / 114 |
| M\* enc2 | 1 | 227 / 294 | 6.7 | 6.8 / 10.0 |
| M\* enc2 | 3 | 287 / 471 | 11.3 | 20 / 102 |

每格 6（单会话）或 18（3 会话）个提问。

## 正确性

- 编码器放置不改变输出：enc2 服务上 `scripts/probes/eq_frames.py` 两次与参照逐字相同（`../../smoke/5880/mstar/m4-profile/eq-enc2*.json`）。
- 本实验的长问题让模型更常说错主色（前 80 字符含期望颜色：vLLM-Omni 4/6、16/18；enc0 5/6、16/18；enc2 6/6、11/18）。逐条核对：每个错答说的都是该会话自己缓冲里更早的颜色（多为上一个场景），没有一条是别的会话的颜色，所以不是会话间串扰。错答集中在最后一个场景只有 2 帧的时刻。`stream_multi.py` 记录里的 `correct` 只看了前 40 字符，汇总已改为 80。

## 结论

1. **编码器独占一卡是 M\* 的配置级优势，效果明显**：3 个会话时回答的每 token 时间从 25.3 降到 11.3 ms（ITL p90 83 → 20 ms），与 vLLM-Omni（13.3 ms）相当或略好。别的会话的帧到达时，视觉编码不再与回答的 decode 抢同一张卡。vLLM-Omni 的编码器在 Thinker 同一步里运行，不能这样分开（要用 vLLM 的编码器分离部署，不在 vLLM-Omni 的 Qwen3-Omni 配置里）。
2. **单会话 decode M\* 更快**（6.5 对 8.6 ms/token），但 **TTFT 仍是 vLLM-Omni 的 2–3 倍**（多会话时 287–315 对 108 ms）：提问要排在别的会话的 warmup prefill 之后（16384 的大步预算让一个 prefill 步很长），且每请求仍重发全部帧。
3. p99 ITL 三者都在 100–115 ms：都是别的会话的 prefill 步插进来。M\* 可以再试 C2/C4（DESIGN 18.4），或给 prefill 单独的步预算（`can_override_mixed_max_tokens`），用更小的 prefill 块换 decode 平滑。
