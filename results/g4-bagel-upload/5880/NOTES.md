# G4：BAGEL "先上传图像、稍后提问"（5880，2026-11-26）

场景：每个会话一张新图（方形 512 或 980），连问 3 个不同问题（都问主色，答一个词）。
- Q 臂：提问时才发 图像 + 问题；
- W 臂：上传时先发一个 `max_tokens=1` 的预热请求（图像 + "Look at this image."），2 s 后开始提问。
脚本 `scripts/run_g4_bagel_upload_cuda.sh`，客户端 `bench/upload.py`，汇总 `summary.md`（`bench/upload_summary.py`）。temperature 0，max_tokens 16。

| 变体 | 服务 | GPU |
|---|---|---|
| M* 基线 | `configs/bagel_mstar_i2t.yaml` | 1 |
| M* M1 | 同上 + ViT 输出缓存 1024 MB（`MSTAR_ENCODER_CACHE_MB`） | 1 |
| vLLM-Omni | `configs/bagel_vllm_omni_audit.yaml`（前缀缓存开） | 2 |

## 目录与有效性

| 目录 | 有效性 |
|---|---|
| `mstar/cache0/`、`mstar/cache1024/`、`vllm-omni/pfx1/` | 有效，每格 8 个会话，全部回答正确，共卡进程 0 |
| `mstar/cache0-imageonly-warmup/` | 无效：预热只含图像时，提问的前缀与预热不一致（token 1383 对 1408），第 1 问 0 命中。之后预热加上固定文本 |
| `mstar/cache1024-failed-dynamo/` | 无效：M1 的锁页拷贝在 BAGEL ViT 的 torch.compile 内，Dynamo 不支持 `pin_memory`。已加 `torch.compiler.disable` |
| `mstar/probe-cache1024/` | 修复后的探针（确认命中） |

最初的 `summary.md` 把两个无效目录也汇总进去（n=14），已改为只汇总三个有效目录并重新生成。

## 结果（TTFT 中位数 ms；预热行为 E2E）

| 分辨率 | 请求 | M* 基线 | M* M1 | vLLM-Omni |
|---|---|---|---|---|
| 512 | Q 第 1 问（冷） | 282 | 284 | 263 |
| 512 | Q 第 2、3 问 | 101–104 | 65–71 | 67–69 |
| 512 | W 第 1 问 | 107 | 81 | 71 |
| 512 | W 预热（上传时，用户不等待） | 289 | 292 | 257 |
| 980 | Q 第 1 问（冷） | 851 | 877 | 1052 |
| 980 | Q 第 2、3 问 | 172–175 | 137–142 | 104–105 |
| 980 | W 第 1 问 | 171 | 139 | 106 |
| 980 | W 预热 | 842 | 883 | 1052 |

第 2、3 问的视觉编码：M* 基线 8/8 次重跑 ViT（前缀已命中，G1 的发现），M1 与 vLLM-Omni 0/8。

## 结论

1. **到达即算对 BAGEL 同样是最大收益**：第 1 问 TTFT，512 时 −62%（M* 基线）/ −71%（M1）/ −73%（vLLM-Omni），980 时 −80% / −84% / −90%。代价是上传时多一次与冷请求等长的预热（用户不等待）。
2. **M1 去掉了 M* 的重复 ViT**：重复提问 512 时 101 → 68 ms，980 时 173 → 140 ms，与 vLLM-Omni 的差距在 512 消失，980 剩约 35 ms。
3. 冷请求 980 时 M* 比 vLLM-Omni 快（851 对 1052 ms，1 卡对 2 卡），这一点与复用无关。
4. 980 时剩下的 35 ms 尚未拆分（候选：ViT 输出从主存拷回、M* 每请求的预处理与传输，同 Qwen3-Omni 的 M4 分析）。B70 上 BAGEL 是唯一能跑的模型，下一步在 B70 上重复本实验。
