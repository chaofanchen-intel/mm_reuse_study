# Smoke 记录：5880，M\* BAGEL（envs/mstar）

目的：确认 M\* 的 BAGEL 能在本机起服务并完成请求；确认 flash-attn 是否必需。

## 结论

- **flash-attn 不是 BAGEL 运行的必需项。** 缺失时 BAGEL 的 ViT 注意力回退到 SDPA，并打印一条 WARNING（`flash_attn is not available; BagelViT will fall back to torch SDPA`）。文本、图生文、文生图都能完成。
- 有 flash-attn 时 ViT 使用 flash 路径（`_FLASH_ATTN_AVAILABLE = True`）。本仓库的 smoke 只做了粗略计时（5 次 i2t，约 0.12–0.17 s 与 0.12–0.31 s），不能据此判断 ViT 的速度差异：重复的同一张图可能命中前缀缓存。ViT 单独的耗时要在阶段 0 画像中用关闭缓存的方式测。
- 既有实验 `exps/results/bagel-attn-backend_rtx5880.md`（基准仓库）显示：屏蔽 flash_attn 后端的 E2E 与保留时几乎相同（i2i 23,577 对 23,566 ms；i2t 2,290 对 2,299 ms；t2i 7,436 对 7,423 ms）。
- **保留 flash-attn** 是为了与基准仓库的最佳配置一致（`setup_envs_cuda.sh` 中安装）。

## 发现的问题

- 默认配置（`external/mstar/configs/bagel_single_gpu.yaml`，未设 `gpu_memory_fraction`）下，CUDA 图捕获会占满 GPU（实测 48.5 GB / 49 GB），随后的文生图在 VAE 解码时报 `CUDNN_STATUS_INTERNAL_ERROR` 或 OOM。M\* 文档（`docs/mintlify/serving.mdx`）说明：未设置时捕获按"当前空闲显存"规划。
- 本仓库的 smoke 配置 `configs/bagel_single_gpu_smoke.yaml` 增加了 `gpu_memory_fraction: 0.8`，文生图即可完成。**这是 smoke 用的配置，不是实验配置**，正式实验前要确定统一的显存设置并写入设计文档。

## 运行记录

| 臂 | GPU | 端口 | 服务日志 | 文本 | 图生文（i2t） | 文生图（t2i 512） |
|---|---|---|---|---|---|---|
| flash-attn 可用 | 1 | 18202 | `results/smoke/bagel-flash-gpu1.server.log` | 通过 | 通过（输出 "red"） | 通过，`results/smoke/t2i_flash.png` |
| flash-attn 屏蔽 | 2 | 18203 | `results/smoke/bagel-noflash-gpu2.server.log` | — | 通过（输出 "Red."） | 通过，`results/smoke/t2i_noflash.json` |

屏蔽方式：`scripts/smoke_noflash/flash_attn/__init__.py` 抛出 ImportError，经 `PYTHONPATH` 作用于所有子进程（包括 worker，ViT 实际在 worker 中运行）。服务日志中两臂的差异可以用 `flash_attn is not available` 这条 WARNING 确认（屏蔽臂 3 条，flash 臂 0 条）。

## 遗留

- 第一次尝试（GPU 0，端口 18201）的服务器仍被记录为运行中；其 shutdown 日志见 `results/smoke/bagel-flash.server.log`。
- 服务进程（端口 18202、18203）仍在运行，需要人工停止（见下方 PID）。
- 结果目录 `results/` 与 `configs/` 中的文件：`results/` 不进 git，`configs/bagel_single_gpu_smoke.yaml` 进 git。
