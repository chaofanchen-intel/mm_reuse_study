# 环境：B70（XPU）

适用机器：本机，3 × Intel Arc Pro B70（PCI `0000:18:00.0`、`0000:54:00.0`、`0000:cc:00.0`）。**与 5880（CUDA）互不混用，见 [README.md](README.md)。**

## 与 5880（CUDA）的区别

| 项 | 5880（CUDA） | B70（XPU） |
|---|---|---|
| 框架 | M\*、vLLM-Omni（同一源码版本） | 同左 |
| torch | M\*：`2.10.0+cu130`；vLLM-Omni：`2.14.0+cu130` | M\*：`2.13.0+xpu`；vLLM-Omni：`2.15.0+xpu`（两个环境各自独立，不共用） |
| vLLM | `vllm==0.34.0`（PyPI，cu130） | `vllm==0.34.0+xpu`（`wheels.vllm.ai/0.34.0/xpu`） |
| 注意力内核 | FlashInfer、flash-attn 2.8.3 | M\* 用 `xpu_paged` 后端；vLLM 用 `vllm-xpu-kernels`；**无 flash-attn**（BAGEL ViT 回退到 SDPA） |
| 工具链 | nvcc 13.0.88 | 不需要单独的编译器；oneAPI 运行时随 XPU 轮子安装 |
| BAGEL 的 extra | `.[bagel]` | `.[bagel_xpu]`（含 `vllm-xpu-kernels==0.1.15.4`） |
| 视频/音频解码 | TorchCodec（CUDA） | TorchCodec `0.16.0+cpu` + `torchcodec-xpu==0.16.0`；需要 FFmpeg（`envs/ffmpeg-xpu`） |
| 两阶段连接器 | 未对比 | Mooncake 可用（TCP）；**NIXL 不可用**（UCX 不能注册 XPU 显存） |
| 部署配置 | `configs/bagel_single_gpu_smoke.yaml`（单卡） | `external/mstar/configs/bagel_xpu_tp2.yaml`（3 张卡：卡 0 编码器与解码器，卡 1–2 张量并行 LLM） |
| 模型缓存 | `models/` → `Inference_Engine_Investigation_Repo/models` | `models/` → `~/workspaces/models`（缓存在 `models/cache`） |

## 路线选择

采用**原生安装**（uv），不用 Docker。理由：

- 本机已有用同一套 M\* 源码（commit `2c8ffde8`）验证过的原生环境（`~/workspaces/mstar_xpu/.venv`，2026-11-18），版本与做法可直接沿用。
- 上游 vLLM-Omni 的 XPU Dockerfile 依赖 `vllm/vllm-openai-xpu:v0.34.0` 镜像，镜像与 vLLM-Omni 的 XPU 运行时验收尚未完成（上游文档写明 "XPU runtime acceptance remains pending"）。Docker 方案保留作备用。

## 源码

固定版本写在 [externals.pins](../externals.pins)，两台机器相同：

- `external/mstar`：commit `2c8ffde8`（`main` 最新提交，2026-11-25 核对）
- `external/vllm-omni`：tag `v0.34.0`（`06a28821`）。这是目前最新的 vLLM-Omni 正式发布（`ls-remote` 核对，没有更高的 v0.3x 正式 tag）。

```bash
scripts/fetch_externals.sh
```

注意：`~/workspaces/mstar_xpu/external/vllm-omni` 检出在 `main`（`a36de509e`），不是 v0.34.0；本仓库不复用它。

## 建立

前提：网络可访问 PyPI、download.pytorch.org、wheels.vllm.ai、github.com、conda-forge；当前用户在 `render` 组；已安装 `cargo`（rustup 会按 `external/mstar/rust-toolchain.toml` 取 Rust 1.99.1）；有 `micromamba`（用于 FFmpeg，可选）。

```bash
scripts/fetch_externals.sh      # 按 externals.pins 检出 external/mstar、external/vllm-omni
ln -sfn ~/workspaces/models models   # 模型缓存（BAGEL 在 models/cache/models--ByteDance-Seed--BAGEL-7B-MoT）
scripts/setup_envs_xpu.sh       # 建立 envs/mstar 与 envs/vllm-omni（首次约 10 分钟，取决于网络）
scripts/check_envs_xpu.sh       # 检查
```

`setup_envs_xpu.sh` 记录的是实际执行过的命令，要点：

- **envs/mstar**：`torch==2.13.0+xpu` 等（`--torch-backend=xpu`，`--overrides` 覆盖 M\* pyproject 中的 `torch<2.13`）；`-e external/mstar[bagel_xpu]`；`external/mstar/rust`（`mstar_rust`）；`cargo build --release --locked` 编译 `mstar-server`，装入 `envs/mstar/bin`；`ffmpeg=6.1`（conda-forge，`envs/ffmpeg-xpu`）。
- **envs/vllm-omni**：`vllm==0.34.0+xpu`（从 `wheels.vllm.ai/0.34.0/xpu` 安装，`--index-strategy unsafe-best-match`）；`external/vllm-omni/requirements/xpu.txt`；`VLLM_OMNI_TARGET_DEVICE=xpu` 可编辑安装 vLLM-Omni（`--no-deps`，依赖已由上一步装好）；`mooncake-transfer-engine-non-cuda`。未装 NIXL。
- 每个环境的完整冻结列表写到 `envs/<name>.freeze.txt`（git 忽略）。

## 验证

```bash
scripts/check_envs_xpu.sh
```

检查项：`xpu-smi discovery` 能否在 20 秒内返回（超时则直接失败，不再 import）；两个环境能否导入框架并看到 XPU；`mstar-server` 是否存在；TorchCodec 能否导入。检查只做设备枚举和导入，不加载模型、不占用显存。

## 运行时环境变量

```bash
source scripts/env_xpu_mstar.sh        # M*：PATH（envs/mstar、可选 ffmpeg）、MSTAR_SERVER_BIN
source scripts/env_xpu_vllm_omni.sh    # vLLM-Omni：PATH
```

两个脚本都会先 source `scripts/env_xpu.sh`，设置：

- `HF_HUB_CACHE=models/cache`、`HF_HUB_OFFLINE=1`
- `OCL_ICD_VENDORS=/etc/OpenCL/vendors`（见下方"已知注意事项"第 1 条）

## 已知注意事项

1. **OpenCL / oneDNN 报错**：`envs/ffmpeg-xpu/lib` 自带的 `libOpenCL.so.1` 会遮蔽系统 ICD loader，导致 oneDNN SDPA 报 `OpenCL device not found`。`env_xpu.sh` 已设置 `OCL_ICD_VENDORS`。不要绕过该脚本直接把 FFmpeg 的 lib 目录加入 `LD_LIBRARY_PATH` 而不设置它。（来源：`~/workspaces/mstar_xpu/docs/VLLM_OMNI_BAGEL_B70_VALIDATION.md`）
2. **设备权限**：访问 `/dev/dri` 需要 `render` 组。旧的 SSH / VS Code 会话不含该组，用 `sg render -c '...'`，并在 `sg` 字符串内重新设置 `LD_LIBRARY_PATH`（`sg` 会过滤它）。
3. **PCI runtime-suspend**：2026-11-23 曾出现三张卡 `runtime_status = suspended`，`xpu-smi` 卡死在 D 状态，需要 root 才能恢复。`check_envs_xpu.sh` 先用 `timeout` 运行 `xpu-smi discovery`，卡住时直接报错，不要在此状态下排队 GPU 任务。当前（2026-11-25）`xpu-smi` 正常。
4. **共享机器**：这台机器的卡可能被其他用户占用（例如 2026-11-22 有 root 进程占用 xpu0 约 30 GiB）。跑任何 GPU 负载前先检查空闲卡，只用空闲卡；空闲卡不够就不运行。`setup`/`check` 不占用显存。
5. **flash-attn**：不存在 XPU 版本，`bagel` 的 ViT 注意力回退到 SDPA（功能正常，较慢）。
6. **FFmpeg**：`torchcodec_xpu` 自动加载依赖 FFmpeg。`envs/ffmpeg-xpu` 不存在时 `check` 会给 WARN。`mstar_xpu` 的经验是：装上 FFmpeg 6.1 后三卡 M\* 服务能正常初始化；如果暂时不装，可设置 `TORCHCODEC_DEVICE_BACKEND_AUTOLOAD=0` 绕过（该变量未在本仓库验证）。
7. **vLLM-Omni 在 XPU 上未经上游正式验收**（上游文档写明 "XPU runtime acceptance remains pending"）。BAGEL 在 vLLM-Omni 的 XPU 上能否跑通，要在 smoke 中确认。
8. **oneCCL 警告**：`MPI launcher` 环境变量缺失、PCIe 拓扑 warning 不影响三卡 TP2 初始化，可忽略。

## 基线与参考

- 历史结果（已在本机运行过的 M\* 与 vLLM-Omni BAGEL）：`~/workspaces/mstar_xpu/results/`（`20261122_*` 为最新，`archive/20261118_initial` 为首轮）。这些结果来自 `~/workspaces/mstar_xpu` 的环境，与本仓库 `envs/` 不是同一个环境。
- 报告与对比：`~/workspaces/mstar_xpu/docs/BAGEL_B70_BEST_CONFIG_COMPARISON.md`。

## 命名与配置约定

- 环境目录：`envs/mstar`、`envs/vllm-omni`、`envs/ffmpeg-xpu`（本机独有，不与 5880 的 `envs/` 混用）。
- 脚本：`scripts/setup_envs_xpu.sh`、`scripts/check_envs_xpu.sh`、`scripts/env_xpu.sh`、`scripts/env_xpu_mstar.sh`、`scripts/env_xpu_vllm_omni.sh`。
- 结果：`results/<实验名>/B70/`，与 `5880/` 分开。

## 记录

| 项 | 值 |
|---|---|
| 机器 | B70（本机）：Linux `6.17.0-1009-intel`，x86_64，Python 3.12.3（系统 `/usr/bin/python3`） |
| GPU | 3 × Intel(R) Arc(TM) Pro B70 Graphics（`0000:18:00.0`、`0000:54:00.0`、`0000:cc:00.0`） |
| 路线 | 原生（uv）。Docker 未使用 |
| `envs/mstar` | Python 3.12；torch `2.13.0+xpu`、torchvision `0.28.0+xpu`、torchaudio `2.11.0+xpu`；torchcodec `0.16.0+cpu`、torchcodec-xpu `0.16.0`；vllm-xpu-kernels `0.1.15.4`；transformers `5.24.1`；mstar 源码 `2c8ffde8`（editable）；`mstar-server` 已编译；冻结列表 `envs/mstar.freeze.txt` |
| `envs/vllm-omni` | Python 3.12；vllm `0.34.0+xpu`；torch `2.15.0+xpu`、triton-xpu `3.9.0`、vllm-xpu-kernels `0.1.17`；vllm-omni 源码 `v0.34.0`（`06a28821`，editable）；mooncake-transfer-engine-non-cuda `0.3.14`；冻结列表 `envs/vllm-omni.freeze.txt` |
| `envs/ffmpeg-xpu` | ffmpeg 6.1（conda-forge） |
| 检查 | `scripts/check_envs_xpu.sh` 通过（2026-11-25）：xpu-smi 3 卡；M\* 与 vLLM-Omni 导入成功；`xpu_available True`，各环境可见 3 张卡；`mstar-server` 存在；TorchCodec 导入成功 |
| 未做 | 未启动任何服务或推理（smoke 待做）；未验证 TorchCodec 的音视频解码；未验证 vLLM-Omni 上的 BAGEL |
