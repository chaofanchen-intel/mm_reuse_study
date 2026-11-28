# 环境：CUDA（5880 / H200）

支持机器：

- 5880：8 × NVIDIA RTX 5880 Ada（sm_89），驱动 580.178.04，CUDA 13.0。
- H200：NVIDIA H200 NVL（compute capability 9.0，FlashInfer sm90a），驱动 580.178.04，CUDA 13.0。

## 环境清单

| 环境 | Python | 关键包 | 来源 |
|---|---|---|---|
| `envs/mstar` | 3.12 | M\* `v0.3.0` 可编辑安装（extras：qwen3_omni、bagel、dev），torch 2.10.0+cu130，flash_attn 2.8.3，`mstar_rust`，`mstar-server` | `external/mstar` |
| `envs/vllm-omni` | 3.12 | torch 2.14.0+cu130，vllm 0.34.0，vLLM-Omni 可编辑安装，flashinfer 0.7.3（cubin、JIT 缓存按架构选择） | `external/vllm-omni` |
| `envs/.uv-bootstrap` | 3.12 | 仅提供 uv | — |

两个环境隔离，不共用 torch。

## 建立

前提：网络可访问 PyPI、github.com、wheels.vllm.ai、flashinfer.ai；已安装 cargo（Rust），用于编译 `mstar-server`；已安装 CUDA 13 兼容的驱动。

```bash
scripts/fetch_externals.sh      # 按 externals.pins 检出 external/mstar、external/vllm-omni
scripts/setup_envs_cuda.sh      # 建立 envs/mstar 与 envs/vllm-omni（首次需下载数 GB，耗时取决于网络）
```

Setup 默认从 `nvidia-smi` 检测计算能力：8.9 使用 sm89 缓存，9.0（H200）使用 sm90a 缓存；可用 `MM_CUDA_ARCH` 覆盖。

`setup_envs_cuda.sh` 记录的是实际执行过的命令，包括：

- M\* `v0.3.0`：torch cu130 与 extras；CUDA 13.0.88 工具链（nvcc、nvvm、crt）与 `nvidia-npp`；flash-attn 2.8.3 wheel；maturin 编译 `mstar_rust`；cargo 编译 `mstar-server`。
- vLLM-Omni：`vllm==0.34.0`（cu130）；可编辑安装 vLLM-Omni；FlashInfer 0.7.3 的 cubin 与 JIT 缓存（sm89 或 sm90a）；nvcc 13.0.88。

FlashInfer 的 wheel 通过 curl 下载到 `envs/downloads/` 再本地安装。原因：PyPI 没有 0.7.x 的 cubin，uv 直接从 GitHub 安装时会卡住。

## 验证

```bash
scripts/check_envs_cuda.sh
```

检查项：两个环境能否导入框架；看到的 GPU 数量；flash-attn 是否加载、BAGEL ViT 是否走 flash 路径；`mstar-server` 是否存在；两个环境的 nvcc 与 ninja 是否可用。

## 运行时环境变量

```bash
source scripts/env_cuda_mstar.sh        # M\*：CUDA_HOME、nvcc（13.0.88）、PATH、MSTAR_SERVER_BIN
source scripts/env_cuda_vllm_omni.sh    # vLLM-Omni：CUDA_HOME、nvcc、ninja
```

两个脚本都会先 source `scripts/env_cuda.sh`，设置 `HF_HUB_CACHE=models/cache` 与离线模式。

## 已知注意事项

- **显存**：M\* 的 BAGEL 默认配置没有设置 `gpu_memory_fraction`，CUDA 图捕获会占满 GPU（实测 48.5 GB / 49 GB），随后的 VAE 解码 OOM。smoke 配置见 `configs/bagel_single_gpu_smoke.yaml`（设为 0.8）。正式实验前要统一设定。详情见 [docs/SMOKE-5880-bagel.md](../docs/SMOKE-5880-bagel.md)。
- **共享机器**：GPU 0–3 在不同时间可能被其他用户占用。运行前先看 `nvidia-smi`，用 `CUDA_VISIBLE_DEVICES` 选择空闲卡。
- **flash-attn**：BAGEL 缺失时回退到 SDPA，能运行。保留它是为了与基准仓库的最佳配置一致。
- **FlashInfer JIT cache**：只安装与当前 GPU 匹配的架构 provider（H200 使用 sm90a）。通用 cache 声明了其他架构 provider 的依赖，因此 `pip check` 可能提示未选中架构的 cache 缺失；这是预期的。
- **模型**：`models/` 符号链接到基准仓库的 `models/`，HF 缓存在 `models/cache/` 下。

## 记录

| 项 | 值 |
|---|---|
| 驱动 | 580.178.04 |
| GPU | 8 × NVIDIA RTX 5880 Ada Generation（compute capability 8.9；FlashInfer sm89） |
| 系统 Python | 3.12.3 |
| envs/mstar | M\* `v0.3.0`（`f63242ff`），mstar-ai 0.3.0，mstar-rust 0.2.0；torch 2.10.0+cu130，CUDA 13.0，8 张 GPU 可见，flash_attn 2.8.3，mstar-server 已编译 |
| envs/vllm-omni | torch 2.14.0+cu130，vllm 0.34.0，vllm_omni 导入成功，flashinfer 0.7.3、cubin 与 sm89 JIT cache 已安装 |
| 检查 | `scripts/check_envs_cuda.sh` 通过（2026-11-29）：M\*、vLLM-Omni 导入成功，8 张 GPU 可见，flash-attn/FlashInfer 与 mstar-server 检查通过 |
