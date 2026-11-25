# 环境：B70（XPU）

适用机器：Intel 独立显卡（B70）。**本仓库尚未在这台机器上建立环境。** 下面是依据两个上游仓库（固定版本，见 `externals.pins`）整理的步骤和需要确认的事项；每一步在实际执行后，把结果填到"记录"部分。

## 与 5880（CUDA）的区别

| 项 | 5880（CUDA） | B70（XPU） |
|---|---|---|
| 框架 | M\*、vLLM-Omni（同版本） | 同左，但 XPU 构建 |
| torch | `torch 2.10.0+cu130`（M\*）；`2.14.0+cu130`（vLLM-Omni）— PyPI | XPU 构建的 torch，**不在 PyPI**，需要按 Intel 的安装说明获取 |
| vLLM | `vllm==0.34.0`（PyPI，cu130） | vLLM 0.34.0 的 XPU 构建（`vllm/vllm-openai-xpu:v0.34.0` 镜像，或按 vLLM 的 XPU 安装文档从源码构建） |
| 内核 | FlashInfer、flash-attn | M\* 使用 `xpu_paged` 注意力后端；vLLM 使用 XPU 内核（`vllm-xpu-kernels`） |
| 工具链 | nvcc 13.0.88 | oneAPI 运行时（上游 Dockerfile 中列出） |
| BAGEL 的 extra | `.[bagel]` | `.[bagel_xpu]`（固定 `vllm-xpu-kernels==0.1.15.4`） |
| 部署配置 | `configs/bagel_single_gpu_smoke.yaml`（单卡） | `external/mstar/configs/bagel_xpu_tp2.yaml`（3 张卡：卡 0 编码器与解码器，卡 1–2 张量并行的 LLM） |

## 两条路线

### 路线 A：Docker（优先，依赖最少）

M\* 与 vLLM-Omni 各自的上游仓库都提供 XPU 的 Dockerfile：

```bash
# M*
cd external/mstar
docker build -t mstar:xpu - < docker/Dockerfile.xpu

# vLLM-Omni（以 v0.34.0 为基础镜像）
cd external/vllm-omni
export VLLM_VERSION=v0.34.0
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g \
  --build-arg VLLM_VERSION="$VLLM_VERSION" .
```

运行容器时需要：

- 宿主机已安装可用的 Intel GPU 驱动，并能访问 `/dev/dri`。
- 映射模型缓存：`-v <models 路径>:/root/.cache/huggingface`（或设置 `HF_HUB_CACHE`）。
- M\* 的编译缓存用命名卷（`-v mstar-cache:/root/.cache/mstar`）。上游文档说明：不要用宿主机的 `$HOME/.cache/mstar`，否则容器（以 root 运行）写入的文件会阻止之后的原生运行。

容器内检查 GPU：

```bash
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
```

### 路线 B：原生环境

1. 安装 Intel GPU 驱动与 oneAPI 运行时（版本按 vLLM 0.34.0 的 XPU 安装文档）。
2. 安装 XPU 构建的 torch 与 vLLM 0.34.0（参考 vLLM 的 XPU 安装文档）。
3. M\*：在 `external/mstar` 下 `uv pip install -e '.[bagel_xpu]'`，并按 torch 的 XPU 构建调整依赖。
4. vLLM-Omni：先装好匹配的 vLLM XPU，再在 `external/vllm-omni` 下按 `docs/getting_started/installation/gpu/xpu.inc.md` 的源码安装步骤执行（`VLLM_OMNI_TARGET_DEVICE=xpu`，`--excludes` 保留 torch 与 Intel 运行时包）。

## 需要先确认的事项

- **B70 的 GPU 数量**：`bagel_xpu_tp2.yaml` 需要 3 张卡。如果 B70 机器卡数不够，需要另写配置。
- **torch XPU 版本**：必须与 vLLM XPU 构建的版本一致；上游文档只写了"保持配对"，没有给出版本号。
- **vLLM-Omni 的 XPU 运行时验收**：上游文档写明"XPU runtime acceptance remains pending"，即 vLLM-Omni 在 XPU 上尚未完成正式验收。BAGEL 在 vLLM-Omni 的 XPU 上能否跑通要在 smoke 中确认。
- **历史运行**：记忆中记录用户已在 B70 上跑过 M\* 与 vLLM-Omni 的 BAGEL，但仓库里没有日志。需要找回这些日志，作为 XPU 基线的参考。

## 建立后要做的检查（待建立脚本）

- `scripts/check_envs_xpu.sh`（待建立）：导入两个框架、`torch.xpu.is_available()`、可见卡数、`mstar-server`（若用 Rust 前端）是否存在。
- smoke：启动 `mstar serve bagel --config configs/bagel_xpu_tp2.yaml`，依次测文本、图生文、文生图；记录与 5880 相同的请求（见 `docs/SMOKE-5880-bagel.md`）。

## 命名与配置约定

- 环境目录仍为 `envs/mstar`、`envs/vllm-omni`（在 B70 机器上），但脚本与运行时变量使用 `*_xpu` 后缀，例如 `scripts/env_xpu.sh`、`scripts/env_xpu_mstar.sh`（待建立）。
- 结果目录 `results/<实验名>/B70/`，与 `5880/` 分开。

## 记录

| 项 | 值 |
|---|---|
| 机器 | B70（待填：主机名、GPU 数量与型号、驱动版本） |
| 路线 | 待定（A：Docker，B：原生） |
| torch XPU 版本 | 待填 |
| vLLM 版本 | 待填（应为 0.34.0 的 XPU 构建） |
| mstar commit | 待填（应为 `externals.pins` 中的 2c8ffde8） |
| vllm-omni commit | 待填（应为 v0.34.0 的 06a28821） |
| 检查 | 待填 |
