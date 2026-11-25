# 本机环境总览

本仓库的实验要在两台机器上跑，环境互不相同，**不要混用**：

| 机器 | 硬件 | 平台 | 环境文档 | 建立脚本 | 检查脚本 |
|---|---|---|---|---|---|
| 5880 | 8 × NVIDIA RTX 5880 Ada（sm_89） | CUDA 13.0 | [CUDA.md](CUDA.md) | `scripts/setup_envs_cuda.sh` | `scripts/check_envs_cuda.sh` |
| B70 | Intel Arc B70（XPU） | XPU（oneAPI） | [XPU.md](XPU.md) | 待建立（`scripts/setup_envs_xpu.sh`） | 待建立（`scripts/check_envs_xpu.sh`） |

两台机器的环境只在各自机器上存在（`envs/` 在 git 中忽略）。它们之间共享的是：

- **同一个仓库**：`docs/`、`scripts/`、`bench/`、`scenarios/`、`configs/` 都是共用的。
- **固定的上游版本**：`externals.pins` 指定 M\* 与 vLLM-Omni 的版本，两台机器都用 `scripts/fetch_externals.sh` 检出同一版本。
- **模型权重**：`models/` 是符号链接到 `../Inference_Engine_Investigation_Repo/models`（HF 缓存格式）。B70 上需要自己指向对应目录。

## 命名与隔离规则

- 脚本、配置、文档中凡是平台相关的名字都带平台后缀：`*_cuda*`、`*_xpu*`。通用的脚本不带后缀（如 `bench/`）。
- 结果按机器分目录：`results/<实验名>/<机器>/`，机器名为 `5880` 或 `B70`。两台机器的结果合并比较时，必须看清机器名。
- 同一台机器上，M\* 与 vLLM-Omni 各自独立的环境（`envs/mstar`、`envs/vllm-omni`）。两个环境的 torch 版本不同，不能共用。
- 不同平台的环境不能拷贝。B70 上的 `envs/` 从零开始建立。

## 共用的环境变量

- `scripts/env_cuda.sh`：CUDA 机器的共用变量（HF 缓存、离线模式）。B70 对应的文件 `scripts/env_xpu.sh` 待建立。

## 每台机器都要记录的内容

每次建立或升级环境后，在对应的文档（`CUDA.md` 或 `XPU.md`）的"记录"部分填写：驱动版本、GPU 型号与数量、系统 Python、各框架的 commit 或版本、torch 版本、导入与检查的结果。这样结果可以追溯到具体的环境。
