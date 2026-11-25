# mm_reuse_study

多模态输入复用实验：验证"不变的部分提前计算并复用"（编码器输出、KV 前缀）在 M\* 与 vLLM-Omni 上的收益。

与 `../robotics_vla_runtime` 并列，互不依赖。

## 目录

| 路径 | 内容 | 进入 git |
|---|---|---|
| `docs/DESIGN.md` | 设计：目的、基线、假设、流程、指标 | 是 |
| `externals.pins` | 上游源码固定版本（M\*、vLLM-Omni） | 是 |
| `scripts/fetch_externals.sh` | 按 pins 拉取并检出 `external/` | 是 |
| `external/` | 上游源码检出（由脚本生成） | 否 |
| `bench/` | 实验矩阵与结果格式（框架适配器后续加入） | 是 |
| `scenarios/` | 场景描述与输入夹具 | 是 |
| `patches/` | 对上游的改动（待建立） | 是 |
| `envs/` | 本机 Python 环境（5880 与 B70 各自不同） | 否 |
| `models/` | 模型权重（符号链接到本机路径） | 否 |
| `results/` | 实验输出，按 `<实验名>/<机器>/` 存放 | 否 |

## 快速开始

```bash
scripts/fetch_externals.sh     # 拉取并检出 external/ 中的固定版本
python3 bench/plan.py          # 打印实验矩阵（不运行推理）
```

## 两台机器

- 5880（CUDA）：Qwen3-Omni 相关实验。
- B70（Intel XPU）：BAGEL 相关实验。

环境搭建与检查：`envs/README.md` 为总览，平台细节分别见 `envs/CUDA.md` 与 `envs/XPU.md`。
