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
| `bench/` | 实验矩阵、结果格式、G0 审计脚本（统一客户端与适配器待建立，见 DESIGN 第 13 节） | 是 |
| `scenarios/` | 场景描述与输入夹具 | 是 |
| `patches/` | 对上游的改动（待建立） | 是 |
| `envs/` | 本机 Python 环境（5880 与 B70 各自不同） | 否 |
| `models/` | 模型权重（符号链接到本机路径） | 否 |
| `configs/` | 服务配置（smoke 与审计用，非性能配置） | 是 |
| `scripts/audit_vo/` | vLLM-Omni 测量 shim（环境变量开启，不改框架代码） | 是 |
| `results/` | 实验输出，按 `<实验名>/<机器>/<框架>/` 存放（G0 的早期目录尚未迁移） | 否 |

## 快速开始

```bash
scripts/fetch_externals.sh     # 拉取并检出 external/ 中的固定版本
python3 bench/plan.py          # 打印实验矩阵（不运行推理）
MM_SETTLE_S=0 envs/mstar/bin/python -m pytest -q tests/test_tooling.py   # 测量工具的离线测试
```

一次 image→text 测量（以 M\* 为例；vLLM-Omni 用 `scripts/serve_vllm_omni_cuda.sh`，参数见脚本头部）：

```bash
source scripts/env_cuda.sh
d=results/g1/$MM_MACHINE/mstar/pfx-arms/server
scripts/serve_mstar_cuda.sh --name mstar-bagel --model bagel --config configs/bagel_mstar_i2t.yaml \
    --gpus 1 --port 18301 --out $d --wait
envs/mstar/bin/python -m bench.runner --server-dir $d --arm enc0-pfx1 --prefix-cache on \
    --out results/g1/$MM_MACHINE/mstar/pfx-arms/records.jsonl
python3 -m bench.summarize results/g1/$MM_MACHINE/mstar/pfx-arms/records.jsonl
scripts/stop_serve.sh mstar-bagel
```

## 两台机器

- 5880（CUDA）：BAGEL 先在这里做 G0 与 image→text 对照；Qwen3-Omni 只在这里做。
- B70（Intel XPU）：BAGEL 的同样审计与对照，结果与 5880 分开，不合并结论。

环境搭建与检查：`envs/README.md` 为总览，平台细节分别见 `envs/CUDA.md` 与 `envs/XPU.md`。
