#!/usr/bin/env bash
# 本机（CUDA 13.0，RTX 5880）建立 envs/mstar 与 envs/vllm-omni。
# 记录的是实际执行过的命令；可重复运行（已存在的步骤会重新安装到相同版本）。
# 前提：先运行 scripts/fetch_externals.sh；需要网络（含 github.com）。
set -euo pipefail
cd "$(dirname "$0")/.."
boot=envs/.uv-bootstrap
dl=envs/downloads
[[ -x $boot/bin/uv ]] || { python3 -m venv $boot && $boot/bin/pip install -q uv; }
UV=$PWD/$boot/bin/uv
mkdir -p "$dl"

# ---- envs/mstar：M* ----
$UV venv --python 3.12 --seed envs/mstar
$UV pip install --python envs/mstar/bin/python --torch-backend=cu130 \
  -e 'external/mstar[qwen3_omni,bagel,dev]' \
  'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0' 'torchcodec==0.10.0'
# JIT 用的 CUDA 13.0.88 工具链（与 torch cu130 同小版本）；DROID 视频解码用 nvidia-npp
$UV pip install --python envs/mstar/bin/python 'nvidia-cuda-nvcc==13.0.88' \
  'nvidia-nvvm==13.0.88' 'nvidia-cuda-crt==13.0.88' 'nvidia-npp==13.0.0.50'
# flash-attn：BAGEL 的 ViT 注意力优先用它，缺失时回退到 SDPA（见 smoke 记录）
$UV pip install --python envs/mstar/bin/python \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
# Rust 扩展 mstar_rust（服务端必需）与 Rust 服务端 mstar-server（需要 cargo）
$UV pip install --python envs/mstar/bin/python maturin
(
  source scripts/env_cuda_mstar.sh
  export VIRTUAL_ENV="$PWD/envs/mstar"
  cd external/mstar && "$PWD/../../envs/mstar/bin/maturin" develop --release -m rust/Cargo.toml
  cd rust/server && cargo build --release
)

# ---- envs/vllm-omni：vLLM-Omni v0.34.0 ----
$UV venv --python 3.12 --seed envs/vllm-omni
$UV pip install --python envs/vllm-omni/bin/python --torch-backend=cu130 'vllm==0.34.0'
$UV pip install --python envs/vllm-omni/bin/python -e external/vllm-omni
# FlashInfer 0.7.3：cubin 与 JIT 缓存（sm89 = RTX 5880）。PyPI 没有 0.7.x 的 cubin，
# uv 直接从 GitHub 发布页安装时会卡住，因此先用 curl 下载到 envs/downloads 再本地安装。
fi_base=https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.3
for whl in flashinfer_cubin-0.7.3-py3-none-any.whl \
           flashinfer_jit_cache-0.7.3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl \
           flashinfer_jit_cache_sm89-0.7.3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl; do
  [[ -s $dl/$whl ]] || curl -L --fail -o "$dl/$whl" "$fi_base/$whl"
done
$UV pip install --python envs/vllm-omni/bin/python "$dl/flashinfer_cubin-0.7.3-py3-none-any.whl"
$UV pip install --no-deps --python envs/vllm-omni/bin/python \
  "$dl/flashinfer_jit_cache-0.7.3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" \
  "$dl/flashinfer_jit_cache_sm89-0.7.3+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
# nvcc 13.0.88（替换 vLLM 依赖带入的 13.5.37）
$UV pip install --python envs/vllm-omni/bin/python 'nvidia-cuda-nvcc==13.0.88' \
  'nvidia-nvvm==13.0.88' 'nvidia-cuda-crt==13.0.88'
