#!/usr/bin/env bash
# envs/mstar（CUDA 13.0）的工具链。用法：source scripts/env_cuda_mstar.sh
# JIT 编译（flashinfer、MoE 对齐核）必须用与 torch cu130 相同小版本的 nvcc（13.0.88），
# 不能用系统的 CUDA 版本。
_FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_FW_ROOT/scripts/env_cuda.sh"
_cuda="$_FW_ROOT/envs/mstar/lib/python3.12/site-packages/nvidia/cu13"
if [[ ! -x "$_cuda/bin/nvcc" ]]; then
  echo "nvcc 不存在：$_cuda/bin/nvcc（运行 scripts/setup_envs_cuda.sh）" >&2
else
  export CUDA_HOME="$_cuda"
  export CUDACXX="$_cuda/bin/nvcc"
  export PATH="$_cuda/bin:$_FW_ROOT/envs/mstar/bin:$PATH"
  export LD_LIBRARY_PATH="$_cuda/lib:$_FW_ROOT/envs/mstar/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$MM_CUDA_ARCH}"
fi
export MSTAR_SERVER_BIN="${MSTAR_SERVER_BIN:-$_FW_ROOT/external/mstar/rust/server/target/release/mstar-server}"
unset _FW_ROOT _cuda
