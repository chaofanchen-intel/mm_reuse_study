#!/usr/bin/env bash
# envs/vllm-omni（CUDA 13.0）的工具链。用法：source scripts/env_cuda_vllm_omni.sh
# FlashInfer 的 JIT 需要 nvcc（13.0.88）和 ninja；ninja 装在环境的 bin 目录里。
_FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_FW_ROOT/scripts/env_cuda.sh"
_env="$_FW_ROOT/envs/vllm-omni"
_cuda="$_env/lib/python3.12/site-packages/nvidia/cu13"
if [[ ! -x "$_cuda/bin/nvcc" ]]; then
  echo "nvcc 不存在：$_cuda/bin/nvcc（运行 scripts/setup_envs_cuda.sh）" >&2
else
  export CUDA_HOME="$_cuda"
  export CUDACXX="$_cuda/bin/nvcc"
  export PATH="$_cuda/bin:$_env/bin:$PATH"
  export LD_LIBRARY_PATH="$_cuda/lib:$_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$MM_CUDA_ARCH}"
fi
unset _FW_ROOT _env _cuda
