#!/usr/bin/env bash
# 本机（CUDA）共用环境变量。用法：source scripts/env_cuda.sh
# 模型缓存指向 models/（符号链接到 Inference_Engine_Investigation_Repo/models），HF 缓存格式，离线。
_ENV_CUDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_CACHE="$_ENV_CUDA_ROOT/models/cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
_gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
_gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]' || true)"
case "$_gpu_name" in
	*H200*) _machine_default=H200 ;;
	*5880*) _machine_default=5880 ;;
	*) _machine_default="${_gpu_name// /_}" ;;
esac
export MM_MACHINE="${MM_MACHINE:-${_machine_default:-5880}}"
export MM_CUDA_ARCH="${MM_CUDA_ARCH:-${_gpu_arch:-8.9}}"
unset _ENV_CUDA_ROOT _gpu_name _gpu_arch _machine_default
