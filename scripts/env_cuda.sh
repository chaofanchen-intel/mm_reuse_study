#!/usr/bin/env bash
# 本机（CUDA）共用环境变量。用法：source scripts/env_cuda.sh
# 模型缓存指向 models/（符号链接到 Inference_Engine_Investigation_Repo/models），HF 缓存格式，离线。
_ENV_CUDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_CACHE="$_ENV_CUDA_ROOT/models/cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset _ENV_CUDA_ROOT
