#!/usr/bin/env bash
# envs/vllm-omni（XPU，vLLM 0.34.0）的运行时变量。用法：source scripts/env_xpu_vllm_omni.sh
_FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_FW_ROOT/scripts/env_xpu.sh"
export PATH="$_FW_ROOT/envs/vllm-omni/bin:$PATH"
unset _FW_ROOT
