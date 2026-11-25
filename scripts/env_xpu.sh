#!/usr/bin/env bash
# 本机（B70，Intel Arc Pro XPU）共用环境变量。用法：source scripts/env_xpu.sh
# 模型缓存指向 models/（符号链接到 ~/workspaces/models），HF 缓存格式，离线。
_ENV_XPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_CACHE="$_ENV_XPU_ROOT/models/cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MM_MACHINE="${MM_MACHINE:-B70}"    # 结果目录的机器层级（results/<实验>/<机器>/<框架>/）

# OpenCL / oneDNN：conda 版 FFmpeg（envs/ffmpeg-xpu）自带的 libOpenCL.so.1 会遮蔽系统 ICD loader，
# 只读到空的 vendor 目录，于是 oneDNN SDPA 报 "OpenCL device not found"。指向系统 vendor 目录。
export OCL_ICD_VENDORS=/etc/OpenCL/vendors

# 访问 /dev/dri 需要 render 组。新开的 SSH / VS Code 会话才会带上该组；旧会话可用 `sg render -c '...'`
# （注意 sg 会过滤继承的 LD_LIBRARY_PATH，需要在 sg 的命令字符串里重新设置）。
if ! id -nG | tr ' ' '\n' | grep -qx render; then
  echo "env_xpu.sh: 当前 shell 不在 render 组，访问 /dev/dri 可能失败" >&2
fi
unset _ENV_XPU_ROOT
