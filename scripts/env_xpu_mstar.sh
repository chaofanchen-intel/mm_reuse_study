#!/usr/bin/env bash
# envs/mstar（XPU）的运行时变量。用法：source scripts/env_xpu_mstar.sh
_FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_FW_ROOT/scripts/env_xpu.sh"
export PATH="$_FW_ROOT/envs/mstar/bin:$PATH"
# torchcodec_xpu 需要 FFmpeg 的 libav*（可选；没有则 TorchCodec 自动加载失败，见 envs/XPU.md）
if [[ -x "$_FW_ROOT/envs/ffmpeg-xpu/bin/ffmpeg" ]]; then
  export PATH="$_FW_ROOT/envs/ffmpeg-xpu/bin:$PATH"
  export LD_LIBRARY_PATH="$_FW_ROOT/envs/ffmpeg-xpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MSTAR_SERVER_BIN="${MSTAR_SERVER_BIN:-$_FW_ROOT/envs/mstar/bin/mstar-server}"
unset _FW_ROOT
