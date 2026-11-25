#!/usr/bin/env bash
# 本机（Intel Arc Pro B70，XPU）建立 envs/mstar 与 envs/vllm-omni。原生安装，不用 Docker。
# 记录的是实际执行过的命令；可重复运行（已存在的步骤会重新安装到相同版本）。
# 前提：先运行 scripts/fetch_externals.sh；需要网络（PyPI、download.pytorch.org、wheels.vllm.ai、github.com）；
#       当前用户在 render 组（可访问 /dev/dri）；已安装 cargo（rustup 会按 rust-toolchain.toml 取 1.99.1）。
# 版本依据：envs/XPU.md。M* 与 vLLM-Omni 都依赖同一套 XPU 运行时，但各自独立的环境。
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_XPU=2.13.0+xpu        # M*：与 external/mstar 的 bagel_xpu extra 配对（pyproject 的 torch<2.13 由 --overrides 覆盖）
VLLM_XPU=0.34.0+xpu         # vLLM-Omni v0.34.0 对应的 vLLM XPU 构建
LOG=envs/logs; mkdir -p "$LOG"

# uv：优先用 PATH 上的；没有则在 envs/.uv-bootstrap 里装一个
if command -v uv >/dev/null 2>&1; then
  UV=$(command -v uv)
else
  boot=envs/.uv-bootstrap
  [[ -x $boot/bin/uv ]] || { python3 -m venv $boot && $boot/bin/pip install -q uv; }
  UV=$PWD/$boot/bin/uv
fi
echo "uv: $UV ($($UV --version))"

# ---- envs/mstar：M* （XPU） ----
$UV venv --python 3.12 --seed envs/mstar
$UV pip install --python envs/mstar/bin/python --torch-backend=xpu \
  --overrides <(printf 'torch==%s\n' "$TORCH_XPU") \
  "torch==$TORCH_XPU" 'torchvision==0.28.0+xpu' 'torchaudio==2.11.0+xpu' \
  'torchcodec==0.16.0+cpu' 'torchcodec-xpu==0.16.0' \
  -e 'external/mstar[bagel_xpu]' ./external/mstar/rust \
  msgspec blake3 py-cpuinfo compressed_tensors gguf partial_json_parser tabulate
# Rust HTTP 服务端 mstar-server（需要 cargo）；装进 envs/mstar/bin，与 CUDA 环境一致
(
  cd external/mstar
  cargo build --release --locked --manifest-path rust/server/Cargo.toml
  install -m 755 rust/server/target/release/mstar-server ../../envs/mstar/bin/mstar-server
) 2>&1 | tee "$LOG/mstar-server.log" | tail -n 3
# torchcodec_xpu 自动加载需要 FFmpeg（libav*）。用 conda-forge 的 ffmpeg 6.1 放在 envs/ffmpeg-xpu。
# 该目录自带 libOpenCL，必须配合 OCL_ICD_VENDORS 使用（见 scripts/env_xpu.sh）。
MAMBA=$(command -v micromamba || true)
if [[ -n $MAMBA && ! -x envs/ffmpeg-xpu/bin/ffmpeg ]]; then
  MAMBA_ROOT_PREFIX="$PWD/envs/.mamba" "$MAMBA" create -y -q -p envs/ffmpeg-xpu -c conda-forge 'ffmpeg=6.1'
elif [[ -z $MAMBA ]]; then
  echo "警告：未找到 micromamba，跳过 FFmpeg（torchcodec_xpu 将不能自动加载）" >&2
fi
$UV pip freeze --python envs/mstar/bin/python > envs/mstar.freeze.txt

# ---- envs/vllm-omni：vLLM 0.34.0 XPU + vLLM-Omni v0.34.0 源码（可编辑） ----
$UV venv --python 3.12 --seed envs/vllm-omni
$UV pip install --python envs/vllm-omni/bin/python \
  --index-url "https://wheels.vllm.ai/0.34.0/xpu" \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pypi.org/simple \
  --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl \
  --extra-index-url https://wheels.vllm.ai/xpu/ \
  --index-strategy unsafe-best-match "vllm==$VLLM_XPU"
$UV pip install --python envs/vllm-omni/bin/python -r external/vllm-omni/requirements/xpu.txt
VLLM_OMNI_TARGET_DEVICE=xpu $UV pip install --python envs/vllm-omni/bin/python \
  --no-build-isolation --no-deps --editable external/vllm-omni
# 可选的两阶段连接器：Mooncake（TCP 协议可用，无需 RDMA）。NIXL 在 XPU 上不可用（UCX 不能注册 XPU 显存），不装。
$UV pip install --python envs/vllm-omni/bin/python mooncake-transfer-engine-non-cuda
$UV pip check --python envs/vllm-omni/bin/python || echo "注意：uv pip check 报告了依赖问题，见上方输出" >&2
$UV pip freeze --python envs/vllm-omni/bin/python > envs/vllm-omni.freeze.txt

echo "完成。下一步：scripts/check_envs_xpu.sh"
