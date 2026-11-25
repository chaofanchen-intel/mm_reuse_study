#!/usr/bin/env bash
# 检查本机（B70，XPU）两个环境：能否导入框架、看到 XPU、工具链是否就绪。失败时返回非零。
# 注意：这是 B70 机器专用的检查（CUDA 机器用 check_envs_cuda.sh）。
# 只做设备枚举和导入，不加载模型、不占用显存。
cd "$(dirname "$0")/.." || exit 1
status=0

echo "== 设备（xpu-smi）"
# 2026-11-23 出现过 PCI runtime-suspend 后 xpu-smi 卡死在 D 状态；此时不要继续，否则检查本身也会挂住。
if ! timeout -s KILL 20 xpu-smi discovery > /tmp/.xpu_check_discovery.$$ 2>&1; then
  echo "FAIL: xpu-smi discovery 超时或失败（可能是 PCI 卡在 runtime-suspend，需要 root 处理）"
  rm -f /tmp/.xpu_check_discovery.$$
  exit 1
fi
grep -o 'Device Name: [^|]*' /tmp/.xpu_check_discovery.$$ | sed 's/ *$//' | sort | uniq -c
grep -o 'PCI BDF Address: [0-9a-f:.]*' /tmp/.xpu_check_discovery.$$ | sed 's/PCI BDF Address: /  /'
rm -f /tmp/.xpu_check_discovery.$$

echo "== envs/mstar"
(
  source scripts/env_xpu_mstar.sh
  envs/mstar/bin/python - <<'PY'
import torch, mstar, vllm_xpu_kernels
from mstar.model.bagel.components import vit_encoder  # noqa: F401  BAGEL 组件可导入
print(f"torch {torch.__version__} xpu_available {torch.xpu.is_available()} xpu_count {torch.xpu.device_count()}")
print(f"mstar {mstar.__file__}")
print(f"vllm_xpu_kernels {vllm_xpu_kernels.__file__}")
PY
  [[ $? -eq 0 ]] || { echo "FAIL: mstar 导入失败"; exit 1; }
  envs/mstar/bin/python -c "import torchcodec; print('torchcodec', torchcodec.__version__)" 2>/dev/null \
    || echo "WARN: torchcodec 不能导入（检查 envs/ffmpeg-xpu 是否存在）"
  [[ -x "$MSTAR_SERVER_BIN" ]] && echo "mstar-server: $MSTAR_SERVER_BIN" || { echo "FAIL: 缺少 mstar-server"; exit 1; }
) || status=1

echo "== envs/vllm-omni"
(
  source scripts/env_xpu_vllm_omni.sh
  envs/vllm-omni/bin/python - <<'PY'
import torch, vllm, vllm_omni, vllm_xpu_kernels
print(f"torch {torch.__version__} xpu_available {torch.xpu.is_available()} xpu_count {torch.xpu.device_count()}")
print(f"vllm {vllm.__version__}; vllm_omni {vllm_omni.__file__}")
print(f"vllm_xpu_kernels {vllm_xpu_kernels.__file__}")
PY
  [[ $? -eq 0 ]] || { echo "FAIL: vllm-omni 导入失败"; exit 1; }
) || status=1

echo "== 结果：$([[ $status -eq 0 ]] && echo 通过 || echo 失败)"
exit $status
