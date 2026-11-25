#!/usr/bin/env bash
# 检查本机（CUDA）两个环境：能否导入框架、看到 GPU、工具链是否就绪。失败时返回非零。
# 注意：这是 CUDA 机器专用的检查（B70 的 XPU 环境用 check_envs_xpu.sh，待建立）。
cd "$(dirname "$0")/.." || exit 1
status=0

echo "== envs/mstar"
(
  source scripts/env_cuda_mstar.sh
  envs/mstar/bin/python - <<'PY' 2>/dev/null
import torch, mstar, flash_attn
from mstar.model.bagel.components import vit_encoder as v
print(f"torch {torch.__version__} cuda {torch.version.cuda} gpus {torch.cuda.device_count()}")
print(f"mstar {mstar.__file__}")
print(f"flash_attn {flash_attn.__version__}; BAGEL ViT flash path: {v._FLASH_ATTN_AVAILABLE}")
PY
  [[ $? -eq 0 ]] || { echo "FAIL: mstar 导入失败"; exit 1; }
  nvcc --version | tail -n 1
  [[ -x "$MSTAR_SERVER_BIN" ]] && echo "mstar-server: $MSTAR_SERVER_BIN" || { echo "FAIL: 缺少 mstar-server"; exit 1; }
) || status=1

echo "== envs/vllm-omni"
(
  source scripts/env_cuda_vllm_omni.sh
  envs/vllm-omni/bin/python - <<'PY' 2>&1 | grep -v -i "warn\|deprecated"
import torch, vllm, vllm_omni, flashinfer, flashinfer_cubin
print(f"torch {torch.__version__} cuda {torch.version.cuda} gpus {torch.cuda.device_count()}")
print(f"vllm {vllm.__version__}; vllm_omni {vllm_omni.__file__}")
print(f"flashinfer {flashinfer.__version__}; flashinfer_cubin {flashinfer_cubin.__version__}")
PY
  nvcc --version | tail -n 1
  ninja --version
) || status=1

exit $status
