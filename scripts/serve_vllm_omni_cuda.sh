#!/usr/bin/env bash
# 启动 vLLM-Omni 服务（5880，CUDA），写启动清单与 PID 文件。
#
# 用法：
#   scripts/serve_vllm_omni_cuda.sh --name vo-bagel --model ByteDance-Seed/BAGEL-7B-MoT --served-name bagel \
#       --deploy-config configs/bagel_vllm_omni_audit.yaml --gpus 0,3 --port 18401 \
#       --out results/<实验>/$MM_MACHINE/vllm-omni/<臂>/server --shim count [--wait] [--allow-busy-gpu]
# --model 是 HF 仓库名，从 models/cache 解析本地快照（离线）。
# --shim：off | count | time。count 只记录编码器调用与缓存检查；time 另加设备同步计时（只用于画像轮）。
# 停止：scripts/stop_serve.sh vo-bagel
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_serve_common.sh"
source "$ROOT/scripts/env_cuda_vllm_omni.sh"

name="" model="" served="" config="" gpus="" port="" out="" shim="count" wait=0 allow_busy=0 timeout=1200
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --served-name) served="$2"; shift 2 ;;
    --deploy-config) config="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --shim) shim="$2"; shift 2 ;;
    --wait) wait=1; shift ;;
    --timeout) timeout="$2"; shift 2 ;;
    --allow-busy-gpu) allow_busy=1; shift ;;
    *) die "未知参数 $1" ;;
  esac
done
[[ -n "$name" && -n "$model" && -n "$served" && -n "$config" && -n "$gpus" && -n "$port" && -n "$out" ]] \
  || die "需要 --name --model --served-name --deploy-config --gpus --port --out"
[[ -f "$config" ]] || die "部署配置不存在：$config"
case "$shim" in off|count|time) ;; *) die "--shim 只能是 off、count、time" ;; esac

snap_root="$ROOT/models/cache/models--${model//\//--}/snapshots"
[[ -d "$snap_root" ]] || die "本地没有模型：$snap_root"
snap="$(ls -1 "$snap_root" | head -n 1)"
model_path="$snap_root/$snap"

check_name_free "$name"
check_port_free "$port"
check_gpus_free "$gpus" "$allow_busy"

cd "$ROOT"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="$gpus"
if [[ "$shim" != off ]]; then
  export PYTHONPATH="$ROOT/scripts/audit_vo${PYTHONPATH:+:$PYTHONPATH}"
  export VO_AUDIT_LOG="$ROOT/$out/shim.log"
  [[ "$out" = /* ]] && export VO_AUDIT_LOG="$out/shim.log"
  export VO_AUDIT_MODE="$shim"
  rm -f "$VO_AUDIT_LOG"
fi
cmd=("$ROOT/envs/vllm-omni/bin/vllm" serve "$model_path" --omni --port "$port"
     --deploy-config "$config" --served-model-name "$served")
launch "$name" "$out" -- "${cmd[@]}"
write_manifest "$out" "$name" vllm-omni "$port" "$gpus" "$config" "$shim" "$LAUNCHED_PID" -- "${cmd[@]}"
echo "serve: $name 已启动（PID $LAUNCHED_PID，GPU $gpus，端口 $port，shim $shim，日志 $out/server.log）"
if [[ "$wait" == 1 ]]; then
  wait_ready "$name" "$out" "http://localhost:$port/health" "$timeout"
fi
