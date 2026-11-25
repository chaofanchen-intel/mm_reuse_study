#!/usr/bin/env bash
# 启动 M* 服务（5880，CUDA），写启动清单与 PID 文件。
#
# 用法：
#   scripts/serve_mstar_cuda.sh --name mstar-bagel --model bagel --config configs/bagel_mstar_i2t.yaml \
#       --gpus 1 --port 18301 --out results/<实验>/$MM_MACHINE/mstar/<臂>/server [--wait] [--allow-busy-gpu]
# 停止：scripts/stop_serve.sh mstar-bagel
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_serve_common.sh"
source "$ROOT/scripts/env_cuda_mstar.sh"

name="" model="" config="" gpus="" port="" out="" wait=0 allow_busy=0 timeout=1200
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --wait) wait=1; shift ;;
    --timeout) timeout="$2"; shift 2 ;;
    --allow-busy-gpu) allow_busy=1; shift ;;
    *) die "未知参数 $1" ;;
  esac
done
[[ -n "$name" && -n "$model" && -n "$config" && -n "$gpus" && -n "$port" && -n "$out" ]] \
  || die "需要 --name --model --config --gpus --port --out"
[[ -f "$config" ]] || die "配置不存在：$config"

check_name_free "$name"
check_port_free "$port"
check_gpus_free "$gpus" "$allow_busy"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$gpus"
cmd=("$ROOT/envs/mstar/bin/mstar" serve "$model" --config "$config" --port "$port" --enable-metrics)
launch "$name" "$out" -- "${cmd[@]}"
write_manifest "$out" "$name" mstar "$port" "$gpus" "$config" off "$LAUNCHED_PID" -- "${cmd[@]}"
echo "serve: $name 已启动（PID $LAUNCHED_PID，GPU $gpus，端口 $port，日志 $out/server.log）"
if [[ "$wait" == 1 ]]; then
  wait_ready "$name" "$out" "http://localhost:$port/metrics" "$timeout"
fi
