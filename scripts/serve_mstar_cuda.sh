#!/usr/bin/env bash
# 启动 M* 服务（5880，CUDA），写启动清单与 PID 文件。
#
# 用法：
#   scripts/serve_mstar_cuda.sh --name mstar-bagel --model bagel --config configs/bagel_mstar_i2t.yaml \
#       --gpus 1 --port 18301 --out results/<实验>/$MM_MACHINE/mstar/<臂>/server [--wait] [--allow-busy-gpu] \
#       [--encoder-cache-mb N] [--log-stats]
# --encoder-cache-mb：开启本研究的 M* 改动 M1（编码器输出缓存，external/mstar 需含该改动），日志 <out>/encoder_cache.log
# 停止：scripts/stop_serve.sh mstar-bagel
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_serve_common.sh"
source "$ROOT/scripts/env_cuda_mstar.sh"

name="" model="" config="" gpus="" port="" out="" wait=0 allow_busy=0 timeout=1200 enc_cache=0 log_stats=0
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
    --encoder-cache-mb) enc_cache="$2"; shift 2 ;;
    --log-stats) log_stats=1; shift ;;
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
export MSTAR_ENCODER_CACHE_MB="$enc_cache"
# 编码项日志总是开启（缓存关时记录"全部编码"），基线与 M1 用同样的口径
mkdir -p "$out"; export MSTAR_ENCODER_CACHE_LOG="$(cd "$out" && pwd)/encoder_cache.log"
# C1（MSTAR_C1=1）：每个视觉 walk 跳过几张图记入 <out>/c1.log；清掉上次运行留下的登记文件
if [[ "${MSTAR_C1:-0}" == 1 ]]; then
  export MSTAR_C1_LOG="$(cd "$out" && pwd)/c1.log"
  rm -rf "${MSTAR_C1_DIR:-/dev/shm/mstar_c1}"
fi
# C4（MSTAR_C4=1）：清掉上次运行存下的帧
[[ "${MSTAR_C4:-0}" == 1 ]] && rm -rf "${MSTAR_C4_FRAMES_DIR:-/dev/shm/mstar_c4_frames}"
cmd=("$ROOT/envs/mstar/bin/mstar" serve "$model" --config "$config" --port "$port" --enable-metrics)
# --log-stats：每个请求的节点/walk 时间线写入 <out>/request_stats.log（画像用，会增加少量主机开销）
[[ "$log_stats" == 1 ]] && cmd+=(--log-stats-file "$(cd "$out" && pwd)/request_stats.log")
launch "$name" "$out" -- "${cmd[@]}"
write_manifest "$out" "$name" mstar "$port" "$gpus" "$config" off "$LAUNCHED_PID" -- "${cmd[@]}"
echo "serve: $name 已启动（PID $LAUNCHED_PID，GPU $gpus，端口 $port，日志 $out/server.log）"
if [[ "$wait" == 1 ]]; then
  wait_ready "$name" "$out" "http://localhost:$port/metrics" "$timeout"
fi
