# 实验脚本共用：等待空闲卡、带重试地启动服务、退出时停止本脚本启动的服务。由 run_*.sh source。
# 需要调用方先设置：WAIT_HOURS、GPU_BUSY_MIB；可用数组 STARTED 记录已启动的服务名。
STARTED=()
log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() { for n in "${STARTED[@]}"; do [[ -n "$n" ]] && scripts/stop_serve.sh "$n" >/dev/null 2>&1 && log "已停止 $n"; done; }
trap cleanup EXIT

free_gpus() {  # 打印空闲卡（逗号分隔）
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v t="$GPU_BUSY_MIB" '$2 < t {print $1}' | paste -sd, -
}

wait_gpus() {  # wait_gpus N → 打印 N 张"连续 3 次检查（间隔 30 s）都空闲"的卡
  local n="$1" deadline=$(( $(date +%s) + WAIT_HOURS * 3600 )) f k
  while true; do
    f="$(free_gpus)"; k=0
    while (( k < 2 )) && [[ -n "$f" ]]; do
      sleep 30
      f="$(comm -12 <(tr ',' '\n' <<< "$f" | sort) <(free_gpus | tr ',' '\n' | sort) | paste -sd, -)"
      k=$(( k + 1 ))
    done
    if [[ -n "$f" ]] && (( $(tr ',' '\n' <<< "$f" | wc -l) >= n )); then
      tr ',' '\n' <<< "$f" | sort -n | head -n "$n" | paste -sd, -; return 0
    fi
    (( $(date +%s) > deadline )) && { log "等待 $n 张空闲卡超时（${WAIT_HOURS} h）" >&2; return 1; }
    sleep 30
  done
}

start_with_retry() {  # start_with_retry NAME N_GPUS OUTDIR -- SERVE_ARGS...（--gpus/--out/--wait 由本函数填）
  local name="$1" n="$2" dir="$3"; shift 4
  local attempt g
  for attempt in 1 2 3; do
    log "$name：等待 $n 张稳定空闲的卡（第 $attempt 次）"
    g="$(wait_gpus "$n")" || return 1
    log "$name：GPU $g"
    if "$@" --gpus "$g" --out "$dir/server" --wait; then
      STARTED+=("$name"); return 0
    fi
    scripts/stop_serve.sh "$name" >/dev/null 2>&1
    mv "$dir/server" "$dir/server-failed-$(date +%H%M%S)"
    log "$name：启动失败（多半是别人的任务同时占了卡），重试"
  done
  return 1
}

stop_started() {  # stop_started NAME：停止并从 STARTED 中移除
  scripts/stop_serve.sh "$1"; STARTED=("${STARTED[@]/$1}")
}
