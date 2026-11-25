#!/usr/bin/env bash
# G1：BAGEL image→text 的 S1–S4，两个框架、各两个前缀臂（DESIGN 5.1、11）。串行运行，结束后汇总。
#
# 用法：scripts/run_g1_bagel_cuda.sh            （环境变量可改：EXP REPEATS RES MAX_TOKENS WAIT_HOURS PHASES）
#   PHASES 默认 "mstar vo-pfx1 vo-pfx0"，可只跑其中几段。
# GPU：每段开始前等待足够的空闲卡（已用显存 < GPU_BUSY_MIB），最多等 WAIT_HOURS 小时；不与他人共卡。
# 退出（包括出错、Ctrl-C）时停止本脚本启动的服务。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env_cuda.sh

EXP="${EXP:-g1}"
REPEATS="${REPEATS:-8}"          # 每个臂的重复次数（M* 的两个臂各分两半交错运行）
RES="${RES:-512,980}"
MAX_TOKENS="${MAX_TOKENS:-32}"
WAIT_HOURS="${WAIT_HOURS:-12}"
PHASES="${PHASES:-mstar vo-pfx1 vo-pfx0}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1000}"
PY="$ROOT/envs/mstar/bin/python"
BASE="results/$EXP/$MM_MACHINE"
STARTED=()

log() { echo "[$(date +%H:%M:%S)] $*"; }
cleanup() { for n in "${STARTED[@]}"; do [[ -n "$n" ]] && scripts/stop_serve.sh "$n" >/dev/null 2>&1 && log "已停止 $n"; done; }
trap cleanup EXIT

free_gpus() {  # 打印空闲卡（逗号分隔）
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v t="$GPU_BUSY_MIB" '$2 < t {print $1}' | paste -sd, -
}

wait_gpus() {  # wait_gpus N → 打印 N 张"连续 3 次检查（间隔 30 s）都空闲"的卡
  local n="$1" deadline=$(( $(date +%s) + WAIT_HOURS * 3600 )) f stable="" k
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

start_with_retry() {  # start_with_retry NAME N_GPUS OUTDIR -- SERVE_ARGS...（--gpus 由本函数填）
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

run_mstar() {
  local d="$BASE/mstar/pfx-arms" half=$(( (REPEATS + 1) / 2 ))
  start_with_retry g1-mstar 1 "$d" -- scripts/serve_mstar_cuda.sh --name g1-mstar --model bagel \
    --config configs/bagel_mstar_i2t.yaml --port 18301 || return 1
  # 两个臂交错：pfx1 前半、pfx0 前半、pfx1 后半、pfx0 后半（种子不同，图像不同）
  for chunk in 0 1; do
    for arm in pfx1 pfx0; do
      local pc=on wu=1; [[ "$arm" == pfx0 ]] && pc=off
      [[ "$chunk" == 0 && "$arm" == pfx1 ]] && wu=3
      "$PY" -m bench.runner --server-dir "$d/server" --arm "enc0-$arm" --prefix-cache "$pc" \
        --sequences s1,s2,s3,s4 --res "$RES" --repeats "$half" --max-tokens "$MAX_TOKENS" \
        --seed "$chunk" --warmup "$wu" \
        --out "$d/records.jsonl" || return 1
    done
  done
  scripts/stop_serve.sh g1-mstar; STARTED=("${STARTED[@]/g1-mstar}")
}

run_vo() {  # run_vo pfx1|pfx0
  local arm="$1" cfg=configs/bagel_vllm_omni_audit.yaml d="$BASE/vllm-omni/$1"
  [[ "$arm" == pfx0 ]] && cfg=configs/bagel_vllm_omni_audit_noprefix.yaml
  # 两张卡：阶段 0 与 DiT 阶段各一张
  start_with_retry "g1-vo-$arm" 2 "$d" -- scripts/serve_vllm_omni_cuda.sh --name "g1-vo-$arm" \
    --model ByteDance-Seed/BAGEL-7B-MoT --served-name bagel --deploy-config "$cfg" --port 18401 --shim count \
    || return 1
  "$PY" -m bench.runner --server-dir "$d/server" --arm "enc1-$arm" --prefix-cache on \
    --sequences s1,s2,s3,s4 --res "$RES" --repeats "$REPEATS" --max-tokens "$MAX_TOKENS" \
    --out "$d/records.jsonl" || return 1
  scripts/stop_serve.sh "g1-vo-$arm"; STARTED=("${STARTED[@]/g1-vo-$arm}")
}

rc=0
for p in $PHASES; do
  case "$p" in
    mstar) run_mstar || { log "M* 段失败"; rc=1; } ;;
    vo-pfx1) run_vo pfx1 || { log "vLLM-Omni pfx1 段失败"; rc=1; } ;;
    vo-pfx0) run_vo pfx0 || { log "vLLM-Omni pfx0 段失败"; rc=1; } ;;
    *) log "未知段 $p"; rc=1 ;;
  esac
done

shopt -s nullglob
recs=("$BASE"/*/*/records.jsonl)
if (( ${#recs[@]} )); then
  python3 -m bench.summarize "${recs[@]}" > "$BASE/summary.md" && log "汇总：$BASE/summary.md"
fi
exit $rc
