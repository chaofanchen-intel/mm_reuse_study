#!/usr/bin/env bash
# E2（DESIGN 18）：多个摄像头会话同时运行，回答受别的会话"到达即算"的干扰有多大；M* 的编码器放置（与 Thinker 同卡 / 独占一卡）。
# 每个服务：先 1 个会话（无干扰的参照），再 CONC 个会话同时运行，各 REPS 轮。W 臂、grow、2 fps、长回答（64 token）。
#
# 用法：scripts/run_e2_multi_cuda.sh   （环境变量：EXP PHASES CONC REPS WAIT_HOURS GPU_BUSY_MIB）
#   PHASES 默认 "vo enc0 enc2"：vo = vLLM-Omni thinker-only TP2（2 卡）；enc0 / enc2 = M*（M1+M3(16)+M3b+M4，大步预算，3 卡），
#   编码器在 rank 0（与 Thinker 同卡）/ rank 2。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env_cuda.sh
EXP="${EXP:-e2-multi}"
PHASES="${PHASES:-vo enc0 enc2}"
CONC="${CONC:-3}"
REPS="${REPS:-2}"
WAIT_HOURS="${WAIT_HOURS:-12}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1000}"
PY="$ROOT/envs/mstar/bin/python"
BASE="results/$EXP/$MM_MACHINE"
source scripts/_queue_common.sh
mkdir -p "$BASE"; echo $$ > "$BASE/run.pid"

run_load() {  # run_load SERVER_DIR OUT
  local sd="$1" out="$2" r
  for r in $(seq "$REPS"); do
    "$PY" -m bench.stream_multi --server-dir "$sd" --arm W --sessions 1 --session-tag "solo-$r-$RANDOM" --out "$out" || return 1
    "$PY" -m bench.stream_multi --server-dir "$sd" --arm W --sessions "$CONC" --session-tag "conc-$r-$RANDOM" --out "$out" || return 1
  done
}

run_vo() {
  local d="$BASE/vllm-omni/multi"
  start_with_retry e2-vo 2 "$d" -- scripts/serve_vllm_omni_cuda.sh --name e2-vo \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct --served-name qwen3_omni --deploy-config configs/qwen3omni_vllm_omni_thinker.yaml \
    --port 18501 --shim count || return 1
  run_load "$d/server" "$d/records.jsonl" || return 1
  stop_started e2-vo
}

run_mstar() {  # run_mstar NAME CONFIG
  local name="$1" cfg="$2" d="$BASE/mstar/$1"
  export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 \
    MSTAR_PREPROCESS_CACHE_MB=4096
  start_with_retry "e2-mstar-$name" 3 "$d" -- scripts/serve_mstar_cuda.sh --name "e2-mstar-$name" --model qwen3_omni \
    --config "$cfg" --port 18502 --encoder-cache-mb 2048 || return 1
  run_load "$d/server" "$d/records.jsonl" || return 1
  stop_started "e2-mstar-$name"
}

rc=0
for p in $PHASES; do
  case "$p" in
    vo) run_vo || { log "vLLM-Omni 段失败"; rc=1; } ;;
    enc0) run_mstar enc0 configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml || { log "M* enc0 段失败"; rc=1; } ;;
    enc2) run_mstar enc2 configs/qwen3omni_mstar_thinker_tp2_bigstep_enc2.yaml || { log "M* enc2 段失败"; rc=1; } ;;
    *) log "未知段 $p"; rc=1 ;;
  esac
done
exit $rc
