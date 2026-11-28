#!/usr/bin/env bash
# G4：BAGEL"先上传图像、稍后提问"（DESIGN 17）。M* 基线、M* M1（ViT 输出缓存）、vLLM-Omni；Q 与 W 两臂；512 与 980。
# W 臂的预热请求在图像后附一句固定文本（WARMUP_TEXT）：M* 不把"最后一个 prefill 步"的 KV 放进前缀缓存，只含图像的预热不会被复用。
# 用法：scripts/run_g4_bagel_upload_cuda.sh   （环境变量：EXP PHASES SESSIONS QUESTIONS WAIT_HOURS GPU_BUSY_MIB）
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env_cuda.sh
EXP="${EXP:-g4-bagel-upload}"
PHASES="${PHASES:-mstar0 mstar1 vo}"
SESSIONS="${SESSIONS:-8}"
QUESTIONS="${QUESTIONS:-3}"
WAIT_HOURS="${WAIT_HOURS:-12}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1000}"
WARMUP_TEXT="${WARMUP_TEXT:-Look at this image.}"
PY="$ROOT/envs/mstar/bin/python"
BASE="results/$EXP/$MM_MACHINE"
source scripts/_queue_common.sh
mkdir -p "$BASE"; echo $$ > "$BASE/run.pid"

run_arms() {  # run_arms SERVER_DIR OUT
  local res arm
  for res in 512 980; do
    for arm in Q W W Q; do   # 交错
      "$PY" -m bench.upload --server-dir "$1" --arm $arm --sessions $(( SESSIONS / 2 )) --questions "$QUESTIONS" \
        --res $res --warmup-text "$WARMUP_TEXT" --session-tag "$arm-$res-$RANDOM" --out "$2" || return 1
    done
  done
}

run_mstar() {  # run_mstar CACHE_MB
  local mb="$1" d="$BASE/mstar/cache$1"
  start_with_retry "g4-ms-c$mb" 1 "$d" -- scripts/serve_mstar_cuda.sh --name "g4-ms-c$mb" --model bagel \
    --config configs/bagel_mstar_i2t.yaml --port 18601 --encoder-cache-mb "$mb" || return 1
  run_arms "$d/server" "$d/records.jsonl" || return 1
  stop_started "g4-ms-c$mb"
}

run_vo() {
  local d="$BASE/vllm-omni/pfx1"
  start_with_retry g4-vo 2 "$d" -- scripts/serve_vllm_omni_cuda.sh --name g4-vo \
    --model ByteDance-Seed/BAGEL-7B-MoT --served-name bagel --deploy-config configs/bagel_vllm_omni_audit.yaml \
    --port 18602 --shim count || return 1
  run_arms "$d/server" "$d/records.jsonl" || return 1
  stop_started g4-vo
}

rc=0
for p in $PHASES; do
  case "$p" in
    mstar0) run_mstar 0 || { log "M* 基线段失败"; rc=1; } ;;
    mstar1) run_mstar 1024 || { log "M* M1 段失败"; rc=1; } ;;
    vo) run_vo || { log "vLLM-Omni 段失败"; rc=1; } ;;
    *) log "未知段 $p"; rc=1 ;;
  esac
done
recs=("$BASE"/mstar/cache0/records.jsonl "$BASE"/mstar/cache1024/records.jsonl "$BASE"/vllm-omni/pfx1/records.jsonl)
"$PY" -m bench.upload_summary "${recs[@]}" > "$BASE/summary.md" && log "汇总：$BASE/summary.md"
exit $rc
