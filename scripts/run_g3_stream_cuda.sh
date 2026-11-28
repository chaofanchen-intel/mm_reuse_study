#!/usr/bin/env bash
# G3：Qwen3-Omni 摄像头流 + 提问（DESIGN 16）。每个框架一个服务；Q 与 W 两臂交错；grow 与 slide 两种负载；
# 每种组合先 1 个 audit 会话（单个提问的计数），再 SESSIONS 个 timing 会话。
#
# 用法：scripts/run_g3_stream_cuda.sh   （环境变量：EXP PHASES SESSIONS FPS WAIT_HOURS GPU_BUSY_MIB VO_CONFIG MS_CONFIG）
#   PHASES 默认 "vo mstar0 mstar1"（mstar0 = 未改 M*，mstar1 = M1 编码器输出缓存 2048 MB）。
#   mstar4 = M1 + M3（每组 ≤ 16 张）+ M3b + M4，Thinker 每步预算 16384（configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml）。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/env_cuda.sh
EXP="${EXP:-g3-stream}"
PHASES="${PHASES:-vo mstar0 mstar1}"
SESSIONS="${SESSIONS:-3}"
FPS="${FPS:-2}"
WAIT_HOURS="${WAIT_HOURS:-12}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1000}"
VO_CONFIG="${VO_CONFIG:-configs/qwen3omni_vllm_omni_thinker.yaml}"
MS_CONFIG="${MS_CONFIG:-configs/qwen3omni_mstar_thinker_tp2.yaml}"
PY="$ROOT/envs/mstar/bin/python"
BASE="results/$EXP/$MM_MACHINE"
source scripts/_queue_common.sh
ARMS_TIMING="${ARMS_TIMING:-Q W W Q}"   # timing 会话的臂顺序（交错）
SKIP_AUDIT="${SKIP_AUDIT:-0}"
RECORDS="${RECORDS:-records.jsonl}"      # 每个服务目录旁的记录文件名
RUN_TAG="${RUN_TAG:-}"                   # 重跑时加后缀，服务目录与记录不与前一轮混在一起
mkdir -p "$BASE"; echo $$ > "$BASE/run.pid"   # 本脚本的 PID（setsid 启动时 $! 不是它）

GROW_Q="8,16,24"          # 帧数约 16、32、48（≤ 50，前缀一直有效）
SLIDE_Q="20,28,36,44"     # 25 s 时缓冲满（50 帧），丢最旧的 25 帧

run_arms() {  # run_arms SERVER_DIR OUT
  local sd="$1" out="$2" load q arm
  for load in grow slide; do
    q=$GROW_Q; [[ $load == slide ]] && q=$SLIDE_Q
    for arm in Q W; do
      [[ "$SKIP_AUDIT" == 1 ]] && break
      "$PY" -m bench.stream --server-dir "$sd" --arm $arm --load $load --mode audit --fps "$FPS" \
        --queries "$q" --sessions 1 --session-tag "audit-$arm" --out "$out" ${STREAM_EXTRA:-} || return 1
    done
    for arm in $ARMS_TIMING; do   # 交错，抵消时间漂移
      "$PY" -m bench.stream --server-dir "$sd" --arm $arm --load $load --mode timing --fps "$FPS" \
        --queries "$q" --sessions $(( (SESSIONS + 1) / 2 )) --session-tag "t-$arm" --out "$out" ${STREAM_EXTRA:-} || return 1
    done
  done
}

run_vo() {
  local d="$BASE/vllm-omni/stream$RUN_TAG"
  start_with_retry g3-vo 2 "$d" -- scripts/serve_vllm_omni_cuda.sh --name g3-vo \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct --served-name qwen3_omni --deploy-config "$VO_CONFIG" \
    --port 18501 --shim count || return 1
  run_arms "$d/server" "$d/$RECORDS" || return 1
  stop_started g3-vo
}

run_mstar() {  # run_mstar CACHE_MB：0 = 未改 M*（基线），>0 = 本研究的 M1（编码器输出缓存）
  local mb="$1" d="$BASE/mstar/stream-cache$1$RUN_TAG"
  # M0：5880 上跳过 Talker/Code2Wav 的 CUDA graph（文本输出不用）；每请求媒体文件上限 16 → 64
  export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64
  start_with_retry "g3-mstar-c$mb" 3 "$d" -- scripts/serve_mstar_cuda.sh --name "g3-mstar-c$mb" --model qwen3_omni \
    --config "$MS_CONFIG" --port 18502 --encoder-cache-mb "$mb" || return 1
  run_arms "$d/server" "$d/$RECORDS" || return 1
  stop_started "g3-mstar-c$mb"
}

run_mstar4() {  # M1 + M3(16) + M3b + M4 + 大步预算
  local d="$BASE/mstar/stream-m4$RUN_TAG"
  export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 \
    MSTAR_PREPROCESS_CACHE_MB=4096
  start_with_retry g3-mstar-m4 3 "$d" -- scripts/serve_mstar_cuda.sh --name g3-mstar-m4 --model qwen3_omni \
    --config configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml --port 18502 --encoder-cache-mb 2048 || return 1
  run_arms "$d/server" "$d/$RECORDS" || return 1
  stop_started g3-mstar-m4
  unset MSTAR_MEDIA_KEYS MSTAR_GROUP_IMAGES MSTAR_PREPROCESS_CACHE_MB
}

run_mstar5() {  # M4 段 + C1（前缀已覆盖的图不编码、不送像素；开头文本单独一步）
  local d="$BASE/mstar/stream-c1$RUN_TAG"
  export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 \
    MSTAR_PREPROCESS_CACHE_MB=4096 MSTAR_C1=1
  start_with_retry g3-mstar-c1 3 "$d" -- scripts/serve_mstar_cuda.sh --name g3-mstar-c1 --model qwen3_omni \
    --config configs/qwen3omni_mstar_thinker_tp2_bigstep.yaml --port 18502 --encoder-cache-mb 2048 || return 1
  run_arms "$d/server" "$d/$RECORDS" || return 1
  stop_started g3-mstar-c1
  unset MSTAR_MEDIA_KEYS MSTAR_GROUP_IMAGES MSTAR_PREPROCESS_CACHE_MB MSTAR_C1
}

run_mstar6() {  # C1 段 + C2（编码提前，编码器在 rank 2，图像 prefill 只捕获到 2048 token）
  local d="$BASE/mstar/stream-c2$RUN_TAG"
  export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 \
    MSTAR_PREPROCESS_CACHE_MB=4096 MSTAR_C1=1 MSTAR_C2=1
  start_with_retry g3-mstar-c2 3 "$d" -- scripts/serve_mstar_cuda.sh --name g3-mstar-c2 --model qwen3_omni \
    --config configs/qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml --port 18502 --encoder-cache-mb 2048 || return 1
  run_arms "$d/server" "$d/$RECORDS" || return 1
  stop_started g3-mstar-c2
  unset MSTAR_MEDIA_KEYS MSTAR_GROUP_IMAGES MSTAR_PREPROCESS_CACHE_MB MSTAR_C1 MSTAR_C2
}

run_mstar7() {  # C2 段 + C4（会话：已送达的帧发引用，c4_warm 提示把已预热的帧并成一组；客户端 --c4）
  local d="$BASE/mstar/stream-c4$RUN_TAG"
  export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 \
    MSTAR_PREPROCESS_CACHE_MB=4096 MSTAR_C1=1 MSTAR_C2=1 MSTAR_C4=1 MSTAR_C4_KEEP=4   # 2026-11-28 G3 C4 轮的规则
  start_with_retry g3-mstar-c4 3 "$d" -- scripts/serve_mstar_cuda.sh --name g3-mstar-c4 --model qwen3_omni \
    --config configs/qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml --port 18502 --encoder-cache-mb 2048 || return 1
  STREAM_EXTRA="--c4 --c4-hint older" run_arms "$d/server" "$d/$RECORDS" || return 1
  stop_started g3-mstar-c4
  unset MSTAR_MEDIA_KEYS MSTAR_GROUP_IMAGES MSTAR_PREPROCESS_CACHE_MB MSTAR_C1 MSTAR_C2 MSTAR_C4 MSTAR_C4_KEEP
}

rc=0
for p in $PHASES; do
  case "$p" in
    vo) run_vo || { log "vLLM-Omni 段失败"; rc=1; } ;;
    mstar0) run_mstar 0 || { log "M* 基线段失败"; rc=1; } ;;
    mstar1) run_mstar 2048 || { log "M* M1 段失败"; rc=1; } ;;
    mstar4) run_mstar4 || { log "M* M4 段失败"; rc=1; } ;;
    mstar5) run_mstar5 || { log "M* C1 段失败"; rc=1; } ;;
    mstar6) run_mstar6 || { log "M* C2 段失败"; rc=1; } ;;
    mstar7) run_mstar7 || { log "M* C4 段失败"; rc=1; } ;;
    *) log "未知段 $p"; rc=1 ;;
  esac
done
exit $rc
