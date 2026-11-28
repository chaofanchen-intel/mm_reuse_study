#!/usr/bin/env bash
# C4 提示规则对照：同一服务（MSTAR_C4_KEEP，默认 0：预热组全部不带像素），W 臂交错跑 --c4-hint latest / older。
# 其余同 scripts/run_g3_stream_cuda.sh 的 mstar7 段。
#   GPUS=0,1,2 scripts/probes/c4_hint.sh [OUT_DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUT="${1:-results/smoke/5880/mstar/c4-profile/hint-keep${MSTAR_C4_KEEP:-0}}"
GPUS="${GPUS:-0,1,2}"
PY="$ROOT/envs/mstar/bin/python"
export MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 \
  MSTAR_PREPROCESS_CACHE_MB=4096 MSTAR_C1=1 MSTAR_C2=1 MSTAR_C4=1 MSTAR_C4_KEEP="${MSTAR_C4_KEEP:-0}"
mkdir -p "$OUT"
scripts/serve_mstar_cuda.sh --name c4-hint --model qwen3_omni \
  --config configs/qwen3omni_mstar_thinker_tp2_bigstep_enc2_c2.yaml --gpus "$GPUS" --port 18512 \
  --out "$OUT/server" --encoder-cache-mb 2048 --wait
trap 'scripts/stop_serve.sh c4-hint || true' EXIT
run() {  # run LOAD QUERIES HINT TAG
  "$PY" -m bench.stream --server-dir "$OUT/server" --arm W --load "$1" --mode timing --fps 2 --queries "$2" \
    --sessions 1 --session-tag "$4" --c4 --c4-hint "$3" --out "$OUT/records.jsonl"
}
run grow 8,16,24 older warm >/dev/null   # 首个大 prefill 的一次性开销
rm -f "$OUT/records.jsonl"
for rep in 1 2; do
  for hint in latest older; do run grow 8,16,24 $hint "$hint-$rep"; done
done
for rep in 1 2; do
  for hint in latest older; do run slide 20,28,36,44 $hint "$hint-$rep"; done
done
