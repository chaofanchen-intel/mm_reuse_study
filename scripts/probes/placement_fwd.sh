#!/usr/bin/env bash
# usage: fwdprobe.sh NAME CONFIG
set -e
cd /home/chaofan/workspaces/mm_reuse_study
D=results/smoke/5880/mstar/placement-profile/$1; mkdir -p $D
export MSTAR_FWD_LOG=$PWD/$D/fwd.log MSTAR_C1=1 MSTAR_SPEECH_NODES_EAGER=1 MSTAR_MAX_MEDIA_FILES=64 MSTAR_MEDIA_KEYS=1 MSTAR_GROUP_IMAGES=16 MSTAR_PREPROCESS_CACHE_MB=4096 ${EXTRA_ENV}
scripts/serve_mstar_cuda.sh --name pp-$1 --model qwen3_omni --config $2 --gpus 0,1,2 --port 18512 --out $D/server --encoder-cache-mb 2048 --log-stats --wait | tail -1
PROF_WARM=1 PYTHONPATH=. envs/mstar/bin/python scripts/probes/cached_frames_prof.py 48 17,41 > $D/prof.txt 2>&1 || true
scripts/stop_serve.sh pp-$1 | tail -1
