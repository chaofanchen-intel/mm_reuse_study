#!/usr/bin/env bash
# serve_*.sh 共用的函数。不要直接运行，由 serve_mstar_cuda.sh / serve_vllm_omni_cuda.sh source。
#
# 约定：
# - 每个服务有一个名字 NAME；PID 文件在 run/NAME.pid（进程组 ID，setsid 启动，停止时整组结束）。
# - 启动清单写到 --out 目录：manifest.json、server.log。runner 从 manifest.json 读取服务信息写进每条记录。
# - 启动前检查：同名服务未在运行；端口空闲；指定 GPU 的已用显存低于阈值（--allow-busy-gpu 跳过）。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/run"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-1000}"

die() { echo "serve: $*" >&2; exit 1; }

check_name_free() {
  local name="$1" pidfile="$RUN_DIR/$1.pid"
  if [[ -f "$pidfile" ]]; then
    local pid; pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      die "服务 $name 已在运行（PID $pid）；先运行 scripts/stop_serve.sh $name"
    fi
    rm -f "$pidfile"
  fi
}

check_port_free() {
  local port="$1"
  python3 - "$port" <<'EOF' || die "端口 $port 已被占用"
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)
EOF
}

check_gpus_free() {
  local gpus="$1" allow="$2" g used
  [[ "$allow" == 1 ]] && return 0
  IFS=',' read -ra arr <<< "$gpus"
  for g in "${arr[@]}"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')"
    [[ -n "$used" ]] || die "读不到 GPU $g 的显存"
    if (( used > GPU_BUSY_MIB )); then
      die "GPU $g 已用 ${used} MiB（阈值 ${GPU_BUSY_MIB}）；换卡或加 --allow-busy-gpu"
    fi
  done
}

# write_manifest OUT NAME FRAMEWORK PORT GPUS CONFIG SHIM_MODE PID -- CMD...
write_manifest() {
  local out="$1" name="$2" fw="$3" port="$4" gpus="$5" config="$6" shim="$7" pid="$8"; shift 9
  local ext="$ROOT/external/$fw"
  [[ "$fw" == "mstar" ]] || ext="$ROOT/external/vllm-omni"
  python3 - "$out" "$name" "$fw" "$port" "$gpus" "$config" "$shim" "$pid" "$ext" "${MM_MACHINE:-unknown}" "$@" <<'EOF'
import hashlib, json, os, socket, subprocess, sys, time
out, name, fw, port, gpus, config, shim, pid, ext, machine, *cmd = sys.argv[1:]
def git(*a):
    try:
        return subprocess.run(["git", "-C", ext, *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
sha = hashlib.sha256(open(config, "rb").read()).hexdigest() if config and os.path.isfile(config) else None
gpu_names = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader", "-i", gpus],
                           capture_output=True, text=True).stdout.strip().splitlines()
manifest = {
    "name": name, "framework": fw, "machine": machine, "host": socket.gethostname(),
    "port": int(port), "base_url": f"http://localhost:{port}", "gpus": gpus, "gpu_info": gpu_names,
    "config": config, "config_sha256": sha, "shim_mode": shim,
    "framework_commit": git("rev-parse", "HEAD"), "framework_dirty": bool(git("status", "--porcelain")),
    # 本研究对框架源码的改动：diff 的哈希（diff 本身用 git -C external/<fw> diff 查看）
    "framework_diff_sha256": hashlib.sha256((git("diff", "HEAD") or "").encode()).hexdigest(),
    "framework_untracked": (git("status", "--porcelain", "--untracked-files=all") or "").count("??"),
    "study_env": {k: v for k, v in os.environ.items() if k.startswith(("MSTAR_", "VO_AUDIT_", "VLLM_OMNI_")) and k != "MSTAR_SERVER_BIN"},
    "command": cmd, "cwd": os.getcwd(), "pid": int(pid),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": os.path.join(out, "server.log"),
}
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
EOF
}

# launch NAME OUT -- CMD...（在当前环境变量下以新会话启动，记录 PID）
launch() {
  local name="$1" out="$2"; shift 3
  mkdir -p "$RUN_DIR" "$out"
  setsid "$@" > "$out/server.log" 2>&1 < /dev/null &
  LAUNCHED_PID=$!
  echo "$LAUNCHED_PID" > "$RUN_DIR/$name.pid"
  echo "$out" > "$RUN_DIR/$name.out"
}

# wait_ready NAME OUT HEALTH_URL TIMEOUT_S
wait_ready() {
  local name="$1" out="$2" url="$3" timeout="$4" pid t0 now
  pid="$(cat "$RUN_DIR/$name.pid")"; t0="$(date +%s)"
  while true; do
    if curl -s -m 2 -o /dev/null -w '%{http_code}' "$url" | grep -q '^200$'; then
      echo "serve: $name 就绪（$url）"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "serve: $name 进程已退出，日志末尾：" >&2; tail -n 20 "$out/server.log" >&2; return 1
    fi
    if grep -q -E "Traceback|OutOfMemoryError|ValidationError" "$out/server.log"; then
      echo "serve: $name 启动出错，日志：" >&2
      grep -m 3 -E "Error|Traceback" "$out/server.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-300 >&2
      return 1
    fi
    now="$(date +%s)"
    (( now - t0 > timeout )) && { echo "serve: $name 启动超时（${timeout}s）" >&2; return 1; }
    sleep 3
  done
}
