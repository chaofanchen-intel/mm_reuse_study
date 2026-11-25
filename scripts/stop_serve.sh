#!/usr/bin/env bash
# 停止由 serve_*.sh 启动的服务：按 run/NAME.pid 结束整个进程组（含 worker 子进程），并在启动清单旁记录停止时间。
# 用法：scripts/stop_serve.sh NAME [NAME...]；scripts/stop_serve.sh --list 列出本仓库启动、仍在运行的服务。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/run"

if [[ "${1:-}" == "--list" ]]; then
  shopt -s nullglob
  for f in "$RUN_DIR"/*.pid; do
    n="$(basename "$f" .pid)"; p="$(cat "$f")"
    if kill -0 "$p" 2>/dev/null; then echo "$n  PID $p  $(cat "$RUN_DIR/$n.out" 2>/dev/null)"; else echo "$n  （已退出）"; fi
  done
  exit 0
fi
[[ $# -ge 1 ]] || { echo "用法：$0 NAME [NAME...] | --list" >&2; exit 1; }

rc=0
for name in "$@"; do
  pidfile="$RUN_DIR/$name.pid"
  if [[ ! -f "$pidfile" ]]; then echo "stop: 没有 $name 的 PID 文件" >&2; rc=1; continue; fi
  pgid="$(cat "$pidfile")"
  out="$(cat "$RUN_DIR/$name.out" 2>/dev/null || true)"
  if kill -0 "$pgid" 2>/dev/null || pgrep -g "$pgid" >/dev/null 2>&1; then
    kill -TERM -- "-$pgid" 2>/dev/null
    for _ in $(seq 1 60); do
      pgrep -g "$pgid" >/dev/null 2>&1 || break
      sleep 1
    done
    if pgrep -g "$pgid" >/dev/null 2>&1; then
      echo "stop: $name 60 s 内未退出，发送 KILL" >&2
      kill -KILL -- "-$pgid" 2>/dev/null
      sleep 2
    fi
  fi
  if pgrep -g "$pgid" >/dev/null 2>&1; then
    echo "stop: $name 仍有进程残留（进程组 $pgid）" >&2; rc=1; continue
  fi
  [[ -n "$out" && -d "$out" ]] && date +%Y-%m-%dT%H:%M:%S%z > "$out/stopped_at"
  rm -f "$pidfile" "$RUN_DIR/$name.out"
  echo "stop: $name 已停止"
done
exit $rc
