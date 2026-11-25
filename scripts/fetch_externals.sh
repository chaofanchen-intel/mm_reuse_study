#!/usr/bin/env bash
# 按 externals.pins 拉取上游源码到 external/<name>，并检出固定版本。
# 已存在的目录只检查版本，不覆盖本地修改（有未提交改动时报错退出）。
set -euo pipefail
cd "$(dirname "$0")/.."
while read -r name url ref; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue
  dir="external/$name"
  if [[ ! -d "$dir/.git" ]]; then
    echo "clone $name -> $dir"
    git clone -q "$url" "$dir"
  fi
  if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
    echo "error: $dir 有未提交改动，跳过检出" >&2; exit 1
  fi
  git -C "$dir" fetch -q --tags origin
  git -C "$dir" checkout -q "$ref"
  echo "$name @ $(git -C "$dir" rev-parse --short HEAD) ($ref)"
done < externals.pins
