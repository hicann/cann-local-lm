#!/bin/bash
# set_env_export.sh — ① converter env (已合并到 set_env.sh, 此为 back-compat 薄 wrapper)
# converter+compile 现复用一套: source scripts/set_env.sh
# 用法: source scripts/set_env_export.sh   (等价 source scripts/set_env.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/set_env.sh"
