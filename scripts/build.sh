#!/usr/bin/env bash
#=============================================================================
# scripts/build.sh — thin entry (反向透传到仓库根 build.sh)
#
# 构建逻辑现回归仓库根 build.sh (CI 流水线需根目录的完整 build.sh, 内容不可移走;
# 见根 build.sh)。本文件保留为 thin entry, 供任何引用 `bash scripts/build.sh` 的
# 历史入口/文档透传到根, 不破坏既有引用。
#   等价: bash build.sh "$@"
#=============================================================================
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../build.sh" "$@"
