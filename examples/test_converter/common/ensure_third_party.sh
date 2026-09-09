#!/usr/bin/env bash
# ensure_third_party.sh — 确保 output/third_party (json/gtest/makeself) 就绪 (幂等, 缺失自动 build)
# 依赖: PROJECT_ROOT (由 check_env.sh 设)
ensure_third_party() {
  if [[ -d "${PROJECT_ROOT}/output/third_party/json" ]]; then
    echo "[env] ✓ third_party: json/gtest/makeself"
    return 0
  fi
  echo "[env] third_party 缺失, 自动构建 (build_third_party.sh)..."
  bash "${PROJECT_ROOT}/scripts/cann_lm_compile/build_third_party.sh" \
    || { echo "[env] ✗ build_third_party.sh 失败"; return 1; }
  [[ -d "${PROJECT_ROOT}/output/third_party/json" ]] || { echo "[env] ✗ build 后 third_party 仍缺失"; return 1; }
  echo "[env] ✓ third_party (built)"
}
