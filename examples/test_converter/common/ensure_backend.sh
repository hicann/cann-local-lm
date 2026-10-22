#!/usr/bin/env bash
# ensure_backend.sh — 确保 libtorch_delegate_backend.so 就绪 (幂等, 缺失自动 build_cpp.sh)
# 依赖: PROJECT_ROOT, ASCEND_HOME_PATH (由 check_env.sh 设)
ensure_backend() {
  if [[ -f "${BACKEND_LIB}" ]]; then
    echo "[env] ✓ backend lib: ${BACKEND_LIB##*/}"
    return 0
  fi
  echo "[env] backend .so 缺失, 自动构建 (build_cpp.sh)..."
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then echo "[env] ✗ ASCEND_HOME_PATH 未设"; return 1; fi
  bash "${PROJECT_ROOT}/scripts/cann_lm_compile/build_cpp.sh" --target torch_delegate_backend --no-tests --no-om \
    || { echo "[env] ✗ build_cpp.sh 失败"; return 1; }
  [[ -f "${BACKEND_LIB}" ]] || { echo "[env] ✗ build 后 .so 仍不存在: ${BACKEND_LIB}"; return 1; }
  echo "[env] ✓ backend lib (built): ${BACKEND_LIB##*/}"
}
