#!/usr/bin/env bash
# check_env.sh — llm_models_run_on_cpu 用例环境前置
# 用法: source check_env.sh   (先 source 本文件再跑 debug_*/ 下的脚本)
# source 共享 env (common/check_env.sh: NPU 探测+CANN+路径+ensure+LD, lite 模式无 stubs)
# HF 路径为增量必填, 各 debug 脚本自行用 os.environ["XXX_HF_PATH"] 强制.
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/../common/check_env.sh"
echo "[check_env] llm_models_run_on_cpu 前置就绪 (NPU_MODE=$NPU_MODE, lite 无 stubs)"
echo "  converter python : ${CONVERTER_PYTHON}"
for v in GEMMA4_HF_PATH QWEN3_HF_PATH QWEN3_5_HF_PATH; do
  if [[ -n "${!v:-}" ]]; then echo "  ${v} = ${!v}"
  else echo "  [!] ${v} 未设 (加载该模型的脚本需先 export ${v}=<model dir>)"; fi
done
