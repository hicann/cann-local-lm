#!/usr/bin/env bash
# check_env.sh — llm_engine_on_cpu 用例环境前置 (① converter stubs, CPU-only)
# 用法: source check_env.sh   (先 source 本文件再跑 debug_*/ 下的脚本)
# 基底 set_env_export.sh 已设 CONVERTER_PYTHON/PYTHONPATH/stubs LD; HF 路径为增量必填,
# 各 debug 脚本自行用 os.environ["XXX_HF_PATH"] 强制 (不需 HF 的 --test-only 模式可不设).
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SELF_DIR}/../../.." && pwd)"
# source 基底 (① converter: CPU venv + torchair stubs; 内部也定义 SCRIPT_DIR, 故本脚本用 SELF_DIR)
source "${PROJECT_ROOT}/scripts/set_env_export.sh"
# 增量: HF 模型路径 (无机器默认; 加载权重的脚本必填, 不需 HF 的脚本可空)
echo "[check_env] llm_engine_on_cpu 前置就绪 (① converter 基底已 source)"
echo "  converter python : ${CONVERTER_PYTHON}"
for v in GEMMA4_HF_PATH QWEN3_HF_PATH QWEN3_5_HF_PATH; do
  if [[ -n "${!v:-}" ]]; then echo "  ${v} = ${!v}"
  else echo "  [!] ${v} 未设 (加载该模型的脚本需先 export ${v}=<model dir>)"; fi
done
