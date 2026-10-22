#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$(pwd):${LD_LIBRARY_PATH:-}"

UTS=(model_executor_ut ut_engine_executor ut_tokenizer ut_c_api ut_scheduler ut_text_embedding)
SMOKE=(api_smoke c_api_smoke)
chmod +x bbit "${UTS[@]}" "${SMOKE[@]}" 2>/dev/null || true

run_args() { echo "===== $* ====="; local rc=0; "$@" || rc=$?; echo "(exit=$rc)"; }

if [ "$#" -gt 0 ]; then
  for t in "$@"; do run_args ./"$t"; done
  exit 0
fi

echo "── 单元测试 UT ──"
for t in "${UTS[@]}"; do run_args ./"$t"; done
echo "── 黑盒集成 bbit ──"
run_args ./bbit --check
run_args ./bbit
echo "── 冒烟 ──"
for t in "${SMOKE[@]}"; do run_args ./"$t"; done
echo "===== 全部完成 ====="
