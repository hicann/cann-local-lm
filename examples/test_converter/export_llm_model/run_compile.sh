#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# run_compile.sh — 一键式 epair→OM 编译 (source 共享 env + build + run + 存日志)
#
# 用法:
#   bash run_compile.sh                       # 默认: 当前目录找 ep_json + weight_info
#   bash run_compile.sh <epair_dir>            # 指定 epair 目录
#   bash run_compile.sh <model_json> <weights_index> [output_name]   # 显式路径
#
# 选项:
#   --no-build    跳过构建 (用已有 build/cann_lm_compile)
#   --keep-om     保留 OM 产物 (默认即保留)
#   --delete-om   跑完删 OM 产物 (OM+权值可达数 GB, 占盘时用)
#   --output <name>  OM 输出名
#   --log <path>     日志路径 (默认 /tmp/opencode/cann_lm_compile_run.log)
#=============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# source 共享 env (NPU 探测+CANN 自动 source+路径派生+ensure 依赖+LD 真 CANN 无 stubs)
source "$SCRIPT_DIR/../common/check_env.sh"

NO_BUILD=0
KEEP_OM=1
OUTPUT=""
LOG="/tmp/opencode/cann_lm_compile_run.log"
POS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)  NO_BUILD=1; shift;;
        --keep-om)   KEEP_OM=1; shift;;
        --delete-om) KEEP_OM=0; shift;;
        --output)    OUTPUT="$2"; shift 2;;
        --log)       LOG="$2"; shift 2;;
        -h|--help)   grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
        *)           POS+=("$1"); shift;;
    esac
done

# 默认 epair (无参时)
DEFAULT_EPAIR="$REPO/examples/test_converter/export_llm_model/cann_lm_cli_export/epair_qwen3_5_single"

# 解析位置参数 → model_json / weights
EPAIR_DIR=""
if [[ ${#POS[@]} -eq 0 ]]; then
    EPAIR_DIR="$DEFAULT_EPAIR"
elif [[ ${#POS[@]} -eq 1 ]]; then
    EPAIR_DIR="${POS[0]}"
else
    MODEL_JSON="${POS[0]}"
    WEIGHTS="${POS[1]}"
    [[ -n "${POS[2]:-}" ]] && OUTPUT="${POS[2]}"
fi

# 从 epair 目录自动找 ep_json + weight_info
if [[ -n "$EPAIR_DIR" ]]; then
    [[ -d "$EPAIR_DIR" ]] || { echo "ERROR: epair 目录不存在: $EPAIR_DIR" >&2; exit 1; }
    if [[ -f "$EPAIR_DIR/ep_json_with_ascend_ir.json" ]]; then
        MODEL_JSON="$EPAIR_DIR/ep_json_with_ascend_ir.json"
    else
        MODEL_JSON="$(ls "$EPAIR_DIR"/*_ep_json_with_ascend_ir.json 2>/dev/null | head -1 || true)"
    fi
    WEIGHTS="$EPAIR_DIR/weight_info.json"
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$REPO/build/om_output/$(basename "${EPAIR_DIR:-${MODEL_JSON%/*}}")"
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUTPUT")"
[[ -f "$MODEL_JSON" ]] || { echo "ERROR: model_json 不存在: $MODEL_JSON" >&2; exit 1; }
[[ -f "$WEIGHTS" ]]    || { echo "ERROR: weights_index 不存在: $WEIGHTS" >&2; exit 1; }

# build (除非 --no-build)
BIN="$REPO/build/cann_lm_compile"
if [[ "$NO_BUILD" -eq 0 ]] || [[ ! -x "$BIN" ]]; then
    echo "### [1/3] build cann_lm_compile (build_cpp.sh)"
    bash "$REPO/scripts/cann_lm_compile/build_cpp.sh" --target cann_lm_compile 2>&1 | tail -3
fi
[[ -x "$BIN" ]] || { echo "ERROR: cann_lm_compile 未构建: $BIN (跑 build_cpp.sh)" >&2; exit 1; }

# run
echo ""
echo "### [2/3] compile epair → OM"
echo "  binary     : $BIN"
echo "  model_json : $MODEL_JSON"
echo "  weights    : $WEIGHTS"
echo "  output     : ${OUTPUT}.om"
echo "  log        : $LOG"
echo ""
"$BIN" "$MODEL_JSON" "$WEIGHTS" "$OUTPUT" 2>&1 | tee "$LOG" || true

# 结果
echo ""
echo "### [3/3] result"
if [[ -f "${OUTPUT}.om" ]]; then
    ls -lh "${OUTPUT}.om" | awk '{print "  OM 产物: "$5, $9}'
    if [[ "$KEEP_OM" -eq 1 ]]; then
        echo "  (OM + 权值已保留, --delete-om 清理)"
    else
        rm -f "${OUTPUT}.om"
        echo "  (OM + 权值已清理)"
    fi
else
    echo "  ERROR: OM 未生成 (见 $LOG)" >&2
    exit 1
fi
echo "  日志: $LOG"
