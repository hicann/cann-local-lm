#!/usr/bin/env bash
# =============================================================================
# run_e2e_pipeline.sh — 串通 cli_export → OM 编译 → npu_verify 的端到端用例
#
# ① 导出 epair (cann_lm_converter.py --mode prefill-decode, lite_export_context, 无 stubs)
# ② OM 编译 prefill+decode → .om (cann_lm_compile, CPU 侧交叉编译)
# ③ NPU 出字验证 (set_env_cann --npu, DOPT_PYTHON+torch_npu; 仅 NPU_MODE=1 时跑)
#
# 用法:
#   bash run_e2e_pipeline.sh --hf-path /path/to/Qwen3.5-0.8B
#   bash run_e2e_pipeline.sh --hf-path /path/to/gemma-4-E2B-it --model-type gemma4
#   bash run_e2e_pipeline.sh --skip-npu                        # 只跑 ①② (无 NPU)
#   bash run_e2e_pipeline.sh --skip-export --hf-path /x        # 复用已有 epair, 只跑 ②③
#   bash run_e2e_pipeline.sh --check-env --model-type gemma4   # 只验 env (不跑全流程)
#
# seq_len 固定 64 (converter 代码层默认, CLI 不再暴露 --seq-len); gemma4 run_gemma4_gen.py PREFILL_SEQ 自定。
# --num-layers N: 只导出前 N 层 (slice/debug); 不传 = 全层。
# dtype: converter 按 --target-soc 派生 (Ascend→bf16, Kirin→fp16); Ascend 需 bf16
#        (ChunkGatedDeltaRule 在 910 只支持 bf16)。
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# source 共享 env (NPU 探测→NPU_MODE; CANN 自动 source; 路径派生; ensure 依赖; LD 真 CANN 无 stubs)
source "${SCRIPT_DIR}/../common/check_env.sh"

MODEL_TYPE="qwen3_5"
HF_PATH="${HF_PATH:-}"
PROMPT="Hello, my name is"
SEQ_LEN=64
OUTPUT_DIR=""
NUM_LAYERS=""
SKIP_EXPORT=0; SKIP_OM=0; SKIP_NPU=0; CHECK_ENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-path)     HF_PATH="$2"; shift 2;;
    --prompt)     PROMPT="$2"; shift 2;;
    --seq-len)    SEQ_LEN="$2"; shift 2;;
    --model-type) MODEL_TYPE="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --num-layers) NUM_LAYERS="$2"; shift 2;;
    --skip-export) SKIP_EXPORT=1; shift;;
    --skip-om)    SKIP_OM=1; shift;;
    --skip-npu)   SKIP_NPU=1; shift;;
    --check-env)  CHECK_ENV=1; shift;;
    -h|--help)    grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
    *) echo "unknown: $1" >&2; exit 1;;
  esac
done
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$SCRIPT_DIR/output/e2e_${MODEL_TYPE}"
if [[ $CHECK_ENV -eq 0 ]]; then
  [[ -n "$HF_PATH" ]] || { echo "需要 --hf-path 或 export HF_PATH=<model dir> (--check-env 不需要)" >&2; exit 1; }
fi

CONVERTER="$PROJECT_ROOT/cann_local_lm/converter/cann_lm_converter.py"
COMPILE_BIN="$PROJECT_ROOT/build/cann_lm_compile"
NPY_RUNNER_DIR="$SCRIPT_DIR/om_npu_verify"
case "$MODEL_TYPE" in
  qwen3_5) RUNNER="$NPY_RUNNER_DIR/run_qwen3_5_gen.py" ;;
  gemma4)  RUNNER="$NPY_RUNNER_DIR/run_gemma4_gen.py" ;;
  *) echo "model-type $MODEL_TYPE 暂只支持 qwen3_5/gemma4" >&2; exit 1 ;;
esac

phase() { echo ""; echo "==== [$1] $2 ===="; }
ok()   { echo "  [✓] $1"; }
ng()   { echo "  [✗] $1" >&2; }
die()  { ng "$1"; exit 1; }

# ── --check-env: 验 env 前置 (不跑全流程) ──
if [[ $CHECK_ENV -eq 1 ]]; then
  echo "################ e2e pipeline 环境检查 [NPU_MODE=$NPU_MODE] ################"
  echo "  MODEL_TYPE=$MODEL_TYPE  SOC=$SOC_VERSION  RUNNER=$(basename "$RUNNER")"
  echo ""
  echo "== ① 导出 epair (lite_export_context, 无 stubs) =="
  [[ -x "$CONVERTER_PYTHON" ]] && ok "CONVERTER_PYTHON" || ng "CONVERTER_PYTHON 缺失"
  ( cd /tmp && "$CONVERTER_PYTHON" -c "import torch,torchair,torch_delegate" ) 2>/dev/null && ok "venv: torch+torchair+torch_delegate" || ng "venv import 失败"
  [[ -f "$CONVERTER" ]] && ok "cann_lm_converter.py" || ng "converter 入口缺失"
  echo ""
  echo "== ② OM 编译 (cann_lm_compile, CPU 侧) =="
  command -v cmake &>/dev/null && ok "cmake" || ng "cmake 未找到"
  [[ -x "$COMPILE_BIN" ]] && ok "cann_lm_compile" || echo "  [!] cann_lm_compile 未构建 (bash scripts/build.sh cpp)"
  echo ""
  echo "== ③ NPU 出字 (DOPT_PYTHON + torch_npu) =="
  if [[ "$NPU_MODE" == "1" ]]; then
    "$DOPT_PYTHON" -c "import torch_npu" 2>/dev/null && ok "torch_npu" || echo "  [!] torch_npu 未装 (③ 需)"
    [[ -f "$RUNNER" ]] && ok "runner: $(basename "$RUNNER")" || ng "runner 缺失"
  else
    echo "  [○] NPU_MODE=0 → ③ 跳过 (无 NPU)"
  fi
  if [[ -n "$HF_PATH" ]]; then
    [[ -f "$HF_PATH/config.json" ]] && ok "HF model: $HF_PATH" || ng "HF model config.json 缺失"
  else echo "  [!] HF_PATH 未设 (e2e 跑需 --hf-path)"; fi
  echo ""
  echo "################ 完成 ################"
  exit 0
fi

# ── ① 导出 epair (lite_export_context; env 已由 check_env.sh 设好) ──
phase ① "导出 epair (lite_export_context, 无 stubs)"
if [[ $SKIP_EXPORT -eq 1 ]]; then ok "skip ① (--skip-export), 复用 $OUTPUT_DIR"; else
  (
    rm -rf "$OUTPUT_DIR"; mkdir -p "$OUTPUT_DIR"
    echo "  converter: --mode prefill-decode${NUM_LAYERS:+ --num-layers $NUM_LAYERS}"
    "$CONVERTER_PYTHON" "$CONVERTER" --hf-path "$HF_PATH" --output-dir "$OUTPUT_DIR" \
        --mode prefill-decode \
        ${NUM_LAYERS:+--num-layers "$NUM_LAYERS"} >"$OUTPUT_DIR.export.log" 2>&1 &
    PID=$!
    for _ in $(seq 1 120); do
      sleep 5
      if grep -q "Done. Output:" "$OUTPUT_DIR.export.log" 2>/dev/null; then break; fi
      if ! kill -0 "$PID" 2>/dev/null; then echo "  converter exited $(date +%T)"; break; fi
    done
    if kill -0 "$PID" 2>/dev/null; then echo "  output ready, kill hung tail $(date +%T)"; kill -9 "$PID"; fi
    pkill -9 -f "cann_lm_converter.*$(basename "$OUTPUT_DIR")" 2>/dev/null || true
  )
  [[ -f "$OUTPUT_DIR/prefill_ep_json_with_ascend_ir.json" ]] || die "① prefill ep_json 未产出 (见 $OUTPUT_DIR.export.log)"
  [[ -f "$OUTPUT_DIR/decode_ep_json_with_ascend_ir.json" ]] || die "① decode ep_json 未产出"
  [[ -f "$OUTPUT_DIR/weight_info.json" ]] || die "① weight_info 未产出"
  ok "① epair: prefill+decode ep_json + weight_info → $OUTPUT_DIR"
fi

# ── ② OM 编译 prefill+decode (cann_lm_compile; SOC 由 check_env.sh 设) ──
phase ② "OM 编译 prefill+decode (cann_lm_compile)"
if [[ $SKIP_OM -eq 1 ]]; then ok "skip ② (--skip-om)"; else
  [[ -x "$COMPILE_BIN" ]] || die "② cann_lm_compile 未构建: bash scripts/build.sh cpp"
  (
    cd "$OUTPUT_DIR"
    for st in prefill decode; do
      mkdir -p "$st"
      echo "  compile $st → $st/$st.om"
      "$COMPILE_BIN" "${st}_ep_json_with_ascend_ir.json" weight_info.json "${st}/${st}" \
          >"$st/compile.log" 2>&1 || true
      [[ -f "$st/$st.om" ]] || { ng "$st/$st.om 未产出 (见 $st/compile.log)"; exit 1; }
      ok "$st.om ($(ls -lh "$st/$st.om" | awk '{print $5}'))"
    done
  ) || die "② OM 编译失败"
  ok "② OM 就绪: $OUTPUT_DIR/{prefill,decode}/"
fi

# ── ③ NPU 出字 (仅 NPU_MODE=1; set_env_cann --npu) ──
phase ③ "NPU 出字"
if [[ $SKIP_NPU -eq 1 ]]; then ok "skip ③ (--skip-npu)"; else
  if [[ "$NPU_MODE" == "1" ]] && "$DOPT_PYTHON" -c "import torch_npu, torch; assert torch.npu.is_available()" 2>/dev/null; then
    ok "检测到 NPU + torch_npu → 跑 npu_verify"
    PREFILL_OM="$OUTPUT_DIR/prefill/prefill.om"
    DECODE_OM="$OUTPUT_DIR/decode/decode.om"
    [[ -f "$PREFILL_OM" && -f "$DECODE_OM" ]] || die "③ 缺 OM (先跑 ② 或 --skip-npu)"
    (
      source "$PROJECT_ROOT/scripts/cann_lm_compile/set_env_cann.sh" --npu >/dev/null 2>&1
      export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib"
      export PREFILL_OM DECODE_OM HF="$HF_PATH"
      export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${PYTHONPATH:-}"
      cd "$NPY_RUNNER_DIR"
      echo "  runner: $(basename "$RUNNER") \"$PROMPT\""
      "$DOPT_PYTHON" "$(basename "$RUNNER")" "$PROMPT"
    ) || die "③ npu_verify 失败"
    ok "③ npu_verify 出字完成"
  else
    echo "  [○] NPU_MODE=$NPU_MODE 或 torch_npu 未装 → 跳过 ③ (①② 已完成)"
  fi
fi

echo ""
echo "==== 端到端完成 (① 导出 epair → ② OM 编译 → ③ NPU 出字) ===="
echo "  epair: $OUTPUT_DIR/"
echo "  OM:    $OUTPUT_DIR/{prefill,decode}/*.om"
