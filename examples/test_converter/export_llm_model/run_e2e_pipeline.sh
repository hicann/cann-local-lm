#!/usr/bin/env bash
# =============================================================================
# run_e2e_pipeline.sh — 串通 cli_export → OM 编译 → npu_verify 的端到端用例
#
# ① 导出 epair (cli_export: cann_lm_converter.py --mode prefill-decode, set_env_export ①, P1+stubs)
# ② OM 编译 prefill+decode → .om (set_env_cann ②, 真 CANN; cann_lm_compile 二进制)
# ③ NPU 出字验证 (set_env_cann --npu ③, om_npu_verify/run_qwen3_5_gen_emb.py)
#    仅当自动检测到 NPU + torch_npu 才跑; 否则跳过 ③ (①② 仍完成)
#
# 用法:
#   bash run_e2e_pipeline.sh                                   # 默认 qwen3.5, prompt="Hello, my name is", seq=8
#   bash run_e2e_pipeline.sh --hf-path /path/to/Qwen3.5-0.8B --prompt "讲个故事" --seq-len 8
#   bash run_e2e_pipeline.sh --skip-export --output-dir ./epair_qwen3_5_no_quant  # 复用已有 epair, 只跑 ②③
#   bash run_e2e_pipeline.sh --skip-npu                        # 只跑 ①② (无 NPU 机器)
#   bash run_e2e_pipeline.sh --model-type qwen3_5|gemma4
#
# 每步在子 shell 内 unset LD_LIBRARY_PATH 隔离, 防 ① stubs-first 污染 ② 真 CANN.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # CANN-LM_5883 (本脚本在 examples/test_converter/export_llm_model/)

MODEL_TYPE="qwen3_5"
HF_PATH="/mnt/workspace/models/Qwen/Qwen3.5-0.8B"
PROMPT="Hello, my name is"
SEQ_LEN=8
DTYPE="bfloat16"
OUTPUT_DIR=""
SKIP_EXPORT=0; SKIP_OM=0; SKIP_NPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-path)     HF_PATH="$2"; shift 2;;
    --prompt)     PROMPT="$2"; shift 2;;
    --seq-len)    SEQ_LEN="$2"; shift 2;;
    --dtype)      DTYPE="$2"; shift 2;;
    --model-type) MODEL_TYPE="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --skip-export) SKIP_EXPORT=1; shift;;
    --skip-om)    SKIP_OM=1; shift;;
    --skip-npu)   SKIP_NPU=1; shift;;
    -h|--help)    grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
    *) echo "unknown: $1" >&2; exit 1;;
  esac
done
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$SCRIPT_DIR/e2e_${MODEL_TYPE}"

CONVERTER="$PROJECT_ROOT/cann_lm/converter/cann_lm_converter.py"
COMPILE_BIN="$PROJECT_ROOT/build/cann_lm_compile"
NPY_RUNNER_DIR="$SCRIPT_DIR/om_npu_verify"
case "$MODEL_TYPE" in
  qwen3_5) RUNNER="$NPY_RUNNER_DIR/run_qwen3_5_gen_emb.py" ;;
  gemma4)  RUNNER="$NPY_RUNNER_DIR/run_gemma4_gen.py" ;;
  *) echo "model-type $MODEL_TYPE 暂只支持 qwen3_5/gemma4" >&2; exit 1 ;;
esac

phase() { echo ""; echo "==== [$1] $2 ===="; }
ok()   { echo "  [✓] $1"; }
ng()   { echo "  [✗] $1" >&2; }
die()  { ng "$1"; exit 1; }

# ── ① 导出 epair (cli_export, set_env_export ①, P1+stubs; 后台+轮询+kill 绕过 GE finalize 挂起) ──
phase ① "导出 epair (cli_export, set_env_export ①)"
if [[ $SKIP_EXPORT -eq 1 ]]; then ok "skip ① (--skip-export), 复用 $OUTPUT_DIR"; else
  (
    unset LD_LIBRARY_PATH
    source "$PROJECT_ROOT/scripts/set_env_export.sh" >/dev/null 2>&1
    rm -rf "$OUTPUT_DIR"; mkdir -p "$OUTPUT_DIR"
    echo "  converter: --mode prefill-decode --seq-len $SEQ_LEN --dtype $DTYPE"
    "$CONVERTER_PYTHON" "$CONVERTER" --hf-path "$HF_PATH" --output-dir "$OUTPUT_DIR" \
        --mode prefill-decode --seq-len "$SEQ_LEN" --dtype "$DTYPE" >"$OUTPUT_DIR.export.log" 2>&1 &
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
  ok "① epair: prefill+decode ep_json + weight_info  → $OUTPUT_DIR"
fi

# ── ② OM 编译 prefill+decode (set_env_cann ②, 真 CANN; 各自子目录隔离权值) ──
phase ② "OM 编译 prefill+decode (set_env_cann ②)"
if [[ $SKIP_OM -eq 1 ]]; then ok "skip ② (--skip-om)"; else
  [[ -x "$COMPILE_BIN" ]] || die "② cann_lm_compile 未构建: bash scripts/build.sh cpp"
  (
    unset LD_LIBRARY_PATH
    cd "$OUTPUT_DIR"
    for st in prefill decode; do
      mkdir -p "$st"
      echo "  compile $st → $st/$st.om (via run_compile.sh)"
      # run_compile.sh 自 source set_env_cann + 加 aarch64 LD (libascend_hal.so) + 处理 GE teardown 崩溃
      bash "$PROJECT_ROOT/scripts/cann_lm_compile/run_compile.sh" --no-build \
          "$OUTPUT_DIR/${st}_ep_json_with_ascend_ir.json" "$OUTPUT_DIR/weight_info.json" "$OUTPUT_DIR/${st}/${st}" \
          >"$st/compile.log" 2>&1 || true
      [[ -f "$st/$st.om" ]] || { ng "$st/$st.om 未产出 (见 $st/compile.log)"; exit 1; }
      ok "$st.om ($(ls -lh "$st/$st.om" | awk '{print $5}'))"
    done
  ) || die "② OM 编译失败"
  ok "② OM 就绪: $OUTPUT_DIR/{prefill,decode}/"
fi

# ── ③ NPU 出字 (set_env_cann --npu ③; 自动检测 NPU + torch_npu) ──
phase ③ "NPU 出字 (set_env_cann --npu ③)"
if [[ $SKIP_NPU -eq 1 ]]; then ok "skip ③ (--skip-npu)"; else
  source "$PROJECT_ROOT/scripts/build.conf"   # 拿 DOPT_PYTHON (仅变量, 不动 LD)
  if "$DOPT_PYTHON" -c "import torch_npu, torch; assert torch.npu.is_available(), 'no npu'" 2>/dev/null; then
    ok "检测到 NPU + torch_npu → 跑 npu_verify"
    PREFILL_OM="$OUTPUT_DIR/prefill/prefill.om"
    DECODE_OM="$OUTPUT_DIR/decode/decode.om"
    [[ -f "$PREFILL_OM" && -f "$DECODE_OM" ]] || die "③ 缺 OM (先跑 ② 或 --skip-npu): $PREFILL_OM / $DECODE_OM"
    (
      unset LD_LIBRARY_PATH
      source "$PROJECT_ROOT/scripts/set_env_cann.sh" --npu >/dev/null 2>&1
      export PREFILL_OM DECODE_OM
      export HF="$HF_PATH"
      export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${PYTHONPATH:-}"   # acl python
      cd "$NPY_RUNNER_DIR"
      echo "  runner: $(basename "$RUNNER") \"$PROMPT\""
      "$DOPT_PYTHON" "$(basename "$RUNNER")" "$PROMPT"
    ) || die "③ npu_verify 失败"
    ok "③ npu_verify 出字完成"
  else
    echo "  [○] 未检测到 NPU 或 torch_npu → 跳过 ③ (①② 已完成). 装 torch_npu + 有 NPU 时重跑可验 ③."
  fi
fi

echo ""
echo "==== 端到端完成 (① cli_export → ② OM 编译 → ③ npu_verify) ===="
echo "  epair: $OUTPUT_DIR/"
echo "  OM:    $OUTPUT_DIR/{prefill,decode}/*.om"
