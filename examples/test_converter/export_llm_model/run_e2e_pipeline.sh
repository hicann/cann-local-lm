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
#   bash run_e2e_pipeline.sh --hf-path /path/to/Qwen3.5-0.8B                  # qwen3.5, prompt="Hello, my name is", seq=8
#   bash run_e2e_pipeline.sh --hf-path /path/to/Qwen3.5-0.8B --prompt "讲个故事" --seq-len 8
#   HF_PATH=/path/to/Qwen3.5-0.8B bash run_e2e_pipeline.sh
#   bash run_e2e_pipeline.sh --skip-export --hf-path /x --output-dir ./output/epair_qwen3_5_no_quant  # 复用已有 epair, 只跑 ②③
#   bash run_e2e_pipeline.sh --skip-npu                        # 只跑 ①② (无 NPU 机器)
#   bash run_e2e_pipeline.sh --model-type qwen3_5|gemma4
#   bash run_e2e_pipeline.sh --check-env --model-type gemma4   # 只验 ①②③ env 前置 (不跑全流程)
#
# gemma4 注意: run_gemma4_gen.py PREFILL_SEQ=128, 故导出需 --seq-len 128 (qwen3.5 用默认 8).
#
# 每步在子 shell 内 unset LD_LIBRARY_PATH 隔离, 防 ① stubs-first 污染 ② 真 CANN.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # CANN-LM_5883 (本脚本在 examples/test_converter/export_llm_model/)

MODEL_TYPE="qwen3_5"
HF_PATH="${HF_PATH:-}"
PROMPT="Hello, my name is"
SEQ_LEN=8
DTYPE="bfloat16"
OUTPUT_DIR=""
SKIP_EXPORT=0; SKIP_OM=0; SKIP_NPU=0; CHECK_ENV=0

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
    --check-env)  CHECK_ENV=1; shift;;
    -h|--help)    grep -E '^# ' "$0" | sed 's/^# //'; exit 0;;
    *) echo "unknown: $1" >&2; exit 1;;
  esac
done
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$SCRIPT_DIR/output/e2e_${MODEL_TYPE}"
if [[ $CHECK_ENV -eq 0 ]]; then
  [[ -n "$HF_PATH" ]] || { echo "需要 --hf-path 或 export HF_PATH=<model dir> (--check-env 不需要)" >&2; exit 1; }
fi

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

# ── --check-env: 验 ①②③ 三套 env 前置 (不跑全流程) ──
check_env() {
  local pass=0 warn=0 fail=0
  _ok() { echo "  [✓] $1"; pass=$((pass+1)); }
  _wn() { echo "  [!] $1"; warn=$((warn+1)); }
  _ng() { echo "  [✗] $1"; fail=$((fail+1)); }
  echo "################ e2e pipeline 环境检查 (①②③) ################"
  echo "  PROJECT_ROOT=$PROJECT_ROOT"
  echo "  MODEL_TYPE=$MODEL_TYPE  RUNNER=$(basename "$RUNNER")"

  # ① 导出 epair (converter venv + torchair stubs; 子 shell source set_env_export 避免污染 ②③)
  echo ""; echo "== ① 导出 epair (converter venv + torchair stubs, CPU) =="
  local venv_py
  venv_py=$( ( unset LD_LIBRARY_PATH; source "$PROJECT_ROOT/scripts/set_env_export.sh" >/dev/null 2>&1; echo "$CONVERTER_PYTHON" ) )
  if [[ -x "$venv_py" ]]; then _ok "CONVERTER_PYTHON=$venv_py"; else _ng "CONVERTER_PYTHON 缺失 (set_env_export.sh)"; fi
  ( unset LD_LIBRARY_PATH; source "$PROJECT_ROOT/scripts/set_env_export.sh" >/dev/null 2>&1; "$CONVERTER_PYTHON" -c "import torch,torchair,torch_delegate" ) 2>/dev/null && _ok "venv: torch+torchair+torch_delegate" || _ng "venv import 失败"
  local stubs; stubs=$( ( unset LD_LIBRARY_PATH; source "$PROJECT_ROOT/scripts/set_env_export.sh" >/dev/null 2>&1; echo "${STUBS_DIR:-${TORCHAIR_ROOT:-}/build/stubs}" ) )
  [[ -d "$stubs" ]] && _ok "stubs: $stubs" || _ng "stubs 缺失"
  [[ -f "$CONVERTER" ]] && _ok "cann_lm_converter.py" || _ng "converter 入口缺失"

  # ② OM 编译 (真 CANN + cann_lm_compile; 只查文件/命令, 不 import)
  echo ""; echo "== ② OM 编译 (真 CANN + cann_lm_compile, CPU 侧) =="
  command -v cmake &>/dev/null && _ok "cmake $(cmake --version | head -1 | awk '{print $3}')" || _ng "cmake 未找到"
  [[ -x "$COMPILE_BIN" ]] && _ok "cann_lm_compile: $COMPILE_BIN" || _wn "cann_lm_compile 未构建 (bash scripts/build.sh cpp)"
  [[ -f "$PROJECT_ROOT/build/libtorch_delegate_backend.so" ]] && _ok "backend lib" || _wn "backend lib 缺失 (bash scripts/build.sh cpp)"
  source "$PROJECT_ROOT/scripts/build.conf" 2>/dev/null || true
  [[ -n "${ASCEND_HOME_PATH:-}" ]] && _ok "ASCEND_HOME_PATH=${ASCEND_HOME_PATH} (CANN)" || _ng "ASCEND_HOME_PATH 未设 (source CANN set_env.sh)"

  # ③ NPU 出字 (系统 python + torch_npu + acl + NPU; 子 shell source set_env_cann --npu, 无 stubs)
  echo ""; echo "== ③ NPU 出字 (DOPT_PYTHON + torch_npu + acl + NPU) =="
  if command -v npu-smi &>/dev/null; then
    _ok "NPU: $(npu-smi info 2>/dev/null | grep -oE 'Ascend[0-9]+' | head -1)"
  else _wn "npu-smi 不可用 (无 NPU → ③ 跳过)"; fi
  if [[ -x "${DOPT_PYTHON:-}" ]]; then
    # 镜像 e2e ③ 的完整 env (unset LD + set_env_cann --npu + aarch64 lib + acl python site-packages)
    if ( unset LD_LIBRARY_PATH ASCEND_VISIBLE_DEVICES ASCEND_HOME_PATH
          source "$PROJECT_ROOT/scripts/set_env_cann.sh" --npu >/dev/null 2>&1
          export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib"
          export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${PYTHONPATH:-}"
          "$DOPT_PYTHON" -c "import torch,torch_npu,acl" ) >/dev/null 2>&1; then
      _ok "DOPT_PYTHON: torch+torch_npu+acl"
    else _wn "DOPT_PYTHON 缺 torch_npu/acl"; fi
  else _ng "DOPT_PYTHON 缺失"; fi
  [[ -f "$RUNNER" ]] && _ok "runner: $(basename "$RUNNER")" || _ng "runner 缺失 ($RUNNER)"
  if [[ -n "$HF_PATH" ]]; then
    [[ -f "$HF_PATH/config.json" ]] && _ok "HF model: $HF_PATH" || _ng "HF model config.json 缺失 ($HF_PATH)"
  else _wn "HF_PATH 未设 (e2e 跑需 --hf-path)"; fi

  echo ""; echo "################ 汇总 ################"
  echo "  ✓ pass=$pass  [!] warn=$warn  [✗] fail=$fail"
  [[ $fail -eq 0 ]] && echo "  → 无硬缺项 (warn 可按子流程忽略)" || echo "  → 有 $fail 项硬缺, 按需补齐"
  echo "  (①导出 / ②OM编译 / ③NPU出字 各自只需对应检查项通过即可)"
}

if [[ $CHECK_ENV -eq 1 ]]; then check_env; exit 0; fi

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
      unset LD_LIBRARY_PATH ASCEND_VISIBLE_DEVICES ASCEND_HOME_PATH
      source "$PROJECT_ROOT/scripts/set_env_cann.sh" --npu >/dev/null 2>&1
      export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${ASCEND_HOME_PATH}/aarch64-linux/devlib"
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
