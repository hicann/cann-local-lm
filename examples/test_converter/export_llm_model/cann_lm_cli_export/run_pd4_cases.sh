#!/usr/bin/env bash
#
# run_pd4_cases.sh — 精简版 4 用例: qwen3.5 / gemma4 × prefill-decode × use-emb × {无量化, 量化}
#
# 三套环境 (互不叠加):
#   converter env  CPU venv + torchair stubs (torch_npu 不装, 走 stub 路径)  → ① 导出 epair
#   dopt env       系统 python + 真 torch_npu + NPU                            → Step A GPTQ (量化前置)
#   OM 编译 / NPU 出字 env                                                      → 见 README (本脚本不含; 量化用例不要求 OM)
#
# 关键: converter venv **不装 torch_npu python 包** (设计决策: converter 不依赖真实 torch_npu,
#   详见 docs/cann_lm_converter/cli/cli_design.md「converter 与 torch_npu 的关系」):
#   - cann_lm/models/fused_ops/_library.py 顶层 `try: import torch_npu` → ImportError → _HAS_TORCH_NPU=False
#     (stub 路径), _library.py 自定义 npu:: schema+fake+CPU impl (含 npu_chunk_gated_delta_rule /
#     npu_dynamic_quant / quant_matmul_s8s4 等), torch.export 能 trace; cann_lm:: 算子本就自定。
#   - 无 torch_npu → torch_npu.dynamo 的 _LazyTorchair 代理不安装 → 真 (源码编译) torchair 直接用
#     (GE 转换能过的版本; 装 torch_npu 会让代理用 bundled wrapper 覆盖源码 torchair, wrapper 的
#     fx2ge_converter.py:1336 parse_input 有 AssertionError 在 input_ids placeholder 上失败, 源码已修);
#     _normalize_ge_option 走 `torch_npu not in sys.modules → "0"` 兜底, 不调 _get_deterministic_level。
#   - lib 层 stub (/mnt/workspace/torchair/build/stubs 在 LD_LIBRARY_PATH 最前) 给 torchair GE
#     引擎提供 C 符号, 避免真 NPU aclrtSetDevice→107001。无运行时 shim。
#   注: bug 只在 ① 导出 epair 的 GE 转换触发; dopt/OM 编译/出字/eager 不跑 FX→GE 转换, 不撞。
#   (历史: venv 曾误装 torch_npu 2.9.0 (无算子) → export 失败; 曾用 sitecustomize shim 阻断
#    import torch_npu 走 stub 路径; 曾试真 torch_npu 2.9.0.post6 + 重命名 dynamo/torchair 禁代理
#    (用官方算子), 能跑但依赖 torch_npu 版本 + 需重命名。现采用 stub 路径: uninstall torch_npu,
#    shim 弃用, converter 正确性不依赖任何 torch_npu 版本。)
#
# converter 导出产出有效产物后进程会在 GE finalize/stub 收尾挂起不退出 → run_conv 用
# "后台跑 + 轮询产物 + kill" 绕过 (产物在挂起前已落盘)。
#
# 用法:
#   bash run_pd4_cases.sh 1        # qwen3.5 pd use-emb 无量化
#   bash run_pd4_cases.sh 2        # gemma4  pd use-emb 无量化
#   bash run_pd4_cases.sh 3        # qwen3.5 pd use-emb 量化 (场景1: dopt GPTQ + --quant-path)
#   bash run_pd4_cases.sh 4        # gemma4  pd use-emb 量化 (场景1)
#   bash run_pd4_cases.sh 5        # qwen3 量化 场景2 (--quant-method gptq auto_quant 一键式)
#   bash run_pd4_cases.sh all      # 1→2→3→4→5
#   bash run_pd4_cases.sh setup    # 一次性: 确保 venv 不装 torch_npu (stub 路径)
#   bash run_pd4_cases.sh env      # 仅打印 env 设置说明
#
set -uo pipefail

PROJECT_ROOT="/mnt/workspace/project/CANN-LM_5883"
CLI_DIR="$PROJECT_ROOT/examples/test_converter/export_llm_model/cann_lm_cli_export"
VENV="/mnt/workspace/torch-delegate-harmony/.venv"
CONV="$PROJECT_ROOT/cann_lm/converter/cann_lm_converter.py"
DOPT_PY="/opt/buildtools/Python-3.11.4/bin/python3"
QD="$PROJECT_ROOT/cann_lm/converter/quant"
WORK_DIR="/tmp/opencode"           # dopt_config / quant_output 临时产物放这里 (不入仓)

# HF 模型 (本地已下载)
HF_QWEN3_5="/mnt/workspace/data/models/Qwen3.5-0.8B"
HF_QWEN3="/mnt/workspace/data/models/qwen3-0.6B"   # 最小, 场景2 量化快测用
HF_GEMMA4="/mnt/workspace/data/models/gemma-4-E2B-it"

# ─────────────────── 一次性 setup: 确保 converter venv 不装 torch_npu (stub 路径) ───────────────────
# 仅首次/环境重建后跑一次. stub 路径要求 venv 无 torch_npu python 包 (import torch_npu → ImportError
# → _library.py 走 stub 分支自提供 npu:: fake). 若误装了 torch_npu, 这里 uninstall + 清残留 dir.
setup_torch_npu() {
  "$VENV/bin/python3" -m pip uninstall -y torch_npu 2>&1 | tail -2
  local SP="$VENV/lib/python3.11/site-packages"
  # pip uninstall 可能留无 __init__.py 的 torch_npu 残留 dir (PEP 420 namespace package 仍可 import)
  if [[ -d "$SP/torch_npu" && ! -f "$SP/torch_npu/__init__.py" ]]; then
    rm -rf "$SP/torch_npu"; echo "removed leftover torch_npu dir (无 __init__.py 的 namespace 残留)"
  fi
  echo "verify (stub 路径, 无 shim):"
  setup_converter_env
  "$VENV/bin/python3" -c "
import torch; print('torch', torch.__version__)
try: import torch_npu; print('torch_npu STILL HERE (bad)')
except ModuleNotFoundError: print('torch_npu: absent (stub path OK)')
import torchair, torch_delegate; print('torchair+torch_delegate OK')
import cann_lm.models.fused_ops._library as lib; print('_HAS_TORCH_NPU:', lib._HAS_TORCH_NPU, '(want False)')
" 2>&1 | grep -v "PTA\|ERR99999\|UserWarning\|warnings.warn\|Failed to import\|Permission\|FunctionLoader" | tail -4
}

# ─────────────────── converter env (CPU venv + torchair stubs, 不装 torch_npu, 走 stub 路径) ───────────────────
setup_converter_env() {
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/mnt/workspace/torchair/build/stubs:${LD_LIBRARY_PATH:-}"
}

# ─────────────────── dopt env (系统 python + 真 torch_npu + NPU) ───────────────────
# 无 stubs (真 CANN lib64 + NPU); dopt 子进程用系统 python 自带的 torch_npu 跑 GPTQ
setup_dopt_env() {
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH="$QD:$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export DEVICE=npu
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
}

# ─────────────────── converter 导出 (后台 + 轮询产物 + kill 挂起) ───────────────────
# run_conv <out_dir> <hf_path> <mode> [extra converter args...]
run_conv() {
  local OUT="$1" HF="$2" MODE="$3"; shift 3
  rm -rf "$OUT"
  echo "### converter mode=$MODE -> $(basename "$OUT")  $(date +%T)"
  "$VENV/bin/python3" "$CONV" \
    --hf-path "$HF" --output-dir "$OUT" --mode "$MODE" --dtype bfloat16 --seq-len 8 "$@" \
    >"$OUT.log" 2>&1 &
  local PID=$!
  for _ in $(seq 1 150); do
    sleep 5
    [[ -f "$OUT/weight_info.json" && ( -f "$OUT/decode_ep_json_with_ascend_ir.json" || -f "$OUT/ep_json_with_ascend_ir.json" ) ]] && break
    kill -0 "$PID" 2>/dev/null || { echo "  exited on its own $(date +%T)"; break; }
  done
  kill -0 "$PID" 2>/dev/null && { echo "  output ready, kill hung tail $(date +%T)"; kill -9 "$PID"; }
  pkill -9 -f "cann-lm-converter.*$(basename "$OUT")" 2>/dev/null
  sleep 1
  tail -3 "$OUT.log" 2>/dev/null | grep -v "Permission\|UserWarning\|warnings.warn\|Failed to import"
  echo "  -> $(basename "$OUT")  done"
}

# ─────────────────── 生成 dopt_config (量化前置, 任意模型通用) ───────────────────
# gen_dopt_config <hf_path> <out_json>
gen_dopt_config() {
  local HF="$1" OUT="$2"
  echo "### generate dopt_config for $(basename "$HF")"
  "$VENV/bin/python3" - "$HF" "$OUT" <<'PYEOF'
import json, sys, shutil
import torch
from cann_lm.converter.loader import load_model
from cann_lm.converter.quant.auto_quant import generate_dopt_config
art = load_model(sys.argv[1], dtype=torch.bfloat16, device="cpu")
cfg = generate_dopt_config(art, "w4a8")   # w4a8 only; w4a16 NotImplementedError
shutil.copy(cfg, sys.argv[2])
from collections import Counter
d = json.load(open(sys.argv[2]))
c = Counter(v["quant_strategy"] for v in d["layer_strategy"].values())
print(f"  dopt_config: {sys.argv[2]}  ({len(d['layer_strategy'])} layers, {dict(c)})")
PYEOF
}

# ─────────────────── Step A: dopt GPTQ (NPU) ───────────────────
# run_dopt <hf_path> <dopt_config> <config_yaml> <out_dir>
run_dopt() {
  local HF="$1" CFG="$2" OPTCFG="$3" OUT="$4"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "### dopt GPTQ -> $(basename "$OUT")  $(date +%T)  (cwd=quant/ 找 calibration.json)"
  ( cd "$QD" && timeout 1200 "$DOPT_PY" -m dopt_lm.opt_main \
      --model-path "$HF" --dopt-config "$CFG" --optimize-config "$OPTCFG" \
      --output-dir "$OUT" --hf_type_save ) 2>&1 | grep -v "Permission\|UserWarning\|warnings.warn" | tail -15
  echo "  -> $OUT  exit=${PIPESTATUS[0]}  $(date +%T)"
}

# ─────────────────────────── Case 1: qwen3.5 无量化 ───────────────────────────
case_1() {
  setup_converter_env
  run_conv "$CLI_DIR/epair_qwen3_5" "$HF_QWEN3_5" prefill-decode
  "$VENV/bin/python3" "$CLI_DIR/verify.py" qwen3_5 2>&1 | tail -4
}

# ─────────────────────────── Case 2: gemma4 无量化 ────────────────────────────
case_2() {
  setup_converter_env
  run_conv "$CLI_DIR/epair_gemma4" "$HF_GEMMA4" prefill-decode
  "$VENV/bin/python3" "$CLI_DIR/verify.py" gemma4 2>&1 | tail -4
}

# ─────────────────────── Case 3: qwen3.5 量化 (场景1) ────────────────────────
# 场景1 (手动 dopt + --quant-path) 已实测通过. 场景2 (--quant-method gptq, auto_quant
# 一键式) 现也可用 (shim 已移除, dopt 子进程不再被污染), 但 dopt "曾静默卡住"+ 实测待做.
case_3() {
  local DOPT_CFG="$WORK_DIR/dopt_config_qwen3_5.json"
  local OPTCFG="$QD/config.yaml"            # ptq_samples=1024 cutoff=64; 快测可改 ptq_samples=16
  local QOUT="$WORK_DIR/qwen3_5_quant_output"
  # Step A: dopt GPTQ (NPU env)
  setup_dopt_env
  gen_dopt_config "$HF_QWEN3_5" "$DOPT_CFG"
  run_dopt "$HF_QWEN3_5" "$DOPT_CFG" "$OPTCFG" "$QOUT"
  # Step B: converter --quant-path (converter env)
  setup_converter_env
  run_conv "$CLI_DIR/epair_qwen3_5_quant" "$HF_QWEN3_5" prefill-decode --quant-path "$QOUT"
  # 量化产物核对 (IR 含 quant_matmul_s8s4 + weights 变小)
  "$VENV/bin/python3" - "$CLI_DIR/epair_qwen3_5_quant" <<'PYEOF' 2>&1 | tail -6
import json,sys; from pathlib import Path
q=Path(sys.argv[1]); s=json.dumps(json.load(open(q/"ep_json_with_ascend_ir.json")))
print("  IR quant_matmul_s8s4:", "quant_matmul_s8s4" in s, "| DynamicQuant:", "DynamicQuant" in s)
mc=json.load(open(q/"model_config.json")); qi=mc.get("quant_info") or {}
print("  quant_info:", {k:qi.get(k) for k in ("scheme","group_size","n_quantized")} )
print(f"  weights_0.bin: {(q/'weights_0.bin').stat().st_size/1e6:.1f} MB (int4 packed)")
PYEOF
}

# ─────────────────────── Case 4: gemma4 量化 (场景1) ──────────────────────────
# 注: dopt 对 gemma4 多模态 (ForConditionalGeneration) 有已知阻塞 (dopt 写死文本-only
# ForCausalLM); 本会话未实测 case 4. 静态 dopt_config.json 是 gemma4-E4B 专用.
case_4() {
  local DOPT_CFG="$WORK_DIR/dopt_config_gemma4.json"   # 或用静态 $QD/dopt_config.json (E4B)
  local OPTCFG="$QD/config.yaml"
  local QOUT="$WORK_DIR/gemma4_quant_output"
  setup_dopt_env
  gen_dopt_config "$HF_GEMMA4" "$DOPT_CFG"
  run_dopt "$HF_GEMMA4" "$DOPT_CFG" "$OPTCFG" "$QOUT"
  setup_converter_env
  run_conv "$CLI_DIR/epair_gemma4_quant" "$HF_GEMMA4" prefill-decode --quant-path "$QOUT"
  "$VENV/bin/python3" "$CLI_DIR/verify.py" gemma4 2>&1 | tail -4
}

# ─────────────────── Case 5: qwen3 量化 场景2 (auto_quant 一键式) ───────────────────
# `--quant-method gptq`: converter 主进程 (stub env) → auto_quant 检测 NPU → spawn dopt subprocess
# (系统 python + torch_npu + NPU) 跑 GPTQ → swap + export → 量化 epair。一条命令全程 (场景1 的两步合一)。
# 用 qwen3 (最小)。为快测, 临时把 quant/config.yaml 的 ptq_samples 改 16 (备份 + trap 保证还原)。
# 注: shim 已移除 (stub 路径), auto_quant 的 dopt 子进程不再被 shim PYTHONPATH 污染 → 场景2 可用。
case_5() {
  setup_converter_env
  local OUT="$CLI_DIR/epair_qwen3_qm"   # qm = quant-method
  # 临时 quick config (ptq_samples=16) 加速 dopt GPTQ; 备份 + trap 还原
  cp "$QD/config.yaml" "$QD/config.yaml.bak"
  printf 'dataset:\n  train_files:\n    - "calibration.json"\n  ptq_samples: 16\ncutoff_len: 64\nextra_training_config:\n  fp16: true\ncutoff_len: 64\n' > "$QD/config.yaml"
  _restore_qm_cfg(){ mv "$QD/config.yaml.bak" "$QD/config.yaml" 2>/dev/null || true; }
  trap _restore_qm_cfg EXIT
  echo "### case 5: --quant-method gptq (场景2 auto_quant, qwen3, 16 样本) $(date +%T)"
  "$VENV/bin/python3" "$CONV" --hf-path "$HF_QWEN3" --output-dir "$OUT" \
    --mode single --dtype bfloat16 --seq-len 8 --quant-method gptq >"$OUT.log" 2>&1 &
  local PID=$!
  for _ in $(seq 1 90); do
    sleep 10
    [[ -f "$OUT/weight_info.json" && -f "$OUT/ep_json_with_ascend_ir.json" ]] && break
    kill -0 "$PID" 2>/dev/null || { echo "  exited self $(date +%T)"; break; }
  done
  kill -0 "$PID" 2>/dev/null && { echo "  output ready, kill hung tail $(date +%T)"; kill -9 "$PID"; }
  pkill -9 -f "dopt_lm.opt_main" 2>/dev/null   # 清可能残留的 dopt 子进程
  _restore_qm_cfg; trap - EXIT
  sleep 1
  echo "=== log 尾 ==="; tail -4 "$OUT.log" 2>/dev/null | grep -v "Permission\|UserWarning\|warnings.warn\|Failed to import"
  # 量化产物核对 (IR 含 quant_matmul_s8s4 + weights 变小, 与场景1 同结构)
  "$VENV/bin/python3" - "$OUT" <<'PYEOF' 2>&1 | tail -4
import json,sys; from pathlib import Path
d=Path(sys.argv[1])
if not (d/"ep_json_with_ascend_ir.json").exists(): print("  FAIL: 无 ep_json (导出失败, 见 log)"); sys.exit()
s=json.dumps(json.load(open(d/"ep_json_with_ascend_ir.json")))
print("  IR quant_matmul_s8s4:", "quant_matmul_s8s4" in s, "| DynamicQuant:", "DynamicQuant" in s)
mc=json.load(open(d/"model_config.json")); qi=mc.get("quant_info") or {}
print("  quant_info:", {k:qi.get(k) for k in ("scheme","group_size","n_quantized")})
print(f"  weights_0.bin: {(d/'weights_0.bin').stat().st_size/1e6:.1f} MB (int4 packed, vs 非量化 bf16 ~1192 MB)")
PYEOF
}

print_env() {
  cat <<'EOF'
=== converter env (① 导出 epair; 不装 torch_npu, 走 stub 路径, 无 shim) ===
# 前置 (一次性): bash run_pd4_cases.sh setup  → 确保 venv 无 torch_npu (uninstall + 清残留 dir)
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:/mnt/workspace/torchair/build/stubs:${LD_LIBRARY_PATH:-}
# venv: /mnt/workspace/torch-delegate-harmony/.venv (torch 2.9.0+cpu, torch_npu 不装, torchair 源码编译)
# stub 路径: _library.py import torch_npu → ImportError → 自定义 npu:: fake (含 npu_chunk_gated_delta_rule)
# 设计决策: converter 不依赖真实 torch_npu (详见 docs/.../cli_design.md「converter 与 torch_npu 的关系」)

=== dopt env (Step A GPTQ, 量化前置; 系统 python + 真 torch_npu + NPU) ===
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883/cann_lm/converter/quant:/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export DEVICE=npu
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
# python: /opt/buildtools/Python-3.11.4/bin/python3 -m dopt_lm.opt_main ...

=== OM 编译 env (③, 真 CANN, CPU 侧不需 NPU) ===
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}/../torch.libs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export SOC_VERSION=Ascend910_9362
export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg   # {"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}

=== NPU 出字 env (④, 系统 python + torch_npu; 无 shim) ===
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# /opt/buildtools/Python-3.11.4/bin/python3 run_<model>_gen.py "prompt"
EOF
}

# ─────────────────────────── main ───────────────────────────
case "${1:-help}" in
  1) case_1 ;;
  2) case_2 ;;
  3) case_3 ;;
  4) case_4 ;;
  5) case_5 ;;
  all) case_1; case_2; case_3; case_4; case_5 ;;
  setup) setup_torch_npu ;;
  env) print_env ;;
  *) echo "用法: bash run_pd4_cases.sh {1|2|3|4|5|all|setup|env}"; echo "  1=qwen3.5无量化 2=gemma4无量化 3=qwen3.5量化(场景1) 4=gemma4量化(场景1) 5=qwen3量化(场景2 auto_quant) all=全部 setup=venv无torch_npu env=打印env" ;;
esac
