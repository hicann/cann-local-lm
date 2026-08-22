#!/usr/bin/env bash
#
# run_pd6_cases.sh — 6 用例: qwen3.5 / gemma4 × {无量化, 场景1 --quant-path, 场景2 --quant-method gptq}
#
# 当前进展 (2026-08-19):
#   case_1  qwen3.5 无量化       ✓ 跑通 (weights 1571.9 MB, host_weights 508.6 MB)
#   case_2  gemma4  无量化       ✓ 跑通
#   case_3  qwen3.5 场景1        ✗ dopt 不支持 qwen3.5 混合架构 (linear_attn 无 self_attn → AttributeError)
#   case_4  gemma4  场景1        ✓ 跑通 (quant_info 8 字段全一致, n_quantized=277, weights 1377.3 MB)
#   case_5  qwen3.5 场景2        ✗ 同 case_3 (dopt 不支持 linear_attn)
#   case_6  gemma4  场景2        ✓ 跑通 (quant_info 与 case_4 一致)
#   gemma4 场景1 vs 场景2 对比:  quant_info + host_weights 完全一致; weights_0.bin md5 不同 (NPU 前向非确定性,
#                                非 dopt_config 差异 — dopt_config 279 层完全一致)
#   待解决: case_3/5 需改 dopt ptq_loop.py:163 (layer.self_attn 硬编码 → 兼容 linear_attn)
#
# 三套环境 (互不叠加):
#   converter env  CPU venv + torchair stubs (torch_npu 不装, 走 stub 路径)  → ① 导出 epair
#   dopt env       系统 python + 真 torch_npu + NPU                            → Step A GPTQ (量化前置)
#   OM 编译 / NPU 出字 env                                                      → 见 README (本脚本不含)
#
# 关键: converter venv 不装 torch_npu (设计决策, 详见 docs/.../cli_design.md
# 「converter 与 torch_npu 的关系」). converter 导出产出有效产物后进程会在 GE
# finalize/stub 收尾挂起 → run_conv 用 "后台跑 + 轮询产物 + kill" 绕过.
#
# 用法:
#   bash run_pd6_cases.sh 1        # qwen3.5 无量化 (基线)
#   bash run_pd6_cases.sh 2        # gemma4  无量化 (基线)
#   bash run_pd6_cases.sh 3        # qwen3.5 场景1 (--quant-path: 手动 dopt + converter 消费)
#   bash run_pd6_cases.sh 4        # gemma4  场景1 (--quant-path)
#   bash run_pd6_cases.sh 5        # qwen3.5 场景2 (--quant-method gptq: 一键式 auto_quant)
#   bash run_pd6_cases.sh 6        # gemma4  场景2 (--quant-method gptq)
#   bash run_pd6_cases.sh all      # 1→2→3→4→5→6
#   bash run_pd6_cases.sh setup    # 一次性: 确保 venv 不装 torch_npu (stub 路径)
#   bash run_pd6_cases.sh env      # 打印 env 设置说明
#
set -uo pipefail

PROJECT_ROOT="/mnt/workspace/project/CANN-LM_5883"
CLI_DIR="$PROJECT_ROOT/examples/test_converter/export_llm_model/cann_lm_cli_export"
VENV="/mnt/workspace/torch-delegate-harmony/.venv"
CONV="$PROJECT_ROOT/cann_lm/converter/cann_lm_converter.py"
DOPT_PY="/opt/buildtools/Python-3.11.4/bin/python3"
QD="$PROJECT_ROOT/cann_lm/converter/quant"
WORK_DIR="/tmp/opencode"

HF_QWEN3_5="/mnt/workspace/data/models/Qwen3.5-0.8B"
HF_GEMMA4="/mnt/workspace/data/models/gemma-4-E2B-it"

# ─── 一次性 setup: 确保 converter venv 不装 torch_npu (stub 路径) ───
setup_torch_npu() {
  "$VENV/bin/python3" -m pip uninstall -y torch_npu 2>&1 | tail -2
  local SP="$VENV/lib/python3.11/site-packages"
  if [[ -d "$SP/torch_npu" && ! -f "$SP/torch_npu/__init__.py" ]]; then
    rm -rf "$SP/torch_npu"; echo "removed leftover torch_npu dir"
  fi
  echo "verify (stub 路径):"
  setup_converter_env
  "$VENV/bin/python3" -c "
import torch; print('torch', torch.__version__)
try: import torch_npu; print('torch_npu STILL HERE (bad)')
except ModuleNotFoundError: print('torch_npu: absent (stub path OK)')
import torchair, torch_delegate; print('torchair+torch_delegate OK')
" 2>&1 | grep -v "PTA\|ERR99999\|UserWarning\|warnings.warn\|Failed to import\|Permission\|FunctionLoader" | tail -3
}

# ─── env: converter (CPU venv + torchair stubs) / dopt (系统 python + 真 NPU) ───
setup_converter_env() {
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/mnt/workspace/torchair/build/stubs:${LD_LIBRARY_PATH:-}"
}

setup_dopt_env() {
  source /home/developer/Ascend/cann/set_env.sh
  export PYTHONPATH="$QD:$PROJECT_ROOT:${PYTHONPATH:-}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export DEVICE=npu
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
}

# ─── 临时改 config.yaml 加速 (ptq_samples=16); 备份 + trap 还原 ───
_quick_config() {
  cp "$QD/config.yaml" "$QD/config.yaml.bak"
  printf 'dataset:\n  train_files:\n    - "calibration.json"\n  ptq_samples: 16\ncutoff_len: 64\nextra_training_config:\n  fp16: true\ncutoff_len: 64\n' > "$QD/config.yaml"
}
_restore_config() { mv "$QD/config.yaml.bak" "$QD/config.yaml" 2>/dev/null || true; }

# ─── converter 导出 (后台 + 轮询产物 + kill 挂起) ───
# run_conv <out_dir> <hf_path> <mode> <seq_len> [extra converter args...]
# 轮询次数/间隔可由环境变量覆盖 (场景2 跑 dopt GPTQ + export 慢, 调大)
run_conv() {
  local OUT="$1" HF="$2" MODE="$3" SEQ_LEN="${4:-8}"; shift 4
  local POLLS="${RUN_CONV_POLLS:-150}" SLEEP="${RUN_CONV_SLEEP:-5}"
  rm -rf "$OUT"
  echo "### converter mode=$MODE seq=$SEQ_LEN -> $(basename "$OUT")  $(date +%T)"
  "$VENV/bin/python3" "$CONV" \
    --hf-path "$HF" --output-dir "$OUT" --mode "$MODE" --dtype bfloat16 --seq-len "$SEQ_LEN" "$@" \
    >"$OUT.log" 2>&1 &
  local PID=$!
  for _ in $(seq 1 "$POLLS"); do
    sleep "$SLEEP"
    # 检查 "Done. Output:" 出现在 log 里 (convert() 写完所有产物后 print Done 再挂起)
    # 不能只检查文件存在 (json.dump 过程中文件已存在但不完整, kill 会截断)
    grep -q "Done. Output:" "$OUT.log" 2>/dev/null && break
    kill -0 "$PID" 2>/dev/null || { echo "  exited on its own $(date +%T)"; break; }
  done
  kill -0 "$PID" 2>/dev/null && { echo "  output ready, kill hung tail $(date +%T)"; kill -9 "$PID"; }
  pkill -9 -f "cann-lm-converter.*$(basename "$OUT")" 2>/dev/null
  sleep 1
  tail -3 "$OUT.log" 2>/dev/null | grep -v "Permission\|UserWarning\|warnings.warn\|Failed to import"
  echo "  -> $(basename "$OUT")  done"
}

# ─── 生成 dopt_config (量化前置, 任意模型通用) ───
gen_dopt_config() {
  local HF="$1" OUT="$2"
  echo "### generate dopt_config for $(basename "$HF")"
  "$VENV/bin/python3" - "$HF" "$OUT" <<'PYEOF'
import json, sys, shutil, torch
from cann_lm.converter.loader import load_model
from cann_lm.converter.quant.auto_quant import generate_dopt_config
art = load_model(sys.argv[1], dtype=torch.bfloat16, device="cpu")
cfg = generate_dopt_config(art, "w4a8")
shutil.copy(cfg, sys.argv[2])
from collections import Counter
d = json.load(open(sys.argv[2]))
c = Counter(v["quant_strategy"] for v in d["layer_strategy"].values())
print(f"  dopt_config: {sys.argv[2]}  ({len(d['layer_strategy'])} layers, {dict(c)})")
PYEOF
}

# ─── Step A: dopt GPTQ (NPU) ───
run_dopt() {
  local HF="$1" CFG="$2" OPTCFG="$3" OUT="$4"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "### dopt GPTQ -> $(basename "$OUT")  $(date +%T)  (cwd=quant/ 找 calibration.json)"
  ( cd "$QD" && timeout 1200 "$DOPT_PY" -m dopt_lm.opt_main \
      --model-path "$HF" --dopt-config "$CFG" --optimize-config "$OPTCFG" \
      --output-dir "$OUT" --hf_type_save ) 2>&1 | grep -v "Permission\|UserWarning\|warnings.warn" | tail -15
  echo "  -> $OUT  exit=${PIPESTATUS[0]}  $(date +%T)"
}

# ─── 验证量化产物 (IR 含 quant 算子 + quant_info + weights 大小) ───
# 无量化产物也适用 (IR 无 quant 算子=False, quant_info 空, weights 大)
verify_quant() {
  local OUT="$1"
  "$VENV/bin/python3" - "$OUT" <<'PYEOF' 2>&1 | tail -10
import json,sys; from pathlib import Path
d=Path(sys.argv[1])
eps = list(d.glob("*ep_json_with_ascend_ir.json"))
if not eps: print("  FAIL: 无 ep_json (导出失败, 见 log)"); sys.exit()
for ep in eps:
    s=json.dumps(json.load(open(ep)))
    print(f"  {ep.name}: quant_matmul_s8s4={'quant_matmul_s8s4' in s}, DynamicQuant={'DynamicQuant' in s}")
mc=json.load(open(d/"model_config.json")); qi=mc.get("quant_info") or {}
print("  quant_info:", {k:qi.get(k) for k in ("scheme","group_size","bit","n_quantized")} if qi else "(无, 未量化)")
w=d/"weights_0.bin"
if w.exists(): print(f"  weights_0.bin: {w.stat().st_size/1e6:.1f} MB")
hw=d/"host_weights_0.bin"
if hw.exists(): print(f"  host_weights_0.bin: {hw.stat().st_size/1e6:.1f} MB (embed host gather)")
PYEOF
}

# ─── 无量化: converter 直接导出 (基线对比) ───
# case_no_quant <name> <hf_path> <out_basename> <seq_len>
case_no_quant() {
  local NAME="$1" HF="$2" BASE="$3" SEQ_LEN="${4:-8}"
  setup_converter_env
  run_conv "$CLI_DIR/$BASE" "$HF" prefill-decode "$SEQ_LEN"
  verify_quant "$CLI_DIR/$BASE"
}

# ─── 场景1: --quant-path (手动 dopt GPTQ + converter 消费) ───
# case_quant_path <name> <hf_path> <out_basename> <seq_len>
case_quant_path() {
  local NAME="$1" HF="$2" BASE="$3" SEQ_LEN="${4:-8}"
  local DOPT_CFG="$WORK_DIR/dopt_config_${NAME}.json"
  local QOUT="$WORK_DIR/${NAME}_quant_output"
  _quick_config; trap _restore_config EXIT
  # Step A: dopt GPTQ (NPU env)
  setup_dopt_env
  gen_dopt_config "$HF" "$DOPT_CFG"
  run_dopt "$HF" "$DOPT_CFG" "$QD/config.yaml" "$QOUT"
  cp "$DOPT_CFG" "$QOUT/dopt_config.json"  # 让 _read_quant_param_map 读到完整参数
  # Step B: converter --quant-path (converter env)
  setup_converter_env
  run_conv "$CLI_DIR/$BASE" "$HF" prefill-decode "$SEQ_LEN" --quant-path "$QOUT"
  _restore_config; trap - EXIT
  verify_quant "$CLI_DIR/$BASE"
}

# ─── 场景2: --quant-method gptq (一键式, converter spawn dopt subprocess) ───
# case_quant_method <name> <hf_path> <out_basename> <seq_len>
case_quant_method() {
  local NAME="$1" HF="$2" BASE="$3" SEQ_LEN="${4:-8}"
  setup_converter_env
  _quick_config; trap _restore_config EXIT
  echo "### 场景2: --quant-method gptq ($NAME, 16 样本, seq=$SEQ_LEN) $(date +%T)"
  # 场景2 跑 dopt GPTQ + export, 慢; 加长轮询到 200×10s=33min (够 gemma4 5B)
  RUN_CONV_POLLS=200 RUN_CONV_SLEEP=10 run_conv "$CLI_DIR/$BASE" "$HF" prefill-decode "$SEQ_LEN" --quant-method gptq
  pkill -9 -f "dopt_wrapper" 2>/dev/null  # 清可能残留的 dopt subprocess
  _restore_config; trap - EXIT
  verify_quant "$CLI_DIR/$BASE"
}

# ─── 6 个用例: 无量化 → 场景1 (--quant-path) → 场景2 (--quant-method gptq) ───
# seq_len: qwen3.5=8 (run_qwen3_5_gen_emb.py PREFILL_SEQ=8), gemma4=128 (run_gemma4_gen.py PREFILL_SEQ=128)
case_1() { case_no_quant     qwen3_5 "$HF_QWEN3_5" epair_qwen3_5_no_quant      8;   }
case_2() { case_no_quant     gemma4  "$HF_GEMMA4"  epair_gemma4_no_quant       128; }
case_3() { case_quant_path   qwen3_5 "$HF_QWEN3_5" epair_qwen3_5_quant_path   8;   }
case_4() { case_quant_path   gemma4  "$HF_GEMMA4"  epair_gemma4_quant_path    128; }
case_5() { case_quant_method qwen3_5 "$HF_QWEN3_5" epair_qwen3_5_quant_method 8;   }
case_6() { case_quant_method gemma4  "$HF_GEMMA4"  epair_gemma4_quant_method  128; }

# ─── env 打印 ───
print_env() {
  cat <<'EOF'
=== converter env (① 导出 epair; 不装 torch_npu, 走 stub 路径) ===
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_HOME=/mnt/workspace/torch-delegate-harmony
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:/mnt/workspace/torchair/build/stubs:${LD_LIBRARY_PATH:-}
# venv: /mnt/workspace/torch-delegate-harmony/.venv (torch 2.9.0+cpu, torch_npu 不装)

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

=== NPU 出字 env (④, 系统 python + torch_npu) ===
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/mnt/workspace/project/CANN-LM_5883:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# /opt/buildtools/Python-3.11.4/bin/python3 run_<model>_gen.py "prompt"
EOF
}

# ─── main ───
case "${1:-help}" in
  1) case_1 ;;
  2) case_2 ;;
  3) case_3 ;;
  4) case_4 ;;
  5) case_5 ;;
  6) case_6 ;;
  all) case_1; case_2; case_3; case_4; case_5; case_6 ;;
  setup) setup_torch_npu ;;
  env) print_env ;;
  *) echo "用法: bash run_pd6_cases.sh {1|2|3|4|5|6|all|setup|env}"; echo "  1=qwen3.5无量化 2=gemma4无量化 3=qwen3.5场景1(--quant-path) 4=gemma4场景1 5=qwen3.5场景2(--quant-method gptq) 6=gemma4场景2 all=全部 setup=venv无torch_npu env=打印env" ;;
esac
