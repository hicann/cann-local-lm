#!/usr/bin/env bash
set -eo pipefail
#=============================================================================
# quant_export.sh — 两步式量化导出 (dopt GPTQ + converter epair 导出)
#
# Step 1  dopt_lm GPTQ 权重量化  (HF AutoModelForCausalLM, NPU)
#         系统 Python + 真 torch_npu (不加 stub LD_PRELOAD, 否则拦截真 NPU op)
#         → <output>/quant_output/model.safetensors  (GPTQ 修正 fp16 权重 + group_scale)
# Step 2  converter 消费 dopt 产物 → 物化 int4 → swap QuantLinear → export epair
#         venv Python + torchair stubs (CPU 导出, 不需 NPU)
#
# 两步用不同 Python 环境: dopt 需真 torch_npu (系统 Python 3.11.4), converter 需
# torchair stubs (venv). 本脚本各自配好环境后串行执行, 失败即停 (set -e).
#
# Usage:
#   ./scripts/quant_export.sh --hf-path /path/to/qwen3-0.6B --output-dir ./out
#   ./scripts/quant_export.sh --hf-path ... --group-size 128        # 自定义 group
#   ./scripts/quant_export.sh --hf-path ... --regen-dopt-config     # 为新模型重生 dopt_config
#   ./scripts/quant_export.sh --hf-path ... --gptq-only             # 只跑 step1
#   ./scripts/quant_export.sh --hf-path ... --export-only           # 只跑 step2 (复用已有 quant_output)
#   ./scripts/quant_export.sh --hf-path ... --num-layers 2          # converter 只导前 N 层 (调试)
#   ./scripts/quant_export.sh --hf-path ... --host-embeddings      # embed_tokens→host gather (W8), 图用外部 embedding (NPU 部署: embed 在 host, lm_head 在 NPU, untie)
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUANT_DIR="${PROJECT_ROOT}/cann_lm/converter/quant"

# ---------- environment paths ----------
CANN_SET_ENV="/home/developer/Ascend/cann/set_env.sh"
NPU_PYTHON="/opt/buildtools/Python-3.11.4/bin/python3"
TORCH_DELEGATE_VENV="/mnt/workspace/torch-delegate-harmony/.venv"
STUBS_DIR="/mnt/workspace/torchair/build/stubs"
TORCH_DELEGATE_HOME="/mnt/workspace/torch-delegate-harmony"
VENV_PYTHON="${TORCH_DELEGATE_VENV}/bin/python"

# ---------- defaults ----------
HF_PATH=""
OUTPUT_DIR="${PROJECT_ROOT}/quant_epair_out"
GROUP_SIZE=256
DTYPE="float16"
QUANT_ACT="w4a8"
NUM_LAYERS=""
DO_GPTQ=true
DO_EXPORT=true
REGEN_DOPT_CONFIG=false
HOST_EMBEDDINGS=false

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-path)             HF_PATH="$2"; shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2"; shift 2 ;;
        --group-size)          GROUP_SIZE="$2"; shift 2 ;;
        --dtype)               DTYPE="$2"; shift 2 ;;
        --quant-act)           QUANT_ACT="$2"; shift 2 ;;
        --num-layers)          NUM_LAYERS="$2"; shift 2 ;;
        --regen-dopt-config)   REGEN_DOPT_CONFIG=true; shift ;;
        --host-embeddings)     HOST_EMBEDDINGS=true; shift ;;
        --gptq-only)           DO_EXPORT=false; shift ;;
        --export-only)         DO_GPTQ=false; shift ;;
        -h|--help)             sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$HF_PATH" ]] || { echo "ERROR: --hf-path required" >&2; exit 1; }
[[ -f "$CANN_SET_ENV" ]] || { echo "ERROR: $CANN_SET_ENV not found" >&2; exit 1; }
[[ -x "$NPU_PYTHON" ]] || { echo "ERROR: NPU python not found: $NPU_PYTHON" >&2; exit 1; }
[[ -x "$VENV_PYTHON" ]] || { echo "ERROR: venv python not found: $VENV_PYTHON" >&2; exit 1; }
[[ -d "$STUBS_DIR" ]] || { echo "ERROR: stubs dir not found: $STUBS_DIR" >&2; exit 1; }
"$NPU_PYTHON" -c "import transformers, accelerate" 2>/dev/null \
    || { echo "ERROR: NPU python missing transformers/accelerate (dopt GPTQ needs them)" >&2; exit 1; }

QUANT_OUTPUT_DIR="${OUTPUT_DIR}/quant_output"
DOPT_CONFIG="${QUANT_OUTPUT_DIR}/dopt_config.json"
DOPT_CONFIG_TEMPLATE="${QUANT_DIR}/dopt_config.json"

# source CANN env once (exported vars inherited by step subshells); set +u guards
# set_env.sh referencing unset vars
set +u; # shellcheck disable=SC1090
source "$CANN_SET_ENV"; set -u

# ---------- prepare dopt_config (regen for new model, or copy template + group override) ----------
prepare_dopt_config() {
    mkdir -p "$QUANT_OUTPUT_DIR"
    if $REGEN_DOPT_CONFIG || [[ ! -f "$DOPT_CONFIG" ]]; then
        echo "  [dopt-config] regenerating scaffold for $HF_PATH"
        rm -f "$DOPT_CONFIG"
        # opt_main step1: dopt_config 不存在 → generate_config 生成脚手架 (type 字段填好) + exit
        (cd "$QUANT_DIR" && "$NPU_PYTHON" -m dopt_lm.opt_main \
            --model-path "$HF_PATH" --dopt-config "$DOPT_CONFIG" \
            --optimize-config "${QUANT_DIR}/config.yaml" --output-dir "$QUANT_OUTPUT_DIR") || true
        # 填策略: Linear→Quant_act_weight_eco (group_size 覆盖), Embedding→Quant_Embed_MinMax
        "$NPU_PYTHON" - "$DOPT_CONFIG" "$GROUP_SIZE" <<'PY'
import json, sys
path, gs = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    cfg = json.load(f)
for k, v in cfg["layer_strategy"].items():
    t = v.get("type", "")
    if "Linear" in t:
        v["quant_strategy"] = "Quant_act_weight_eco" if gs == 256 else f"Quant_act_weight_eco;{gs}"
    elif "Embedding" in t:
        v["quant_strategy"] = "Quant_Embed_MinMax"
with open(path, "w") as f:
    json.dump(cfg, f, indent=4)
print(f"  [dopt-config] filled strategies (group_size={gs}) -> {path}")
PY
    elif [[ "$GROUP_SIZE" -ne 256 ]]; then
        # 复用已有 config 但需覆盖 group_size (dopt Quant_act_weight_eco 默认 256)
        echo "  [dopt-config] override group_size=$GROUP_SIZE on template (assumes config matches model)"
        "$NPU_PYTHON" - "$DOPT_CONFIG_TEMPLATE" "$DOPT_CONFIG" "$GROUP_SIZE" <<'PY'
import json, sys
src, dst, gs = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = json.load(f)
for k, v in cfg["layer_strategy"].items():
    if str(v.get("quant_strategy", "")).startswith("Quant_act_weight_eco"):
        v["quant_strategy"] = f"Quant_act_weight_eco;{gs}"
with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)
PY
    else
        DOPT_CONFIG="$DOPT_CONFIG_TEMPLATE"  # 默认 group=256 直接用模板
    fi
}

# ---------- step 1: dopt GPTQ (NPU env, system python, real torch_npu) ----------
run_dopt_gptq() (
    echo "========== [Step 1/2] dopt GPTQ (NPU, ${NPU_PYTHON##*/}) =========="
    unset LD_PRELOAD          # 清掉 converter 步遗留的 stub LD_PRELOAD (会拦截真 NPU op)
    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1  # 预占 static TLS 槽, 修 set_env 后 transformers→torchaudio→libgomp 的 "cannot allocate memory in static TLS block"
    export DEVICE=npu
    export PYTHONPATH="${QUANT_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    prepare_dopt_config
    echo "  [gptq] model=$HF_PATH group=$GROUP_SIZE config=$DOPT_CONFIG"
    cd "$QUANT_DIR"  # calibration.json 相对 CWD, dopt_lm 包可被 -m 找到
    "$NPU_PYTHON" -m dopt_lm.opt_main \
        --model-path "$HF_PATH" \
        --dopt-config "$DOPT_CONFIG" \
        --optimize-config "${QUANT_DIR}/config.yaml" \
        --output-dir "$QUANT_OUTPUT_DIR" \
        --hf_type_save
    echo "  [gptq] done -> ${QUANT_OUTPUT_DIR}/model.safetensors"
)

# ---------- step 2: converter epair export (venv env, torchair stubs, CPU) ----------
run_converter() (
    echo "========== [Step 2/2] converter epair export (venv/stubs, ${VENV_PYTHON##*/}) =========="
    export LD_LIBRARY_PATH="${STUBS_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="${STUBS_DIR}/libacl_stub.so:${STUBS_DIR}/libge_local_stub.so:${STUBS_DIR}/libhccl_stub.so"
    export TORCH_DELEGATE_HOME="$TORCH_DELEGATE_HOME"
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
    unset DEVICE  # converter CPU 导出, 不需 DEVICE=npu
    local -a FLAGS=(
        --hf-path "$HF_PATH"
        --output-dir "$OUTPUT_DIR"
        --dtype "$DTYPE"
        --quant-path "$QUANT_OUTPUT_DIR"
        --quant-group-size "$GROUP_SIZE"
        --quant-act "$QUANT_ACT"
    )
    [[ -n "$NUM_LAYERS" ]] && FLAGS+=(--num-layers "$NUM_LAYERS")
    # host-embeddings: embed_tokens → host gather (W8), graph uses external embeddings.
    # Required for NPU deployment where embed is on host (untie from NPU-side lm_head).
    $HOST_EMBEDDINGS && FLAGS+=(--quant-embeddings --use-embeddings)
    echo "  [export] cann-lm-converter ${FLAGS[*]}"
    cd "$PROJECT_ROOT"
    "$VENV_PYTHON" "${PROJECT_ROOT}/cann_lm/converter/cann_lm_converter.py" "${FLAGS[@]}"
)

# ---------- run ----------
$DO_GPTQ && run_dopt_gptq
$DO_EXPORT && run_converter
echo "========== Done. Output: $OUTPUT_DIR =========="
