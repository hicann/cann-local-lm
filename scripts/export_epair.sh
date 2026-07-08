#!/usr/bin/env bash
set -euo pipefail
#=============================================================================
# export_epair.sh — Export SimpleTransformer → .pt2 → .epair
#
# Sets up environment (correct Python, LD_PRELOAD stubs, PYTHONPATH)
# then delegates to export_to_epair.py.
#
# Usage:
#   ./scripts/export_epair.sh                       # export + compile
#   ./scripts/export_epair.sh --export-only         # export .pt2 only
#   ./scripts/export_epair.sh --compile-only        # compile existing .pt2
#   ./scripts/export_epair.sh --output-dir ./out    # custom output dir
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${PROJECT_ROOT}/generate"
DO_EXPORT=true
DO_COMPILE=true

# ---------- environment paths (tweak as needed) ----------
TORCH_DELEGATE_VENV="/mnt/workspace/torch-delegate-harmony/.venv"
STUBS_DIR="/mnt/workspace/torchair/build/stubs"
PYTHON_BIN="${TORCH_DELEGATE_VENV}/bin/python"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --export-only)  DO_COMPILE=false; shift ;;
        --compile-only) DO_EXPORT=false;  shift ;;
        --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---------- prerequisites ----------
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: .venv python not found at $PYTHON_BIN" >&2
    exit 1
fi
for lib in libacl_stub.so libge_local_stub.so libhccl_stub.so; do
    if [[ ! -f "${STUBS_DIR}/${lib}" ]]; then
        echo "ERROR: stub missing: ${STUBS_DIR}/${lib}" >&2
        exit 1
    fi
done

# ---------- flags to forward ----------
FLAGS=("--output-dir" "$OUTPUT_DIR")
$DO_EXPORT   && true   # export is default behaviour, no flag needed
$DO_COMPILE  || FLAGS+=("--export-only")
$DO_EXPORT   || FLAGS+=("--compile-only")

# ---------- run ----------
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_PRELOAD="${STUBS_DIR}/libacl_stub.so:${STUBS_DIR}/libge_local_stub.so:${STUBS_DIR}/libhccl_stub.so"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m cann_lm.converter.export.export_to_epair "${FLAGS[@]}"
