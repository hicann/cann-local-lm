#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Source after activating Python; external setup scripts are optional overrides.
export EXECUTORCH_NPU_ROOT="${EXECUTORCH_NPU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -n "${CANN_ET_CONDA_ENV:-}" && "${CONDA_DEFAULT_ENV:-}" != "$CANN_ET_CONDA_ENV" ]]; then
  if [[ -n "${CONDA_SETUP_SCRIPT:-}" ]]; then
    source "$CONDA_SETUP_SCRIPT" || return
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || return
  else
    echo "Activate Python first, or set CONDA_SETUP_SCRIPT for CANN_ET_CONDA_ENV." >&2
    return 1
  fi
  conda activate "$CANN_ET_CONDA_ENV" || return
fi

if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
  source "$CANN_ENV_SCRIPT" || return
elif [[ -f "${ASCEND_HOME_PATH:-}/set_env.sh" ]]; then
  source "$ASCEND_HOME_PATH/set_env.sh" || return
fi
if [[ -n "${TORCH_DELEGATE_ENV_SCRIPT:-}" ]]; then
  source "$TORCH_DELEGATE_ENV_SCRIPT" || return
fi

export PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="$EXECUTORCH_NPU_ROOT:$EXECUTORCH_NPU_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${TORCH_DELEGATE_REPO:-}" ]]; then
  export PYTHONPATH="$TORCH_DELEGATE_REPO:$PYTHONPATH"
  if [[ -d "$TORCH_DELEGATE_REPO/build/release/csrc" ]]; then
    export LD_LIBRARY_PATH="$TORCH_DELEGATE_REPO/build/release/csrc${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  elif [[ -d "$TORCH_DELEGATE_REPO/build/bridge-delegate/csrc" ]]; then
    export LD_LIBRARY_PATH="$TORCH_DELEGATE_REPO/build/bridge-delegate/csrc${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

# Discover wheel locations without importing torch or initializing its backend.
_cann_et_paths=$(python - <<'PYTHON'
import importlib.util
import sys
from pathlib import Path
libraries = []
spec = importlib.util.find_spec("torch")
if spec and spec.submodule_search_locations:
    libraries.append(str(Path(next(iter(spec.submodule_search_locations))) / "lib"))
libraries.append(str(Path(sys.prefix) / "lib"))
print(":".join(p for p in libraries if Path(p).is_dir()))
spec = importlib.util.find_spec("executorch")
flatc = Path(next(iter(spec.submodule_search_locations))) / "data/bin/flatc" if spec and spec.submodule_search_locations else None
print(str(flatc) if flatc and flatc.is_file() else "-")
PYTHON
) || return
_cann_et_libs="${_cann_et_paths%%$'\n'*}"
_cann_et_flatc="${_cann_et_paths#*$'\n'}"
if [[ -n "$_cann_et_libs" ]]; then
  export LD_LIBRARY_PATH="$_cann_et_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -z "${FLATC_EXECUTABLE:-}" && "$_cann_et_flatc" != "-" ]]; then
  export FLATC_EXECUTABLE="$_cann_et_flatc"
fi
unset _cann_et_paths _cann_et_libs _cann_et_flatc
