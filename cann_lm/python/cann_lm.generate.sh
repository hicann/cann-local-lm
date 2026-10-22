#!/bin/sh
# cann_lm.generate — device shim for the CANN-LM generate CLI (no pip needed).
# Mirrors `mlx_lm.generate`: a command that invokes the python CLI.
#
# Deploy layout: this shim, the cann_lm/ package, bbit.py, and
# libcann_lm_engine.so all in one directory (e.g. deploy_ohos/). If your layout
# differs, override CANN_LM_LIB / PYTHONPATH in the environment.
#
#   cann_lm.generate --config testcase.json --prompt "你好"
#   cann_lm.generate --config testcase.json --prompt xxx --max-tokens 128 --stop </s>
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CANN_LM_LIB:=$HERE/libcann_lm_engine.so}"      # .so alongside the package
export CANN_LM_LIB
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"        # find cann_lm/ + bbit.py
export PYTHONPATH
# libc++_shared.so sits next to the engine .so; libneural_network_core.so is a
# system lib on the device (Hiai/NNRT). $HERE lets the .so resolve libc++.
LD_LIBRARY_PATH="$HERE${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
exec python -m cann_lm.generate "$@"
