#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Source CANN and activate executorch_npu before invoking this script.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="${1:-$repo_dir/build/gdn_proto/libchunk_gdn_proto.so}"
cann_root="${ASCEND_HOME_PATH:?Source CANN set_env.sh first}"
if [[ -f "$cann_root/include/graph/operator_reg.h" ]]; then
  cann_sdk="$cann_root"
else
  cann_sdk="$cann_root/aarch64-linux"
fi
mkdir -p "$(dirname "$output")"
"${CXX:-c++}" -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
  "$repo_dir/tools/chunk_gdn_proto.cpp" -I"$cann_sdk/include" \
  -L"$cann_sdk/lib64" -lgraph -o "$output"
printf '%s\n' "$output"
