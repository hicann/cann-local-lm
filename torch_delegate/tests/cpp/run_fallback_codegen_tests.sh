#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Compile and run test_fallback_codegen.cc independently

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="$PROJECT_ROOT/build"

CANN_ROOT="${CANN_ROOT:-/home/thj/Ascend/cann-9.0.0}"
TORCH_PKG_DIR="${TORCH_PKG_DIR:-/home/thj/code/pytorch/torch_build_env/lib/python3.10/site-packages/torch}"
TORCH_INCLUDE="${TORCH_INCLUDE:-${TORCH_PKG_DIR}/include}"
TORCH_LIB="${TORCH_LIB:-${TORCH_PKG_DIR}/lib}"
GTEST_ROOT="${GTEST_ROOT:-$PROJECT_ROOT/output/third_party/gtest}"
JSON_ROOT="${JSON_ROOT:-$PROJECT_ROOT/output/third_party/json}"

TEST_BINARY="$BUILD_DIR/test_fallback_codegen"

echo "Compiling test_fallback_codegen.cc..."
echo "CANN_ROOT: $CANN_ROOT"
echo "TORCH_PKG_DIR: $TORCH_PKG_DIR"

mkdir -p "$BUILD_DIR"

g++ -std=c++17 -g -Wall -Wextra \
    -I"$PROJECT_ROOT/csrc" \
    -I"$PROJECT_ROOT/csrc/inc/external" \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE/torch/csrc/api/include" \
    -I"$CANN_ROOT/include" \
    -I"$JSON_ROOT/include" \
    -I"$GTEST_ROOT/include" \
    "$SCRIPT_DIR/test_fallback_codegen.cc" \
    "$PROJECT_ROOT/csrc/fallback_codegen/fallback_codegen.cc" \
    -L"$BUILD_DIR/stub_lib" \
    -L"$TORCH_LIB" \
    -L"$CANN_ROOT/lib64" \
    -Wl,-rpath,"$BUILD_DIR/stub_lib" \
    -Wl,-rpath,"$TORCH_LIB" \
    -Wl,-rpath,"$CANN_ROOT/lib64" \
    -Wl,-rpath-link,"$BUILD_DIR/stub_lib" \
    "$GTEST_ROOT/lib/libgtest.a" \
    "$GTEST_ROOT/lib/libgtest_main.a" \
    -ltorch -ltorch_cpu -lc10 \
    -lascendcl -lascend_hal \
    -ldl \
    -lpthread \
    -o "$TEST_BINARY"

echo "Compilation successful!"
echo "Running tests..."
echo ""

export LD_LIBRARY_PATH="$BUILD_DIR/stub_lib:$TORCH_LIB:$CANN_ROOT/lib64:$LD_LIBRARY_PATH"
"$TEST_BINARY"

echo ""
echo "All tests passed!"
