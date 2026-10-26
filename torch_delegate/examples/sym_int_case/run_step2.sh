#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/../venv_test"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

cd "${SCRIPT_DIR}"

echo "========================================"
echo " Step 2: 导出模型 (Python)"
echo "========================================"

python export_model.py

if [[ -f "output/model_simple.epair" ]]; then
  echo "[✓] model_simple.epair 已生成"
else
  echo "[✗] model_simple.epair 未生成"
  exit 1
fi

echo ""
echo "解压并检查 JSON..."
mkdir -p output/extracted
cd output
unzip -o model_simple.epair -d extracted/ > /dev/null

echo ""
echo "检查 placeholders..."
python3 -c "
import json
with open('extracted/model_simple/extra/ep_json_with_ascend_ir.json') as f:
    data = json.load(f)

inputs = data['graph_module']['graph']['inputs']

print(f'找到 {len(inputs)} 个 placeholder:')
for i, inp in enumerate(inputs):
    if 'as_tensor' in inp:
        tensor_name = inp['as_tensor'].get('name', 'unknown')
        print(f'  [{i}] as_tensor: {tensor_name}')
    elif 'as_sym_int' in inp:
        sym_name = inp['as_sym_int'].get('as_name', 'unknown')
        print(f'  [{i}] as_sym_int: {sym_name} ✓ 动态非 tensor')
    elif 'as_int' in inp:
        value = inp['as_int']
        print(f'  [{i}] as_int: {value} ✓ 静态非 tensor')
    else:
        print(f'  [{i}] {inp}')

print()
print('检查 nodes...')
nodes = data['graph_module']['graph']['nodes']
converter_count = sum(1 for n in nodes if n.get('ascend_ir', {}).get('source') == 'converter')
fallback_count = sum(1 for n in nodes if n.get('ascend_ir', {}).get('source') == '')
print(f'  Converter 节点: {converter_count}')
print(f'  Fallback 节点: {fallback_count}')

print()
print('检查 range_constraints...')
range_constraints = data.get('range_constraints', {})
if range_constraints:
    print(f'找到 {len(range_constraints)} 个动态符号:')
    for sym, constraint in range_constraints.items():
        print(f'  - {sym}: min={constraint.get(\"min_val\")}, max={constraint.get(\"max_val\")}')
else:
    print('未找到 range_constraints')
"

echo ""
echo "[✓] 模型导出完成"
