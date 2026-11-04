/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/operator_reg.h"
namespace ge {
REG_OP(ChunkGatedDeltaRule)
    .INPUT(query, TensorType({DT_BF16}))
    .INPUT(key, TensorType({DT_BF16}))
    .INPUT(value, TensorType({DT_BF16}))
    .INPUT(beta, TensorType({DT_BF16}))
    .INPUT(initial_state, TensorType({DT_BF16}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(g, TensorType({DT_FLOAT}))
    .OUTPUT(out, TensorType({DT_BF16}))
    .OUTPUT(final_state, TensorType({DT_BF16}))
    .ATTR(scale_value, Float, 1.0)
    .OP_END_FACTORY_REG(ChunkGatedDeltaRule) IMPLEMT_COMMON_INFERFUNC(ChunkGdnInfer) {
  if (op.UpdateOutputDesc("out", op.GetInputDesc("value")) != GRAPH_SUCCESS) return GRAPH_FAILED;
  return op.UpdateOutputDesc("final_state", op.GetInputDesc("initial_state"));
}
COMMON_INFER_FUNC_REG(ChunkGatedDeltaRule, ChunkGdnInfer);
}  // namespace ge
