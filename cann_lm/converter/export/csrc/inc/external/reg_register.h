// =============================================================================
// reg_register.h — 自研算子注册 (仅保留 CANN 未提供的自研算子)
//
// 标准算子 (MatMul/Const/Data/...) 不在此注册: CANN opp (libopsproto, GERT V2)
// 运行时由 libopp_registry 加载, 供 aclgrphBuildModel 编译用。
//
// 注意: 图构建 (GetGEGraph, ge::Operator V1 schema) 与 CANN GERT V2 是两套注册表,
//   libgraph 的 V1 OperatorFactory 默认空 (CANN 不再 V1 注册标准算子)。若 GetGEGraph
//   需标准算子 V1 schema, 需把对应 REG_OP 补回 (原 86 算子版备份在
//   /tmp/opencode/reg_register.h.bak, 见 docs/adr/0002)。
// =============================================================================
#include <graph/operator_reg.h>

namespace ge {

// qwen3.5 线性注意力融合算子 (cann_lm::chunk_gated_delta_rule -> ge.ChunkGatedDeltaRule)
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
    .ATTR(scale_value, Float, 1.0f)
    .OP_END_FACTORY_REG(ChunkGatedDeltaRule)

}  // namespace ge
