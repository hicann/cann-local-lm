// =============================================================================
// reg_register.h — CANN-LM 自研算子注册
//
// CANN 库 (libge_compiler.so + opp 算子包) 已内置全部标准算子 (MatMul/Add/
// FusedInferAttentionScore/AddRmsNorm/RotaryMul 等 80+), 无需在此重复注册。
// 本文件仅注册 CANN 库中不包含的自研算子。
//
// 验证方式: 注释掉本文件后编译, GE 报 "not registered" 的算子即为需保留项。
// 参见 csrc/tools/README.md: "自研算子注册（仅 ChunkGatedDeltaRule；标准算子由 CANN opp 运行时加载）"
// =============================================================================

#include <graph/operator_reg.h>

namespace ge {

// ChunkGatedDeltaRule: qwen3.5 线性注意力融合算子
// (cann_local_lm::chunk_gated_delta_rule -> ge.ChunkGatedDeltaRule)
// CANN 库无此算子, 必须在此注册才能编译 qwen3.5 模型。
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

} // namespace ge
