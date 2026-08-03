#include <graph/operator_reg.h>

namespace ge {

std::vector<std::string> kAttrListString = {""};
std::vector<int64_t> kAttrListInt = {0};
std::vector<float> kAttrListFloat = {0.0f};
std::vector<bool> kAttrListBool = {false};
std::vector<ge::DataType> kAttrListType = {ge::DT_FLOAT};

REG_OP(Data)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

REG_OP(Const)
    .OUTPUT(y, ge::TensorType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_UINT8,
                               ge::DT_INT32, ge::DT_INT64, ge::DT_UINT32, ge::DT_UINT64, ge::DT_BOOL, ge::DT_DOUBLE}))
    .ATTR(value, Tensor, ge::Tensor())
    .OP_END_FACTORY_REG(Const)

REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)

REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(Mul)

REG_OP(Add)
    .INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8,
                           DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add)

REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_QINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT32, DT_INT16, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_QINT8, DT_BF16}))
    .OP_END_FACTORY_REG(Relu)

REG_OP(LayerNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(begin_norm_axis, Int, -1)
    .ATTR(begin_params_axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5)
    .OP_END_FACTORY_REG(LayerNorm)

REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo)

REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                          DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                          DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16,      DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,   DT_INT64,       DT_INT32,
                           DT_UINT8,     DT_UINT16,    DT_UINT32,     DT_UINT64,   DT_INT8,        DT_INT16,
                           DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,    DT_QUINT8,      DT_QINT16,
                           DT_QUINT16,   DT_QINT32,    DT_BOOL,       DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)

REG_OP(TransposeD)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16}))
    .REQUIRED_ATTR(perm, ListInt)
    .OP_END_FACTORY_REG(TransposeD)

REG_OP(InplaceUpdate)
    .INPUT(x, TensorType::ALL())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(InplaceUpdate)

REG_OP(GreaterEqual)
    .INPUT(x1, TensorType::ALL())
    .INPUT(x2, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(GreaterEqual)

REG_OP(LessEqual)
    .INPUT(x1, TensorType::ALL())
    .INPUT(x2, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LessEqual)

REG_OP(LogicalAnd)
    .INPUT(x1, TensorType({DT_BOOL}))
    .INPUT(x2, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalAnd)

REG_OP(LogicalNot)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(LogicalNot)

REG_OP(MaskedFill)
    .INPUT(x, TensorType::ALL())
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(value, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(MaskedFill)

REG_OP(TensorMove)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(TensorMove)

REG_OP(RmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(RmsNorm)

// AddRmsNorm: 残差加 + RMSNorm 融合算子 (cann_lm custom_op, ge_converter ge.AddRmsNorm)
// inputs x1/x2/gamma, outputs y/rstd/x, attr epsilon. (与 export_and_load_epair/reg_register.h 一致)
REG_OP(AddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNorm)

// RotaryMul: 半旋转位置编码融合算子 (cann_lm custom_op, ge_converter ge.RotaryMul)
// inputs x/r1(cos)/r2(sin) -> output y. op_info: aic-ascend910_93-ops-info-legacy.json
REG_OP(RotaryMul)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(r1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(r2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(RotaryMul)

REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32,
                          DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE,
                          DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16,
                          DT_QUINT16, DT_QINT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32,
                           DT_UINT8, DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE,
                           DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16,
                           DT_QUINT16, DT_QINT32, DT_BF16}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

REG_OP(ReduceMean)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceMean)

REG_OP(Rsqrt)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Rsqrt)

REG_OP(GatherV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherV2)

REG_OP(Reshape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

REG_OP(ConcatD)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32,
                                  DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_INT32,
                           DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatD)

REG_OP(SoftmaxV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE, DT_BF16,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE, DT_BF16,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(scale, Float, 1.0)
    .OP_END_FACTORY_REG(SoftmaxV3)

REG_OP(SoftmaxV2)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .ATTR(half_to_float, Bool, false)
    .OP_END_FACTORY_REG(SoftmaxV2)

REG_OP(Sigmoid)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sigmoid)

REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8, DT_UINT16,
                           DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8, DT_UINT16,
                           DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8, DT_UINT16,
                           DT_INT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Sub)

REG_OP(SliceD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(offsets, ListInt)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(SliceD)

REG_OP(StridedSliceV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceV2)

REG_OP(Unsqueeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)

REG_OP(ExpandDims)
    .INPUT(x, TensorType::BasicType())
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(ExpandDims)

REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_INT64, DT_BOOL, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                           DT_INT8, DT_INT64, DT_BOOL, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(Fill)

REG_OP(Triu)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .ATTR(diagonal, Int, 0)
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
    .OP_END_FACTORY_REG(Triu)

REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)

REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)

REG_OP(RealDiv)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(RealDiv)

REG_OP(Div)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(Div)

REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
                          DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
                           DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(Identity)

REG_OP(Softplus)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(Softplus)

REG_OP(IndexByTensor)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL}))
    .ATTR(indices_mask, ListInt, {})
    .OP_END_FACTORY_REG(IndexByTensor)

REG_OP(Reciprocal)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX128, DT_BF16}))
    .OP_END_FACTORY_REG(Reciprocal)

REG_OP(Gelu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Gelu)

REG_OP(Tanh)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Tanh)

REG_OP(Swish)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(scale, Float, 1.0)
    .OP_END_FACTORY_REG(Swish)

REG_OP(Range)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(limit, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(delta, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .ATTR(is_closed, Bool, false)
    .OP_END_FACTORY_REG(Range)

REG_OP(Cos)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Cos)

REG_OP(Sin)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sin)

REG_OP(Pow)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT32, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(Pow)

REG_OP(RepeatInterleave)
    .INPUT(x, TensorType::BasicType())
    .INPUT(repeats, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(RepeatInterleave)

REG_OP(Embedding)
    .INPUT(weight, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(Embedding)

REG_OP(Neg)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(Neg)

REG_OP(Abs)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT16, DT_UINT16,
                          DT_INT32, DT_INT64, DT_UINT8, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT16, DT_UINT16,
                           DT_INT32, DT_INT64, DT_UINT8, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Abs)

REG_OP(Exp)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Exp)

REG_OP(Log)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Log)

REG_OP(Sqrt)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sqrt)

REG_OP(Where)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(x, TensorType::BasicType())
    .INPUT(y, TensorType::BasicType())
    .OUTPUT(y_out, TensorType::BasicType())
    .OP_END_FACTORY_REG(Where)

REG_OP(Muls)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(Muls)

REG_OP(Adds)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(Adds)

REG_OP(ZerosLike)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(ZerosLike)

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

// 动态 shape 算子 (torchair 为符号维度图引入, CANN GE OperatorFactory 默认未注册)
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, 3)
    .OP_END_FACTORY_REG(Shape)

REG_OP(Expand)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_UINT8, DT_INT8, DT_BOOL, DT_INT64, DT_BF16}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_UINT8, DT_INT8, DT_BOOL, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(Expand)

REG_OP(Gather)
    .INPUT(x, TensorType({DT_INT64, DT_UINT64, DT_UINT32, DT_INT32, DT_FLOAT, DT_FLOAT16, DT_BF16,
                          DT_INT8, DT_UINT8, DT_INT16, DT_UINT16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT64, DT_UINT64, DT_UINT32, DT_INT32, DT_FLOAT, DT_FLOAT16, DT_BF16,
                           DT_INT8, DT_UINT8, DT_INT16, DT_UINT16}))
    .ATTR(validate_indices, Bool, true)
    .ATTR(batch_dims, Int, 0)
    .ATTR(is_preprocessed, Bool, false)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(Gather)

REG_OP(Minimum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT8, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT8, DT_BF16}))
    .OP_END_FACTORY_REG(Minimum)

REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(axis, Int)
    .ATTR(N, Int, 0)
    .OP_END_FACTORY_REG(Pack)

REG_OP(ScatterElements)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8, DT_BF16}))
    .ATTR(axis, Int, 0)
    .ATTR(reduction, String, "none")
    .OP_END_FACTORY_REG(ScatterElements)

REG_OP(SplitV)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL, DT_BF16}))
    .INPUT(size_splits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                                   DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL, DT_BF16}))
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

// Conv2D (aten.conv1d -> ge.Unsqueeze + Conv2D + Squeeze)
REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)

REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

// ── Quantization ops (W4A8 s8s4 export path) ──────────────────────────────
// QuantBatchMatmulV3: INTEGER int4/int8 matmul (act int8 + weight int4/int8).
//   Matches the integer-int4 materialize path (cann_lm materialize_int4 +
//   npu_quant_matmul). per-tensor/per-channel (no group_size attr) — for
//   per-group int4 the converter/materialize must align (遗留: int4 CANN 约定).
REG_OP(QuantBatchMatmulV3)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4}))
    .INPUT(scale, TensorType({DT_UINT64, DT_FLOAT, DT_INT64, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_INT32}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(QuantBatchMatmulV3)

// QuantBatchMatmulV4: A8W4 / MxA8W4 / A8W8 / A4W4 per-group matmul (antiquant+matmul fusion).
//   Per official proto (cann/ops-nn quant_batch_matmul_v4_proto.h): x1 supports
//   int8/int4 (integer) + float8 (mxfp8); x2 supports int4/int8 (integer) + float4_e2m1 (mxfp4).
//   W4A8 = x1 DT_INT8 + x2 DT_INT4 (prebuilt ND_ND_int8_int4 kernel, tilingKey=24).
REG_OP(QuantBatchMatmulV4)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_FLOAT4_E2M1, DT_INT4, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .OP_END_FACTORY_REG(QuantBatchMatmulV4)

// DynamicQuant: per-token dynamic act quant (fp16/bf16 x -> int8 y + fp32 scale).
//   W4A8 act side (npu_dynamic_quant -> ge.DynamicQuant via custom_op converter).
REG_OP(DynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuant)

}  // namespace ge
