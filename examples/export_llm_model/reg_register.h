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

REG_OP(RmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(RmsNorm)

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
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axes, ListInt, {0})
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

}  // namespace ge
