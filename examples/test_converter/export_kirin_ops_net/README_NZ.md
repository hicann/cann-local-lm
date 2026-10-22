# NZ 权重格式验证示例

本目录包含 NZ (FractNZ) 权重预转换的端到端验证用例(CLI-015 L2)。

## 用例

| 脚本 | 场景 | 说明 |
|------|------|------|
| `matmul_single_net.py` | bare-matmul (`x @ weight[K,N]`, 无 nn.Linear) | 最小验证(无 Transpose_1) |
| `transdata_probe.py` | 多场景探针(含 `single`/`bare_matmul` 等) | NZ 触发条件 + TransData 分析 |

## 通路验证步骤(A3, `Ascend910_9362`)

### 1. 导出(converter env, ①)

```bash
source scripts/set_env_export.sh
cd examples/test_converter/export_kirin_ops_net

# bare-matmul(无 Transpose_1)
python matmul_single_net.py export --output out/matmul_single.pt2
CANN_LM_ENABLE_WEIGHT_TRANS=1 python matmul_single_net.py compile \
    --ep-path out/matmul_single.pt2 --output-dir out/matmul_single_nz_epair

# nn.Linear(有 Transpose_1)
python transdata_probe.py export --net single --output out/single.pt2
CANN_LM_ENABLE_WEIGHT_TRANS=1 python transdata_probe.py compile \
    --ep-path out/single.pt2 --output-dir out/single_nz_epair

# 对照(ND, env off)
python matmul_single_net.py compile \
    --ep-path out/matmul_single.pt2 --output-dir out/matmul_single_nd_epair
python transdata_probe.py compile \
    --ep-path out/single.pt2 --output-dir out/single_nd_epair
```

### 2. OM 编译(CANN env, ②)

```bash
source scripts/set_env_cann.sh

# NZ 版
SOC_VERSION=Ascend910_9362 GE_EXTERNAL_WEIGHT=0 \
    build/cann_lm_compile out/matmul_single_nz_epair/ep_json_with_ascend_ir.json \
    out/matmul_single_nz_epair/weight_info.json out/matmul_single_nz

# ND 对照
SOC_VERSION=Ascend910_9362 GE_EXTERNAL_WEIGHT=0 \
    build/cann_lm_compile out/matmul_single_nd_epair/ep_json_with_ascend_ir.json \
    out/matmul_single_nd_epair/weight_info.json out/matmul_single_nd
```

### 3. NPU 输出对比(NPU env, ③)

```bash
source /home/developer/Ascend/cann/set_env.sh

# bare-matmul 对比
python3 examples/test_converter/export_kirin_ops_net/run_nz_npu_compare.py \
    --nz-om out/matmul_single_nz.om \
    --nd-om out/matmul_single_nd.om \
    --input-shape 1 128 --input-dtype float16 --input-value ones

# nn.Linear 对比
python3 examples/test_converter/export_kirin_ops_net/run_nz_npu_compare.py \
    --nz-om out/single_nz.om \
    --nd-om out/single_nd.om \
    --input-shape 1 128 --input-dtype float16 --input-value ones
```

期望输出:
```
[PASS] All outputs match (max_diff=0 expected). NZ OM == ND OM.
```

## 四件套(C++ 侧, 缺一不可)

| # | 组件 | 文件 | 作用 |
|---|------|------|------|
| ① | Python `nd2nz` | `weight_format/layout.py` | `[K,N]` → `permute(2,0,1,3)` → `[n1,k1,k0,n0]` |
| ② | Const 2D + FRACTAL_NZ | `ge_graph_builder.cc:CreateConstNode` | 不 reshape, GE 自己算 4D storage |
| ③ | skip Transpose | `ge_graph_builder.cc:PlanConverterNode` | 跳过 NZ 权重上的 Transpose(nn.Linear) |
| ④ | input NZ + `transpose_x2=True` | `ge_graph_builder.cc:PlanConverterNode` | MatMul input FRACTAL_NZ + 内部转置 |

## 关键配置

- **环境变量**: `CANN_LM_ENABLE_WEIGHT_TRANS=1`(开启 NZ 转换, 默认关)
- **SOC**: `Ascend910_9362`(A3)
- **GE_EXTERNAL_WEIGHT**: `0`(内嵌权重, 便于验证)

## 已验证结果(2026-09-07)

| 场景 | OM 结构 | NPU max_diff |
|------|---------|-------------|
| bare-matmul | Const FRACTAL_NZ + MatMul input NZ + 0 TransData | **0** |
| nn.Linear | Const FRACTAL_NZ + MatMul input NZ + 0 TransData + transpose_x2=True | **0** |
