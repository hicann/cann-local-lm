# SymPack 优化验证

验证 torchair `optimize_sym_pack` Pass 消除冗余 Pack 算子的效果。对比两条路径：torchair torch.compile 路径和 epair 导出 + C++ 加载推理路径。

---

## 测试模型

### PackReproduceModel

int 参数 (m, n) 不等于任何 tensor 维度时，`_optimize_sym_input` 无法将 m/n 匹配到 tensor.size()。SymInt 作为独立 placeholder 进入 GE 图，变成 CPU Data 节点。Pack 以 Data 为输入，`_inputs_all_sym=True`：

```
Group A: view([m, n]) × 4 → 4 Pack[Data_m, Data_n] (3 冗余)
Group B: view([n, m]) × 2 → 2 Pack[Data_n, Data_m] (1 冗余)
```

optimize_sym_pack Pass 把 6 个冗余 Pack 合并成 2 个唯一 Pack，在每个 Data 的子图中消除全部冗余。

## 目录结构

```
sym_pack_optimization_case/
├── torchair_pack_optimization.py   # torchair torch.compile 路径 Pack 优化验证
├── export_pack_model.py            # torch.export + epair 导出 + Pack 分析
├── main.cpp                        # C++ EpairModelLoader 加载 + 推理 + 精度比对
├── CMakeLists.txt                  # C++ 构建配置
├── run.sh                          # 一键式验证脚本
└── README.md                       # 本文档
```

## 环境要求

1. **torch 2.9.0+cpu**
2. **torch_npu** + **torchair**
3. **torch-delegate** 已构建
4. **CANN 9.0.0+**
5. **ASCEND_HOME_PATH** 和 **TORCH_DELEGATE_ROOT** 已设置

## 快速开始

```bash
# 1. 加载 CANN 环境
source /path/to/cann/set_env.sh

# 2. 加载 TorchDelegate 环境（whl 安装后自动推导，无需 set_env.sh）
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)

# 3. 执行测试
cd examples/sym_pack_optimization_case
bash run.sh
```

## 详细步骤

### step 1: torchair torch.compile 路径 Pack 优化

```bash
python3 torchair_pack_optimization.py
```

使用 `torch.compile(dynamic=True)` + torchair backend，验证 `optimize_sym_pack` Pass 消除冗余 Pack。输出 pbtxt dump 到 `graph_dump_output/`。

### step 2: torch.export + epair 导出

```bash
python3 export_pack_model.py
```

生成文件：
- `generated/model.epair` — Epair 文件
- `generated/model.pt2` — PT2 归档
- `generated/input_*.bin` / `generated/input_meta.txt` — 测试输入数据
- `generated/ref_output.bin` — 参考输出

### step 3: 构建 C++ 示例

```bash
mkdir -p build && cd build
cmake .. \
    -DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT} \
    -DCANN_ROOT=${ASCEND_HOME_PATH} \
    -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cmake --build . -j$(nproc)
```

### step 4: C++ 推理 + 精度比对

```bash
TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${LD_LIBRARY_PATH:-}"

# 开启 GE Dump，导出编译后的图供 step 5 校验
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH="${PWD}/ge_graph_dump"

./build/sym_pack_optimization_case generated/model.epair
```

### step 5: 校验 GE Dump 图中不存在 Pack 算子

```bash
grep -rl 'type: "Pack"' ge_graph_dump/ --include='ge_proto*.txt' | wc -l
# 预期: 0
```

## 技术要点

### SymPack 优化原理

原始 GE 图中，每次 `view([m, n])` 产生一个 `Pack([Data_m, Data_n])`，相同参数的多次 view 产生重复 Pack。`optimize_sym_pack` 按输入语义分组去重：

```
优化前: GE Graph 有 3 个 Data([m, n, x]) + 6 个 Pack
优化后: GE Graph 有 3 个 Data([x, merged_A([m,n]), merged_B([n,m])]) + 0 个 Pack
```

EpairModelLoader 内部通过 InputBindingMap 自动适配输入：`user_inputs[x, m, n] → ge_inputs[x, merged_A([m,n]), merged_B([n,m])]`。
