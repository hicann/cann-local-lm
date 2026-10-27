# epair_runner — C++ epair加载与精度验证

简体中文 | [English](README_en.md)

本示例演示如何使用C++ `td::EpairModelLoader` 加载 `.epair` 文件并执行GE图推理，用于 `accuracy_compare` 流程中的Delegate路径精度验证。

## 目录结构

```text
epair_runner/
├── CMakeLists.txt       # 构建配置
├── epair_runner.cpp     # 主程序
├── tensor_io.h          # Tensor 二进制 I/O
├── tensor_io.cpp
└── README.md            # 本文档
```

## 前置条件

| 依赖 | 说明 |
|------|------|
| TorchDelegate | 已安装的torch_delegate whl（或源码构建的install目录） |
| CANN | 已安装，`ASCEND_HOME_PATH` 已设置 |
| libtorch | 独立libtorch分发包或Python torch wheel |
| libtorch_npu | （可选）用于自定义算子注册 |

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `ASCEND_HOME_PATH` | 是 | CANN安装路径 |
| `TORCH_DELEGATE_ROOT` | 是 | TorchDelegate install目录 |
| `TORCH_PKG_DIR` | 是 | Python torch包目录 |
| `TORCH_NPU_DIR` | 否 | libtorch_npu源码路径（用于 `LIBTORCH_NPU_ROOT`） |

## 构建与运行

### 设置环境

```bash
source /usr/local/Ascend/cann/set_env.sh
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
```

### 构建

```bash
cmake -S . -B build \
  -DTorchDelegate_ROOT=${TORCH_DELEGATE_ROOT} \
  -DCANN_ROOT=${ASCEND_HOME_PATH} \
  -DTORCH_PKG_DIR=${TORCH_PKG_DIR}
cmake --build build -j$(nproc)
```

### 运行

```bash
./build/epair_runner <epair_file> <output_dir> <device_id> <input0.bin> [input1.bin ...]
```

参数说明：
- `epair_file` — `.epair` 文件路径
- `output_dir` — 输出目录（保存 `output_*.bin`）
- `device_id` — NPU设备ID（如 `"0"`）
- `input*.bin` — 输入二进制文件（支持多输入）

### 与accuracy_compare配合使用

`accuracy_compare/run.sh compare` 会自动调用本二进制进行Delegate路径推理。

## 关键API

| API | 说明 |
|-----|------|
| `td::EpairModelLoader(archive_path)` | 加载epair并构建GE Graph |
| `loader.GetGEGraph(&graph)` | 提取GE Graph |
| `ge::GeSession(options)` | 创建GE会话 |
| `session.AddGraph(id, graph)` | 注册图 |
| `session.RunGraph(id, inputs, outputs)` | 同步运行 |

## 自定义算子

可通过 `CUSTOM_OP_LIBRARY_PATH` 环境变量指定自定义算子库路径：

```bash
export CUSTOM_OP_LIBRARY_PATH=/path/to/custom_op/lib
```

或在cmake中通过 `-DCUST_OPAPI_DIR=/path/to/custom_op/build` 指定 `libcust_opapi.so`。
