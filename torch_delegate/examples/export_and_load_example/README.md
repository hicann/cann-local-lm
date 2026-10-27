# 示例 从EPAIR获取Graph并使用GeSession运行

简体中文 | [English](README_en.md)

本示例演示TorchDelegate安装包的最小C++接入路径：加载 `.epair`，通过 `td::EpairModelLoader` 提取 `ge::Graph`，再用 `ge::GeSession` 和 `gert::Tensor` 直接执行模型。

## 目录结构

```text
export_and_load_example/
├── CMakeLists.txt         # C++ 构建配置
├── main.cpp               # 最小 C++ 示例
├── export_ep_and_epair.py # 导出 pt2 和 epair
├── export_ep_and_epair_lite.py # [实验特性]无 CANN 运行时导出 pt2 和 epair
├── load_epair_run_eager.py      # Python 侧直接加载 epair 运行
├── compare_single_op_vs_delegate.py  # 单算子 vs Delegate 精度比对
├── run.sh                         # 一键式验证脚本
└── README.md                      # 本文档
```

## 使用方法

### 前置条件

1. 安装CANN并source环境
2. 构建并安装TorchDelegate whl（仓库根目录 `./build.sh` 后 `pip install output/package/torch_delegate-*.whl`，或直接 `bash examples/prepare_env.sh`）

### 一键式运行

```bash
# 1. 设置 CANN 环境（根据实际安装路径替换）
source /usr/local/Ascend/cann/set_env.sh

# 2. 设置 TorchDelegate 环境（whl 安装后自动推导，无需 set_env.sh）
export TORCH_DELEGATE_ROOT=$(torch-delegate-config --cmake-prefix)
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_DELEGATE_ROOT}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# 3. 一键式运行
cd examples/export_and_load_example
bash run.sh
```

`run.sh` 自动完成：
1. 环境检查（CANN、TorchDelegate、Python、cmake）
2. 生成示例epair（Python）
3. 构建C++示例（cmake）
4. 执行Delegate整图推理
5. 单算子vs Delegate精度比对（Python）

### 分步运行

#### 生成示例epair

```bash
python export_ep_and_epair.py
```

默认会在当前目录生成 `epair_model.epair`。

#### 无CANN运行时导出

如果导出环境没有CANN、`torch_npu` 或NPU设备，可使用Lite模式脚本：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python export_ep_and_epair_lite.py
```

该脚本使用CPU `ExportedProgram`，在 `lite_export_context` 中构建GE IR，
然后在context外保存：

- `ep_model_lite.pt2`
- `epair_model_lite.epair`
- `input_lite_0.bin`

Lite模式只负责生成EPAIR，不在本机加载、编译或执行GE图。生成的
`epair_model_lite.epair` 可以复制到具备匹配CANN环境的部署侧，再使用后续
C++步骤加载运行。

#### 构建C++示例

```bash
cmake -S . -B build \
  -DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT} \
  -DCANN_ROOT=${ASCEND_HOME_PATH} \
  -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cmake --build build -j$(nproc)
```

#### 运行示例

```bash
TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${LD_LIBRARY_PATH:-}"
./build/export_and_load_example epair_model.epair
```

程序成功时会完成：

- `EpairModelLoader::EpairModelLoader(archive_path)` — 加载epair
- `EpairModelLoader::GetGEGraph(&graph)` — 提取GE图
- `GeSession::AddGraph(graph_id, graph)` — 注册图
- `GeSession::RunGraph(graph_id, inputs, outputs)` — 运行图

#### 单算子vs Delegate精度比对

```bash
python compare_single_op_vs_delegate.py
```

该脚本自动完成：
1. **单算子执行**：用PyTorch eager模式逐算子执行模型（mm → add → transpose），每步输出中间结果
2. **Delegate执行**：调用C++ `export_and_load_example` 二进制，通过GE整图执行
3. **精度比对**：逐元素比较输出，报告最大绝对误差和相对误差

如果需要验证Python侧直接执行epair，可额外运行：

```bash
python load_epair_run_eager.py
```

## 关键API

| API | 头文件 | 说明 |
|-----|--------|------|
| `td::EpairModelLoader` | `torch_delegate/epair_model_loader.h` | 加载 `.epair` 并构建GE Graph |
| `td::EpairModelLoader::GetGEGraph` | `torch_delegate/epair_model_loader.h` | 提取 `ge::Graph` |
| `ge::GEInitializeV2` | `ge/ge_api_v2.h` | GE全局初始化 |
| `ge::GeSession` | `ge/ge_api_v2.h` | 在线编译会话 |
| `GeSession::AddGraph` | `ge/ge_api_v2.h` | 注册图 |
| `GeSession::RunGraph` | `ge/ge_api_v2.h` | 使用 `gert::Tensor` 同步执行 |
| `gert::Tensor` | `ge/ge_api_v2.h` | GeSession v2输入输出张量 |

## 流程说明

1. **初始化GE**：`ge::GEInitializeV2(options)`
2. **加载epair**：`td::EpairModelLoader loader(epair_path)` — 构造时解析归档并构建GE Graph
3. **提取GE图**：`loader.GetGEGraph(&graph)` — 获取内部GE Graph副本
4. **创建会话**：`ge::GeSession ge_session(options)`
5. **添加图并运行**：`ge_session.AddGraph(graph_id, graph)` → `ge_session.RunGraph(graph_id, inputs, outputs)`
6. **清理资源**：`ge_session.RemoveGraph(graph_id)` → `ge::GEFinalizeV2()`
