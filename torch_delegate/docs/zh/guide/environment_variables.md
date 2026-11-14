# TorchDelegate环境变量参考

## 环境变量总览

| 变量名 | 读取位置 | 默认值 | 简述 |
|--------|---------|--------|------|
| `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN` | `csrc/backend/ge_graph_builder.cc`（`GeGraphBuilder` 构造函数） | `1`（启用） | 是否在构图时触发Fallback Codegen |
| `TORCH_PKG_DIR` | `csrc/fallback_codegen/fallback_codegen.cc`（`FallbackCodegen::GenerateFromModelJson()`）；`cmake/third_party/modules/Findlibtorch_cpu.cmake` | — | Python torch包目录 |
| `TORCH_NPU_DIR` | `csrc/fallback_codegen/fallback_codegen.cc`（`FallbackCodegen::GenerateFromModelJson()`） | — | torch_npu库目录 |
| `TORCH_DELEGATE_OUTPUT_DIR` | `csrc/fallback_codegen/fallback_codegen.cc`（`FallbackCodegen::GenerateFromModelJson()`） | `./output_dir` | fallback编译产物保留目录 |
| `TORCH_DELEGATE_KEEP_ARTIFACTS` | `csrc/fallback_codegen/fallback_codegen.cc`（`FallbackCodegen::GenerateFromTargets()`） | `0`（不保留） | 编译后是否保留 `.cpp` / `CMakeLists.txt` / `.so` |
| `TORCH_DELEGATE_DEBUG` | `torch_delegate/core/utils.py`（`_get_logger()`） | `0`（WARN） | Python侧日志级别切换 |

---

## `TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN`

**作用**：控制是否启用ATen算子的Fallback Codegen功能。

**取值**：
- `1`：启用（默认，未设置时也启用）
- `0` 或其他任意值：禁用

**说明**：
- 禁用后，若epair中存在未转换的节点（`ascend_ir.source` 为空），GE图构建会失败报错
- 若所有节点均已成功转换，即使启用也不会触发Fallback Codegen，不校验 `TORCH_PKG_DIR` / `TORCH_NPU_DIR`

---

## `TORCH_PKG_DIR`

**作用**：指向Python torch包目录，用于g++编译时查找头文件和库文件，以及CMake构建时定位 `TorchConfig.cmake`。

**取值**：Python torch包的绝对路径（包含 `include/`、`lib/`、`share/cmake/Torch` 子目录）。

**示例**：
```bash
export TORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
# 示例值：/path/to/site-packages/torch
```

**说明**：
- 仅在Fallback Codegen触发时必需；所有节点的 `ascend_ir.source` 为 `converter` 时不校验

---

## `TORCH_NPU_DIR`

**作用**：指向**libtorch_npu**安装目录，用于g++编译时查找 `libtorch_npu.so` 和头文件。

**取值**：源码编译产出的libtorch_npu目录（包含 `lib/libtorch_npu.so` 和 `include/` 子目录）。

**示例**：
```bash
export TORCH_NPU_DIR=/path/to/libtorch_npu
```

**说明**：
- 生成的fallback代码依赖 `libtorch_npu.so`
- 仅在Fallback Codegen触发时必需；所有节点的 `ascend_ir.source` 为 `converter` 时不校验
- **不支持pip wheel版的 `torch_npu`**；`--system_python` 模式下环境里pip安装的 `torch_npu` 同样不可用
- `build.sh --with_libtorch_npu` 按 `build.conf` 的 `LIBTORCH_NPU_BRANCH`（当前 `v2.9.0`）从源码编译到 `third_party/lib/libtorch_npu`；不使用build脚本时需自行编译libtorch_npu并把本变量指向它
- 获取 `libtorch_npu`：从 [Ascend/pytorch](https://gitcode.com/Ascend/pytorch) 仓库下载 **v2.9.0分支**源码后，按 [libtorch_npu编译指南](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/installation_guide/references/building_libtorch_npu.md) 编译产出

---

## `TORCH_DELEGATE_OUTPUT_DIR`

**作用**：指定Fallback Codegen编译产物的保留目录。

**取值**：任意可写目录路径。

**说明**：
- 仅在 `TORCH_DELEGATE_KEEP_ARTIFACTS=1` 时实际使用
- 产物保留到 `${TORCH_DELEGATE_OUTPUT_DIR}/` 下的子目录中

---

## `TORCH_DELEGATE_KEEP_ARTIFACTS`

**作用**：控制在线编译完成后是否保留生成的 `.cpp`、`CMakeLists.txt`、`.so` 文件。

**取值**：
- `1`：保留到 `${TORCH_DELEGATE_OUTPUT_DIR}/` 下（未设置 `OUTPUT_DIR` 时为 `./output_dir/`）
- `0` 或未设置：不保留（默认），编译后清理临时目录

**说明**：
- 保留时产物包含 `custom_op.cpp`（头部含完整g++命令注释）、`CMakeLists.txt`、`libcust_opapi.so`
- 主要用于调试编译问题或跨环境部署

---

## `TORCH_DELEGATE_DEBUG`

**作用**：控制Python侧torch_delegate库的日志级别。

**取值**：
- `1`：`logging.DEBUG` 级别
- 未设置或其他值：`logging.WARN` 级别（默认）
