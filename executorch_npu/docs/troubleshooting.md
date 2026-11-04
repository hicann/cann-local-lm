# 故障排查

先确定错误属于源码构建、模型导出、C++ 应用编译还是设备运行。不要在一个环境中同时
尝试修复所有阶段。

## 源码构建

### ExecuTorch full wheel未安装

现象：

```text
ExecuTorch 1.5.0 full wheel with the C++ SDK is not installed
```

处理：

```bash
python -m pip install --no-deps "executorch==1.5.0"
```

确认wheel中存在：

```text
executorch/include/executorch/runtime/backend/interface.h
executorch/include/executorch/extension/module/module.h
executorch/lib/libexecutorch.so
executorch/lib/cmake/executorch/
```

ExecuTorch 1.0–1.4不属于当前构建支持范围。

### `Could not find CMAKE_ROOT`

CMake可执行文件与其 `share/cmake-*` 数据目录不匹配。重新安装当前环境中的CMake，
并确认 `command -v cmake`、`cmake --version` 指向同一环境。

### 找不到CANN头文件

先执行：

```bash
source /path/to/cann/set_env.sh
```

不要手工拼接多个CANN安装目录。`ASCEND_HOME_PATH` 必须指向当前使用的Toolkit。

### `nlohmann/json.hpp` 缺失

仓库固定依赖位于 `third_party/nlohmann_json/include`。确认仓库内容完整，不需要额外设置
`NLOHMANN_JSON_INCLUDE_DIR`。

## 模型导出

### 安装了错误的Torch wheel

现象包括CUDA动态库缺失、TorchAir ABI未定义符号，或导出前直接被
ExecuTorch-NPU拒绝。

确认：

```bash
python -c 'import torch; print(torch.__version__)'
```

版本必须位于2.9.0–2.13.0，且包含 `+cpu`。重新安装方式见
[模型导出](export.md#两阶段共享依赖)。

### `No module named ge` 或 `No module named tbe`

通常是没有加载CANN环境，或执行命令时覆盖了 `set_env.sh` 注入的 `PYTHONPATH`。

错误：

```bash
PYTHONPATH="$TORCH_DELEGATE_ROOT" python export.py
```

正确：

```bash
export PYTHONPATH="$TORCH_DELEGATE_ROOT:${PYTHONPATH:-}"
```

长任务、`nohup` 和子进程必须继承同一CANN环境。

### `_torchair.so: undefined symbol`

TorchAir C++ 扩展不是使用当前Torch编译。为每个Torch版本重新构建独立TorchAir，
不要复制其他环境生成的 `_torchair.so`。

### 独立TorchAir被torch_npu替换

独立TorchAir profile不应安装torch_npu。不要在同一环境混用独立TorchAir和
torch_npu内置TorchAir。

### 找不到 `libtorch_delegate_backend.so`

安装仓内 `../torch_delegate` 构建出的whl（`bash build.sh` 后位于其
`output/package/`，动态库在包内 `torch_delegate/lib/`），或直接安装已发布的
torch_delegate whl；也可将源码构建产物目录 `../torch_delegate/build/release/csrc`
追加到 `LD_LIBRARY_PATH`。不要只添加Python源码路径。

### `guard_or_true` 等导入错误

环境中混入了其他Torch或ExecuTorch源码路径。确认使用ExecuTorch 1.5.0和一个
受支持CPU Torch，并清理 `PYTHONPATH` 中的旧ExecuTorch路径。

### `program.fbs` 缺失

当前安装不是ExecuTorch 1.5.0 full wheel，或旧源码路径覆盖了wheel。重新安装full
wheel，不要手工复制schema。

### 没有生成PTD

必须调用：

```python
program.write_tensor_data_to_file(output_dir)
```

PTE和PTD必须从同一次导出产生并成对保存。

### `Disk quota exceeded`

导出会产生EPAIR、GE临时文件、OM和外置权重。大模型编译前检查 `/tmp` 和产物目录
空间，不要删除当前进程仍在使用的 `executorch_npu_*` 临时目录。

GE编译的临时权重峰值可达37 GB（Gemma4 E2B）。若报
`Write data failed, errno: -1` 或 `Failed to save weight to file`，是
`EXECUTORCH_NPU_STAGING` 所在文件系统写满，将其指向大容量磁盘。失败残留的
`/tmp/tmp_weight_*` 目录不会自动清理，需手动删除。

### GE `SaveModel` 跨文件系统失败

GE可能先在当前工作目录生成临时权重再移动到目标目录。将工作目录和外置权重目录放在
同一文件系统，例如都位于 `/tmp`。

### 编译期报 `ACL set device id failed, return 107001`

TorchAir的 `InitializeGraphEngine` 无条件绑定设备。设置
`EXECUTORCH_NPU_HOST_ONLY=1` 启用Lite导出，见
[模型导出](export.md#无npu设备导出lite导出)。

若已设置仍报错，检查 `fx2ge_converter.initialize_graph_engine` 是否被覆盖。该模块以
`from torchair.core._backend import initialize_graph_engine` 直接导入名字，只替换
`_backend` 上的属性对其无效。

### `TdEpairModelLoaderCreate failed with status=1`

EPAIR加载器要求GE已初始化，但不要求设备。Lite导出下应由
`ge.offline_compile.build_initialize()` 完成初始化，确认 `_initialize_ge()` 走的是
host-only分支。

### `CANN produced no external weight files`

host-only的 `build_initialize` 传入了 `ge.externalWeightDir`，外置权重被写到
session workspace。该调用不得设置此选项，GE会按输出路径写入partition的
`weight/` 目录。

### `Offline_compile did not produce expected OM`

无设备时GE可能给OM追加host os/arch后缀（如 `model_linux_aarch64.om`）。
打包侧按 `model*.om` 前缀定位，不要硬编码 `.om`。

### `reduce_sum() takes from 3 to 5 positional arguments but 6 were given`

OPP算子实现与TBE Python框架来自不同CANN版本。不要只覆盖 `ASCEND_OPP_PATH`，
应 `source` 同一份Toolkit的 `set_env.sh`。该错误只在动态Shape编译时出现。

### `aten.index_put.default ge_converter is not implemented`

KV cache的CPU分支使用了 `index_copy_`，TorchAir对该算子只注册了抛异常的桩。
cache应统一使用 `torch_npu.scatter_update_`：该算子在CPU张量上也可被
`torch.export` 追踪，且与设备侧导出产出同一算子，不引入算子替换或精度差异。

## C++ 应用编译

### 找不到ExecuTorch CMake package

确认 `CMAKE_PREFIX_PATH` 指向安装后的ExecuTorch包根目录，而不是源码仓库：

```bash
export EXECUTORCH_ROOT="$(python -c 'import executorch; print(next(iter(executorch.__path__)))')"
```

### 仍要求 `executorch_core` 或 `extension_module_static`

使用了旧版ExecuTorch-NPU CMake文件。当前ExecuTorch 1.5接口使用
`executorch::runtime`，应用只需要链接 `executorch_npu::backend`。

### ExecuTorch头文件的warning被 `-Werror` 放大

使用wheel SDK自带的CMake接口。它会把ExecuTorch include标记为外部SYSTEM
头文件，同时保留用户源码自身的 `-Werror`。

## 设备运行

### `aclrtSetDevice` 参数无效

`device_id` 是ACL逻辑编号，不一定等于 `npu-smi` 展示的物理卡号。只有一张可见卡时，
逻辑编号通常为0。

### `DelegateInvalidCompatibility` 或错误码 `0x30`

常见原因：

- 产物 `acl_soc_name` 与设备实际SoC（`aclrtGetSocName()`）不一致。
- PTE/PTD、Backend SDK或ExecuTorch Runtime版本不匹配。
- Delegate identity校验失败：Python端 `to_compile_specs()` 只输出单个
  `ascend_compile_options` JSON CompileSpec，而Runtime按扁平的
  `method_name`/`delegate_index` 键读取，取到空串后必然不等于manifest中的
  method。此时日志为 `Ascend delegate identity does not match runtime context`，
  且**任何产物都无法加载**。修复方式是让Runtime解析该JSON
  （`ParseAscendCompileOptions`），而不是回退产物。

重新按实际 `aclrtGetSocName()` 导出，不要用相近型号替代。

### PTD不存在或校验失败

确认PTD路径正确、文件未修改，并与PTE来自同一次导出。Manifest会校验key、逻辑
长度和SHA-256。

### `ldd` 显示 `not found`

执行前加载CANN环境，并确认ExecuTorch Runtime库可见：

```bash
source /path/to/cann/set_env.sh
ldd build/my_runner
```

### 输出误差超限

确认输入文件、参考输出、PTE/PTD、SoC、CANN、驱动和固件属于同一验证组合。先使用
固定输入完成单次验证，再排查模型精度问题。


## 同版本CANN的多安装目录与端到端基线

目录名或 `version.info` 相同，不代表实际动态库一致。先核对项目构建目录
`CMakeCache.txt` 中的 `ASCEND_HOME_PATH`、导出脚本选择的
`CANN_ENV_SCRIPT`，再检查新进程实际加载的GE/AscendCL路径。

本开发环境验证通过的安装是 `/mnt/workspace/Ascend/cann-9.1.0`。
另一个 `/home/developer/Ascend/cann-9.1.0` 安装中的 `libgraph_base.so`
缺少其 `librts_engine.so` 要求的 `ge::ATTR_NAME_RTS_OP_IMPL_TYPE`，
会导致RTS引擎加载失败，并表现为 `NetOutput` 找不到支持的引擎。
应选择完整、匹配的安装启动新进程，避免混合不同安装的Python包和动态库。

CANN保存外置权重时使用重命名操作。调用底层编译接口时，暂存与输出需要位于
同一文件系统；推荐将 `EXECUTORCH_NPU_STAGING` 明确设到有足够容量的工作盘，
并提前创建该目录。完整导出优先使用公开的PTE/PTD导出入口，由backend
管理其内部暂存路径。

验证重构时，先在固定环境下让旧版完成导出、PTE/PTD加载和设备执行，再对新版
使用相同模型、输入与C++ runtime。比较实际输出、I/O与状态契约，覆盖
host-only/设备模式、动态范围边界、跨方法状态、请求隔离及重置。
PTE/PTD文件摘要可能因编译元数据或布局不同而变化，不能代替数值比较。


### 源码已改，但runner仍执行旧校验

Backend和manifest parser静态链接进runner。仅重建静态库或测试程序，不会更新
已有runner；保持artifact ABI 1也不意味着旧二进制自动支持新增的状态行为。
修改解析或状态逻辑后，应在已选择正确CANN的环境中重新构建实际使用的应用：

```bash
bash examples/scripts/build_runner.sh
```

例如源码已删除single-buffer的ABI 2门禁，但运行仍报该错误时，核对实际
runner路径、构建时间和二进制中的报错字符串，再重链接验证。
