# TorchDelegate — Agent 协作指南

本文档为参与 TorchDelegate 开发的 AI agent（及人类开发者）提供项目结构、模块边界和开发约定的快速参考。

---

## 项目结构

```text
torch-delegate/
├── cmake/                  ← CMake 模块（FindXxx.cmake）、三方库构建、打包配置
├── torch_delegate/         ← Python 包（converter、epair、api、core）
├── csrc/                   ← C++ 源码（backend、common、fallback_codegen、对外头文件）
├── tests/                  ← 测试（python/ pytest、cpp/ GTest + CANN/GE stub）
├── examples/               ← 示例项目（导出加载、精度对比、异步执行等）
├── docs/                   ← 设计文档、构建指南、调研报告
├── build.sh                ← 一站式构建（产出单一 whl）
├── build_test.sh           ← 编译并运行测试（含 ASAN）
├── build.conf              ← 版本号集中管理（唯一来源）
├── setup.py                ← whl 装配（内嵌 CMake 安装树、平台 tag、版本注入）
└── pyproject.toml          ← Python 包元数据与 entry point
```

---

## 模块边界

| 模块 | 语言 | 职责 | 输入 | 输出 |
|------|------|------|------|------|
| `torch_delegate/core/` | Python | 核心工具函数 | — | 公共辅助 |
| `torch_delegate/converter/` | Python | EP node → Ascend IR 转换 | `torch.fx.Node` | `NodeIR` |
| `torch_delegate/epair/` | Python | epair 序列化 / 反序列化 | `ExportedProgram` | `.epair` 文件 |
| `torch_delegate/api.py` | Python | 顶层入口，组合上面两者 | `ExportedProgram`路径 | `.epair` 文件 / `EpairContents` |
| `csrc/backend/` | C++ | epair 加载 → GE Graph 构建 → 权重填充 | `.epair` 归档 | GE Graph + 权重 |
| `csrc/fallback_codegen/` | C++ | Fallback 代码生成 | source="" 的 fallback 节点 | GE 自定义算子代码 |
| `csrc/common/` | C++ | 公共工具（日志、归档读取、Base64、ABI 兼容） | — | 基础设施 |

---

## 开发约定

### 通用

- 构建指南参考：[build.md](./docs/zh/guide/build.md)
- 依赖：`torch`（必需）+ `torchair`（Ascend 后端必需，用于 IR 转换）
- 测试：`pytest`，测试文件放置于 `tests/python/`；单文件运行 `pytest tests/python/test_api.py`，单用例 `pytest tests/python/test_api.py::test_name`
- **protobuf ABI 陷阱**：`tests/python/conftest.py` 强制设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，因为 protobuf C++ 扩展与 PyTorch 编译的 protobuf 版本 ABI 不兼容，调用 `serialize()` 时会 segfault；写脚本直接调用 `torch.export` / `serialize()` 时须手动设置此环境变量
- **严格报错原则**：对所有外部输入（`.epair` 文件内容、EP JSON 结构、字段类型）采用严格解析策略；遇到不认识的字段类型、缺失必要字段、类型不符，一律报错（C++ 侧用 `TD_ASSERT*` 宏 + `TDLOGE`，Python 侧抛异常或 `logger.error`），不静默跳过、不猜测意图
- **组件规模原则**：单函数实现代码最好控制在 50 行以内；若明显超过，应优先按单一职责拆分辅助函数，但不要为了拆分而制造无意义跳转，优先聚合而非拆分；实现复杂度上升后再重构拆分
- **函数参数原则**：函数/方法参数个数默认不超过 5 个；若超过，优先通过结构体、context 对象或明确的数据聚合来收敛参数，而不是继续横向追加位置参数
- **CANN 路径获取**：CANN 安装路径始终从环境变量 `ASCEND_HOME_PATH` 获取（`echo $ASCEND_HOME_PATH`），不要猜测或硬编码路径；cmake 时通过 `-DASCEND_INSTALL_PATH=$ASCEND_HOME_PATH` 传入（CMake 变量名为 `ASCEND_INSTALL_PATH`，不是 `CANN_ROOT`）

### OpenSpec 文档

- `proposal.md`、`design.md`、`tasks.md` 统一使用中文
- `spec.md` 默认也优先中文；但修改既有英文 capability 时，可保持该 capability 内语言一致，避免同一 capability 下中英文混写
- 同一个 change 的 `proposal`、`design`、`tasks` 不允许无规则混用中英文；代码符号、路径、capability 名称、OpenSpec 关键字保持原样
- 新增 capability 优先直接使用中文 spec；修改既有英文 capability 时可继续英文，但应保持整份 capability 文本语言一致

### converter 模块

- `NodeIRConverter` 是抽象基类；具体实现为 `TorchAirGEConverter`（在 `torchair_converter.py`）
- `NodeIR` 数据类仅在 `converter/base.py` 定义，单一数据源
- `is_node_supported` 和 `convert_node` 必须保持幂等，不产生副作用
- `convert_node` 返回类型为 `NodeIR | None`，实现类需确保类型一致（torchair `_NodeIR` 需转换为本地 `NodeIR`）

### epair 模块

- `EpairWriter` 类负责注入 `ascend_ir` 到 EP JSON
- `epair_load` 函数负责加载 epair 文件，返回 `EpairContents`
- writer 写入的每个字段，`epair_load` 都能读取还原（manifest、ep_json_with_ir）
- 通过 `torch.export.save` / `torch.export.load` 的 `extra_files` 参数读写 epair 扩展文件，复用 PT2 归档格式
- epair 扩展文件名（`epair_manifest.json`、`ep_json_with_ascend_ir.json`）统一定义在 `epair/constants.py`，不在代码中散落硬编码
- `epair_manifest.json` 和 `ep_json_with_ascend_ir.json` 是 epair 对 PT2 格式的两个新增文件

### 格式约定

- `models/model.json` 是标准 EP JSON，唯一扩展是 node 对象上的可选 `ascend_ir` 字段
- `ascend_ir` 字段内含 `source`（str）、`ops`（GE opdef 列表）、`mapping`（输出对应关系），不重复 EP JSON 中已有的 shape/dtype 信息
- `epair_manifest.json` 存储 `EpairManifest` 的 JSON 序列化结果（含 `cann_version` 字段）
- `ep_json_with_ascend_ir.json` 存储带 `ascend_ir` 标注的完整 EP JSON

### C++ 模块

- 编码风格遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- 函数名、方法名使用大驼峰（UpperCamelCase），如 `Partition()`、`BuildGraph()`
- 成员变量加 `_` 后缀，如 `archive_path_`
- 类名使用大驼峰，如 `EpairPartitioner`、`GraphPartitioner`
- **零警告原则**：C++ 代码必须在 `-Wall -Wextra -Werror` 下无警告编译；CMake 已配置 `-Werror`，PR 中不允许引入新的编译器警告
- **公共 Flag 原则**：所有 C++ target（库、可执行文件）必须通过 `DELEGATE_COMMON_FLAGS` 统一管理编译和链接选项；禁止在单个 target 上直接硬编码 `-Wall`/`-Wextra`/`-Werror` 等公共 flags，新增 target 时须显式调用 `target_compile_options` 和 `target_link_options` 并引用此变量。
- **ASAN 强制原则**：开发期间运行 C++ 测试时**必须**开启 ASAN（`./build_test.sh` 默认启用；手动 cmake 时传 `-DENABLE_ASAN=ON`）；提交前须确认测试在 ASAN 构建下全部通过且无内存错误报告。
- **C++ 函数约束**：C++ 函数/方法同样遵循“参数不超过 5 个、单函数最好不超过 50 行”的通用原则；复杂逻辑优先提取为私有辅助方法，但保持调用链可读，避免过度切碎。

### 测试 stub

- **CANN stub**：`tests/cpp/cann_stub/ascendcl.cc` → `libascendcl.so`，编译时直接 `#include <acl/acl_rt.h>` 使用真实 CANN 头文件，运行时通过 `LD_LIBRARY_PATH` 替换；每个 GTest fixture 须在 `SetUp()` 调用 `cann_stub::Reset()`，`TearDown()` 断言无流/内存泄漏
- **GE stub**：`tests/cpp/ge_stub/ge_stub.cc` → `ge_runner_v2` stub，通过 `TorchDelegateTestStubs.cmake` 在 `BUILD_TESTS=ON` 时自动替换真实链接

### 日志模块

TorchDelegate 提供统一的日志模块（`csrc/common/logging/`），封装 CANN `ACL_APP_LOG` 接口，支持分级日志输出和校验宏。

#### 日志宏

- `TDLOGI(fmt, ...)`：INFO，关键入口/出口、重要状态变更
- `TDLOGE(fmt, ...)`：ERROR，错误返回前打印（含错误码、节点名称、类型等）
- `TDLOGW(fmt, ...)`：WARNING，非紧急问题（如参数降级、兼容性警告）
- `TDLOGD(fmt, ...)`：DEBUG，详细执行流程

#### 校验宏

TorchDelegate 使用 `ErrorResult` 结构体实现自动返回值类型推导，参考 CANN `checker.h` 设计。校验宏在失败时返回 `ErrorResult`，通过类型转换运算符自动推导为正确的返回值类型（`bool`、`Status`、指针、`unique_ptr`、`vector`、`string` 等）。

- `TD_ASSERT(expr, ...)`：通用断言宏，失败时输出 ERROR 日志并返回 `ErrorResult`
- `TD_ASSERT_NOTNULL(ptr, ...)`：断言指针非空，基于 `TD_ASSERT`
- `TD_ASSERT_TRUE(expr, ...)`：断言表达式为真，基于 `TD_ASSERT`
- `TD_CHK_STATUS(expr)`：检查 `td::Status` 对象，失败时输出 ERROR 日志并提前返回该 Status

**ErrorResult 类型转换**：
- `operator bool()` → 返回 `false`
- `operator Status()` → 返回 `INVALID_PARAM`
- `operator T*()` → 返回 `nullptr`
- `operator std::unique_ptr<T>()` → 返回 `nullptr`
- `operator std::shared_ptr<T>()` → 返回 `nullptr`
- `operator std::vector<T>()` → 返回空 `vector`
- `operator std::string()` → 返回空 `string`
- `operator T()` → 返回默认构造的 `T()`

详细用法示例见 `csrc/common/logging/logging.h` 和 `assert_macros.h`。

### 对外 ABI 安全接口

- `inc/external/torch_delegate/epair_model_loader.h`、`inc/external/torch_delegate/td_types.h`、`backend/epair_archive.h`、`common/epair_archive_reader.h` 视为 ABI 边界，禁止重新引入 `std::string`、`std::unordered_map`、`std::filesystem::path`
- 对外字符串使用 `ge::AscendString`，路径参数使用 `const char*`
- 对外类通过 PIMPL 隔离内部 STL 实现；新增公开字段或方法时，先检查 ABI 暴露面
- `torch_delegate_backend` 对外只公开 CANN graph/metadef 相关依赖，禁止把 libtorch 编译定义继续向外传播

### 不应做的事（避免过度工程）

- 不在 Python 侧做切图（切图推迟到 C++ 模块 2）
- converter 类型选择不暴露给用户（内部实现可独立演进）
- 不为假想的未来需求添加额外字段或抽象层
- stub 中的 `attrs` 可以为空 `{}`，等待正式 GE 组件接入后填充

---

## C++ 工程构建

详见 [docs/zh/guide/build.md](docs/zh/guide/build.md)。主要命令：

- `./build.sh` — 一站式构建（三方库 → C++ 库 → 单一 whl）
- `./build.sh --skip_third_party` — 跳过三方库
- `./build_test.sh` — 编译并运行测试（默认启用 ASAN）
- `source build.conf` — 版本号集中管理

---

## 发布约定

- **单一 whl 发布**：唯一发布产物是 `output/package/torch_delegate-<ver>-py3-none-linux_<arch>.whl`，同时承载 Python API 与 C++ 制品（`torch_delegate/inc/`、`torch_delegate/lib/`、`torch_delegate/lib/cmake/TorchDelegate/`）。不再生成 `.run` 安装包，也不提供 C++ 独立分发包；C++-only 用户直接解压 whl（zip）取用。
- **版本号单一来源**：`build.conf` 的 `TORCH_DELEGATE_VERSION` 是唯一来源，构建时经 `TORCH_DELEGATE_PACKAGE_VERSION` 注入 whl 元数据与 `torch_delegate/_version.py`，CMake 用同一变量；`pyproject.toml` 使用 dynamic version，不得硬编码版本号。debug 构建追加 PEP 440 local version `+debug`。
- **多 Python 版本**：whl 的 tag 固定 `py3-none`（动态库不链接 libpython），单包覆盖 3.10~3.13；`requires-python = ">=3.10,<3.14"`。
- **不提供 set_env.sh**：C++ 前缀通过 `torch_delegate.cmake_prefix()` 或 `torch-delegate-config --cmake-prefix` 获取；运行期 `LD_LIBRARY_PATH`（libtorch、CANN）由用户按 docs/zh/guide/build.md 手动配置，动态库不注入指向 torch/CANN 的 RUNPATH。
- **依赖声明**：whl 不声明任何运行时依赖（由 `test_wheel_package.py` 守护此行为）；环境依赖 `torch>=2.9.0`，完整导出模式另需配套 `torch_npu`，均由用户预装。
