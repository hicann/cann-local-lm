# CANN-LM — Agent 协作指南

## 项目结构

```
CANN-LM_5883/
├── cann_lm/      # 核心源码：模型定义 (gemma4/qwen3/qwen3_5) + converter 导出工具链 + C++ 后端
├── ci/           # CI/CD 配置：流水线定义、构建脚本、镜像配置
├── cmake/        # CMake 公共模块：DELEGATE_COMMON_FLAGS 等共享构建变量定义
├── docs/         # 设计文档：converter 设计、模型统一方案 (短期/长期)
├── examples/     # 端到端用例：export_and_load_epair (SimpleNetwork) + export_llm_model (qwen3 LLM) + 生成产物
├── generate/     # 代码生成器：基于模板自动生成模型骨架或配置文件
├── scripts/      # 辅助脚本：环境初始化、批量构建、部署工具
└── tests/        # 单元测试：Python 测试 (tests/python/) + C++ 测试 (tests/c++/)
```

## 架构文档加载
重要： 在探索仓库/项目、回答问题、修改代码、输出设计文档、需求Spec或做代码检视时，根据下表加载对应文档。每个文档只需加载一次，匹配触发词、涉及目录或场景中任意一条即加载。同时，修改了代码后，要更新docs/和docs/下的对应文档。
| 文档 | 触发词 | 涉及目录 |
|------|----------------|----------|
|[`export_design.md`](docs/cann_lm_converter/cann_lm_conveter_export/export_design.md)|导出和加载epair模型|cann_lm/converter/export/|
|[`converter_design.md`](docs/cann_lm_converter/converter_design.md)|大模型导出全流程|cann_lm/converter/|

## 开发约定
- 测试：`pytest`，测试文件放置于 `tests/python/`；单文件运行 `pytest tests/python/test_api.py`，单用例 `pytest tests/python/test_api.py::test_name`
- **端到端测试原则**：所有端到端测试用例及其生成产物（模型文件、权重、OM 等）统一放置于 `examples/` 路径下，按用例名分目录组织；禁止散落到 `cann_lm/` 或其他路径；在进行测试的时候先不要修改主仓代码，先修改测试代码，如果无法通过进行交互
- **组件规模原则**：单函数实现代码最好控制在 50 行以内；若明显超过，应优先按单一职责拆分辅助函数，但不要为了拆分而制造无意义跳转，优先聚合而非拆分；实现复杂度上升后再重构拆分
- **函数参数原则**：函数/方法参数个数默认不超过 5 个；若超过，优先通过结构体、context 对象或明确的数据聚合来收敛参数，而不是继续横向追加位置参数


### C++ 模块

- 编码风格遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- 函数名、方法名使用大驼峰（UpperCamelCase），如 `Partition()`、`BuildGraph()`
- 成员变量加 `_` 前缀，如 `_archive_path`
- 类名使用大驼峰，如 `EpairPartitioner`、`GraphPartitioner`
- **零警告原则**：C++ 代码必须在 `-Wall -Wextra -Werror` 下无警告编译；CMake 已配置 `-Werror`，PR 中不允许引入新的编译器警告
- **公共 Flag 原则**：所有 C++ target（库、可执行文件）必须通过 `DELEGATE_COMMON_FLAGS` 统一管理编译和链接选项；禁止在单个 target 上直接硬编码 `-Wall`/`-Wextra`/`-Werror` 等公共 flags，新增 target 时须显式调用 `target_compile_options` 和 `target_link_options` 并引用此变量。
- **ASAN 强制原则**：开发期间运行 C++ 测试时**必须**开启 ASAN（`./build_test.sh` 默认启用；手动 cmake 时传 `-DENABLE_ASAN=ON`）；提交前须确认测试在 ASAN 构建下全部通过且无内存错误报告。
- **C++ 函数约束**：C++ 函数/方法同样遵循“参数不超过 5 个、单函数最好不超过 50 行”的通用原则；复杂逻辑优先提取为私有辅助方法，但保持调用链可读，避免过度切碎。