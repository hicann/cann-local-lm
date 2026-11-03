# tests/ 目录说明

> 维护者：engine/converter 团队。
> 仓库目录重组的总决策见 `docs/adr/0003-repo-layout-restructure.md`（已执行）。
> 测试分类规则的纠正见 `docs/adr/0004-classify-tests-by-semantics.md`。
> 本文件描述 `tests/` 的**当前真实结构**与尚未完成的清理债务。

## 1. 边界原则与分类规则

- `tests/` 只放 **ut**；`examples/` 放 **st / 端到端 / 集成**（`examples/test_engine/bbit/` C ABI 黑盒、`examples/test_converter/`）。
- **分类看测试语义，不看构建工具链**：一个测试是 UT 还是集成测试，取决于**它断言什么、运行期调用什么**，不取决于它链接什么库、需要什么工具链编译。判定方法：grep 测试源里的设备/执行 API（`aclrt*` / `aclmdl*` / `GeExecutor` / `RunGraph` / `CompileGraph` / `->Run`）；若无，则即便需要 CANN 库链接，也是 UT。
- `tests/cpp/epair_archive/` 即此例：需 CANN 工具链（`ASCEND_HOME_PATH` + CANN 库）才能编译/链接，但 7 个 `.cc` 不调用任何 NPU/图执行 API，断言全是 host 侧解析/注册表/映射逻辑——**是 UT**，归 `tests/`。C7（348d28c）曾误按"需 CANN"当成集成测试挪到 `examples/`，现纠正回 `tests/cpp/`（见 ADR 0004）。

## 2. 当前结构

```
tests/
├── cpp/                 # C++ UT 根
│   ├── epair_archive/   # converter C++ UT（自带 CMakeLists + gen_fixtures.py）；需 CANN 工具链编译，无 NPU 设备
│   └── engine/          # engine C++ 测试 + 其桩
│       ├── test_*.cpp       # engine GTest（tokenizer/c_api/scheduler/sampler/text_embedding/engine_executor...）；源文件统一 test_ 前缀，CMake target 名仍保 ut_*（见 §3 债务 1）
│       ├── model_executor/  # test_model_executor.cpp / test_kv_cache_pool.cpp / test_tensor_manager.cpp / ...
│       ├── stderr_capture.h # 共享 helper
│       └── stub_ndk/        # host 桩 .so（OH_NN* 假实现，engine UT 的 test double）+ backend bug-sweep probe（stale，EXCLUDE_FROM_ALL）
├── python/              # 单一 Python UT 根
│   ├── conftest.py      # engine FFI bindings 根 conftest（2 级深，_REPO=../..）
│   ├── test_*.py        # engine ctypes FFI bindings 测试（扁平）
│   ├── converter/       # 镜像 cann_local_lm/converter（自带 conftest.py + UT_DESIGN.md，3 级深 parents[3]=repo）
│   ├── dopt_lm/         # 镜像 cann_local_lm/converter/quant/dopt_lm
│   └── llm_models/      # 镜像 cann_local_lm/models
└── cpp/engine/  # fixture 文件（gitignored）
```

`tests/cpp/engine/`：**gitignored 生成物**（`.gitignore: fixtures/`），由 `tests/cpp/engine/gen_fixture.py`（需 HF `tokenizers`）生成；`tests/python/conftest.py` 开跑前自动生成（有库时），`test_token_count.py` 缺则 skip。C++ 侧 `ut_tokenizer` 经 CMake 注入同路径，目前**无对等的自动生成/skip 机制**（见 §3 债务 5）。

C++ UT 注册分两处：engine 侧（`tests/cpp/engine/`）在 `cann_local_lm/engine/CMakeLists.txt`（`UT_DIR=tests/cpp/engine`）；converter 侧（`tests/cpp/epair_archive/`）自带 `CMakeLists.txt`，由顶层 `CMakeLists.txt` 在 `BUILD_CONVERTER`（OFF 默认，需 torch+CANN 工具链）下 `add_subdirectory`。

## 3. 已知债务（尚未执行）

本轮（Phase B/C1/C2）已消除：Python 双根（原 `py/`+`python/` 已合并为单一 `tests/python/`，`py/*` 移入对应子目录，conftest 的 `parents[3]`/`../..` 路径算术因深度不变而保持正确）、`stub_ndk/` 归位（原 top-level sibling 已并入 `tests/cpp/engine/stub_ndk/`，CMake target 名不变故 `add_dependencies`+`CANN_LM_NDK_LIB_DIR` 接线不破，仅重算 stub CMakeLists 路径算术 2→4 级深）、C++ 双根合并（原 `tests/ut/` 已并入 `tests/cpp/engine/`，源文件 `ut_*.cpp`→`test_*.cpp` 统一前缀；CMake target 名仍保 `ut_*` 减小 diff，见债务 1）。下列仍欠，**单独立 PR 处理**：

1. **【低】C++ CMake target 名仍保 `ut_*` 前缀**：源文件已统一 `test_*.cpp` 并并入 `tests/cpp/engine/`，但 CMake target/binary 名仍为 `ut_*`（`ut_tokenizer`/`ut_engine_executor`/`ut_c_api`/`ut_scheduler`/`ut_sampler`/`ut_request_handle`/`model_executor_ut`/`tokenizer_realdata`），属外部契约——`scripts/cann_lm_engine/{build_ohos,sign_ohos,package_ohos,package_run_ohos,run_ut}.sh` + `scripts/build.sh` 产物清单 + ctest 过滤模式 + docs 均按现名引用。改名冲击面广、收益纯审美，缓做。
2. **【低】`stub_ndk/probe_*.cpp` stale**：仍 `#include "backend_base/..."`（已删接口），CMake `EXCLUDE_FROM_ALL` 不编、ctest 注册注释。待迁 hiai 新接口后解禁（Phase D，functional plan，见 memory `ndkproxy-stub-ndk-wiring`）。
3. **【低】`.cc` 与 `.cpp` 扩展名混用**：`tests/cpp/epair_archive/` 用 `.cc`，其余 C++ 用 `.cpp`。目标：统一 `.cpp`。
4. **【中】C++ 侧 fixture 供给缺口**：`ut_tokenizer` / `ut_tokenizer_realdata` 缺 fixture 时无 skip-guard（无 `NEEDS_FIXTURES` / `GTEST_SKIP`），fresh build 缺 `tests/cpp/engine/*` 会直接 fail。应对齐 Python 侧的自动生成 + skip 机制。

## 4. 新增测试放哪（迁移完成前，按目标结构放，不加重存量债）

- Python，被测在 `cann_local_lm/converter*` / `cann_local_lm/models`：`tests/python/<对应包>/test_*.py`，包内补 `conftest.py`。
- Python，被测在 engine ctypes FFI bindings：`tests/python/test_*.py`。
- C++ GTest（engine，host 可跑）：`tests/cpp/engine/test_<unit>.cpp`（CMake target 名仍保 `ut_*`，见 §3 债务 1）。
- C++ GTest（converter，需 CANN 工具链但无 NPU 设备）：`tests/cpp/<模块>/test_*.cpp`（参照 `tests/cpp/epair_archive/` 模式）。
- 黑盒端到端（打真实 C ABI / 真 NPU 集成）：`examples/`（如 `examples/test_engine/bbit/`）。
- 共享测试数据：`tests/cpp/engine/`（生成物 `.gitignore`，除非 golden 需入库）。

判定拿不准时：**看测试断言什么、运行期调用什么**——不碰设备的就是 UT（进 `tests/`），跑模型/打 NPU 的才是集成（进 `examples/`）。需要什么工具链编译不影响这个判定。
