# 测试按语义分类，不按工具链依赖（纠正 C7 对 epair_archive 的误判）

## 背景

C7（348d28c，refactor(layout): tests/bbit + tests/c++/epair_archive → examples/）把 `tests/c++/epair_archive/` 挪到 `examples/epair_archive/`，commit message 写"converter C++ 集成, 需 CANN"，据此归 `examples/`（ADR 0003 原则：`tests/=ut only`、`examples/=st/端到端`）。

epair_archive 的作者（同事）提出质疑：**这些就是 UT**。

## 误判根因

把"**需 CANN 工具链编译**"当成了"**NPU 集成测试**"。两者不同：

- `examples/epair_archive/CMakeLists.txt` 确实 `ASCEND_HOME_PATH` 未设即 `FATAL_ERROR`、`find_package(ascendcl/acl_rt/graph/...)`、运行时 `LD_LIBRARY_PATH=CANN_LIB_DIRS`——**编译/链接/加载需要 CANN 库**，这是真的。
- 但 7 个 `.cc` 全量 grep 设备/执行 API（`aclInit`/`aclrtCreate`/`aclmdlLoad`/`aclmdlExecute`/`GeExecutor`/`RunGraph`/`CompileGraph`/`->Run`）**全空**。断言全是 host 侧纯逻辑：`test_base64.cc` 测 base64 解码；`test_epair_model_parser.cc` 测 `parsed_model.nodes.size()`/`parser.SchemaVersionValue()`/`GraphInputs().size()`/`PlaceholderKindOf`/`FindTensorValue`。**不碰设备、不在 NPU 上跑模型。**

即：epair_archive 是一个"需要 CANN 库才能编译链接、但运行期不调用 NPU 执行"的 **UT**，不是集成测试。C7 按"需 CANN"归类，把**构建工具链依赖**误当**测试语义**。

## 决策

**测试按语义分类（断言什么、运行期调用什么），不按构建工具链依赖（链接什么库、需要什么工具链编译）。**

判定方法：grep 测试源里的设备/执行 API（`aclrt*`/`aclmdl*`/`GeExecutor`/`RunGraph`/`CompileGraph`/`->Run`）。若无，即便需要 CANN 库链接，也是 UT，归 `tests/`。需要什么工具链编译，不进入此判定。

据此：`epair_archive` 纠正回 `tests/cpp/epair_archive/`（3 级深，并行 `tests/py/` 作 per-language 根，避开 `c++/` 的 shell 不友好）。ADR 0003 的"tests/=ut only、examples/=集成"原则**本身一直正确**，是 C7 **误用**了它——纠正后原则才被正确执行。

## 附带修复

C7 的 move 是纯 rename（git show 显示 0 insert/0 delete），但 `epair_archive/CMakeLists.txt:11` 的 `set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")` 原为 3 级深 `tests/c++/epair_archive` 校准（3 级上 = repo root）；挪到 2 级深 `examples/epair_archive` 后 `../../..` 指向 repo 上一级（路径错）。因 `BUILD_CONVERTER` 默认 OFF 且需 CANN，长期未构建、bug 潜伏。挪回 3 级深 `tests/cpp/epair_archive` 即恢复正确（`../../..` = repo root），无需改 CMakeLists。

## 下次如何避免

分类一个测试前，先看它**测什么**（断言）、**运行期调什么**（grep 设备/执行 API），再看它**需要什么编译**。三者顺序不能颠倒——工具链依赖最显眼但最不该作为分类依据。需要 CANN 库链接 ≠ 集成测试。

## 关联

- `docs/adr/0003-repo-layout-restructure.md`（tests/=ut、examples/=集成 原则；本 ADR 是其正确应用，非推翻）
- `tests/README.md` §1 分类规则、`examples/epair_archive/UT_DESIGN.md`（已随目录迁回 `tests/cpp/epair_archive/`）
