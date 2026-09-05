# epair_archive 单元测试设计报告

## 1. 文档信息

- 被测对象：`cann_lm/converter/export/csrc/`（C++ backend：epair 加载/解析/权重打包/aten 映射/value registry）
- 测试归档目录：`tests/c++/epair_archive`
- 测试框架：GoogleTest（GTest）+ CMake `ctest`
- 基准环境：② 真 CANN（`source scripts/set_env_cann.sh`，无 stubs）+ ASAN（`-static-libasan`）
- 编译器约束：C++17、`-Wall -Wextra -Werror`（经 `DELEGATE_COMMON_FLAGS`）
- 设计日期：2026-09-05

## 2. 测试目标与范围

本轮 UT 验证 `csrc` 在 CPU 侧的纯逻辑/解析/打包层：epair 目录加载、model JSON 解析、consolidated 权重打包与索引、aten target/schema 到 GE dtype/layout 的映射、value registry 别名解析，以及 base64 解码工具。测试必须满足以下原则：

1. 在 CPU 离线运行 csrc 逻辑，不调用 NPU acl 设备接口、不编译 OM、不触发 torchair GE 转换。
2. 随机 fixture 由 `gen_fixtures.py`（纯 Python + 标准库）确定性生成，复现小尺寸 mm+add 模型与异常索引/缺 meta 场景。
3. 开发期间强制开启 ASAN，ctest 裸跑（`-static-libasan` 静态链入，无需 `LD_PRELOAD`）。
4. 编译零警告：`-Wall -Wextra -Werror` 由 `DELEGATE_COMMON_FLAGS` 统一注入；PR 不许引入新警告。
5. 错误路径通过 `ValidationResult` 累积 issue code（如 `MISSING_TENSOR_META`/`MISSING_WEIGHTS_INDEX`/`UNSUPPORTED_FUNCTIONALIZED_V1`），同时检查 `result.ok` 与 issue code。

范围内模块（`cann_lm/converter/export/csrc/`）：

- `backend/aten_schema_resolver`、`backend/aten_tensor_desc_mapper`
- `backend/epair_model_parser`、`backend/ge_weight_loader`
- `backend/value_registry`、`backend/model_types`
- `common/base64`、`common/abi_compat_utils`、`common/epair_archive_constants`、`common/pt2_type_constants`、`common/logging`

范围外项目：

- `backend/ge_graph_builder`、`backend/epair_model_loader` 的 OM 编译/图构建执行体（归 e2e，`examples/test_converter/export_llm_model/run_e2e_pipeline.sh`）。
- `cann_lm_compile` OM 编译与 NPU acl 运行时验证。
- torchair FX→GE 转换、真实 LLM 权重导出（converter venv + stubs 路径，归 ① converter 测试）。

## 3. 代码分析结论

### 3.1 架构与调用关系

顶层 `CMakeLists.txt`（仓库根）`if(BUILD_TESTS) add_subdirectory(tests/c++/epair_archive) endif()`，`BUILD_TESTS` 默认 ON。本目录 `CMakeLists.txt`：

- `add_cann_lm_test(name src [NEEDS_FIXTURES])` helper：建可执行文件 → include `csrc` + CANN 头 → link `torch_delegate_backend`（顶层 in-tree target 或 fallback 预编译 `build/libtorch_delegate_backend.so`）+ CANN 重库（`graph`/`graph_base`/`metadef`/`ascendcl`/`acl_rt`/`runtime`/`securec`/`ascend_hal`/`ge_runner_v2`）+ `nlohmann_json::nlohmann_json` + `GTest::gtest_main` → `delegate_apply_common_flags`（走 `DELEGATE_COMMON_FLAGS`，`-Wall -Wextra -Werror`）→ rpath-link 透传 → 注册 ctest（`LD_LIBRARY_PATH=CANN_LIB_DIRS`）。
- `gen_fixtures.py` 经 `add_custom_command`（stamp file）+ `add_custom_target(generate_fixtures ALL)` 产 `.epair` 目录到 `${BINARY_DIR}/fixtures`；`FIXTURES_DIR` 宏注入测试可执行文件。3 个 `NEEDS_FIXTURES` 测试 `add_dependencies(generate_fixtures)`，4 个纯单测不依赖。
- 调用链：`EpairModelParser::Parse` → `ParseRootJson`/`ParseModelJson` 解析 `graph_module`（inputs/outputs/nodes/tensor_values/scalar value maps）；`GeWeightLoader::Create` 消费 `weights_index.json` + `weights.bin` 并经 `EpairModelParser.Model()` 解析 placeholder 名/类型；`AtenSchemaResolver`/`AtenTensorDescMapper` 把 aten target/pt2 meta 映射为 GE schema/desc；`ValueRegistry` 维护 placeholder/producer/alias→`RegisteredValue`（GNode + output_index）。

### 3.2 主要风险

| 风险域 | 代码风险 | UT 关注点 |
|---|---|---|
| CANN 重库依赖 | test exe 直接 link `graph`/`metadef`/`ascendcl` 等，需 ② 真 CANN env（`ASCEND_HOME_PATH`），非 stubs | 用 `source scripts/set_env_cann.sh` 起 env；ctest 注册 `LD_LIBRARY_PATH=CANN_LIB_DIRS` |
| env 互斥 | ① converter（stubs-first `LD_LIBRARY_PATH`）与 ② 真 CANN lib64 不能叠加 | C++ 测试只在 ② 跑，不与 converter venv 混用 |
| fixture 生成依赖 | 3 个 NEEDS_FIXTURES 依赖 `gen_fixtures.py` 产 `.epair` 目录（model.json + weights.bin + weights_index.json）；缺则失败 | `gen_fixtures` 作 `ALL` custom target，CMake 自动先跑；4 纯单测不依赖 |
| 零警告 `-Werror` | `-Wall -Wextra -Werror` 经 `DELEGATE_COMMON_FLAGS`；CANN `ge_error_codes.h` deprecated 提示属 CANN 侧 | PR 不引入新警告；deprecated 归 CANN 侧，非本仓 |
| ASAN 静态链入 | ASAN flags 含 `-static-libasan`，ctest 裸跑无需 `LD_PRELOAD` | `bash scripts/build.sh cpp --asan` 重配 `build/`；ASAN 无内存错误才准入 |
| 覆盖率与 ASAN 互斥 | gcov `--coverage` 编译与 ASAN 不能同 build 目录 | 覆盖率用独立 `build_cov`（`ENABLE_COVERAGE=ON`），与 ASAN build 分开跑 |
| OM 编译路径零覆盖 | `ge_graph_builder.cc`/`epair_model_loader.cc` 的图构建/OM 编译体需真实 epair→OM 链路 | 归 e2e，UT 只测加载/解析/映射层，不以全行覆盖作准入 |

## 4. 测试方法与判定准则

- 断言用 GTest 宏：`EXPECT_EQ`/`EXPECT_TRUE`/`EXPECT_FALSE`/`EXPECT_FLOAT_EQ`/`EXPECT_DOUBLE_EQ` 做非致命校验，`ASSERT_TRUE`/`ASSERT_NE`/`ASSERT_EQ` 守卫关键前提（loader 创建成功、指针非空、容器大小）以避免后续解引用空指针。
- 错误路径同时检查 `result.ok`（`ValidationResult`）与 `result.issues[].code`（如 `MISSING_TENSOR_META`/`MISSING_WEIGHTS_INDEX`/`UNSUPPORTED_FUNCTIONALIZED_V1`），用 `std::any_of` 在 issues 中定位 code。
- fixture 依赖：`gen_fixtures.py` 产 `.epair` 目录到 `${BINARY_DIR}/fixtures`，`FIXTURES_DIR` 宏注入；测试用内联 JSON 字面量（`R"JSON(...)"`）或读 fixture 文件。每条 fixture 测试独立 `ValidationResult{true, {}}`，避免跨用例状态泄漏。
- ASAN：开发期间 `--asan` 构建，ctest 裸跑，无内存错误（heap-use-after-free/buffer-overflow 等）才算通过。
- 覆盖率目标：`csrc` 纯逻辑/映射模块（`aten_schema_resolver`/`aten_tensor_desc_mapper`/`base64`/`value_registry`）行覆盖率不低于 80%；解析/加载层（`epair_model_parser`/`ge_weight_loader`）以分支覆盖为主；`ge_graph_builder`/`epair_model_loader`（OM 编译路径）归 e2e，不以全行覆盖率作为 UT 准入条件。
- 所有 UT 必须可重复执行；`ctest` 整体退出码为 0 才算通过。

## 5. 用例设计

### 5.1 aten schema 解析（test_aten_schema_resolver.cc，10 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| ASR-001 | `AtenSchemaResolver.ParseTarget` | `torch.ops.aten.add.Tensor` → ns=`aten`/op=`add`/overload=`Tensor`/dispatcher=`aten::add`，`result.ok` | P0 |
| ASR-002 | `AtenSchemaResolver.ParseTarget` | `torch.ops.aten.mm.default` → overload 为空串，`result.ok` | P0 |
| ASR-003 | `AtenSchemaResolver.ParseTarget` | `torch.ops.aten.add_.Tensor` → op_name=`add_`/overload=`Tensor`（inplace 后缀） | P1 |
| ASR-004 | `AtenSchemaResolver.ParseTarget` | `torch.ops.xllm_ascend.rope_in_place.default` → 自定义 ns=`xllm_ascend`/overload 空 | P1 |
| ASR-005 | `AtenSchemaResolver.ParseTarget` | `invalid_target`（无点分隔）→ `result.ok=false` | P0 |
| ASR-006 | `AtenSchemaResolver.MakeFallbackOpType` | aten/add/Tensor → `aten__add_Tensor` | P1 |
| ASR-007 | `AtenSchemaResolver.MakeFallbackOpType` | aten/mm（无 overload）→ `aten__mm` | P1 |
| ASR-008 | `AtenSchemaResolver.MakeFallbackOpType` | xllm_ascend/rope_in_place → `xllm_ascend__rope_in_place` | P2 |
| ASR-009 | `AtenSchemaResolver.Resolve` | `torch.ops.aten.add.out` 节点 → `is_out=true`/`is_inplace=false` | P0 |
| ASR-010 | `AtenSchemaResolver.Resolve` | `torch.ops.aten.add_.Tensor` + self tensor 输入 → `is_inplace=true`/`is_out=false` | P0 |

### 5.2 aten tensor desc 映射（test_aten_tensor_desc_mapper.cc，18 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| ATM-001 | `MapPt2Dtype` | pt2 float → `DT_FLOAT`，`result.ok` | P0 |
| ATM-002 | `MapPt2Dtype` | pt2 half → `DT_FLOAT16`，`result.ok` | P0 |
| ATM-003 | `MapPt2Dtype` | pt2 long → `DT_INT64`，`result.ok` | P0 |
| ATM-004 | `MapPt2Dtype` | pt2 bool → `DT_BOOL`，`result.ok` | P0 |
| ATM-005 | `MapPt2Dtype` | 999（不支持）→ `DT_UNDEFINED`，`result.ok=false` | P0 |
| ATM-006 | `MapPt2Layout` | pt2 strided → `FORMAT_ND`，`result.ok` | P0 |
| ATM-007 | `MapPt2Layout` | 0（不支持）→ `result.ok=false` | P1 |
| ATM-008 | `MapGeDtypeString` | `"DT_FLOAT"` → `DT_FLOAT` | P1 |
| ATM-009 | `MapGeDtypeString` | `"DT_INT64"` → `DT_INT64` | P1 |
| ATM-010 | `MapGeDtypeString` | `"DT_BOOL"` → `DT_BOOL` | P1 |
| ATM-011 | `MapGeDtypeString` | `"DT_COMPLEX64"`（不支持）→ `DT_UNDEFINED`，`result.ok=false` | P0 |
| ATM-012 | `MapGeDtypeString` | 空串 → `result.ok=true`/`DT_UNDEFINED` | P1 |
| ATM-013 | `MapGeLayoutString` | `"ND"` → `FORMAT_ND` | P1 |
| ATM-014 | `MapGeLayoutString` | 空串 → `FORMAT_ND`，`result.ok=true` | P1 |
| ATM-015 | `MapGeLayoutString` | `"NCDHW"`（不在 {empty,ND,NCHW,NHWC}）→ `result.ok=false` | P0 |
| ATM-016 | `MapPt2TensorMeta` | meta {dtype float, sizes [2,3], strided} → desc {DT_FLOAT, [2,3], ND, has_* 全 true} | P0 |
| ATM-017 | `MapAscendIrDesc` | desc 有 dtype/shape/layout → 用 desc（`{4,4}`/DT_FLOAT） | P0 |
| ATM-018 | `MapAscendIrDesc` | desc 全空 → fallback 到 meta（`{1,2,3}`/DT_FLOAT） | P0 |

### 5.3 base64 解码（test_base64.cc，6 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| B64-001 | `DecodeBase64ToString` | `"SGVsbG8gV29ybGQ="` → `"Hello World"` | P0 |
| B64-002 | `DecodeBase64` | `"QlNI"` → `['B','S','H']` 三字节 | P0 |
| B64-003 | `DecodeBase64ToString` | `"not_base64***"`（非法 payload）→ `false` | P0 |
| B64-004 | `DecodeBase64` | 空串 → 空 vector，`true` | P1 |
| B64-005 | `DecodeBase64` | `"QQ=="`（padding）→ `['A']` 单字节 | P1 |
| B64-006 | `DecodeBase64` | `"abc"`（长度非 4 倍数）→ `false` | P0 |

### 5.4 consolidated 权重加载（test_consolidated_weights.cc，9 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| CWL-001 | `GeWeightLoader.ReadTensorData` | `p_weight`{3,3} + `b_bias`{3} holder dims/dtype/format/ByteSize/requires_grad 全对 | P0 |
| CWL-002 | `GeWeightLoader.ReadTensorData` | `"does_not_exist"` → `result.ok=false`/`!HasValue` | P0 |
| CWL-003 | `GeWeightLoader.Create` | `missing_meta` fixture → `MISSING_TENSOR_META` issue | P0 |
| CWL-004 | `GeWeightLoader.Create` | `no_index` fixture → `MISSING_WEIGHTS_INDEX` issue | P0 |
| CWL-005 | `GeWeightLoader.ReadTensorData` | weight float `[1,5,9]` + bias `[0.5,1.5,2.5]` 二进制值精确 | P0 |
| CWL-006 | `GeWeightLoader.GetTensorRecordInfo` | `p_weight` is_parameter=true、`b_bias`=false | P1 |
| CWL-007 | `GeWeightLoader.GetTensorRecordInfo` | weight offset=0/size=36、bias offset=36/size=12 | P1 |
| CWL-008 | `GeWeightLoader.GetTensorMeta` | weight{3,3}/dtype7/grad、bias{3}/dtype7/!grad | P1 |
| CWL-009 | `GeWeightLoader.HasTensor` | weight/bias true、unknown false | P1 |

### 5.5 epair 目录结构（test_epair_archive.cc，5 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| EAD-001 | fixture 目录 | `extra/ep_json_with_ascend_ir.json` 存在（`file.good()`） | P1 |
| EAD-002 | fixture 目录 | `data/weights.bin` 存在（binary 模式） | P1 |
| EAD-003 | fixture 目录 | `data/weights_index.json` 存在 | P1 |
| EAD-004 | fixture 目录 | model.json 内容非空 | P1 |
| EAD-005 | fixture 目录 | `/nonexistent/...json` → `!file.good()`（缺目录 fail） | P2 |

### 5.6 epair model parser（test_epair_model_parser.cc，7 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| EMP-001 | `EpairModelParser::Parse` | annotated fixture：2 节点、schema 8.7、3 input/1 output、`LogicalNameOfPlaceholder(p_weight)=weight`/`PlaceholderKindOf(b_bias)=Buffer`/`FindTensorValue(mm)` 非空 | P0 |
| EMP-002 | `EpairModelParser::Parse` | 内联 JSON sym_int user input `batch_size` → GraphInputs[0].name | P0 |
| EMP-003 | `EpairModelParser::Parse` | int/float/bool/string 四种 value ref → 4 个 inputs，`result.ok` | P1 |
| EMP-004 | `EpairModelParser::Parse` | `auto_functionalized` v1 节点 → `UNSUPPORTED_FUNCTIONALIZED_V1`，`result.ok=false` | P0 |
| EMP-005 | `ParseModelJson` | tensor meta `as_expr`（Symbol('s0'), hint 4）→ sizes[0]=-1（sym 占位）/sizes[1]=3 | P1 |
| EMP-006 | `ParseModelJson` | int/float/bool/string_values maps 各非空，键值正确 | P1 |
| EMP-007 | `EpairModelParser::Parse` | `add.Tensor` 节点含 `as_int` 标量输入 → `NodeInputKind::ScalarInt`/name=`other` | P0 |

### 5.7 value registry（test_value_registry.cc，7 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| VREG-001 | `RegisterPlaceholderSource` | 注册 `"input"` → `HasRegisteredRef(input)`/(input:0) true，`Resolve` output_index=0 | P0 |
| VREG-002 | `RegisterProducerOutput` | 注册 `"mm:0"` → `HasRegisteredRef` true，`Resolve` output_index=1 | P0 |
| VREG-003 | `RegisterProducerOutput` | 重复 key `"dup:0"` → 第二次 `false`/`result.ok=false` | P0 |
| VREG-004 | `RegisterAlias` | alias `"alias_name"`→`"real_output:0"` → `Resolve` output_index=0 | P1 |
| VREG-005 | `RegisterAlias` | 无 port alias→target:0 → `HasRegisteredRef(alias_no_port)`/(alias_no_port:0) 均 true | P1 |
| VREG-006 | `Resolve`/`HasRegisteredRef` | `"does_not_exist"` → `Resolve=nullptr`/`HasRegisteredRef=false` | P1 |
| VREG-007 | `RegisterAlias`（链式） | origin:0→mid→end → `Resolve(end)` output_index=2 | P1 |

## 6. 计划目录

```text
tests/c++/epair_archive/
├── UT_DESIGN.md
├── CMakeLists.txt
├── gen_fixtures.py
├── test_aten_schema_resolver.cc
├── test_aten_tensor_desc_mapper.cc
├── test_base64.cc
├── test_consolidated_weights.cc
├── test_epair_archive.cc
├── test_epair_model_parser.cc
└── test_value_registry.cc
```

构建产物（`${BINARY_DIR}`，不入仓）：`fixtures/`（`.epair` 目录，由 `gen_fixtures.py` 生成）、7 个 test 可执行文件。

## 7. 执行方式

ASAN 构建 + 测试（强制）：

```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build --output-on-failure
```

覆盖率（可选自检，独立 build 目录，与 ASAN 互斥）：

```bash
BUILD_DIR=build_cov bash scripts/cann_lm_compile/build_cpp.sh --coverage
ctest --test-dir build_cov
lcov --capture --directory build_cov --output-file build_cov/coverage.info
lcov --extract build_cov/coverage.info "$(pwd)/cann_lm/*" --output-file build_cov/coverage.cann_lm.info
lcov --summary build_cov/coverage.cann_lm.info
```

> 注：`--no-external` 会误排除 csrc 源（路径解析），用「全抓 + extract cann_lm」替代；ASAN build 裸跑无需 `LD_PRELOAD`（flags 含 `-static-libasan`）。

## 8. 准入与遗留风险

准入条件：7 个可执行文件全 PASS、ASAN 无内存错误、`-Wall -Wextra -Werror` 零新增警告、`ctest` 退出码 0、CPU 环境可重复。

遗留风险与限制：

1. **fixtures 依赖**：3 个 NEEDS_FIXTURES 测试（`test_consolidated_weights`/`test_epair_archive`/`test_epair_model_parser`）需 `gen_fixtures.py` 产 `.epair` 目录；缺则这 3 个失败，4 个纯单测（`test_base64`/`test_value_registry`/`test_aten_schema_resolver`/`test_aten_tensor_desc_mapper`）仍可独立跑。
2. **CANN 重库依赖**：test exe 直接 link `graph`/`metadef`/`ascendcl` 等，需 ② 真 CANN env（`ASCEND_HOME_PATH`，非 stubs）；与 Python UT（converter venv + stubs-first `LD_LIBRARY_PATH`）env 互斥，分开跑。
3. **`ge_error_codes.h` deprecated**：CANN 侧头文件 deprecated 提示属 CANN，非本仓代码，`-Werror` 不阻断本仓目标编译。
4. **不测 OM 编译/运行**：`ge_graph_builder.cc`/`epair_model_loader.cc` 的 OM 编译/图构建执行体归 e2e（`examples/test_converter/export_llm_model/run_e2e_pipeline.sh`），UT 只测 csrc 的 epair 加载/解析/映射层，故二者覆盖率 0%，不作为 UT 准入条件。

当前没有 DISABLED 或跳过用例；所有已收集用例均作为普通回归用例执行。

## 9. 执行结果（2026-09-05）

### 9.1 测试结果

```text
100% tests passed, 0 tests failed out of 7
Total Test time (real) =  30.95 sec
ctest exit code: 0
```

- 收集用例总数：62
- 可执行文件：7，全部通过
- DISABLED/跳过：0
- ASAN 构建：`ENABLE_ASAN=ON`（`build/`），ctest 裸跑无 `LD_PRELOAD`，无内存错误报告

各可执行文件结果：

| # | 可执行文件 | 用例数 | 结果 | 耗时 |
|---|---|---:|---|---:|
| 1 | test_consolidated_weights | 9 | Passed | 4.43s |
| 2 | test_epair_archive | 5 | Passed | 4.42s |
| 3 | test_epair_model_parser | 7 | Passed | 4.42s |
| 4 | test_base64 | 6 | Passed | 4.41s |
| 5 | test_value_registry | 7 | Passed | 4.42s |
| 6 | test_aten_schema_resolver | 10 | Passed | 4.42s |
| 7 | test_aten_tensor_desc_mapper | 18 | Passed | 4.43s |

单 exe 耗时含 ASAN 初始化与 CANN 重库加载；测试本体为毫秒级。

### 9.2 覆盖率结果

执行命令：

```bash
ctest --test-dir build_cov
lcov --capture --directory build_cov --output-file build_cov/coverage.info
lcov --extract build_cov/coverage.info "$(pwd)/cann_lm/*" \
  --output-file build_cov/coverage.cann_lm.info
lcov --summary build_cov/coverage.cann_lm.info
```

整体行覆盖率为 **37.7%（774 of 2054 lines）**，函数覆盖率为 **50.7%（69 of 136 functions）**，branch 无数据（lcov 默认不报，需 `--rc lcov_branch_coverage=1`）。纯逻辑/映射模块达到不低于 80% 的准入目标；OM 编译路径 0% 归 e2e。核心模块覆盖情况：

| csrc 模块 | 行覆盖率 | 函数覆盖率 |
|---|---:|---:|
| `common/base64.cc` | 97.7% | 100% |
| `backend/value_registry.cc` | 87.9% | 100% |
| `backend/aten_schema_resolver.cc` | 86.0% | 100% |
| `backend/aten_tensor_desc_mapper.cc` | 85.9% | 100% |
| `backend/epair_model_parser.cc` | 55.1% | 69.8% |
| `backend/ge_weight_loader.cc` | 45.9% | 47.8% |
| `backend/epair_model_loader.cc` | 0.0% | 0.0% |
| `backend/ge_graph_builder.cc` | 0.0% | 0.0% |

剩余未覆盖代码主要位于 `ge_graph_builder.cc`（676 行）与 `epair_model_loader.cc`（55 行）的 OM 编译/图构建执行体——需真实 epair→OM 链路（`cann_lm_compile` + GE 图引擎 + NPU acl），留给 e2e（`examples/test_converter/export_llm_model/run_e2e_pipeline.sh`），避免在 UT 中引入真 NPU 设备与重图构建依赖。`ge_weight_loader.cc`/`epair_model_parser.cc` 未覆盖部分为非 consolidated 分支与异步加载路径，后续可补 fixture 扩展。

### 9.3 缺陷修复记录

| ID | 原现象 | 修复内容 | 回归结果 |
|---|---|---|---|
| — | — | 本会话只验证 ASAN 构建 + ctest 通过与覆盖率采集，未改动 `cann_lm/` 或 test/CMake 文件 | 7/7 PASS、ASAN 无内存错误、覆盖率 37.7% 行/50.7% 函数 |

本轮无缺陷修复记录：测试代码与 `csrc` 在 ASAN 构建下全部通过，覆盖率基线（37.7%/50.7%）作为后续回归对比的基线供参考。
