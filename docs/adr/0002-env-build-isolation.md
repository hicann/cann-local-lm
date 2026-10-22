# ADR-0002: 环境与构建隔离 — 四套 env + 顶层统一 build 树

> 状态: Accepted
> 日期: 2026-08-27
> 相关设计: [AGENTS.md](../../../AGENTS.md)「运行环境设置」+ [export_design.md](../cann_lm_converter/export/export_design.md)

## 背景 (Context)

仓库存在三类开发/构建活动，原本边界模糊、脚本硬编码、env 逻辑三处重复：

1. **converter 工具**：HF→epair 导出 + 量化，需 Linux CPU venv + torchair stubs（不装 torch_npu）。
2. **模型转换工具**：epair→编译后模型（OM），需真 CANN + C++ 编译。
3. **generate**：面向鸿蒙 PC 的交叉编译（本仓 `generate/`）。

探查发现的具体问题（P1-P6）：

- **P1**：`DELEGATE_COMMON_FLAGS` 全仓只有引用、无定义（`TorchDelegateTestStubs.cmake:30-31` 引用，但无任何 `set()`），当前等于空串，违反 AGENTS.md「公共 Flag 原则」。
- **P2**：AGENTS.md 写「ASAN 强制」，但 4 个 CMakeLists 无一实现 `ENABLE_ASAN` option；`build_and_test.sh` 也不传 `-DENABLE_ASAN=ON`。名存实亡。
- **P3**：CANN 版本漂移——`build_csrc.sh`/`build_and_test.sh`/`tests/.../CMakeLists.txt` 默认 `cann-9.1.0`，但本机仅 `cann-9.2.0`（`cann`→`cann-9.2.0` 符号链接），脚本默认值直接不可用。
- **P4**：env 逻辑三处重复——`set_env.sh` + `run.sh` + `run_pd6_cases.sh`（`setup_converter_env`/`setup_dopt_env`）各抄一份硬编码路径。
- **P5**：4 个 CMakeLists 全在 compile+link 阶段硬编码 `-Wall -Wextra`（linker 不认 link 阶段的 `-Wall`，且违反公共 Flag 原则）。
- **P6**：cmake module path 二义——csrc `CMakeLists.txt` 指本仓 `cmake/modules`，但 `build_csrc.sh` 又 `-DCMAKE_MODULE_PATH` 指 sibling `torch-delegate-harmony/cmake`，两份同名 `Find*.cmake` 都在搜索路径。

## 决策 (Decision)

按「venv（运行时）+ build/（产物）+ toolchain（CMake 目标）」三层隔离，建立**四套互不叠加的 env** + **顶层统一 CMake build 树**：

- env：`converter`(①) / `compile`(②, 原 `om` 重命名+借①venv torch cmake) / `generate`(③, 预留) / `npu`(④, verify-runtime)。`om` 作 `compile` 兼容别名。
- C++ 构建：新建**顶层 `CMakeLists.txt`**（`add_subdirectory` 串 csrc+examples+tests），单入口 `scripts/cann_lm_compile/build_cpp.sh`；**删除** `csrc/build_csrc.sh`（原 csrc 内独立构建脚本）与 `tests/build_and_test.sh`（原独立 cmake 测试 runner）——前者 6 处调用方 fallback 全切到 `build/libtorch_delegate_backend.so` + `bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend`，后者由 `ctest --test-dir build` 替代（tests CMakeLists 已给每个测试自带 `ENVIRONMENT LD_LIBRARY_PATH`，裸跑即可）。csrc CMakeLists 设 `LIBRARY_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}` 使 .so 落 build 根而非按源码树嵌套（保 fallback 路径稳定）。ASAN flags 用 `-static-libasan`，ASAN 二进制也可裸 `ctest`（免 LD_PRELOAD）。
- 公共 Flag：新建 `cmake/modules/DelegateCommonFlags.cmake`，**定义** `DELEGATE_COMMON_FLAGS=-Wall -Wextra -Werror` + `ENABLE_ASAN` option + `delegate_apply_common_flags()` 辅助函数；所有 C++ target 经此接入，禁止单 target 硬编码。
- ② compile 借 ① converter venv 的 CPU torch cmake（torch 2.9.0+cpu ABI 稳定，backend .so 不调 NPU）；不另建 build venv。
- ③ generate 仅预留：`generate/` + 注释化 OHOS toolchain stub，待目标产物/NDK 确定后补全。

## 后果 (Consequences)

- **正面**：
  - env 边界清晰，四套互不叠加；直 source `scripts/set_env_<env>.sh`（export/compile/generate/npu 各自具名脚本，公共部分 `_env_common.sh`）。
  - `-Werror` 落地（零警告原则兑现），ASAN option 接入（`build_cpp.sh --asan`）。
  - CANN 路径统一走 `ASCEND_HOME_PATH`（修 P3 cann-9.1.0 坏默认值）；venv/torchair 路径走 `TORCH_DELEGATE_HOME` 可覆盖。
  - 顶层统一 build 树：一条 `bash scripts/cann_lm_compile/build_cpp.sh` 出 backend .so + OM binary + tests。
  - env 逻辑收敛到 `set_env_<env>.sh` + 公共 `_env_common.sh`，`run.sh`/`run_pd6_cases.sh` 不再抄一份（修 P4）。
- **负面/代价**：
  - `cann-9.1.0` 默认值移除后，未 source CANN env 直接跑 `build_cpp.sh` 会 FATAL（改此前静默用错误路径）。这是有意的 fail-fast。
  - ② compile 借 ① venv：torch 2.9.0+cpu 链进 backend .so，若将来 .so 需在 NPU/鸿蒙上与不同 torch ABI 共载，需重新评估是否建 ② 专用 venv。
  - `-Werror` 暴露预存死代码 `PlanFallbackNode`（ge_graph_builder.cc:1543，0 引用），已加 `[[maybe_unused]]` 静默；死码清理留作后续。
- **引入的约束**：
  - 新增 C++ target **必须**经 `delegate_apply_common_flags()` 接入公共 Flag，禁止硬编码 `-Wall/-Wextra/-Werror`。
  - C++ 测试运行前**必须** `bash scripts/cann_lm_compile/build_cpp.sh --asan && ctest --test-dir build`（ASAN flags 含 `-static-libasan`，ctest 裸跑无需 LD_PRELOAD）。
  - 新增构建脚本**不得**硬编码 `cann-9.x.0`/venv 绝对路径，一律走 `ASCEND_HOME_PATH`/`TORCH_DELEGATE_HOME` 等可覆盖变量。
  - `generate/` 下新增内容须遵循"预留 stub → 工具链确定后补全"流程，不得在 stub 阶段引入可用假设。

## 替代方案 (Alternatives)

### 替代方案 A: ② compile 建专用 build venv

否决：torch 2.9.0+cpu ABI 稳定，backend .so 不调 NPU（仅链接期借 torch cmake + CANN 头），借 ① venv 零成本且与 converter 同 torch 版本一致；建 ② venv 维护双份 torch、收益不抵成本。保留作未来 ABI 冲突时的回退。

### 替代方案 B: 顶层 CMakeLists 完全取代 build_csrc.sh（删除 build_csrc.sh）

**采纳**（原"薄封装留 csrc/"为过渡，最终删除）：`build_csrc.sh` 已删除，6 处调用方（2 example CMakeLists + tests CMakeLists + `export_epair_and_compile/run.sh` + `export_quant_matmul_single_op/run.sh` + `om_npu_verify/check_env.sh`）的 fallback 全切到 `build/libtorch_delegate_backend.so` + `bash scripts/cann_lm_compile/build_cpp.sh --target torch_delegate_backend`。单一 C++ 构建逻辑（build_cpp.sh），cann_lm/ 源码树内不再留构建脚本。

### 替代方案 C: npu 折叠进 converter

否决：converter venv **不装 torch_npu**（设计决策，走 stub 路径），CANN 轻量化工具 GPTQ / NPU 出字 / OM 验证需真 torch_npu，无法折叠。npu 必须独立作 verify-runtime。

## 关联

- 上游：[ADR-0001](./0001-load-model-cpu-quant-subprocess-device.md)（CPU 量化 + subprocess device）
- 实现：`scripts/set_env_{export,compile,generate,npu}.sh` + `_env_common.sh`、`scripts/cann_lm_compile/build_cpp.sh`、`cmake/modules/DelegateCommonFlags.cmake`、顶层 `CMakeLists.txt`、`generate/`
