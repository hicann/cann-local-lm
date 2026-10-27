# CANN-LM 编译指南

## 1. 概览：build.sh 做什么

仓库根 `build.sh` 是**全量构建/打包/UT 的唯一入口**，只支持 3 个子命令：

| 命令 | 作用 | 产物 |
|------|------|------|
| `bash build.sh --pkg` | 编译三部分并打包 | `dist/` 下两个 `.run` 分发物 |
| `bash build.sh -u` | 编译含 UT 的部分并运行 | 测试结果（无持久产物） |
| `bash build.sh -h` | 打印帮助 | — |

### `--pkg` 的 4 个阶段

```
[1/4] compile     build_cpp.sh --no-tests --target cann_lm_compile
                  → build/cann_lm_compile (epair→OM 编译工具, ~308K)
                  → build/libtorch_delegate_backend.so (C++ backend, ~845K)
[2/4] converter   纯源码 pip wheel (--no-deps)
                  → dist/cann_lm-<ver>-py3-none-any.whl (converter 导出 whl)
[3/4] engine      OHOS 交叉编译 + 签名 + 打包 (build_package_ohos.sh)
                  → engine_ohos.run (libcann_lm_engine.so + 6 UT + bbit + 2 smoke + run.sh)
[4/4] 打包        两个 .run 自解压分发物
                  → dist/cann_lm_compile_converter-<ver>-<arch>.run (compile+converter)
                  → dist/cann_lm_engine_ohos-<ver>-<arch>.run       (engine 部署包)
```

> ⚠️ **任一阶段失败，`--pkg` 立即中断**（`|| return 1` 链式退出），不会继续后续阶段、也不会打包不完整的现有产物。

### `-u` 的 3 段

```
[UT/1] engine host C++ UT     GTest, 编译 engine + UT 目标, 用 stub NDK 跑
[UT/2] converter C++ UT       tests/cpp/epair_archive, ctest 跑 7 个用例 (需 CANN 工具链, 无 NPU 设备)
[UT/3] Python UT              pytest: tests/python/converter + tests/python/llm_models + tests/python
```

---

## 2. 环境准备清单

按"必须项 / engine 交叉编译项 / 可选项"三档列出。**每一项都是 `build.sh` 会真实检查的**（`need_cann` / `need_third_party` / `build_engine` 5 项硬检查）。

### 2.1 必须项（`--pkg` 和 `-u` 都会用到）

| # | 项 | 检查逻辑 | 准备方法 |
|---|----|---------|---------|
| 1 | **CANN toolkit** | `ASCEND_HOME_PATH` 已设 | 安装后 `source <cann>/set_env.sh`。compile 阶段与 converter C++ UT 需要 |
| 2 | **C++ 三方库** | `output/third_party/{json,gtest,makeself}` 均存在 | `bash scripts/build.sh third_party`编译前执行，下载三方库到指定路径然后执行脚本安装 |
| 3 | **cmake / make** | `command -v cmake && command -v make` | `apt install cmake build-essential` |
| 4 | **pip** | `command -v pip`（`--pkg` 的 converter 阶段） | `--no-deps` 纯源码打包，系统 pip 即可 |
| 5 | **python torch** | `pip list` | torch版本需要大于等于2.4 |

### 2.2 engine 交叉编译项（`--pkg` 的 `[3/4]` 强制检查，缺任一直接中断）

`build_engine()` 会**逐项打印 ✓/✗，任一 ✗ 即 `return 1` 中断整个 `--pkg`**。共 5 项：

| # | 项 | 检查逻辑 | 准备方法 |
|---|----|---------|---------|
| 1 | **编译器** | `$OHOS_SDK_NATIVE/llvm/bin/clang` 存在且可执行| 用完整 SDK（华为官方command-line-tools组件 的 native 子树带有，需要在环境内使用对应组件） |
| 2 | **OHOS SDK** | `OHOS_SDK_NATIVE` 已设且含 `sysroot/` 子目录 | `export OHOS_SDK_NATIVE=<sdk>/linux/<ver>/native` |
| 3 | **java + 签名 jar** | `java` 在 PATH；`tools/binary-sign-tool.jar` 或 `cann_lm/engine/ohos/binary-sign-tool.jar` 存在 | `apt install default-jre`；从 SDK `.../26.0.0/toolchains/lib/binary-sign-tool.jar` 拷到 `tools/` |
| 4 | **GTest 源码 tarball** | `GTEST_TAR`（三方库gtest`gtest-1.14.0.tar.gz`）文件存在 | 执行`export GTEST_TAR=<path>/googletest-1.14.0.tar.gz`设置|
| 5 | **cmake / make** | 同 2.1-3 | — |

### 2.3 可选项 / 说明

| 项 | 说明 |
|----|------|
| `VERSION` / `ARCH` | 默认 `0.1.0` / `$(uname -m)`，可 `export` 覆盖（影响 .run 文件名） |
| `THIRD_PARTY` | 默认 `$REPO/output/third_party`，可覆盖 |
| `BUILD_DIR` / `UT_BUILD_DIR` | 默认 `build/` / `/tmp/llm_ut_host`，可覆盖 |
| `PIP` | 默认系统 pip；若 `deps/torch-delegate/.venv/bin/pip` 存在则优先用它 |
| patchelf | 缺失则跳过 rpath 设置（`[4/5]` 打 .run 时 `(patchelf 未装, 跳过; 运行靠 set_env.sh 的 LD_LIBRARY_PATH)`），**不阻塞** |


## 3. 打包流程（`bash build.sh --pkg`）

### 3.1 运行

```bash
cd <repo>   # /srv/workspace/.../cann/CANN-LM-main
# 先按 §2.4 设好环境变量
bash build.sh --pkg
```

### 3.2 成功日志长什么样（对照判读）

```
▶ === CANN-LM 全量构建 + 打包 (--pkg) ===

▶ [1/4] compile: build_cpp.sh --no-tests --target cann_lm_compile
  ✓ CANN 环境path
...
[100%] Built target cann_lm_compile
=== Build complete ===
  308K .../build/cann_lm_compile
  845K .../build/libtorch_delegate_backend.so
  ✓ compile 产物就绪: build/cann_lm_compile + build/libtorch_delegate_backend.so   ← 阶段1 成功标志

▶ [2/4] converter: build cann_lm wheel (--no-deps 纯源码打包)
  Created wheel for cann-lm: filename=cann_lm-0.1.0-py3-none-any.whl size=163313   ← 阶段2 关键行
  ✓ converter whl: .../dist/cann_lm-0.1.0-py3-none-any.whl (160K)                  ← 阶段2 成功标志

▶ [3/4] engine: OHOS 交叉编译 (build_package_ohos.sh, 强制环境检查)
  ✓ 编译器: SDK clang (.../23/native/llvm/bin/clang)                               ← 5 项前置检查全 ✓
  ✓ OHOS_SDK_NATIVE: .../23/native
  ✓ java: /usr/bin/java
  ✓ binary-sign-tool.jar: 已就绪 (tools/)
  ✓ GTEST_TAR: /tmp/gtest-1.14.0.tar.gz
  ✓ cmake/make: 已就绪
  ✓ engine 交叉编译前置就绪 (...)
=== [1/2] OHOS 交叉编译 + 签名(build_ohos.sh) ===
=== cmake configure (OHOS cross) ===
=== build (make -j32) ===
[100%] Built target ...
=== 产物校验 ===
  [ok]   libcann_lm_engine.so
  [ok]   libc++_shared.so
  [ok]   bbit / api_smoke / c_api_smoke / ut_engine_executor / ut_tokenizer / ut_c_api / ut_scheduler / model_executor_ut / ut_text_embedding   ← 11 项全 [ok]
=== sign binaries ===
  ✓ libcann_lm_engine.so -> libcann_lm_engine_signed.so
  ✓ ... (其余 10 项)                                                            ← 签名全 ✓
=== [2/2] 打包部署包(package_ohos.sh → engine_ohos.run) ===
  ✓ libcann_lm_engine_signed.so -> libcann_lm_engine.so
  ✓ ... (11 项全 ✓)                                                            ← 打包全 ✓
  ✓ engine 交叉编译 + 打包完成: engine_ohos.run (libcann_lm_engine.so + UT/bbit/smoke)   ← 阶段3 成功标志

▶ [4/4] 打包 .run (compile_converter + engine_ohos)
  ... makeself 压缩 ...
  Self-extractable archive ".../cann_lm_compile_converter-0.1.0-x86_64.run" successfully created.
  Self-extractable archive ".../cann_lm_engine_ohos-0.1.0-x86_64.run" successfully created.
▶ 打包结果 (dist/)
  533K .../dist/cann_lm_compile_converter-0.1.0-x86_64.run
  2.7M .../dist/cann_lm_engine_ohos-0.1.0-x86_64.run
▶ === --pkg 完成 ===                                                                 ← 整体成功标志
```

**成功判读要点**：每个阶段的 `✓` 成功标志都出现 + 末尾 `▶ === --pkg 完成 ===`。若中途有 `✗` 或 `FATAL_ERROR`，该阶段 `return 1`，不会走到 `--pkg 完成`。

### 3.3 常见失败场景与日志特征

| 阶段 | 失败症状（打屏） | 根因 | 处置 |
|------|-----------------|------|------|
| **前置** | `✗ CANN ... ASCEND_HOME_PATH 未设` / `✗ 缺 third_party/json ...` | 没 source CANN / 没建三方库 | 补 §2.1-1/2 |
| **[1/4] compile** | `CMake Error ... Could NOT find GTest / nlohmann_json` | `output/third_party` 里 json/gtest 只有 tar 没解压产物 | 补 §2.1-2 |
| **[3/4] engine 前置** | 多个 `✗`（缺编译器 / OHOS_SDK_NATIVE 无效 / 缺 jar / 缺 GTest tar）+ `✗ engine 交叉编译前置不满足; 已中断 --pkg` | §2.2 任一未满足 | 按 §2.2 补齐，`export` 环境变量 |
| **[3/4] engine configure** | `CMake Error at .../ohos/CMakeLists.txt:97: FATAL_ERROR "GTest source tarball not found: /mnt/workspace/gtest-1.14.0.tar.gz"` | 没 `export GTEST_TAR`，用了写死的默认路径 | `export GTEST_TAR=<真实存在的 tar>` 重跑 |
| **[4/4] 打包** | `✗ engine_ohos.run 未生成, 打包失败` | 阶段3 没跑出产物（前置被中断，或产物校验有 `[缺]`） | 回阶段3 修 |
| **[4/4] 打包** | `⚠ compile/converter 产物不全, 跳过 compile_converter.run` | 阶段1/2 产物缺失但脚本继续 | 完整 `--pkg` 一次跑通 |


### 3.4 产物验证（打包后）

```bash
# ① .run 自解压完整性（"All good" = 通过）
bash dist/cann_lm_compile_converter-0.1.0-x86_64.run --extract=/tmp/x1 --noexec
bash dist/cann_lm_engine_ohos-0.1.0-x86_64.run     --extract=/tmp/x2 --noexec
# 预期输出: Verifying archive integrity... All good.

# ② compile 工具冒烟（需 CANN 环境）
source <cann>/set_env.sh && ./build/cann_lm_compile   # 打印 usage 即正常
# ③ engine 部署包: 推到鸿蒙设备 解压后 ./run.sh
```

---

## 4. UT 流程（`bash build.sh -u`）

### 4.1 运行

```bash
cd <repo>
source CANN toolkit path   # converter C++ UT 需要 CANN
bash build.sh -u
```

> 若只跑 Python UT / engine host UT 可不用 CANN；但 `-u` 会顺序跑三段，converter C++ UT 依赖 CANN，缺则该段 `✗`。

### 4.2 成功日志特征（三段各一个成功标志）

```
▶ === CANN-LM UT 编译 + 运行 (-u) ===

▶ [UT/1] engine host C++ UT (GTest)
  • cmake configure (BUILD_ENGINE=ON, BUILD_CONVERTER=OFF)
  • cmake build (engine + UT 目标)
  • 运行 UT 二进制 (含 stub NDK)
  == ut_request_handle
  ✓   ut_request_handle: PASS
  == ut_scheduler
  ✓   ut_scheduler: PASS
  ...（每个 ut_* 一行 PASS/FAIL）
  == model_executor_ut
  ✓   model_executor_ut: PASS
  • bbit --check (黑盒自检)
  ✓ bbit --check: PASS

▶ [UT/2] converter C++ UT (tests/cpp/epair_archive, 需 CANN 工具链, 无 NPU 设备)
  ✓ CANN CANN toolkit path
  • ctest
100% tests passed, 0 tests failed out of 7        ← 7/7 全过

▶ [UT/3] Python UT (pytest)
  ✓ tests/python/converter: PASS
  ✓ tests/python/llm_models: PASS
  ✓ tests/python: PASS

▶ === -u 全部通过 ===
```

**成功判读要点**：
- engine host UT：所有 `ut_*` 行 PASS + `bbit --check: PASS` → `✓ engine host UT 全部通过`（若出现此句则**整体无失败**）
- converter C++ UT：`100% tests passed, 0 tests failed out of 7`
- Python UT：三个 `✓ ...: PASS`
- **末尾**：`▶ === -u 全部通过 ===` = 整体成功；`▶ === -u 有失败 (见上) ===` = 至少一段失败

### 4.3 常见失败场景

| 段 | 失败症状 | 根因 / 处置 |
|----|---------|-------------|
| 前置 | `✗ 缺 third_party/gtest ...` | 未建三方库，补 §2.1-2 |
| [UT/1] | 单个 `ut_*: FAIL (exit=...)` | 看具体断言 |
| [UT/2] | `✗ CANN ... ASCEND_HOME_PATH 未设` | 没 source CANN |
| [UT/3] | `✗ tests/...: FAIL` | pytest 断言失败，看上方具体输出，重点检查torch版本有无问题 |
---

## 5. 日志判读速查表（成功/失败一眼判断）

| 打印内容 | 含义 |
|---------|------|
| `▶ === --pkg 完成 ===` | **--pkg 整体成功**（4 阶段全绿） |
| `▶ === -u 全部通过 ===` | **-u 整体成功**（三段全绿） |
| `▶ === -u 有失败 (见上) ===` | -u 有失败，往上找 `✗`；多数情况是 model_executor_ut 已知失败 |
| `✓ <项> ...` | 该项检查/产物成功 |
| `✗ <项> ...` | 该项失败（前置检查/产物缺失/UT 失败） |
| `⚠ <项> ...` | 警告（不中断，如 patchelf 缺失、已知 UT 失败） |
| `[ok] <file>` | engine 交叉编译产物存在（build_ohos.sh 校验） |
| `[缺] <file>` | engine 交叉编译产物缺失 → 阶段3 必失败 |
| `FATAL_ERROR` | cmake configure 致命错误（如 GTEST_TAR 默认路径不存在） |
| `Verifying archive integrity... All good` | .run 自解压完整性通过 |
| `100% tests passed, 0 tests failed out of 7` | converter C++ UT 全过 |

---

## 6. FAQ

**Q1：只想编译 compile+converter，不想被 engine 前置卡住？**
`--pkg` 会强制 engine 交叉编译（缺一中断）。只编前两部分的场景用：
```bash
source <cann>/set_env.sh
bash scripts/cann_lm_compile/build_cpp.sh --no-tests --target cann_lm_compile   # compile C++
bash scripts/cann_lm_converter/build_wheel.sh                                    # converter whl
```
（engine 前置检查在 `build_engine()`，不调 `--pkg` 就不会触发。）

**Q2：没有 OHOS SDK / 不想交叉编译，`-u` 能跑吗？**
能。`-u` 不依赖 OHOS SDK（engine host UT 用 host gcc + stub NDK）。只有 `--pkg` 的 `[3/4]` 需要 SDK。

**Q3：engine_ohos.run 装到设备跑不起来？**
见 `engine_ohos.run` 解压后 README：需开发者模式（自签名）、目录可写可执行（避开 noexec）、`.so` 名字不可改。
