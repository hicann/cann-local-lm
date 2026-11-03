# CANN-LM编译指南

简体中文 | [English](../../en/reference/compilation_guide.md)

## 概览：build.sh做什么

仓库根 `build.sh` 是**全量构建/打包/UT的唯一入口**。主入口为 `--pkg`/`-u`/`-h`；另支持 `check`/`deps`/`cpp`/`wheel`/`third_party`/`package`/`all`/`smoke` 等子命令（日常开发用，详见 [build_guide.md](../build_guide.md) Part 3 速查）。`--pkg` 全量流程：

| 命令 | 作用 | 产物 |
|------|------|------|
| `bash build.sh --pkg` | 编译三部分并打包 | `build_out/` 下 `compile_converter-*.run`（必备）+ `dist/` 下签名 engine wheel（需完整 OHOS 环境，缺则仅产 compile_converter.run） |
| `bash build.sh -u` | 编译含UT的部分并运行 | 测试结果（无持久产物） |
| `bash build.sh -h` | 打印帮助 | — |

### `--pkg` 的5个阶段

```
[1/5] compile      build_cpp.sh --no-tests --target cann_lm_compile
                   → build/cann_lm_compile (epair→OM 编译工具, ~308K)
                   → build/libtorch_delegate_backend.so (C++ backend, ~845K)
[2/5] converter    纯源码 pip wheel (--no-deps)
                   → build_out/cann_local_lm-<ver>-py3-none-any.whl (converter 导出 whl)
[3/5] engine whl   engine python ctypes 绑定 whl (纯 python, 不依赖 OHOS)
                   → build_out/cann_local_lm_engine-<ver>-py3-none-any.whl
[4/5] engine OHOS  OHOS 交叉编译 + 签名 (build_package_ohos.sh, 需 OHOS SDK)
                   → deploy_ohos/ (signed libcann_lm_engine.so + UT/bbit/smoke)
[5/5] 打包         .run 自解压分发物
                   → build_out/cann_local_lm_converter-<ver>-<arch>.run (compile+converter, 必备)
```

> ⚠️ **任一阶段失败，`--pkg` 立即中断**（`|| return 1` 链式退出），不会继续后续阶段、也不会打包不完整的现有产物。

### `-u` 的3段

```
[UT/1] engine host C++ UT     GTest, 编译 engine + UT 目标, 用 stub NDK 跑
[UT/2] converter C++ UT       tests/cpp/epair_archive, ctest 跑 7 个用例 (需 CANN 工具链, 无 NPU 设备)
[UT/3] Python UT              pytest: tests/python/converter + tests/python/llm_models + tests/python
```

---

## 环境准备清单

按"必须项 / engine交叉编译项 / 可选项"三档列出。**每一项都是 `build.sh` 会真实检查的**（`need_cann` / `need_third_party` / `build_engine` 5项硬检查）。

### 必须项（`--pkg` 和 `-u` 都会用到）

| # | 项 | 检查逻辑 | 准备方法 |
|---|----|---------|---------|
| 1 | **CANN toolkit** | `ASCEND_HOME_PATH` 已设 | 安装后 `source <cann>/set_env.sh`。compile阶段与converter C++ UT需要 |
| 2 | **C++ 三方库** | `output/third_party/{json,gtest,makeself}` 均存在 | `bash scripts/build.sh third_party`编译前执行，下载三方库到指定路径然后执行脚本安装 |
| 3 | **cmake / make** | `command -v cmake && command -v make` | `apt install cmake build-essential` |
| 4 | **pip** | `command -v pip`（`--pkg` 的converter阶段） | `--no-deps` 纯源码打包，系统pip即可 |
| 5 | **python torch** | `pip list` | torch版本需要大于等于2.4 |

### engine交叉编译项（`--pkg` 的 `[3/4]` 强制检查，缺任一直接中断）

`build_engine()` 会**逐项打印 ✓/✗，任一 ✗ 即 `return 1` 中断整个 `--pkg`**。共5项：

| # | 项 | 检查逻辑 | 准备方法 |
|---|----|---------|---------|
| 1 | **编译器** | `$OHOS_SDK_NATIVE/llvm/bin/clang` 存在且可执行| 用完整SDK（华为官方command-line-tools组件 的native子树带有，需要在环境内使用对应组件） |
| 2 | **OHOS SDK** | `OHOS_SDK_NATIVE` 已设且含 `sysroot/` 子目录 | `export OHOS_SDK_NATIVE=<sdk>/linux/<ver>/native` |
| 3 | **java + 签名jar** | `java` 在PATH；`tools/binary-sign-tool.jar` 或 `cann_local_lm/engine/ohos/binary-sign-tool.jar` 存在 | `apt install default-jre`；从SDK `.../26.0.0/toolchains/lib/binary-sign-tool.jar` 拷到 `tools/` |
| 4 | **GTest源码tarball** | `GTEST_TAR`（三方库gtest`gtest-1.14.0.tar.gz`）文件存在 | 执行`export GTEST_TAR=<path>/googletest-1.14.0.tar.gz`设置|
| 5 | **cmake / make** | 同2.1-3 | — |

### 可选项 / 说明

| 项 | 说明 |
|----|------|
| `VERSION` / `ARCH` | 默认 `0.1.0` / `$(uname -m)`，可 `export` 覆盖（影响 .run文件名） |
| `THIRD_PARTY` | 默认 `$REPO/output/third_party`，可覆盖 |
| `BUILD_DIR` / `UT_BUILD_DIR` | 默认 `build/` / `/tmp/llm_ut_host`，可覆盖 |
| `PIP` | 默认系统pip；若 `deps/torch-delegate/.venv/bin/pip` 存在则优先用它 |
| patchelf | 缺失则跳过rpath设置（`[4/5]` 打 .run时 `(patchelf未装, 跳过; 运行靠set_env.sh的LD_LIBRARY_PATH)`），**不阻塞** |


## 打包流程（`bash build.sh --pkg`）

### 运行

```bash
cd <repo>   # /srv/workspace/.../cann/CANN-LM-main
# 先按 §2.4 设好环境变量
bash build.sh --pkg
```

### 成功日志长什么样（对照判读）

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

▶ [2/4] converter: build cann_local_lm wheel (--no-deps 纯源码打包)
  Created wheel for cann-lm: filename=cann_local_lm-0.1.0-py3-none-any.whl size=163313   ← 阶段2 关键行
  ✓ converter whl: .../build_out/cann_local_lm-0.1.0-py3-none-any.whl (160K)                  ← 阶段2 成功标志

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
=== [2/2] 打包签名 wheel (build_package_ohos.sh → cann_local_lm_engine-*+signed*.whl) ===
  ✓ copy signed .so → cann_local_lm/lib/libcann_lm_engine.so
  ✓ copy libc++_shared.so → cann_local_lm/lib/
  ✓ pip wheel → build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl   ← 阶段3 成功标志

▶ [4/4] 打包 .run (compile_converter)
  ... makeself 压缩 ...
  Self-extractable archive ".../cann_local_lm_converter-0.1.0-x86_64.run" successfully created.
▶ 打包结果
  533K .../build_out/cann_local_lm_converter-0.1.0-x86_64.run
  845K .../build_out/cann_local_lm_engine-0.1.0+signed-py3-none-any.whl
▶ === --pkg 完成 ===                                                                 ← 整体成功标志
```

**成功判读要点**：每个阶段的 `✓` 成功标志都出现 + 末尾 `▶ === --pkg完成 ===`。若中途有 `✗` 或 `FATAL_ERROR`，该阶段 `return 1`，不会走到 `--pkg完成`。

### 常见失败场景与日志特征

| 阶段 | 失败症状（打屏） | 根因 | 处置 |
|------|-----------------|------|------|
| **前置** | `✗ CANN ... ASCEND_HOME_PATH未设` / `✗ 缺third_party/json ...` | 没source CANN / 没建三方库 | 补 §2.1-1/2 |
| **[1/4] compile** | `CMake Error ... Could NOT find GTest / nlohmann_json` | `output/third_party` 里json/gtest只有tar没解压产物 | 补 §2.1-2 |
| **[3/4] engine前置** | 多个 `✗`（缺编译器 / OHOS_SDK_NATIVE无效 / 缺jar / 缺GTest tar）+ `✗ engine交叉编译前置不满足; 已中断 --pkg` | §2.2任一未满足 | 按 §2.2补齐，`export` 环境变量 |
| **[3/4] engine configure** | `CMake Error at .../ohos/CMakeLists.txt:97: FATAL_ERROR "GTest source tarball not found: /mnt/workspace/gtest-1.14.0.tar.gz"` | 没 `export GTEST_TAR`，用了写死的默认路径 | `export GTEST_TAR=<真实存在的tar>` 重跑 |
| **[4/4]打包** | `✗ engine 签名 wheel 未生成, 打包失败` | 阶段3没跑出产物（前置被中断，或产物校验有 `[缺]`） | 回阶段3修 |
| **[4/4]打包** | `⚠ compile/converter产物不全, 跳过compile_converter.run` | 阶段1/2产物缺失但脚本继续 | 完整 `--pkg` 一次跑通 |


### 产物验证（打包后）

```bash
# ① .run 自解压完整性（"All good" = 通过）
bash build_out/cann_local_lm_converter-0.1.0-x86_64.run --extract=/tmp/x1 --noexec
# 预期输出: Verifying archive integrity... All good.

# ② compile 工具冒烟（需 CANN 环境）
source <cann>/set_env.sh && ./build/cann_lm_compile   # 打印 usage 即正常
# ③ engine 签名 wheel: 推到鸿蒙设备 pip install 后 cllm generate --config testcase.json
```

---

## UT流程（`bash build.sh -u`）

### 运行

```bash
cd <repo>
source CANN toolkit path   # converter C++ UT 需要 CANN
bash build.sh -u
```

> 若只跑Python UT / engine host UT可不用CANN；但 `-u` 会顺序跑三段，converter C++ UT依赖CANN，缺则该段 `✗`。

### 成功日志特征（三段各一个成功标志）

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
- engine host UT：所有 `ut_*` 行PASS + `bbit --check: PASS` → `✓ engine host UT全部通过`（若出现此句则**整体无失败**）
- converter C++ UT：`100% tests passed, 0 tests failed out of 7`
- Python UT：三个 `✓ ...: PASS`
- **末尾**：`▶ === -u全部通过 ===` = 整体成功；`▶ === -u有失败 (见上) ===` = 至少一段失败

### 常见失败场景

| 段 | 失败症状 | 根因 / 处置 |
|----|---------|-------------|
| 前置 | `✗ 缺third_party/gtest ...` | 未建三方库，补 §2.1-2 |
| [UT/1] | 单个 `ut_*: FAIL (exit=...)` | 看具体断言 |
| [UT/2] | `✗ CANN ... ASCEND_HOME_PATH未设` | 没source CANN |
| [UT/3] | `✗ tests/...: FAIL` | pytest断言失败，看上方具体输出，重点检查torch版本有无问题 |
---

## 日志判读速查表（成功/失败一眼判断）

| 打印内容 | 含义 |
|---------|------|
| `▶ === --pkg完成 ===` | **--pkg整体成功**（4阶段全绿） |
| `▶ === -u全部通过 ===` | **-u整体成功**（三段全绿） |
| `▶ === -u有失败 (见上) ===` | -u有失败，往上找 `✗`；多数情况是model_executor_ut已知失败 |
| `✓ <项> ...` | 该项检查/产物成功 |
| `✗ <项> ...` | 该项失败（前置检查/产物缺失/UT失败） |
| `⚠ <项> ...` | 警告（不中断，如patchelf缺失、已知UT失败） |
| `[ok] <file>` | engine交叉编译产物存在（build_ohos.sh校验） |
| `[缺] <file>` | engine交叉编译产物缺失 → 阶段3必失败 |
| `FATAL_ERROR` | cmake configure致命错误（如GTEST_TAR默认路径不存在） |
| `Verifying archive integrity... All good` | .run自解压完整性通过 |
| `100% tests passed, 0 tests failed out of 7` | converter C++ UT全过 |

---

## FAQ

**Q1：只想编译compile+converter，不想被engine前置卡住？**
`--pkg` 会强制engine交叉编译（缺一中断）。只编前两部分的场景用：
```bash
source <cann>/set_env.sh
bash scripts/cann_lm_compile/build_cpp.sh --no-tests --target cann_lm_compile   # compile C++
bash scripts/cann_lm_converter/build_wheel.sh                                    # converter whl
```
（engine前置检查在 `build_engine()`，不调 `--pkg` 就不会触发。）

**Q2：没有OHOS SDK / 不想交叉编译，`-u` 能跑吗？**
能。`-u` 不依赖OHOS SDK（engine host UT用host gcc + stub NDK）。只有 `--pkg` 的 `[3/4]` 需要SDK。

**Q3：engine wheel 装到设备跑不起来？**

`.so` 加载失败常见原因：未开发者模式（自签名 `.so` 被拒）、`libc++_shared.so` 不在同目录、`.so` 名被改。用 `cllm generate --config testcase.json --show-paths` 排查路径，或 `--lib <path>` 显式指定 `.so`。
