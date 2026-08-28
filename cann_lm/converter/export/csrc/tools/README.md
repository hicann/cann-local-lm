# cann_lm_compile — epair→OM 编译工具

把 converter 导出的 epair（`ep_json_with_ascend_ir.json` + `weight_info.json`）编译为 CANN OM 模型（`aclgrphBuildModel`）。

> **不依赖 torch/venv**：csrc 源码零 torch 引用（无 `#include <torch/c10/ATen>`、无 `torch::/at::` 调用）。`.so` 不链 libtorch，**构建只需 CANN + third_party(json/gtest)，运行只需 CANN env**。

## 目录文件

| 文件 | 作用 |
|------|------|
| `compile_main.cc` | 工具主程序：EpairModelLoader 加载 epair → 构建 GE Graph → `aclgrphBuildInitialize`/`BuildModel`/`SaveModel`/`Finalize` → OM |
| `reg_register.h` | 自研算子注册（仅 `ChunkGatedDeltaRule`，qwen3.5 线性注意力用；标准算子由 CANN opp 运行时注册，不在此重复） |
| `fusion_autoub_off.cfg` | GE 融合开关配置（关 AUB + AddRmsNorm/InplaceAddRmsNorm），由 `set_env_compile.sh` 经 `GE_FUSION_SWITCH_FILE` 指向 |

## 流程（3 阶段）

```
[1/3] Load epair       EpairModelLoader(model_json, weights_index)
[2/3] Extract GE Graph  loader.GetGEGraph(&graph)   ← 需 op schema (CANN opp + reg_register.h)
[3/3] Compile graph     aclgrphBuildInitialize → aclgrphBuildModel → aclgrphSaveModel → aclgrphBuildFinalize
```

产物：`<output>.om`（可达数 GB）。

---

## 安装方式

### 方式一：源码安装（build.sh）

适用：本机有源码仓 + CANN + third_party。`build.sh` 一键：前置校验 → source env → 构建。

**前置**：
- CANN 已装（`/home/developer/Ascend/cann/set_env.sh` 可读 → `ASCEND_HOME_PATH`）
- `torch-delegate-harmony/output/third_party`（nlohmann_json + GTest 预构建依赖；与 torch 无关）

```bash
# 仅校验前置（不构建）
bash scripts/cann_lm_compile/build.sh --check

# 校验 + source env + 构建（全树: backend .so + cann_lm_compile + tests）
bash scripts/cann_lm_compile/build.sh

# 校验 + env + 仅工具（跳 tests）
bash scripts/cann_lm_compile/build.sh --no-tests --target cann_lm_compile

# 校验 + env + ASAN（开发期）
bash scripts/cann_lm_compile/build.sh --asan

# 仅 env（source 模式，持久到当前 shell，不构建）
source scripts/cann_lm_compile/build.sh
```

`build.sh` 透传参数给 `build_cpp.sh`（`--no-tests`/`--asan`/`--debug`/`--target ...`/`--no-compile-tool`）。产物：`build/cann_lm_compile` + `build/libtorch_delegate_backend.so`。

**运行**（源码安装后）：

```bash
source scripts/set_env_compile.sh           # 设 CANN env + SOC + fusion cfg
./build/cann_lm_compile <model_json> <weights_index> [output_name]
# 或一键（build + run + 存日志）:
bash scripts/cann_lm_compile/run_compile.sh [epair_dir|--no-build|--keep-om|--output <name>]
```

### 方式二：run 包安装（.run，自包含）

适用：安装机无源码、无 venv，**只需 CANN**。先在构建机用 `package_run.sh` 造 `.run`，再拷到安装机安装。

**构建机造包**（需源码 + CANN + third_party）：

```bash
bash scripts/cann_lm_compile/package_run.sh
# → dist/cann_lm_compile-<ver>-<arch>.run  (约 300 KB)
```

`package_run.sh` 流程：`build_cpp.sh` 构建 → stage(binary + lib/.so + cfg + set_env.sh) → `patchelf --set-rpath '$ORIGIN'`（可重定位）→ `makeself` 打自解压包。

`.run` 包内容：
```
cann_lm_compile                         # 二进制
lib/libtorch_delegate_backend.so        # backend (rpath=$ORIGIN)
fusion_autoub_off.cfg                   # 融合开关
set_env.sh                              # 安装后 source (LD_LIBRARY_PATH + GE_FUSION_SWITCH_FILE + SOC_VERSION)
```

**安装机安装 + 运行**（仅需 CANN）：

```bash
# 安装
bash cann_lm_compile-0.1.0-aarch64.run --target /opt/cann-lm

# 运行
source /home/developer/Ascend/cann/set_env.sh   # CANN env (lib64/simulator/driver/opp)
source /opt/cann-lm/set_env.sh                   # 工具 env (INSTALL_DIR/lib + cfg + SOC)
/opt/cann-lm/cann_lm_compile <model_json> <weights_index> [output_name]
# → <output_name>.om
```

---

## 运行期环境变量（编译选项，免重编调图优化）

| 变量 | 作用 |
|------|------|
| `SOC_VERSION` | 目标 SoC（默认 `Ascend910_9362`；源码模式由 `set_env_compile.sh` 设，run 包由 `set_env.sh` 设） |
| `GE_FUSION_SWITCH_FILE` | 融合开关配置（默认 `tools/fusion_autoub_off.cfg` 或安装目录的同名文件） |
| `GE_BUFFER_OPT` | `ge.bufferOptimize`（`off_optimize`/`l1_optimize`/`l2_optimize`） |
| `GE_OO_LEVEL` | `ge.oo.level`（`O1`/`O3`） |
| `GE_OPT_SWITCH` | `ge.optimizationSwitch`（`pass:off;...`） |
| `GE_RUN_MODE` | `ge.graphRunMode`（`0`=推理/静态，`1`=训练/动态） |
| `GE_SKIP_DSP=1` | 设 graph 属性 `_no_need_dynamic_shape_partition=true`，跳过动态 shape 分区 |
| `GE_EXTERNAL_WEIGHT` | 权值分离：`0`=内嵌(默认)/`1`=每个权值分文件/`2`=全合一 .bin。`=2` 时 `.om` 变小(仅结构), 权值存到 `<output 目录>/weight/delegate_graph_weight_combined`。运行时用 `aclmdlSetExternalWeightAddress` 加载。**约束**：output 路径须与 CWD 同文件系统(GE 临时权值在 CWD, `MoveFilePath` 跨 fs rename 会 EXDEV 失败) |

---

## 使用样例

### qwen3.5（含 ChunkGatedDeltaRule 线性注意力）

```bash
bash scripts/cann_lm_compile/run_compile.sh \
  examples/test_converter/export_llm_model/cann_lm_cli_export/epair_qwen3_5_single \
  --keep-om --output /tmp/qwen35
# → /tmp/qwen35.om (~1.6 GB)
```

### gemma4（不使用 ChunkGatedDeltaRule）

```bash
bash scripts/cann_lm_compile/run_compile.sh \
  examples/test_converter/export_llm_model/om_npu_verify/epair_gemma4 \
  --keep-om --output /tmp/gemma4
# → /tmp/gemma4.om (~4.6 GB)
```

gemma4 用 22 种 GE 算子（Const/Reshape/Transpose/MatMul/Cast/RmsNorm/Add/Mul/Unsqueeze/TransposeD/ConcatV2/BatchMatMul/Gelu/RotaryMul/Muls/GatherV2/SoftmaxV2/Identity/AddRmsNorm/TensorMove/InplaceUpdate/IndexByTensor），全部 CANN 自带 → 不依赖 `reg_register.h`。

### prefill-decode 两图

prefill-decode 模式导出的 epair 目录含 `prefill_ep_json_with_ascend_ir.json` + `decode_ep_json_with_ascend_ir.json`，分别编译：

```bash
D=examples/test_converter/export_llm_model/cann_lm_cli_export/epair_gemma4_no_quant
bash scripts/cann_lm_compile/run_compile.sh "$D/prefill_ep_json_with_ascend_ir.json" "$D/weight_info.json" /tmp/gemma4_prefill --keep-om
bash scripts/cann_lm_compile/run_compile.sh "$D/decode_ep_json_with_ascend_ir.json"  "$D/weight_info.json" /tmp/gemma4_decode  --keep-om
```

### 权值分离（GE_EXTERNAL_WEIGHT=2，全合一 .bin）

`.om` 仅含结构（小）+ 权值单独存 `.bin`，运行时用 `aclmdlSetExternalWeightAddress` 加载外置权值。

```bash
source scripts/set_env_compile.sh
D="examples/test_converter/export_llm_model/cann_lm_cli_export/epair_qwen3_5_single"
# output 必须与 CWD 同文件系统 (GE 临时权值在 CWD, 跨 fs rename 会失败)
OUT=/mnt/workspace/project/CANN-LM_5883/build/sep/qwen35
GE_EXTERNAL_WEIGHT=2 ./build/cann_lm_compile \
  "$D/ep_json_with_ascend_ir.json" "$D/weight_info.json" "$OUT"
# → $OUT.om (~33 MB, 仅结构)
# → $(dirname $OUT)/weight/delegate_graph_weight_combined (~1.5 GB, 全合一条值)
```

对比内嵌模式：`qwen35.om` = 1.6 GB（权值内嵌）vs 权值分离 = 33 MB `.om` + 1.5 GB `.bin`。

---

## 脚本速查（scripts/cann_lm_compile/）

| 脚本 | 职责 |
|------|------|
| `build.sh` | 前置校验（CANN + json + gtest）+ source env + 构建（源码安装入口） |
| `set_env_compile.sh` | env（CANN lib64 + SOC + fusion cfg，被各脚本 source） |
| `build_cpp.sh` | cmake 构建（需 env 已 source） |
| `run_compile.sh` | 跑工具（build + run + 存日志） |
| `package_run.sh` | 造 `.run` 自解压包（build + stage + patchelf + makeself） |
| `_env_common.sh` | env 公共部分（CANN source + PYTHONPATH，被 set_env_compile source） |

## 备注

- **reg_register.h**：仅注册 `ChunkGatedDeltaRule`（CANN 没有的自研算子）。标准算子的 V1 schema 由 CANN opp（`libopsproto`，GERT V2，`aclgrphBuildInitialize` 时由 `libopp_registry` 加载）提供，无需在此重复。原全 86 算子版备份在 `/tmp/opencode/reg_register.h.bak`。
- **退出告警**：`cann_lm_compile` 退出时可能打印 `corrupted size vs. prev_size` + `Aborted`（exit 134）—— 是 ge `REG_OP` 静态注册器 teardown 的 glibc 噪声，OM 在此前已正常产出，不影响产物。`run_compile.sh` 用 `|| true` 兜住，靠 OM 文件存在判定成功。
- **OM 解析**：用 `atc --mode=6 --om=<path>.om`（真 CANN env）查看算子/大小，见 AGENTS.md「查看 OM 模型算子」。
