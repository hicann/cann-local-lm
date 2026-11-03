# cann_lm_compile — epair→OM/OMC 编译工具

把 converter 导出的 epair（`ep_json_with_ascend_ir.json` + `weight_info.json`）编译为 CANN OM/OMC 模型（`aclgrphBuildModel`）。

> 零 torch 依赖：csrc 源码不引用 torch，`.so` 不链 libtorch。构建只需 CANN + third_party(json/gtest)，运行只需 CANN env。

## 目录文件

| 文件 | 作用 |
|------|------|
| `compile_main.cc` | 工具主程序：EpairModelLoader 加载 epair → 构建 GE Graph → `aclgrphBuildInitialize`/`BuildModel`/`SaveModel`/`Finalize` → OM/OMC |
| `reg_register.h` | 自研算子注册（仅 `ChunkGatedDeltaRule`；标准算子由 CANN opp 运行时加载） |
| `fusion_autoub_off.cfg` | GE 融合开关配置（关 AUB + AddRmsNorm/InplaceAddRmsNorm） |

## 流程（3 阶段）

```
[1/3] Load epair       EpairModelLoader(model_json, weights_index)
[2/3] Extract GE Graph  loader.GetGEGraph(&graph)
[3/3] Compile graph     aclgrphBuildInitialize → aclgrphBuildModel → (mobile rename / aclgrphSaveModel) → aclgrphBuildFinalize
```

产物：`<output>.om`（Ascend）或 `<output>.omc`（Kirin mobile 路径）。

---

## 安装方式

### 方式一：源码安装（build_cpp.sh）

适用：本机有源码仓 + CANN + third_party。

**前置**：
- CANN 已装（`ASCEND_HOME_PATH` 可用）
- `torch-delegate-harmony/output/third_party`（nlohmann_json + GTest 预构建依赖）

```bash
# source CANN env (需先设 ASCEND_HOME_PATH 或 source CANN set_env.sh)
source scripts/set_env_compile.sh

# 全量构建 (backend .so + cann_lm_compile + tests)
bash scripts/cann_lm_compile/build_cpp.sh

# 仅工具 (跳 tests)
bash scripts/cann_lm_compile/build_cpp.sh --target cann_lm_compile --no-tests

# ASAN (开发期强制)
bash scripts/cann_lm_compile/build_cpp.sh --asan
```

产物：`build/cann_lm_compile` + `build/libtorch_delegate_backend.so`。

### 方式二：run 包安装（.run，自包含）

适用：安装机无源码、无 venv，仅需 CANN。先在构建机造包，再拷到安装机安装。

**构建机造包**：

```bash
bash scripts/cann_lm_compile/package_run.sh
# → build_out/cann_local_lm_converter-<ver>-<arch>.run  (约 300 KB)
```

`.run` 包内容：
```
cann_lm_compile                   # 二进制
lib/libtorch_delegate_backend.so  # backend (rpath=$ORIGIN)
fusion_autoub_off.cfg             # 融合开关
set_env.sh                        # 安装后 source (LD_LIBRARY_PATH + GE_FUSION_SWITCH_FILE + SOC_VERSION)
```

**安装机安装 + 运行**：

```bash
# 安装
bash cann_lm_compile-0.1.0-aarch64.run --target /opt/cann-local-lm

# 运行
source /home/developer/Ascend/cann/set_env.sh   # CANN env
source /opt/cann-local-lm/set_env.sh                   # 工具 env
/opt/cann-local-lm/cann_lm_compile <model_json> <weights_index> [output_name]
```

---

## 用法

```bash
cann_lm_compile <model_json_path> <weights_index_path> [output_name]
```

- `model_json_path`：`ep_json_with_ascend_ir.json` 路径
- `weights_index_path`：`weight_info.json` 路径
- `output_name`：产物名（默认取 model_json 文件名 stem）

产物扩展名：Ascend → `.om`，Kirin mobile → `.omc`（compile_main 自动检测 mobile 路径改名）。

---

## 运行期环境变量

| 变量 | 作用 |
|------|------|
| `SOC_VERSION` | 目标 SoC（如 `KirinX90`、`Ascend910_9362`；注意大小写匹配 platform_config `.ini`） |
| `GE_FUSION_SWITCH_FILE` | 融合开关配置文件路径（默认 `fusion_autoub_off.cfg`） |
| `GE_BUFFER_OPT` | `ge.bufferOptimize`（`off_optimize`/`l1_optimize`/`l2_optimize`） |
| `GE_OO_LEVEL` | `ge.oo.level`（`O1`/`O3`） |
| `GE_OPT_SWITCH` | `ge.optimizationSwitch`（`pass:off;...`） |
| `GE_EXTERNAL_WEIGHT` | 权值分离：`0`=内嵌(默认)/`1`=分文件/`2`=全合一 |
| `SAVE_AIR_FILE` | 指定路径 → `ge::Graph::SaveToFile` 存编译前 GE 图为二进制 protobuf `.air` |
| `SAVE_AIR` | `=1` → 存 `<output_name>.air`（同上，用 output_name 命名） |

### 权值分离（GE_EXTERNAL_WEIGHT）

`=0`（默认）：权值内嵌 `.om`/`.omc`。
`=2`：`.om`/`.omc` 仅含结构，权值存 `<output 目录>/weight/<model>_weight_combined`，运行时 `aclmdlLoadWithConfig` 自动加载旁路 weight/ 目录。
`=1`：per-file（kirinx90 mobile 路径不支持，BuildModel 失败）。

**注意**：`ge.externalWeightDir`（权值落盘目录）走 `init_options`（`aclgrphBuildInitialize`），不走 `build_options`（会被拒 50331649）。compile_main 已处理：`GE_EXTERNAL_WEIGHT!=0` 时自动设 `ge.externalWeightDir = <output>/weight/`。

### 保存 .air（SAVE_AIR_FILE / SAVE_AIR）

在 GetGEGraph 后、BuildModel 前调用 `ge::Graph::SaveToFile`，存编译前 GE 图为二进制 protobuf `.air`。该 `.air` 可直接用 `atc --framework=1 --model=.air` 编译。

---

## 脚本速查（scripts/cann_lm_compile/）

| 脚本 | 职责 |
|------|------|
| `build_cpp.sh` | cmake 构建（需 env 已 source） |
| `set_env_compile.sh` | env（CANN lib64 + SOC + fusion cfg） |
| `package_run.sh` | 造 `.run` 自解压包（build + stage + patchelf + makeself） |
