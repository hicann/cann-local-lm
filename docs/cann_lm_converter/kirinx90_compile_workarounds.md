# kirinx90 .omc 编译规避清单（操作留档）

> 目的：记录把 CANN-LM 小网络（`examples/test_converter/export_kirin_ops_net`，覆盖 8 个 kirinx90 算子）编译成 kirinx90 `.omc` 过程中遇到的所有卡点与规避方案，供复现/后续固化。
> 范围：aarch64 鸿蒙宿主 + cann-9.2.0（kirinx90 移动端 model 路径）。
> 结论：`.omc` 成功生成（`kirin_ops_kirinx90.omc`，`atc --mode=6` 确认 `soc_version[KirinX90]` + 权重内嵌）。**但该 .omc 含 aicpu 算子（MatMul/BatchMatMul/SoftmaxV2/SplitV，见 aicpu 异构分析），kirinx90 不支持 AICPU → 运行无效**；`aicore_only_kirinx90.omc`（零 aicpu）才是真正可用产物。

## 背景

- 目标 SOC：`KirinX90`（鸿蒙 Kirin 芯片，CANN 9.2.0；产物扩展名 `.omc`，非 Ascend 的 `.om`）。
- 宿主：aarch64（鸿蒙内核 `hce2.aarch64`，非 x86；故 kirinx90 OM 编译是**原生**，无交叉编译）。
- kirinx90 算子库：`aic-kirinx90-ops-info-nn.json`（118 算子）+ `kernel/kirinx90/ops_nn`+`ops_math`（预编 kernel）。
- 原始痛点：kirinx90 的 kernel/engine 覆盖比 Ascend910 稀疏，且几个算子（Gelu/L2Normalize 正向入口）在 ops-nn 包里**打包缺失**。

> ⚠️ **硬约束：kirinx90 不支持 AICPU，凡异构到 AICPU 的算子一定有问题**。
> kirinx90 是移动端 mobile 路径（`MobileModelHelper`），**无 aicpu 执行能力**（不同于 Ascend910 的 `DNN_VM_AICPU`/`DNN_VM_AICPU_ASCEND`）。任何算子被 `EnginePlacer` 分到 aicpu engine（`DNN_VM_AICPU*`）或 converter fallback 到 aicpu 算子（如 `ConcatV2` aicpu 变体）→ 编译期 mobile 路径不认 aicpu 子图 / 运行期无 aicpu kernel 执行 → **必然失败**。
> **推论**：所有规避方案必须把算子留在 aicore（或 host/GE_LOCAL/RTS），**不能 fallback 到 aicpu**。B4 原拟的 `cat→ConcatV2（落 aicpu）` 方案**无效**——需改为 aicore 可用算子或 host 侧预处理（如 scatter indices 在 converter 阶段固定/常量化，不进运行图）。

## 环境准备（toolkit 隔离，不污染原 A3 环境）

原 A3 算子环境在 `/home/developer/Ascend`（不动）；另建独立 toolkit：

```
mkdir -p /mnt/workspace/cann-kirinx90
# ① toolkit（lib64/GE/平台 ini/opp config）
bash Ascend-cann-toolkit_9.2.0~weekly.20260909.01_linux-aarch64.run --full \
    --install-path=/mnt/workspace/cann-kirinx90 --force --quiet
# ② A3-ops（补全 opp 结构 op_tiling/op_api/... + lib/linux/aarch64，GE OS 校验用）
bash Ascend-cann-A3-ops_9.2.0~weekly.20260909.01_linux-aarch64.run --install \
    --type=toolkit --install-path=/mnt/workspace/cann-kirinx90 --force --quiet
# ③ kirinx90 ops-nn + ops-math + ops-transformer（kirinx90 预编 kernel + config）
#    ⚠️ 必须用 --upgrade --quiet，不能用 --full --force！
#    --full --force 不覆盖 toolkit 自带的 opp/built-in ops-info（只读），导致 ops-info 缺算子
#    --upgrade --quiet 正确覆盖 opp/built-in ops-info + kernel binary
bash cann-kirinx90-ops-nn_9.1.0_linux-aarch64.run --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
bash cann-kirinx90-ops-math_9.1.0_linux-aarch64.run --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
bash cann-kirinx90-ops-transformer_9.1.0_linux-aarch64.run --upgrade --quiet --install-path=/mnt/workspace/cann-kirinx90
```

跑前 env（须先 unset 旧 CANN 变量防污染，通用原则见 AGENTS.md「多套 CANN 并存须先 unset」）：
```bash
# 清多 CANN 污染（只 unset CANN 相关，保留系统 Python/numpy/HOME）
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH TOOLKIT_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH PYTHONPATH SOC_VERSION GE_FUSION_SWITCH_FILE GE_EXTERNAL_WEIGHT ASCEND_VISIBLE_DEVICES DEVICE
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/buildtools/python-3.12.9/bin:/usr/local/cmake/bin
source /mnt/workspace/cann-kirinx90/cann-9.2.0/set_env.sh
export SOC_VERSION=KirinX90                 # CamelCase，匹配 KirinX90.ini 的 SoC_version=KirinX90
unset  GE_FUSION_SWITCH_FILE                # 首编先用 GE 默认融合
export GE_EXTERNAL_WEIGHT=0                 # 内嵌权重
# libpython 前置（修复 _ctypes undefined symbol _PyErr_SetLocaleString，见 A5）+ build_kx + CANN lib64
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:$PROJECT_ROOT/build_kx:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
# A7 修复: sitecustomize 屏蔽 torch_npu (避免 GE 嵌入 python 加载系统 torch_npu → cann-9.0.0 GE 双重加载堆损坏)
#   sitecustomize.py 内容: import sys; [sys.modules.__setitem__(m, None) for m in ('torch_npu','torch','torchair','torch_delegate')]
#   PYTHONPATH 前置含 sitecustomize.py 的目录; GE 嵌入 libpython 加载 site 时阻止 torch_npu 被 import
export PYTHONPATH="<sitecustomize_dir>:${PYTHONPATH:-}"
# ASAN build (定位堆损坏用, D 节): cann_lm_compile 编到 build_kx_asan/, GE lib 未插桩但 cann_lm 代码插桩
#   cmake -S $PROJECT_ROOT -B build_kx_asan -DASCEND_INSTALL_PATH=<cann-kirinx90>/aarch64-linux -DENABLE_ASAN=ON ...
# cd 到输出目录跑（onlinec→.omc rename 同目录同设备，见 A6）
```

## 规避清单

### A. toolkit/环境层

| # | 卡点 | 规避 |
|---|---|---|
| A1 | kirinx90 算子散落、原 A3 环境不能动 | 隔离安装独立 toolkit `/mnt/workspace/cann-kirinx90`（上节三步），原 `/home/Developer/Ascend` 零改动 |
| A2 | OS 校验失败 `The OS linux is not within the support list of {}` | 装 **A3-ops** 补全 `tbe/{op_tiling,op_api,op_master_device,op_tiling_device,impl_mode}` + `lib/linux/aarch64`（GE 的 `GetOppSupportedOsAndCpuType` 扫这些 os/cpu 子目录构建支持列表；toolkit 单装只有 `config/impl/kernel/op_host`） |
| A3 | `scene.info`/`version.info` 缺失（安装被超时打断没生成） | `chmod u+w` 只读 `tbe/kernel/` 目录后，从现有 `cann-9.2.0/cann-9.2.0` 拷 `scene.info`（`os=linux`）+ `version.info` 到新 opp；修 OS 校验 |
| A4 | SOC 字符串大小写 | `SOC_VERSION=KirinX90`（CamelCase，匹配 `KirinX90.ini`）；小写 `kirinx90` 只是 opp config 目录名，不是 socVersion，GE 找不到 `kirinx90.ini` |
| A5 | **新 weekly（≥20260909）** source CANN 后 `_ctypes` 导入失败（`undefined symbol: _PyErr_SetLocaleString`）→ TBE/tvm 不可用 → `aclgrphBuildModel failed` | `python3` NEEDED `libpython3.12.so.1.0`，source CANN 后 `LD_LIBRARY_PATH` 改变使其被解析为**系统旧版** `/lib/aarch64-linux-gnu/libpython3.12.so.1.0`（缺 `_PyErr_SetLocaleString`，该符号为较新 patch 才有；`/opt/buildtools/python-3.12.9/lib/libpython3.12.so.1.0` 才有）。单独 `python3 -c "import ctypes"`（不 source）正常，source 后才复现。**修复**：`LD_LIBRARY_PATH` 前置 `/opt/buildtools/python-3.12.9/lib`（含正确 libpython），或 `LD_PRELOAD=/opt/buildtools/python-3.12.9/lib/libpython3.12.so.1.0` |
| A6 | **新 weekly（≥20260909）** `compile_main.cc` 的 `onlinec→{output}.omc` rename 失败：`std::filesystem::rename: Invalid cross-device link` | mobile 路径把模型写到 **CWD** 的 `onlinec`，而 `compile_main.cc` T1 用 `std::filesystem::rename("onlinec", output+".omc")` 改名；CWD 与 output 目录跨分区（如 CWD=`/mnt/workspace`、output=`/tmp`）时 rename 失败 → 进程异常退出 + TBE subprocess 连锁报错。**修复**：`cd` 到输出目录跑（`onlinec` 与目标 `.omc` 同目录同设备）；或输出目录与 CANN-LM 仓同分区 |
| A7 | **torch_npu 双重 GE 堆损坏**（掩盖真正错误）：GE BuildModel 阶段嵌入 libpython（`ge_python_runtime_manager.cc:228 EnsureLibphanLoaded`，加载 `/opt/buildtools/python-3.12.9/lib/libpython3.12.so.1.0`），嵌入 python 用**系统 sys.path** → 加载系统 site-packages 的 `torch_npu` → `libtorch_npu.so` NEEDED 链接 **cann-9.0.0** 的 `libge_runner`/`libge_compiler`/`libascendcl`（`ldd libtorch_npu.so` 可见），与 cann_lm_compile 已加载的 **kirinx90 cann-9.2.0** GE lib 双重加载 → 全局 GE 状态/堆冲突 → glibc `corrupted size vs. prev_size` 堆损坏（非 ASAN 可检，因 GE lib 未插桩）。**此堆损坏掩盖真正编译错误**（如 B4 `get_kernel_type_enum` 异常），需先消除才能定位。**修复**：`sitecustomize.py` 设 `sys.modules['torch_npu']=None`（+ `torch`/`torchair`/`torch_delegate`），`PYTHONPATH` 前置含 `sitecustomize.py` 的目录 → GE 嵌入 python 加载 site 时阻止 torch_npu 被后续 import。注意：TBE fusion `fusion_util.py:get_torch_npu_version` 会 `import torch_npu` 失败（`import of torch_npu halted; None in sys.modules`），但属 DEBUG 级 fallback，非致命 | `sitecustomize.py` + `PYTHONPATH`（见环境准备段）；venv python 不可直接作 PYTHONHOME（venv 无标准库 `encodings`，会崩）；`PYTHONHOME=合成目录`（系统标准库符号链接 + venv/tbe_pylibs site-packages）虽能隔离但 BuildInitialize 阶段崩，不采用 |
| A8 | **`TOOLCHAIN_HOME` 污染致 ops-math/ops-nn 源码仓编译 ccec 用旧 CANN asc 头**（区别于 A1-A7 的 OM 编译场景，本条是 ops 源码仓 `build.sh --pkg` 编译场景）：系统 `~/.bashrc` 启动时 source `/home/developer/Ascend/cann/set_env.sh`（cann-9.0.0）设 `TOOLCHAIN_HOME=/home/developer/Ascend/cann-9.0.0/toolkit`。编译 ops-math（`/mnt/workspace/gitCode/ops-math`，`bash build.sh --pkg --soc=kirinx90`）时 ccec/asc_opc 经 `TOOLCHAIN_HOME` 解析到 **cann-9.0.0** 的 `aarch64-linux/asc` 头文件（而非 cann-kirinx90 cann-9.2.0），致 `int4x2_t` redefinition（cann-9.0.0 `kernel_utils_macros.h` vs bisheng `__clang_cce_types.h`）+ `GetBlockIdxImpl`/`GetSubBlockIdxImpl` 等 undeclared + `recursive aicore function call is not allowed` → ccec 编译 cast 等 AIV_ONLY kernel 全挂（`ld.lld: cannot open ..._0.o`，.o 没生成）→ 整包 `binary` 阶段失败。AGENTS.md unset 列表原写 `TOOLKIT_HOME`（旧名），**实际系统变量是 `TOOLCHAIN_HOME`**（多个 CHAIN），漏 unset 即踩。**修复**：unset 列表补 `TOOLCHAIN_HOME`（与 `TOOLKIT_HOME` 都 unset 保险），source cann-kirinx90 set_env 后 ccec 用 cann-9.2.0 的 asc 头，cast 等编译通过 | env unset 补 `TOOLCHAIN_HOME`（见环境准备段 + AGENTS.md「多套 CANN 并存须先 unset」） |

### B. 网络/converter 层（规避 kirinx90 无 engine 的算子）

> 8 目标算子：`MatMul`/`BatchMatMul`/`RmsNorm`/`Gelu`/`L2Normalize`/`Sigmoid`/`GatherV2`/`ScatterNdUpdate`。
> 其中 6 个（MatMul/BatchMatMul/RmsNorm/Gelu/Sigmoid/GatherV2/ScatterNdUpdate）装好 kirinx90 ops 包后即有 engine；另需规避 3 处：

| # | 卡点 | 规避 | 落点 |
|---|---|---|---|
| B1 | `Muls`（`scores*1/√d` 标量乘）无 engine | `self.scale` 从 python `float` 改 **tensor buffer** → `scores*scale` 变 `ge.Mul`（有 engine），非 `ge.Muls` | `kirin_ops_net.py` |
| B2 | `TensorMove`（scatter out-of-place converter 的 const→mutable 拷贝）无 engine | ① `kv_cache` 从 `nn.Parameter` 改 **forward 输入**（placeholder→Data 可变变量；`ge.ScatterNdUpdate` 是 ref 节点，ref 输入不能引用 Const，故 cache 必须是输入而非 Parameter）；② `compile_to_epair` 里 **monkeypatch** `npu_scatter_nd_update.default` converter，直发 `ge.ScatterNdUpdate`（跳过 `ge.TensorMove`） | `kirin_ops_net.py` + `kirin_ops_net.py::compile_to_epair` |
| B3 | `L2Normalize` 正向 impl 缺失 + tvm CCE codegen 崩 | ops-nn 包打包缺口（缺正向 `ops_nn/dynamic/l2_normalize.py` 动态入口，只有 `_grad`；legacy `ops_legacy/dynamic/l2_normalize.py` 是 tvm/tbe-DSL，kirinx90 上 `tvm CCECodegenImpl` 崩 EB9999）→ **手写 AscendC-apt wrapper** `l2_normalize.py`（仿 `rms_norm.py`/`gelu_apt.py`），用 `asc_op_compiler.compile_op`→ccec 编已打包的 `ascendc/l2_norm/l2_normalize.cpp`，绕开 tvm CCE 路径 | 写入 kirinx90 opp `ops_nn/dynamic/l2_normalize.py` |

> 以上 B1–B3 是 `kirin_ops_net.py` 小网络的**就地规避**（改 modeling/monkeypatch）。真实模型层（gemma4/qwen3）走 **`KIRINX90_CONVERTER_PATCHES`** 机制（converter 导出时 patch GE converter，见 B4–B6）。

### B.2 KIRINX90_CONVERTER_PATCHES 机制（真实模型层规避）

> 环境变量 `KIRINX90_CONVERTER_PATCHES=1`，由 `cann_lm/converter/export/ge_converters/ge_converter.py:80` 检测，加载 `_kirinx90_converter_patches.py`，在 `register_ge_converters()` 之后 `register_kirinx90_patches()` 覆盖（last-write-wins）6 类算子 converter：
> 1. `aten.mm`/`matmul`/`bmm` → `MatMulV3`/`BatchMatMulV3`（kirinx90 aicore 只 V3，无 plain MatMul）
> 2. `npu_rotary_mul` → `RotaryPositionEmbedding`（kirinx90 有此算子，走 arch35）
> 3. `aten.mul`（标量 `Muls`）→ `Mul`（标量转 `Const` tensor）
> 4. `npu_scatter_nd_update` → 直发 `ScatterNdUpdate`（跳过 `TensorMove`）
> 5. `aten.slice` → `StridedSlice`（non-V2，int32；kirinx90 的 `ConstToAttrStridedSliceV2Fusion` 是 forbidden closed 会把 V2 常量转属性 → `StridedSliceD` 无 aicore kernel → 落 host CPU → unknown shape → mobile 失败）
> 6. `aten.cat` → `ConcatD`（非 V2）—— **此条在 gemma4 真实层失败，见 B4**

| # | 卡点 | 规避 | 落点 |
|---|---|---|---|
| B4 | **`cat→ConcatD` apt 编译失败**：`concat_d_apt.py:216` 硬编码 `ascendc_src_file="arch31/concat_d.cpp"`，kernel 用 `KERNEL_TYPE_AIV_ONLY`（`arch31/concat_d.cpp:17`）。`compile_op`→`KernelInfoInfer.get_kernel_type_enum("AIV_ONLY")`（`asc_op_compile_base/asc_op_compiler/kernel_info_infer.py:195`）走 **else 分支 `raise Exception("...not support in current core version")`**——因为 kirinx90 的 soc（`KirinX90`）不在 `is_v200`/`is_v220`/`is_c310` 白名单（`ascendc_common_utility.py`：`is_v200` 只认 `Ascend310P/610`，`is_v220` 只认 `Ascend910B/910_93`，`is_c310` 只认 `Ascend950/350`），三者皆 False → else 抛异常 → `opRes.result=exception` → `second gen tbe op_info not found` → `BuildSingleOp ConcatD failed`。**注意**：① 不是 int32 不支持（ConcatD op_info dtype 含 int32、`concat_d_apt.py` DTYPE_MAP 有 int32、kernel 按字节 copy 不依赖 dtype）；② 不是 arch31/arch32 不匹配（`get_kernel_type_enum` 在 ccec 编译前抛异常，与 arch 无关）；③ `GatherV2` 也是 `AIV_ONLY` 但 GE 编译成功（走不同路径未触发 `get_kernel_type_enum`，待复测）；④ 源码注释 `KERNEL_TYPE_AIV_ONLY`（`conversion/concat_d/op_kernel/arch31/concat_d.cpp` 未提交改动）可过 `get_kernel_type_enum`，但**引入新错误**——tiling info 缺 `_pattern` + `dsl_compile_info is nullptr`（`TilingPrepare4ConcatD`），注释非完整修复。触发场景：gemma4 真实层 KV cache scatter indices 构建（`modeling_gemma4.py:68` `torch.cat([col_b,col_n,col_pos],dim=0)`，int32，K/V cache 各一次） | ⚠️ **不能 fallback `cat→ConcatV2`（落 aicpu）——kirinx90 不支持 AICPU（见背景硬约束）**。正确方向：scatter indices 的 int32 cat 是**坐标拼接**，不进 aicore 计算——在 converter 阶段把 indices 常量化/固定（K/V cache scatter 的 b/n/pos 坐标在 seq_len 固定时可预算），或用 host 侧（`DNN_VM_GE_LOCAL`/RTS）算子替代，**不留 aicpu 算子在图里** | `cann_lm/converter/export/ge_converters/_kirinx90_converter_patches.py::_kirinx90_aten_cat` + `cann_lm/models/gemma4/modeling_gemma4.py::_scatter_kv_cache` |

### C. 产物层（.omc 其实已生成）

| # | 卡点 | 规避 |
|---|---|---|
| C1 | `aclgrphSaveModel` 失败 "model is illegal"（`model.data==nullptr`）——红鲱鱼 | kirinx90 走 `mobile_model_helper.cc` 移动端 model 路径，**`aclgrphBuildModel` 期间直接把模型写到文件 `onlinec`**（默认名，CWD），不填 in-memory `ModelBufferData.data`；故 `aclgrphSaveModel` 的 null-data 校验失败，但模型已存 → **改名 `onlinec → kirin_ops_kirinx90.omc`** |

### D. 诊断手段（非常驻，定位用）

- `--asan` 重建 `cann_lm_compile`（针对 cann-kirinx90，`build_kx_asan/`）：确认非我方堆损坏——无 ASAN 报错，`model.length` 由垃圾值（~2^48 指针）变 sane（6778732）；定位到 `model.data null` 是 mobile 路径行为，非内存损坏。
- debug 日志 `ASCEND_GLOBAL_LOG_LEVEL=0`：定位到 `mobile_model_helper.cc::SaveToOmModel → model_file_saver.cc::SaveCompiledModelToFile output_model_file: onlinec`。
- **ASAN + sitecustomize 定位 torch_npu 双重 GE（A7）+ 揭示 B4**：非 ASAN build + 系统 python 直接崩 `corrupted size vs. prev_size`（A7 堆损坏），崩点非确定（BuildInitialize/BuildModel/Finalize 随堆布局变），无法定位真正错误。**关键三步**：① `sitecustomize.py` 设 `sys.modules['torch_npu']=None` 屏蔽 torch_npu（消除 A7 堆损坏）；② ASAN build（`build_kx_asan`，`-DENABLE_ASAN=ON`）改变内存布局使 glibc 不再 crash（GE lib 未插桩但 cann_lm 代码插桩，exit 255 非 crash）；③ `ASCEND_GLOBAL_LOG_LEVEL=0` SLOG 重定向后 grep ERROR 定位到 `get_kernel_type_enum`/`Cannot find engine`/`failed to recompile` 等真正编译错误（B4 的 `AIV_ONLY not support` 即此法定位）。
- **手动调 apt compile_op 复现单算子**：`python3 -c "from impl.ops_math.dynamic import concat_d_apt; concat_d_apt.concat_d(...)"`（PYTHONPATH 含 cann-kirinx90 python site-packages + tbe opp），直接看 `compile_op` 抛的 Exception（B4 的 `current kernel type: KERNEL_TYPE_AIV_ONLY is not support in current core version`），比 GE 全流程快且无 A7 干扰。

## 验证

```bash
atc --mode=6 --om=.../kirin_ops_kirinx90.omc
# system   info: atc_version[9.2.0], soc_version[KirinX90], framework_type[].
# resource info: memory_size[16801792 B], weight_size[78848 B], stream_num[2], event_num[3].
```
- `soc_version[KirinX90]` ✓
- `weight_size[78848 B]` = 权重内嵌（与 epair `weights_0.bin` 77KB 一致）✓
- 文件 677K，magic `494d4f44`("IMOD") = OM/OMC 格式 ✓
- 8 目标算子在导出的 `ep_json_with_ascend_ir.json` ascend_ir 全部出现（`check_op_coverage.py` 验过 8/8 命中 kirinx90 ops-info-nn.json）

## 端到端流程（复现）

```bash
# 1. 导出 epair（CPU venv + torchair stubs，SOC 无关）
source scripts/set_env_export.sh
python examples/test_converter/export_kirin_ops_net/kirin_ops_net.py export  --output .../kirin_ops.pt2
python examples/test_converter/export_kirin_ops_net/kirin_ops_net.py compile --ep-path .../kirin_ops.pt2 --output-dir .../kirin_ops_epair
# 2. 编译 OM/OMC（kirinx90）
source /mnt/workspace/cann-kirinx90/cann-9.2.0/set_env.sh
export SOC_VERSION=KirinX90; unset GE_FUSION_SWITCH_FILE; export GE_EXTERNAL_WEIGHT=0
export LD_LIBRARY_PATH=$PWD/build_kx:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
./build_kx/cann_lm_compile .../kirin_ops_epair/ep_json_with_ascend_ir.json .../kirin_ops_epair/weight_info.json kirin_ops_kirinx90
# 3. mobile 路径产物改名（aclgrphSaveModel 会报红鲱鱼失败，忽略）
mv onlinec .../kirin_ops_kirinx90.omc
```

## aicpu 异构分析（⚠️ kirinx90 不支持 AICPU，异构到 aicpu 一定有问题）

> **重申硬约束**：kirinx90 mobile 路径无 aicpu 执行能力。下表"8-op 网络编出 .omc 虽成功"是**早期误判**——aicpu 算子在 mobile 路径实际无法运行（编译期可能过，但运行/产物无效）。凡 `EnginePlacer` 分到 `DNN_VM_AICPU*` 或 converter fallback 到 aicpu 算子的，**一律视为失败，必须规避到 aicore**。

8-op 网络（`kirin_ops_net.py`）build 日志 `IsUnknownShapeNode` 显示**有算子落到 aicpu**（`DNN_VM_AICPU*`）——这些**必然有问题**，不可依赖：

| aicpu 算子 | engine | 数量 | 根因 |
|---|---|---|---|
| `MatMul` | DNN_VM_AICPU | 5 | kirinx90 aicore 只有 `MatMulV3`（ops-info-nn.json 无 plain MatMul），torchair 发 plain `ge.MatMul`，无自动改写 → aicpu |
| `BatchMatMul` | DNN_VM_AICPU | 2 | 同上（只有 `BatchMatMulV3`） |
| `SoftmaxV2` | DNN_VM_AICPU_ASCEND | 1 | kirinx90 只有 `MaskedSoftmaxWithRelPosBias`/`ScaledMaskedSoftmaxV2`，无 plain SoftmaxV2 |
| `SplitV` | DNN_VM_AICPU_ASCEND | 1 | 来自 `qkv.chunk(3)`，kirinx90 aicpu 承接 |

aicore 上（`AIcoreEngine`）：Add / Cast / Gelu / L2Normalize / Mul / RmsNorm / ScatterNdUpdate / Sigmoid / Transpose。`DNN_VM_GE_LOCAL`（Const/Reshape）、`DNN_VM_RTS`（Identity/Output）非 aicpu（host/RTS，可用）。

### 为什么 MatMul 不能映射到 MatMulV3（根因）
1. torchair converter 发 **plain `MatMul`**：`aten/mm.py → ge.MatMul(self, mat2, None)`，op 类型就是 `"MatMul"`；`aten/bmm.py → ge.BatchMatMul`。
2. **kirinx90 aicore 只认 V3 变体**：`aic-kirinx90-ops-info-nn.json` 的 matmul 类只有 `MatMulV3`/`BatchMatMulV3`/`GemmV2`/`TransposeBatchMatMul`，**无 plain MatMul/BatchMatMul**。
3. **GE 不做 MatMul→MatMulV3 改写**：GE 源码里 `MatMulV3` 只有 `InferSymbolShape` + `REGISTER_LOWERING_WITH_EXISTED(MatMulV3, LowerMatMul)`（给**已是 MatMulV3** 的节点做 lowering，非从 plain MatMul 改写）。两个是不同 op 类型，无改写 pass。
4. **Ascend910 对比**：`kirin_ops_910.om`（atc --mode=1）里 op 类型就是 plain `MatMul`(5)/`BatchMatMul`(2)，**没映射成 V3**——910 aicore 原生支持 plain MatMul；kirinx90 把 plain MatMul 砍了只留 V3，torchair 还在发 plain → 错位 → aicpu。
5. `ge.MatMulV3` 在本 torchair 的 `ge_apis`（auto_generated）**未生成**（`hasattr(ge,'MatMulV3')=False`；`BatchMatMulV3`/`ScaledMaskedSoftmaxV2` 同；`MaskedSoftmaxWithRelPosBias` 可调用），不能直接 `ge.MatMulV3(...)`，须 `torchair.ge.custom_op("MatMulV3",...)` 手建 V3 节点（inputs x1/x2/bias/offset_w，attrs transpose_x1/x2/offset_x/op_impl_mode）。

> 结论：**不是"不能映射"，是 torchair 这个版本的 converter 没发 MatMulV3、ge_apis 也没暴露 MatMulV3 可调用函数**；kirinx90 aicore 只收 V3。两边版本/注册没对齐。最干净修法：给 torchair `aten.mm`/`aten.bmm` converter 加按 SOC 发 `MatMulV3`/`BatchMatMulV3` 的分支（或 monkeypatch + `custom_op`）。

## 纯 aicore 网络（零 aicpu）—— `aicore_only_net.py`

为"零 aicpu 异构"重写的网络（按"即使没有完全覆盖"取舍）：只用已验证 aicore 的算子，**规避 MatMul/BatchMatMul/SoftmaxV2/SplitV**（无可用 aicore 变体）。

| 用（全 aicore） | 避（kirinx90 落 aicpu） |
|---|---|
| GatherV2 · Add · RmsNorm · Sigmoid · Gelu · Mul · L2Normalize · ScatterNdUpdate | MatMul/BatchMatMul（aicore 只 V3，ge_apis 不可调用）· SoftmaxV2（仅 Masked 变体）· SplitV（chunk） |

结构（无线性层/无 attention，纯 norm+activation+residual+scatter）：
`x → GatherV2(pos) → Add → RmsNorm → Sigmoid(gate)/Gelu(ffn) → Mul → Add(残差) → L2Normalize → ScatterNdUpdate(kv_cache) → out, new_cache`

### 验证（IsUnknownShapeNode 引擎分配）
- **DNN_VM_AICPU: 0** ✓✓✓（零 aicpu）
- AIcoreEngine: 8（Add/Add_1/Gelu/L2Normalize/Mul/RmsNorm/ScatterNdUpdate/Sigmoid）
- DNN_VM_GE_LOCAL: 7（Const/Reshape）· DNN_VM_RTS: 2（Identity/Output）

> 注：日志 `runtime.cc InitAiCpuCnt:aicpuNum=6` 是 runtime init 固定值（非图里 aicpu 算子数；8-op 网络 9 个 aicpu 算子时也是 6）。实际 aicpu 算子数看 `IsUnknownShapeNode` 的 `DNN_VM_AICPU*` 计数。

### 产物
- `out/aicore_only_kirinx90.omc`（350K，`soc_version[KirinX90]`、`weight_size[2560 B]`、`stream_num[1]`，权重内嵌，atc 验证有效）
- `out/golden_aicore/`：`input_x/input_kv_cache/output_out/output_new_cache.bin` + `golden_meta.json`（输入 x[2,4,64]/kv_cache[2,8,64] fp16）
- CLI 同 `kirin_ops_net.py`：`eager/export/compile/golden`

## 权重分离 .omc（externalWeight=2 + ge.externalWeightDir 修复）

mobile 路径默认把外置权值写到 `tmp_weight_<pid>_<sid>/` 临时目录，构建后被清 → .omc 引用丢失权值（见前 =2 残缺分析）。修复：设 `ge.externalWeightDir` 指定权值落盘到 output 同目录 `weight/`。

### 三种模式结论（修复后）

| GE_EXTERNAL_WEIGHT | 结果 | 详情 |
|---|---|---|
| `=0` 内嵌 | ✅ 可用 | .omc 350K，`weight_size=2560B`（权值全内嵌），自包含 |
| `=1` 分文件 | ❌ 失败 | 权值已外置到 `weight/`（3 个 `weight_<hash>` + meta.json），但 `aclgrphBuildModel failed 1343225857`（per-file 模式 mobile 路径不支持） |
| `=2` 全合一 | ✅ **权重分离成功** | .omc 349K + `weight/delegate_graph_weight_combined`（1.5K 外置权值）+ `weight/meta.json`（hash→offset 映射） |

### =2 权重分离产物（三件套）
```
out/aicore_only_kirinx90.omc              349K  (结构+kernel+1024B 内嵌权值)
out/weight/delegate_graph_weight_combined  1.5K  (3 个外置权值, offset 0/512/1024)
out/weight/meta.json                       800B  (hash→file/offset 映射)
```
atc: `weight_size[1024 B]`（部分内嵌）+ `memory_size[16789504 B]`。.omc 引用外置 `delegate_graph_weight_combined`，运行时 .omc + `weight/` 一起加载。

### 修复要点（compile_main.cc）
- **`ge.externalWeightDir` 必须走 `init_options`（`aclgrphBuildInitialize`，全局 context），不能放 `build_options`**——放 build_options 会被拒（`50331649 options illegal`，不在合法 build option 白名单）。
- 设它 = output 同目录 `weight/` → build 期 `graph_external_weight_manager.cc:104 GetThreadLocalContext().GetOption(OPTION_EXTERNAL_WEIGHT_DIR)`（`"ge.externalWeightDir"`，定义于 `ge_common_api_types.h:180`）读到 → 权值直接落 `weight/`（fallback 是 `tmp_weight_<pid>_<sid>/`，`file_constant_utils.cc:748`）。
- `ge.externalWeight`（0/1/2 值）仍走 `build_options`（合法）。
- 改动：`init_options` 段（`GE_EXTERNAL_WEIGHT≠0` 时建 `weight/` 目录 + `emplace("ge.externalWeightDir", weight/)`）；`build_options` 段只留 `ge.externalWeight`。

## AIR 导出（SAVE_AIR 选项）

可选：编译前把 GE 图 IR（AIR）dump 成可读文件，便于查看/调试。

用法（与 OM 编译并行，env 触发）：
```bash
SAVE_AIR=1 ./build_kx/cann_lm_compile <epair>/ep_json_with_ascend_ir.json <epair>/weight_info.json <output_name>
```
产物（output 同目录）：
- `ge_proto_00000_graph_0_<output>_graph.txt` — GE 图 proto 文本（图结构/算子/attr，可读）
- `ge_onnx_00000_graph_0_<output>_graph.pbtxt` — ONNX pbtxt

实现：GetGEGraph 后、BuildModel 前，`SAVE_AIR=1` → `aclgrphDumpGraph(graph, "<output>_graph", len)`（`ge_ir_build.h:170`）→ `GraphUtils::DumpGEGrph`(.txt) + `DumpGrphToOnnx`(.pbtxt)，子图也 dump。这是**编译前** GE 图 IR；编译后融合图 IR 已在 `ep_json_with_ascend_ir.json`（JSON）。独立于 OM/权重分离（`GE_EXTERNAL_WEIGHT`/`SOC_VERSION` 等），可同时跑。

## 待办（固化，不影响已出的 .omc）

| # | 项 | 位置 | 状态 |
|---|---|---|---|
| T1 | `compile_main.cc` 适配 mobile 路径：BuildModel 后检测 `onlinec` → 改名 `{output_name}.omc`（Kirin SOC）/`.om`（否则），跳过多余的 `aclgrphSaveModel`，消除红鲱鱼报错 + 自动正确命名 | `cann_lm/converter/export/csrc/tools/compile_main.cc` | ✅ 已落地（`All 3 stages passed`，直接出 `kirin_ops_kirinx90.omc`，无 `onlinec` 残留） |
| T2 | ops-nn 构建补 l2_normalize 正向 apt 入口打包（`norm/l2_norm` 加 `gelu_src_copy` 同款 CMake target），彻底替代手写 wrapper | `/mnt/workspace/gitCode/ops-nn`（外部仓） | ⬜ 待办 |
| T3 | network 的 `Muls→Mul`、`kv_cache 作输入`、scatter converter monkeypatch 固化为 kirinx90-target 正式改动（或 converter 层按 SOC 条件不发 TensorMove） | `examples/.../kirin_ops_net.py` + `cann_lm/converter/export/ge_converters/_scatter_nd_update_converter.py` | ⬜ 待办 |
| T4 | 确认 Gelu/L2Normalize 以外的 6 算子 kernel 在 kirinx90 上无 tvm-CCE 类残留问题（均已编过，留复测） | — | ⬜ 待办 |
| T5 | 把 MatMul/BatchMatMul 挪回 aicore：monkeypatch `aten.mm`/`aten.bmm` converter，用 `torchair.ge.custom_op("MatMulV3"/"BatchMatMulV3",...)` 手建 V3 节点（须先验 converter stub 环境 `get_registered_ir_def("MatMulV3")` 放行）；SoftmaxV2→`MaskedSoftmaxWithRelPosBias`（可调用，需 rel_pos_bias） | `examples/.../kirin_ops_net.py` + `compile_to_epair` | ⬜ 待办（aicore_only_net.py 已规避，此项为补回 matmul） |
| T6 | 权重分离：`compile_main.cc` 设 `ge.externalWeightDir`（走 init_options 非 build_options）→ =2 出权重分离 .omc（.omc + `weight/delegate_graph_weight_combined` + meta.json） | `cann_lm/converter/export/csrc/tools/compile_main.cc` | ✅ 已落地（=2 验证有效；=1 仍失败 per-file 不支持） |
| T7 | ~~`cat→ConcatV2`（落 aicpu）~~ **方案废弃**——kirinx90 不支持 AICPU（见背景硬约束），ConcatV2 落 aicpu 同样失败。正确方向：scatter indices 的 int32 cat（`modeling_gemma4.py:68` `[col_b,col_n,col_pos]`）是坐标拼接不进 aicore，在 converter 阶段常量化（seq_len 固定时 b/n/pos 可预算）或用 host 侧算子，不留 aicpu 算子。B4 根因（`get_kernel_type_enum` AIV_ONLY + kirinx90 不在白名单；注释 KERNEL_TYPE 引入 tiling `_pattern`/`dsl_compile_info` 缺失）说明 concat_d apt 本身在 kirinx90 不可用，需绕开 apt 路径 | `cann_lm/converter/export/ge_converters/_kirinx90_converter_patches.py::_kirinx90_aten_cat` + `cann_lm/models/gemma4/modeling_gemma4.py::_scatter_kv_cache` | ⬜ 待办 |

## 遗留 bug：mobile 路径 `aclgrphBuildModel` 不填 `ModelBufferData` 且硬编码输出文件名

**问题**：`aclgrphBuildModel` 在 mobile（kirinx90）路径下走 `MobileModelHelper::SaveToOmModel`（`mobile_model_helper.cc:681`），该方法：
1. **`model` 参数被 `(void)model` 忽略**（line 684）——不填充 `ModelBufferData.data`（留 null），导致后续 `aclgrphSaveModel` 的 `model.data == nullptr` 校验失败（"model is illegal"）。
2. **硬编码输出文件名 `"onlinec"`**——`output_file` 用 GE 内部默认值 `"online"`（因为 `output_name` 只传给 `aclgrphSaveModel`，不传给 `aclgrphBuildModel`），文件名 = `"online" + "c"` = `"onlinec"`，不反映用户指定的 output_name。

**影响**：
- `aclgrphSaveModel` 必然失败（红鲱鱼报错 `50331649 "model is illegal"`），但模型文件 `onlinec` 已落盘。
- 产物文件名不直观（`onlinec` 而非 `{output_name}.omc`），需外部 rename。
- `ModelBufferData.length` 在 mobile 路径下是垃圾值（堆损坏/未初始化），打印的 "model size" 不可信。

**根因**：`MobileModelHelper` 的设计是 mobile 场景编译完直接落盘（`SaveCompiledModelToFile` 写磁盘），绕过内存 buffer；但 `output_file` 用了 GE 内部默认值 `"online"` 而非用户的 `output_name`（`aclgrphBuildModel` 的签名不接收 output_file 参数）。

**临时规避**（已落地，T1）：`compile_main.cc` 检测 `onlinec` 存在后 rename → `{output_name}.omc`，跳过 `aclgrphSaveModel`。

**正确修法（GE 侧）**：
- 方案 A：`aclgrphBuildModel` 接收 `output_file` 参数，传给 `MobileModelHelper`，让 mobile 路径直接写 `{output_name}.omc`（避免硬编码 `onlinec`）。
- 方案 B：`MobileModelHelper::SaveToOmModel` 填充 `ModelBufferData`（写完文件后回填 data/length），让 `aclgrphSaveModel` 正常工作（但 mobile 场景避免大内存拷贝的设计意图与此矛盾）。
- 方案 C：GE 在 `aclgrphBuildModel` 返回后，若检测到 mobile 路径（model.data==null 但 onlinec 文件存在），自动 rename + 返回 success（而不是让 `aclgrphSaveModel` 报错）。

**涉及 GE 源码**：
- `base/common/helper/mobile_model_helper.cc:681-718`（`SaveToOmModel`，`(void)model` + 硬编码 `onlinec`）
- `base/common/helper/mobile_model_helper.cc:706-711`（`output_weight_dir` 从 `output_file` 目录部分推导，`output_file="online"` 无目录 → 空）
- `compiler/api/aclgrph/ge_ir_build.cc:1640-1655`（`aclgrphSaveModelInner`，`model.data==nullptr` 校验）

## 涉及文件/产物

| 路径 | 说明 |
|---|---|
| `examples/test_converter/export_kirin_ops_net/kirin_ops_net.py` | 小网络 + export/compile/eager CLI；含 B1/B2 规避 |
| `examples/test_converter/export_kirin_ops_net/aicore_only_net.py` | 纯 aicore 网络（零 aicpu，规避 MatMul/BatchMatMul/SoftmaxV2/SplitV）+ golden CLI |
| `examples/test_converter/export_kirin_ops_net/check_op_coverage.py` | 静态算子覆盖率核对（epair ascend_ir ∩ SOC ops-info json） |
| `cann_lm/converter/export/csrc/tools/compile_main.cc` | mobile 路径收尾（T1：onlinec→.omc 改名）+ 权重分离（T6：ge.externalWeightDir 走 init_options） |
| `cann_lm/converter/export/ge_converters/_kirinx90_converter_patches.py` | KIRINX90_CONVERTER_PATCHES 机制（B.2）：6 类算子 converter 覆盖；B4 的 cat→ConcatD 失败待改 ConcatV2（T7） |
| `cann_lm/converter/export/ge_converters/ge_converter.py:80` | `KIRINX90_CONVERTER_PATCHES=1` env 检测 + 加载 patches |
| `asc_op_compile_base/asc_op_compiler/kernel_info_infer.py:195` | B4 根因：`get_kernel_type_enum` else 分支抛 `AIV_ONLY not support`（kirinx90 soc 不在 `is_v200/v220/c310` 白名单） |
| `asc_op_compile_base/asc_op_compiler/ascendc_common_utility.py:516-609` | soc 白名单：`is_v220`(Ascend910B/910_93) / `is_v200`(Ascend310P/610) / `is_c310`(Ascend950/350)，均不含 KirinX90 |
| `/mnt/workspace/cann-kirinx90/cann-9.2.0/opp/.../tbe/impl/ops_math/dynamic/concat_d_apt.py:216` | 硬编码 `arch31/concat_d.cpp`（AIV_ONLY），触发 B4 |
| `sitecustomize.py` + `PYTHONPATH` | A7 规避：屏蔽 torch_npu（避免 GE 嵌入 python 双重加载 cann-9.0.0 GE） |
| `/mnt/workspace/cann-kirinx90/` | 隔离 toolkit（toolkit + A3-ops + kirinx90 ops） |
| `/mnt/workspace/cann-kirinx90/cann-9.2.0/opp/.../tbe/impl/ops_nn/dynamic/l2_normalize.py` | 手写 AscendC-apt wrapper（B3 规避） |
| `/mnt/workspace/cann-kirinx90/cann-9.2.0/opp/.../tbe/kernel/scene.info` | 手补 `os=linux`（A3 规避） |
| `examples/.../out/kirin_ops_kirinx90.omc` | 8-op 网络产物（677K，soc=KirinX90，权重内嵌，有 aicpu 算子） |
| `examples/.../out/aicore_only_kirinx90.omc` | 纯 aicore 网络产物（350K，零 aicpu） |
| `examples/.../out/aicore_only_kirinx90.omc` + `weight/` | 权重分离产物（=2：.omc 349K + `delegate_graph_weight_combined` 1.5K + meta.json） |
