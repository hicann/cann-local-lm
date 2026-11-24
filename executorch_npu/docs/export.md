# 导出模型为PTE/PTD

“模型导出”包含两个不同阶段：先由 `torch.export` 捕获模型，再由ExecuTorch-NPU
完成ExecuTorch lowering和CANN AOT编译。两个阶段可以在同一环境连续执行，也可以
通过PT2文件分离。

源码构建见 [源码构建](build.md)，C++ 加载和执行见 [C++ 运行](runtime.md)。

## 两阶段导出流程

```text
阶段 A：模型捕获
模型实现 + Checkpoint + 示例输入
→ torch.export.export
→ ExportedProgram
→ 可选：torch.export.save(..., "model.pt2")

阶段 B：CANN AOT 导出
ExportedProgram 或 model.pt2
→ ExecuTorch Edge lowering
→ AscendPartitioner
→ torch-delegate
→ TorchAir 提供者
→ CANN OfflineCompiler
→ model.pte + ascend.weights.package.ge_aot.ptd
```

| 阶段 | 主要输入 | 主要输出 | 依赖归属 |
|---|---|---|---|
| A. `torch.export` | PyTorch模型、权重、示例输入 | `ExportedProgram` / PT2 | 模型实现负责 |
| B. ExecuTorch-NPU | `ExportedProgram` / PT2、目标SoC | PTE/PTD | ExecuTorch-NPU负责 |

## 依赖边界

### 两阶段共享依赖

两个阶段必须使用兼容的Python和Torch。阶段A本身不依赖ExecuTorch。当前支持：

| 依赖 | 要求 |
|---|---|
| PyTorch | 官方CPU wheel 2.11.0–2.13.0 |

ExecuTorch-NPU不自动安装Torch。所有支持版本都必须先从PyTorch官方CPU index
安装带 `+cpu` 标识的wheel。版本组合见 [支持版本](supported_versions.md)。

默认组合：

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0+cpu" \
  "torchvision==0.26.0+cpu"
```

如果阶段A和阶段B使用不同环境，建议保持完全相同的Torch版本。PT2不是跨任意
Torch版本的稳定交换格式。

### 阶段A：`torch.export` 最小依赖

通用文档只能规定以下最小集合：

| 依赖 | 是否固定 | 说明 |
|---|---:|---|
| CPU Torch | 是 | `torch.export` 基础能力 |
| 模型实现包 | 否 | Transformers、Torchvision、自定义模型代码等 |
| Checkpoint/Tokenizer依赖 | 否 | 由模型加载方式决定 |
| torch_npu | 否 | 仅当模型构造、融合算子或NPU tensor捕获需要时安装 |
| 自定义算子注册 | 否 | PT2加载和后续lowering可能仍需要 |

### 阶段B：CANN AOT固定依赖

| 依赖 | 用途 |
|---|---|
| ExecuTorch-NPU wheel | Partitioner、Backend和产物协议 |
| ExecuTorch 1.5.0 | Edge lowering和PTE/PTD序列化 |
| CANN Toolkit GE/TBE | 图优化和离线编译 |
| torch-delegate | EPAIR、GE Graph和native bridge |
| TorchAir提供者 | ATen/FX到GE图转换 |

安装ExecuTorch-NPU wheel会安装ExecuTorch 1.5.0及其普通Python依赖，但不会
安装或替换Torch：

```bash
python -m pip install /path/to/executorch_npu-*.whl
```

### 选择TorchAir提供者

TorchAir只需要一个来源：

| 条件 | 选择 |
|---|---|
| 模型不依赖torch_npu | 安装与当前Torch匹配的独立TorchAir |
| 同一环境已经因模型实现安装torch_npu，且其内置TorchAir与Torch/CANN兼容 | 直接复用torch_npu内置TorchAir |
| 阶段A使用torch_npu，但阶段B是独立环境 | 阶段B重新选择独立TorchAir，或安装同一兼容torch_npu |

不要在同一环境同时安装独立TorchAir和torch_npu内置TorchAir。仅仅因为阶段A
曾经使用torch_npu，不代表另一个阶段B环境自动拥有TorchAir。

独立TorchAir的源码安装入口：

```bash
git clone https://gitcode.com/ascend/torchair.git
cd torchair
bash build.sh -i
```

TorchAir必须针对当前Torch版本构建。torch-delegate无需单独下载：其源码位于
本仓库同级的 `../torch_delegate` 目录。两种安装方式任选其一：

```bash
# 方式一：使用仓内源码构建 whl 并安装
cd ../torch_delegate
bash build.sh
python -m pip install output/package/torch_delegate-*.whl

# 方式二：直接安装已发布的 torch_delegate whl 包
python -m pip install /path/to/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

构建细节见仓内 `../torch_delegate/docs/build.md`。

### 无NPU设备导出（Lite导出）

阶段A和阶段B都**不需要NPU设备在位**。目标芯片只通过 `ACL_SOC_NAME` 传入，
设备仅在运行阶段必需。这也是设计约束第11条的要求，见
[设计文档](design/executorch_npu_design.md)。

```bash
# 阶段 A：DEVICE=cpu
export DEVICE=cpu
python -m examples.qwen3_5.export.pt2

# 阶段 B：host-only
export ACL_SOC_NAME=Ascend910_9362
export EXECUTORCH_NPU_HOST_ONLY=1
python -m examples.qwen3_5.export.pte
```

TorchAir的GE引擎初始化把「初始化GE」和「绑定设备」耦合在一次C++ 调用里，
而阶段B只需要前者。设备依赖分三层，`EXECUTORCH_NPU_HOST_ONLY=1` 分别处理：

| 层 | 位置 | 无设备时的表现 | 处理方式 |
|---|---|---|---|
| Python | `torchair.core._backend._get_device_id()` | 因torch_npu已导入而调用 `npu.current_device()`，抛 `aclInit 107001` | 替换为读取 `ASCEND_DEVICE_ID` |
| C++ | `_torchair.InitializeGraphEngine()` | 内部无条件 `acl.rt.set_device()`，返回107001 | 打桩C++ 叶子函数 |
| 算子转换器 | `torchair._utils.check_platform.is_arch35()` | 经 `torch_npu.npu.get_device_name()` 读物理设备名；无设备时设备下标解析为 `None`，`None < 0` 抛 `TypeError`，TorchAir只捕获 `ImportError` | 把 `get_device_name()` 替换为返回 `acl_soc_name`，架构判断仍由TorchAir自己的设备名规则得出 |

第三层在当前锁定的TorchAir revision下涉及25个converter模块（含
`native_batch_norm_legit_no_training`、`max_pool2d_with_indices`、
`native_group_norm`、`std`、`npu_rotary_mul` 等），判断结果决定GE算子版本、
属性、layout和输出语义，因此不能固定返回 `False`，只能替换设备名来源。
未安装torch_npu时该函数走 `ImportError` 分支返回 `False`，与当前登记的SoC一致；
新增arch35目标（设备名含 `Ascend910_95` 或 `Ascend950`）时必须安装torch_npu，
让 `acl_soc_name` 参与判断。

C++ 层无法用参数绕过：`deviceId` 传 `0` 报 `ACL set device id failed`，传 `-1`
触发 `assert device_index_ >= 0`，`ge_run_with_torch_npu` 置 `0`、改用独立
TorchAir、乃至完全屏蔽torch_npu均无效。

打桩后 `EpairModelLoader` 仍要求GE已初始化。EPAIR加载与离线编译
共用 `OfflineCompiler` 管理的一次初始化：

```python
ge.offline_compile.build_initialize({
    "ge.socVersion": soc,
    "ge.graphRunMode": "0",
    "ge.externalWeightDir": session_workspace,
})
```

初始化时由会话持有专用 `ge.externalWeightDir` 临时目录；当前CANN不支持
在 `build_model` 的图级选项中设置该目录。保存每个OM后，编译器将这次构建的
权重转移到对应partition的 `weight/`，失败构建的残留文件也会清理。
GE先完成finalization，再删除会话目录；不同method的权重不会混入后续产物。
会话目录中的 `meta.json` 是CANN的宿主端索引，不是FileConstant张量，
转移权重时必须排除，不能写入PTD或登记为外置权重绑定。
编译器不改变调用进程的工作目录和 `ASCEND_WORK_PATH`。

同进程编译入口、兼容补丁、临时环境变量和GE操作共用一把可重入锁。
重入用于编译入口调用加载/初始化等内层操作；不同线程的编译串行执行。
临时环境变量即使遇到异常也会恢复。GE会话固定首次使用的SoC和
`EXECUTORCH_NPU_HOST_ONLY` 模式，后续不一致的请求会明确报错；切换配置需使用新进程。
GE关闭后不接受新请求，正常关闭与进程退出只执行一次native finalization。
TorchAir兼容适配集中在 `compiler/compatibility.py`。

设备模式下，TorchAir的初始化由torch-delegate转换器使用完整编译配置触发。
不要在转换器运行前无参数调用 `initialize_graph_engine()`：这会缓存空配置，
使后续带 `ge.exec.enableEngineParallel` 等选项的初始化报冲突。
离线GE会话初始化与这一步职责不同，两种模式都需要真实导出和运行回归。


内存采样默认关闭。设置 `ASCEND_EDGE_MEMORY_DIAGNOSTICS=1` 可启用；
仅在同时设置 `ASCEND_EDGE_MEMORY_COLLECT_GC=1` 时主动触发GC，避免日志采样
默认改变编译耗时。

使用Lite导出的前置条件：

| 要求 | 说明 |
|---|---|
| `EXECUTORCH_NPU_STAGING` 指向大容量磁盘 | GE编译临时权重峰值可达37 GB（Gemma4 E2B）。默认 `/tmp` 常在根分区，写满后报 `Write data failed, errno: -1` |
| CANN Toolkit同源 | 必须 `source` 同一份Toolkit的 `set_env.sh`。只覆盖 `ASCEND_OPP_PATH` 会造成OPP算子实现与TBE框架跨版本混用 |
| `PYTHONPATH` 追加而非覆盖 | 覆盖会丢掉 `set_env.sh` 注入的CANN路径，导致 `No module named 'ge'` |

验证当前进程确实看不到设备：

```bash
ASCEND_RT_VISIBLE_DEVICES="" python -c \
  'import torch, torch_npu; print("device_count =", torch.npu.device_count())'
```

Lite导出的边界：

- 不改变运行阶段要求：加载PTE/PTD并执行仍需匹配的NPU、驱动和AscendCL。
- 无设备时GE产出的OM可能带host os/arch后缀（如 `model_linux_aarch64.om`），
  产物因此绑定该host平台；跨平台部署需在目标平台重新编译。打包侧按
  `model*.om` 前缀定位，不硬编码 `.om`。
- 打桩对象是TorchAir内部符号，TorchAir升级后必须重新验证。
- Lite导出只改变GE引擎的初始化方式，不替换任何算子，因此与设备侧导出产出
  数值等价的图。

## 环境组织方式

### 单环境模式

模型捕获和CANN AOT在同一Python环境连续执行。环境依赖是两个阶段的并集：

```text
模型依赖
+ CPU Torch
+ 可选 torch_npu
+ ExecuTorch-NPU / ExecuTorch
+ CANN GE/TBE
+ torch-delegate
+ 一个 TorchAir 提供者
```

这是小模型和简单导出脚本最直接的方式。如果模型已经要求torch_npu，并且内置
TorchAir兼容，就不再编译独立TorchAir。

### PT2分环境模式

阶段A保存PT2，阶段B再加载：

```text
模型环境                        CANN AOT 环境
模型实现 + 可选 torch_npu       ExecuTorch-NPU + ExecuTorch
torch.export.export             CANN GE/TBE + torch-delegate
torch.export.save               一个 TorchAir 提供者
          │
          └──────── PT2 ───────→ torch.export.load → PTE/PTD
```

PT2可以移除阶段B对Checkpoint、Tokenizer和大部分模型构造代码的依赖，但不能保证
移除以下依赖：

- 图中保留的自定义算子注册。
- NPU device或torch_npu类型注册。
- 加载PT2所需的模型补丁模块。
- 与保存端兼容的Torch版本。

仓库中的Qwen和Gemma示例已经采用PT2/PTE分阶段脚本，可作为复杂模型参考。

## 初始化CANN AOT环境

```bash
conda activate executorch_npu
source /path/to/cann/set_env.sh
export PYTHONNOUSERSITE=1
```

使用源码形式的torch-delegate（仓内 `../torch_delegate`）时，追加路径必须保留
CANN注入的原值：

```bash
export TORCH_DELEGATE_ROOT="$(cd ../torch_delegate && pwd)"
export PYTHONPATH="$TORCH_DELEGATE_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TORCH_DELEGATE_ROOT/build/release/csrc:${LD_LIBRARY_PATH:-}"
```

不要覆盖 `PYTHONPATH`，否则会丢失 `ge` 或 `tbe`。

## 阶段A：捕获并保存PT2

以下代码只表示最小通用步骤，`model`、`example_inputs`、动态shape和模型补丁由模型
实现负责：

```python
from pathlib import Path

import torch

exported = torch.export.export(model.eval(), example_inputs)
torch.export.save(exported, Path("model.pt2"))
```

多Method或Prefill/Decode模型通常分别保存多个PT2。真实示例：

- `examples/qwen3_5/export/pt2.py`
- `examples/gemma4/export/pt2.py`

## 阶段B：从PT2生成PTE/PTD

```python
from pathlib import Path

import torch
from executorch.exir import to_edge_transform_and_lower

from executorch_npu import (
    AscendCompileOptions,
    AscendPartitioner,
    aten_edge_compile_config,
)

exported = torch.export.load("model.pt2")
options = AscendCompileOptions(
    method_name="forward",
    delegate_index=0,
    acl_soc_name="Ascend910B3",
)
program = to_edge_transform_and_lower(
    exported,
    partitioner=[AscendPartitioner(options.to_compile_specs())],
    compile_config=aten_edge_compile_config(),
).to_executorch()

output_dir = Path("artifacts")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "model.pte").write_bytes(program.buffer)
program.write_tensor_data_to_file(output_dir)
```

如果不需要分环境，可以直接把 `torch.export.load("model.pt2")` 替换为阶段A产生的
`ExportedProgram`，不必落盘PT2。

`acl_soc_name` 必须与运行设备 `aclrtGetSocName()` 返回的名称精确一致。

## 导出前检查

两种TorchAir profile都先确认CPU Torch和CANN Python环境：

```bash
python - <<'PY'
import importlib.util
import torch

assert "+cpu" in torch.__version__, torch.__version__
assert importlib.util.find_spec("ge.offline_compile") is not None
assert importlib.util.find_spec("tbe") is not None
print("torch:", torch.__version__)
print("torch_npu:", importlib.util.find_spec("torch_npu"))
PY
```

独立TorchAir profile：

```bash
python -c 'import torchair; print(torchair.__file__)'
```

torch_npu内置TorchAir profile必须先导入torch_npu：

```bash
python -c 'import torch_npu, torchair; print(torch_npu.__file__); print(torchair.__file__)'
```

确认实际 `torchair` 路径来自所选择的提供者。独立TorchAir profile中不应加载
torch_npu。

## 检查产物

```text
artifacts/
├── model.pte
└── ascend.weights.package.ge_aot.ptd
```

- PTE保存ExecuTorch Program、Delegate processed bytes、Manifest和OM named data。
- PTD保存外置权重named data。
- PTE和PTD必须由同一次阶段B导出产生并成对交付。

CANN AOT通常不需要实际NPU，但需要完整GE/TBE环境。阶段A如果在NPU上构造模型
或执行融合算子，则需要相应设备和torch_npu环境。

## 下一步

使用C++ 加载产物见 [C++ 运行](runtime.md)。导出错误见
[故障排查：模型导出](troubleshooting.md#模型导出)。
