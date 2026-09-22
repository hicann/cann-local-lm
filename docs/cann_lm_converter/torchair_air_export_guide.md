# torchair air 模型导出指南（无 NPU 环境）

> 目的：在**无 NPU 设备**的机器上，仅用 `torchair` + `cann-toolkit`（+ 一个新 `.venv`）跑通 `torchair.dynamo_export` 导出 `.air` 模型的完整流程。
> 适用：纯 `aten`/`air::` 算子模型（如 `mm/add/relu`）。含 `npu::` 量化算子（`npu_quant_matmul` 等）的模型需额外依赖 `cann_lm`（见末节）。
> 验证环境：python 3.12.9 + torch 2.9.0+cpu + torchair 0.1 + cann-toolkit 9.2.0。

## 1. 原理

`torchair.dynamo_export` 把 PyTorch 模型经 Dynamo 捕获为 FX 图，再 FX→GE 转换成 GE（Graph Engine）IR，序列化为 `.air`（GE `ModelDef` protobuf 二进制）。FX→GE 转换需要 GE 引擎（`libge_runner`）。

无 NPU 时，真 CANN lib64 的 `aclrtSetDevice → 107001`。解法：把 **torchair 编译生成的 stubs**（`libge_runner`/`libascendcl` 等的桩 `.so`）放 `LD_LIBRARY_PATH` 最前，桩接住 GE 引擎调用（日志 `[STUB] Session::Session created`），不真设设备。

- stubs **必须用 torchair 编译生成的**（ABI 匹配），不能用真 CANN lib64 替代（真 lib64 会调真 `aclrtSetDevice` → 107001）。
- 只有 "torchair + 其自带 stubs" 这套组合在 CPU 上可用。

## 2. 前置条件

| 组件 | 说明 |
|---|---|
| cann-toolkit | 提供 lib64 + 头文件（无 NPU 设备也行，只用作 GE 库来源）|
| python 3.12 | `/opt/buildtools/python-3.12.9/bin/python3`（含 `libpython3.12.so.1.0`）|
| torchair 源码/whl | 已编译产物：`<torchair>/build/dist/dist/torchair-0.1-py3-none-any.whl` + `<torchair>/build/stubs/*.so` |

torchair 是**纯 Python whl**（无 C 扩展），与 torch 版本松耦合（API 兼容即可），但编译时用的 torch 版本建议对齐（这里 torch 2.9.0+cpu）。

### 2.1 torchair 源码编译（从源码生成 whl + stubs）

若没有预编译的 torchair whl + stubs，从源码编译（CANN-LM 的 `deps/torchair` 即此源码，已编译产物在 `build/`）：

**1) 获取源码**：
```bash
git clone https://gitcode.com/ascend/torchair.git
cd torchair
# 或用 CANN-LM 已有源码: cd /mnt/workspace/CANN-LM_5883/deps/torchair
```

**2) configure**（检测 python + torch + 可选 CANN SDK，写入 `tools/TORCH_VERSION` / `tools/PYTHON_BIN_PATH` / `tools/ASCEND_SDK_PATH`）：
```bash
# 指定 venv python + 真 CANN SDK (生成完整 stubs, 链真 CANN 头文件):
TARGET_PYTHON_PATH=<venv>/bin/python ASCEND_SDK_PATH=/home/developer/Ascend/cann/latest bash ./configure
# 或无 SDK (NO_ASCEND_SDK=1, stubs 为纯桩不链 CANN, 但足够接住 GE 引擎的 aclrtSetDevice 调用):
TARGET_PYTHON_PATH=<venv>/bin/python NO_ASCEND_SDK=1 bash ./configure
```

**3) 编译**（生成 whl + stubs）：
```bash
bash build.sh -c          # ci build: cmake + make torchair -j8
# 或手动:
mkdir build && cd build && cmake .. && make torchair -j8
```
产物：
- `build/dist/dist/torchair-0.1-py3-none-any.whl`（纯 Python 安装包）
- `build/stubs/*.so`（stub 库；CMakeLists `set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/stubs)`）

**4) 安装到 venv**：
```bash
bash build.sh -i          # ci build + install (pip 装到 configure 指定的 python)
# 或手动:
<venv>/bin/pip install build/dist/dist/torchair-0.1-py3-none-any.whl
```

**5) 关键产物路径**（后续 air 导出用）：
- whl：`build/dist/dist/torchair-0.1-py3-none-any.whl`（`build.sh -c` 额外移到 `output/`）
- stubs：`build/stubs/*.so`（放 `LD_LIBRARY_PATH` 最前绕过 `aclrtSetDevice`）

**`build.sh` 选项速查**：

| 选项 | 作用 |
|---|---|
| `-c` | ci build（cmake + make，生成 whl + stubs）|
| `-i` | ci build + install（pip 装到当前 python）|
| `-u` | 跑 C++ 单元测试（需 `ASCEND_HOME_PATH` 或 `ASCEND_CUSTOM_PATH`）|
| `-s` | 跑系统测试（`tests/st/`）|
| `-j[n]` | 编译线程数（默认 8）|
| `-g prefix` | 指定 GCC 工具链前缀（交叉编译）|

**无 NPU 跑 C++ 测试**：`source tools/env.sh` 设 `LD_LIBRARY_PATH` 含 stub .so + SDK 库（C++ 测试无需真 NPU 即可链接）。

## 3. 环境搭建

### 3.1 创建新 .venv

```bash
mkdir -p /mnt/workspace/torchair_air_test
/opt/buildtools/python-3.12.9/bin/python3 -m venv /mnt/workspace/torchair_air_test/.venv
/mnt/workspace/torchair_air_test/.venv/bin/python -m pip install --upgrade pip
```

### 3.2 装包

```bash
VENV=/mnt/workspace/torchair_air_test/.venv
TORCHAIR=/mnt/workspace/CANN-LM_5883/deps/torchair   # torchair 源码根 (含 build/stubs + build/dist/dist/*.whl)

# torch (CPU, 对齐 torchair 编译版本)
"$VENV/bin/pip" install torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu

# torchair 依赖 (METADATA 只列 protobuf, 实际 import 还需 numpy/psutil/scipy/packaging)
"$VENV/bin/pip" install "protobuf>=3.13,<4" packaging numpy psutil scipy

# torchair (纯 Python whl)
"$VENV/bin/pip" install "$TORCHAIR/build/dist/dist/torchair-0.1-py3-none-any.whl"
```

### 3.3 验证 import

```bash
"$VENV/bin/python" -c "import torch, torchair; print(torch.__version__); print(torchair.dynamo_export)"
# torch 2.9.0+cpu
# <function dynamo_export at ...>
```

## 4. 环境变量（顺序关键）

每次跑 `dynamo_export` 前，在一个干净 shell 里按顺序设置：

```bash
# ① 清多 CANN 污染（残留变量指向旧版本 → GE/atc 链错库/OS 校验失败）
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH TOOLKIT_HOME CMAKE_PREFIX_PATH PYTHONPATH SOC_VERSION GE_FUSION_SWITCH_FILE GE_EXTERNAL_WEIGHT ASCEND_VISIBLE_DEVICES DEVICE
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/buildtools/python-3.12.9/bin:/usr/local/cmake/bin

# ② source cann-toolkit（lib64 + 头文件）
source /home/developer/Ascend/cann/set_env.sh

# ③ libpython 前置（source CANN 后 libpython3.12.so.1.0 被解析为系统旧版 → _ctypes undefined symbol _PyErr_SetLocaleString → TBE/tvm 不可用 → aclgrphBuildModel failed）
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:$LD_LIBRARY_PATH

# ④ torchair stubs 前置（绕过 aclrtSetDevice 107001，GE 引擎走 stub）
export LD_LIBRARY_PATH=/mnt/workspace/CANN-LM_5883/deps/torchair/build/stubs:$LD_LIBRARY_PATH

# ⑤ 阻止 torch autoload npu backend
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`LD_LIBRARY_PATH` 最终顺序（最前优先）：`libpython` → `stubs` → `cann lib64` → ...。stubs 必须在 cann lib64 前，否则真 lib64 的 `libge_runner` 会调真 `aclrtSetDevice`。

## 5. 导出 air（纯 aten 模型）

### 5.1 模型（纯 aten 算子，无 npu::）

```python
# aten_net.py
import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torchair
from torch import nn
from torchair.configs.compiler_config import CompilerConfig


class AtNet(nn.Module):
    """纯 aten 算子网络: mm + add + relu."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 32))
        self.b = nn.Parameter(torch.zeros(32))

    def forward(self, x):
        return torch.relu(x @ self.w + self.b)


def main():
    torch.manual_seed(0)
    model = AtNet().eval()
    sample = (torch.randn(2, 64),)
    os.makedirs("out", exist_ok=True)
    torchair.dynamo_export(*sample, model=model,
                           export_path="out", export_name="aten_net",
                           config=CompilerConfig())
    print("AIR saved: out/aten_net.air")


if __name__ == "__main__":
    main()
```

### 5.2 跑导出

```bash
cd /mnt/workspace/torchair_air_test
.venv/bin/python aten_net.py
```

预期输出（`[STUB]` 日志 = stub 生效）：
```
[STUB] GEInitialize
[STUB] Session::Session created
AIR saved: out/aten_net.air
[STUB] Session::Session destroyed
[STUB] GEFinalize
```

aten 算子经 torchair FX→GE converter 转成对应 `ge.*` op：

| aten | GE op |
|---|---|
| `aten.mm` | `ge.MatMul` |
| `aten.add` | `ge.Add` |
| `aten.relu` | `ge.Relu` |
| `Parameter` | `ge.Const` |
| 输入 | `ge.Data` |
| 输出 | `ge.NetOutput` |

这些 `ge.*` op 在 stub GE 已注册，`is_cann_compat` 放行（无需 monkey-patch）。

## 6. 验证 air（protobuf 解析）

`.air` 是 GE `ModelDef` protobuf 二进制（不是文本）。用 torchair 的 `ge_ir_pb2` 解析：

```python
from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef

m = ModelDef()
m.ParseFromString(open("out/aten_net.air", "rb").read())
g = m.graph[0]   # air 是 ModelDef, m.graph[0] 是 GraphDef
print(f"graph '{g.name}': {len(g.op)} ops")
for op in g.op:
    print(f"  {op.type:18s} {op.name}")
```

预期：
```
graph 'graph_1': 7 ops
  Const              arg0_1
  Data               arg1_1
  Const              arg2_1
  MatMul             MatMul
  Add                Add
  Relu               Relu
  NetOutput          NetOutput
```

### TensorDescriptor.dtype 枚举

`op.input_desc[i].dtype` / `op.output_desc[i].dtype` 用 **ge_ir_pb2 的 DataType enum**（和 CANN `C_DT_*` 编号不同）：

| 值 | ge_ir_pb2 | CANN C_DT |
|---|---|---|
| 1 | `DT_FLOAT` | `C_DT_FLOAT16` |
| 2 | `DT_FLOAT16` | `C_DT_INT8` |
| 3 | `DT_INT8` | `C_DT_INT32` |
| 8 | `DT_INT64` | `C_DT_UINT32` |
| 10 | `DT_UINT64` | `C_DT_UINT64` |
| 28 | `DT_INT4` | `C_DT_UNDEFINED` |
| 29 | — | `C_DT_INT4` |

查完整枚举用 proto 反射：
```python
from torchair._ge_concrete_graph.ge_ir_pb2 import TensorDescriptor
fd = TensorDescriptor.DESCRIPTOR.fields_by_name["dtype"]
dt_map = {v.number: v.name for v in fd.enum_type.values}
```

## 7. 坑点速查

| 坑 | 现象 | 修复 |
|---|---|---|
| 无 NPU | `ACL set device id failed, return 107001` | stubs 前置 `LD_LIBRARY_PATH`（torchair 编译产物，ABI 匹配）|
| libpython | `_ctypes undefined symbol _PyErr_SetLocaleString` → TBE/tvm 不可用 → `aclgrphBuildModel failed` | libpython 前置 `LD_LIBRARY_PATH`（`/opt/buildtools/python-3.12.9/lib`）|
| 真 CANN lib64 | 替代 stubs 会 `aclrtSetDevice → 107001` | stubs 必须在 LD 最前，不混真 lib64 |
| 多 CANN 并存 | 残留变量指向旧版本 → 链错库/OS 校验失败 | 先 `unset` 所有 CANN 变量再 `source` 目标 set_env |
| torch autoload npu | `torch.npu` backend 自动加载报错 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |

## 8. 限制与扩展

### 当前流程的限制

仅 torchair（无 `cann_lm` + 无 `torch_npu`）只能转：
- **aten 算子**（torch 自带 schema+fake-meta，`torch.export` 能 trace）+ torchair 自带的不 `import torch_npu` 的 converter
- **air:: 算子**（torchair `ops/` 自带 schema+fake+converter，命中 `torch.ops.air.*`，自包含）

**不能用 npu:: 算子**（`npu_quant_matmul`/`npu_dynamic_quant`/`npu_rms_norm` 等）：
- 无 `torch_npu` → npu:: 算子没 schema+fake-meta → `torch.export` trace 不了
- torchair 的 npu:: converter 多数 `import torch_npu` → ModuleNotFoundError
- 未注册 op（`QuantBatchMatmulV4`/`DynamicQuant`）会被 stub GE 的 `is_cann_compat` 拒绝（`OperatorFactory find optype X failed`）

### 扩展：转 npu:: 算子模型（需 cann_lm）

若要转含 `npu::` 量化算子的模型，需加 `cann_lm`（提供 fake-meta + converter 覆盖 + is_cann_compat patch）：

```python
import cann_lm.models.fused_ops  # noqa: F401  (注册 npu:: 算子 schema+fake, 无 torch_npu 时 _library.py 自提供)
from cann_lm.converter.export.ge_converters.ge_converter import register_ge_converters
register_ge_converters()
# ① 注册 cann_lm 自带 converter 覆盖 torchair(import torch_npu 版, 如 quantmatmul)
# ② monkey-patch is_cann_compat = lambda *a, **k: ""  绕过 stub GE 对未注册 op 的拒绝
torchair.dynamo_export(*sample, model=model, ...)
```

cann_lm 还需源码 `PYTHONPATH` 前置（wheel 可能没打包新模块如 `quant_swap`）：
```bash
export PYTHONPATH=/mnt/workspace/CANN-LM_5883:$PYTHONPATH
```

## 9. 涉及产物

| 路径 | 说明 |
|---|---|
| `/mnt/workspace/torchair_air_test/.venv` | 新 venv（torch 2.9.0+cpu + torchair 0.1）|
| `/mnt/workspace/torchair_air_test/aten_net.py` | 纯 aten 算子示例网络 |
| `/mnt/workspace/torchair_air_test/out/aten_net.air` | 导出的 air 模型（MatMul+Add+Relu）|
| `/mnt/workspace/CANN-LM_5883/deps/torchair/build/stubs` | torchair 编译生成的 stub .so |
| `/mnt/workspace/CANN-LM_5883/deps/torchair/build/dist/dist/torchair-0.1-py3-none-any.whl` | torchair 安装包 |
