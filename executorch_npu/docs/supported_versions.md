# 支持版本

## 当前支持基线

ExecuTorch 1.5.0是ExecuTorch-NPU首个支持的正式发布版本。更早版本不属于
支持范围。

| 组件 | 支持版本或范围 |
|---|---|
| 平台 | Linux aarch64 |
| Python | 3.10–3.14；当前验证使用3.11 |
| ExecuTorch | `1.5.0` |
| PyTorch | CPU wheel `>=2.9.0,<2.14.0`；2.11.0为默认开发profile |
| torchao | `>=0.18.0,<0.19`，由ExecuTorch安装 |
| Torchvision | 与Torch对应，见下表；仅模型示例需要 |
| Transformers | `5.17.0`；仅Qwen3.5 / Gemma4示例需要，与 `requirements-examples.txt` 保持一致 |
| CMake | Backend构建 `>=3.24` |
| C++ | C++17，当前验证使用GCC 9.4 |
| CANN | 9.1.0、9.2.0验证环境 |
| Manifest framing | 1 |
| Runtime protocol | 1 |
| Artifact ABI | 1 |

ExecuTorch 1.5和ExecuTorch-NPU均不自动安装Torch。所有支持组合都必须先从
PyTorch官方CPU index显式安装CPU wheel。默认profile为：

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0+cpu" "torchvision==0.26.0+cpu"
python -m pip install "executorch==1.5.0"
```

已验证的模型依赖组合：

| Torch | Torchvision |
|---|---|
| 2.9.0+cpu | 0.24.0+cpu |
| 2.10.0+cpu | 0.25.0+cpu |
| 2.11.0+cpu | 0.26.0+cpu |
| 2.12.0+cpu | 0.27.0+cpu |
| 2.13.0+cpu | 0.28.0+cpu |

禁止使用PyPI自动选出的非CPU wheel。Torch 2.11.0–2.13.0的PyPI Linux aarch64
wheel会引入CUDA运行库，不属于Ascend导出profile；2.9.0和2.10.0也统一使用
官方CPU index，避免安装来源随版本变化。

必须使用ExecuTorch完整wheel。minimal wheel不包含ExecuTorch-NPU构建和C++
应用集成所需的头文件与Runtime库。

## 验证状态

当前已完成：

- 使用ExecuTorch 1.5.0 Linux aarch64 wheel编译ExecuTorch-NPU Backend。
- C++ 单元测试通过。
- 74项Python核心测试通过，包括EXIR lowering、named data、PTE/PTD序列化、
  multi-method preprocess和state contract。
- 使用wheel内 `libexecutorch.so` 编译并链接普通Runner和Stateful Runner。
- Torch 2.9.0+cpu、2.10.0+cpu、2.11.0+cpu、2.12.0+cpu、2.13.0+cpu
  均完成74项Python核心测试。
- 上述每个Torch版本均使用对应环境重新构建独立TorchAir，并通过MobileNetV2
  真实GE/OM编译、PTE/PTD导出和Ascend910B3 C++ Runner验证：
  `ASCEND_EXECUTORCH_OK values=1000`。

当前尚需补充：

- 其他已登记SoC的ExecuTorch 1.5.0端到端回归。
- Stateful Prefill/Decode的ExecuTorch 1.5.0真实NPU验证。

在这些验证完成前，文档不得把“Python核心测试通过”表述为“所有模型和设备已完成
端到端验证”。

## 旧版本边界

| ExecuTorch | 状态 | 原因 |
|---|---|---|
| 0.x | 不支持 | PyPI没有当前Linux aarch64 wheel，接口和产物协议早于当前实现 |
| 1.0.0–1.2.0 | 不支持 | wheel缺少 `executorch/runtime/backend/interface.h`，不能执行当前默认构建 |
| 1.3.1–1.4.1 | 不支持 | 可以编译Backend，但wheel没有当前Runner所需的独立C++ Runtime SDK |
| 1.5.0 | 支持基线 | wheel首次提供完整C++ SDK，构建、导出核心接口和Runner编译已验证 |

旧版本即使能够通过部分Python测试，也不能据此成为支持版本。

## Nightly

Nightly只用于前向兼容验证，不属于稳定支持范围。安装时必须显式包含Torch：

```bash
python -m pip install --upgrade --pre executorch torch \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

CI应记录完整nightly版本和git commit，例如：

```text
executorch-1.6.0.dev20260916+cpu
ac4f33f90f7dff650e3c982fa34a3cc84848b366
```

nightly构建结果不能自动扩大稳定版支持范围。

## 已验证设备

以下ACL SoC名称已完成验证：

```text
Ascend910_9362
Ascend910B3
Ascend910B4
Ascend310P3
Ascend310B1
```

导出期 `acl_soc_name` 按SoC族前缀受理，不再逐个枚举型号：`Ascend910_93*`、
`Ascend910B*`、`Ascend310P*`、`Ascend310B*`。同族新型号无需改动
`compile_spec.py`，具体型号是否可用由CANN离线编译自行判定；不属于上述任一族
时 `AscendCompileOptions` 校验失败并列出支持的族。

上面的清单是**已完成验证**的型号，不是受理范围。

导出产物记录的 `acl_soc_name` 与运行设备 `aclrtGetSocName()` 返回值精确
匹配即可加载，不匹配时拒绝（`DelegateInvalidCompatibility`）。运行期始终是
全词匹配，不做前缀放宽。
