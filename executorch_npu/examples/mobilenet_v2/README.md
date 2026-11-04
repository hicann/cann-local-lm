# MobileNetV2示例

简体中文 | [English](README_en.md)

这个示例使用 `torchvision.models.mobilenet_v2(weights=None)` 导出固定形状
`[1, 3, 224, 224]` 的float32模型。它不下载ImageNet权重，而是固定随机种子
生成输入，并保存CPU输出作为NPU执行的参考结果。

先按 [`docs/quickstart.md`](../../docs/quickstart.md) 和
[`docs/build.md`](../../docs/build.md) 完成CANN、
ExecuTorch、torch-delegate和runtime runner的构建，然后执行：

```bash
export CANN_BUILD_DIR="$EXECUTORCH_NPU_ROOT/build/release/install"
export DEVICE_ID=0
export STATE_NAMESPACE=mobilenet-smoke-001
bash examples/mobilenet_v2/run.sh all
```

也可以只导出或只执行：

```bash
bash examples/mobilenet_v2/run.sh export
DEVICE_ID=1 STATE_NAMESPACE=mobilenet-smoke-002 \
  bash examples/mobilenet_v2/run.sh run
```

可通过环境变量覆盖默认配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OUTPUT_DIR` | `examples/mobilenet_v2/artifacts` | PTE、PTD和输入输出文件目录 |
| `CANN_BUILD_DIR` | `$EXECUTORCH_NPU_ROOT/build/release/install` | runtime安装前缀 |
| `ACL_SOC_NAME` | `Ascend910_9362` | 已验证的ACL SoC名称，同时作为CANN离线编译目标 |
| `DEVICE_ID` | 无默认值 | 必填的Ascend设备编号 |
| `STATE_NAMESPACE` | 无默认值 | 必填的请求唯一状态命名空间 |

执行成功时runner输出 `ASCEND_EXECUTORCH_OK values=1000`。输出比较使用
`atol=1e-3` 和 `rtol=1e-3`，与当前CANN混合精度配置一致。
