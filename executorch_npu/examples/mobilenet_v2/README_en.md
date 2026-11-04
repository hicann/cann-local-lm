# MobileNetV2 Example

English | [简体中文](README.md)

This example uses `torchvision.models.mobilenet_v2(weights=None)` to export a float32
model with the fixed shape `[1, 3, 224, 224]`. It does not download ImageNet weights;
instead, it generates inputs with a fixed random seed and saves the CPU outputs as the
reference results for NPU execution.

First follow [`docs/quickstart.md`](../../docs/quickstart_en.md) and
[`docs/build.md`](../../docs/build_en.md) to complete the builds of CANN,
ExecuTorch, torch-delegate, and the runtime runner, then run:

```bash
export CANN_BUILD_DIR="$EXECUTORCH_NPU_ROOT/build/release/install"
export DEVICE_ID=0
export STATE_NAMESPACE=mobilenet-smoke-001
bash examples/mobilenet_v2/run.sh all
```

You can also only export or only run:

```bash
bash examples/mobilenet_v2/run.sh export
DEVICE_ID=1 STATE_NAMESPACE=mobilenet-smoke-002 \
  bash examples/mobilenet_v2/run.sh run
```

The default configuration can be overridden through environment variables:

| Variable | Default | Description |
|---|---|---|
| `OUTPUT_DIR` | `examples/mobilenet_v2/artifacts` | Directory for the PTE, PTD, and input/output files |
| `CANN_BUILD_DIR` | `$EXECUTORCH_NPU_ROOT/build/release/install` | runtime installation prefix |
| `ACL_SOC_NAME` | `Ascend910_9362` | Validated ACL SoC name, also used as the CANN offline compilation target |
| `DEVICE_ID` | No default | Required Ascend device ID |
| `STATE_NAMESPACE` | No default | Required per-request unique state namespace |

On successful execution, the runner outputs `ASCEND_EXECUTORCH_OK values=1000`. Output
comparison uses `atol=1e-3` and `rtol=1e-3`, consistent with the current CANN
mixed-precision configuration.
