# Epair Packaging Guide

This document describes how to use the torch_delegate Python API to package a PyTorch ExportedProgram into an epair file for loading and execution by the C++ delegate library.

---

## Prerequisites

- **Python >= 3.10** (the whl supports 3.10 ~ 3.13)
- **torch >= 2.9.0** (the whl does not declare runtime dependencies and will not install it automatically; install the matching version yourself)
- **torch_npu** (required for the full export mode, provides `torchair`; the whl does not install it automatically, install the matching version yourself)

---

## Installing torch_delegate

### Install from the whl package (recommended)

```bash
pip install output/package/torch_delegate-<version>-py3-none-linux_<arch>.whl
```

The whl is the only release artifact and contains both the Python API and the C++ artifacts (headers, dynamic libraries, CMake configuration package).

### Build from source

```bash
cd torch-delegate
./build.sh  # One-stop build, produces a single whl in output/package/
```

See **[build.md](build.md)** for details.

### Verify the installation

```bash
python -c "from torch_delegate import export_to_epair, epair_save, epair_load; print('torch_delegate installed successfully')"
```

---

## Python API

torch_delegate provides three core APIs:

### export_to_epair

Converts an ExportedProgram into EpairContents, including Ascend IR annotations:

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents:
    """
    Convert an ExportedProgram to EpairContents

    Args:
        ep: ExportedProgram exported by torch.export
        config: EpairConfig (optional), controls optimization switches

    Returns:
        EpairContents: contains the manifest, ep_json, and an executable ExportedProgram
    """
```

### epair_save

Saves EpairContents to a .epair file:

```python
def epair_save(
    contents: EpairContents,
    f: str | os.PathLike
) -> None:
    """
    Save EpairContents to an epair file

    Args:
        contents: the EpairContents object
        f: output file path (e.g. "model.epair")
    """
```

### epair_load

Loads EpairContents from a .epair file:

```python
def epair_load(f: str | os.PathLike) -> EpairContents:
    """
    Load an epair file

    Args:
        f: epair file path

    Returns:
        EpairContents: contains the manifest, ep_json, and an executable ExportedProgram
    """
```

---

## EpairConfig Configuration

Controls graph optimization switches and operator decomposition behavior:

```python
from torch_delegate.epair import EpairConfig

config = EpairConfig(
    remove_noop_ops=True,       # Remove no-op operators (default True)
    pattern_fusion_pass=True,   # Pattern fusion optimization (default True)
    whitelist_decompositions=None  # Operator decomposition whitelist (decomposes linear/matmul/layernorm etc. by default)
)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|--------|------|
| `remove_noop_ops` | `bool` | `True` | Remove no-op operators (nodes such as identity and reshape that perform no actual computation) |
| `pattern_fusion_pass` | `bool` | `True` | Enable pattern fusion optimization, fusing multiple operators into a single Ascend operator to improve NPU execution efficiency |
| `whitelist_decompositions` | `set \| None` | `None` | Whitelist of operators allowed to be decomposed. `None` means use the default whitelist, and `set()` disables all decomposition |

### whitelist_decompositions in Detail

Controls which PyTorch operators are decomposed into more basic operators so that the Ascend backend can handle them.

**The default whitelist contains the following operators:**

| Operator | Description |
|------|------|
| `torch.ops.aten.linear.default` | Linear layer, decomposed into matmul + add |
| `torch.ops.aten.matmul.default` | Matrix multiplication |
| `torch.ops.aten.layer_norm.default` | LayerNorm |
| `torch.ops.aten.softmax.int` | Softmax |
| `torch.ops.aten.log_softmax.int` | LogSoftmax |
| `torch.ops.aten.__and__.Tensor` | Bitwise AND |
| `torch.ops.aten.__or__.Tensor` | Bitwise OR |
| `torch.ops.aten.isfinite.default` | Checks whether elements are finite, decomposed into abs + ne + mul |
| `torch.ops.aten.negative.default` | Negation, decomposed into neg (alias normalization) |
| `torch.ops.aten.absolute.default` | Absolute value, decomposed into abs (alias normalization) |
| `torch.ops.aten.relu6.default` | ReLU6, decomposed into hardtanh |
| `torch.ops.aten.arccos.default` | Arccosine, decomposed into acos (alias normalization) |
| `torch.ops.aten.arcsin.default` | Arcsine, decomposed into asin (alias normalization) |
| `torch.ops.aten.arctan.default` | Arctangent, decomposed into atan (alias normalization) |
| `torch.ops.aten.arccosh.default` | Inverse hyperbolic cosine, decomposed into acosh (alias normalization) |
| `torch.ops.aten.arcsinh.default` | Inverse hyperbolic sine, decomposed into asinh (alias normalization) |
| `torch.ops.aten.conv2d.default` | 2D convolution, decomposed into convolution |
| `torch.ops.aten.square.default` | Square, decomposed into pow.Tensor_Scalar |

**Configuration examples:**

```python
import torch
from torch_delegate.epair import EpairConfig

# Disable all operator decomposition
config = EpairConfig(whitelist_decompositions=set())

# Decompose linear only
config = EpairConfig(
    whitelist_decompositions={torch.ops.aten.linear.default}
)

# Custom decomposition list
config = EpairConfig(
    whitelist_decompositions={
        torch.ops.aten.linear.default,
        torch.ops.aten.layer_norm.default,
    }
)
```

---

## Complete Examples

### Basic example export and save

```python
import torch
import torch.nn as nn
from torch.export import export
from torch_delegate import export_to_epair, epair_save

# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = SimpleModel()

# Export to ExportedProgram
example_inputs = (torch.randn(1, 10),)
ep = export(model, example_inputs)

# Convert to EpairContents
contents = export_to_epair(ep)

# Inspect conversion statistics
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")
print(f"CANN version: {contents.manifest.cann_version}")
print(f"Backend: {contents.manifest.backend_name}")

# Save to an epair file
epair_save(contents, "model.epair")
print("Saved to model.epair")
```

### Load and run

```python
from torch_delegate import epair_load

# Load the epair file
contents = epair_load("model.epair")

# Inspect metadata
print(f"Backend: {contents.manifest.backend_name}")
print(f"Converted nodes: {contents.manifest.converted_nodes}/{contents.manifest.total_nodes}")

# Run the model
inputs = torch.randn(1, 10)
outputs = contents.ep.module()(inputs)
print(f"Output shape: {outputs.shape}")
```

### Optimize with configuration

```python
from torch_delegate import export_to_epair, epair_save
from torch_delegate.epair import EpairConfig

# Custom configuration
config = EpairConfig(
    remove_noop_ops=True,
    pattern_fusion_pass=False  # Disable fusion optimization
)

# Export and convert
ep = export(model, example_inputs)
contents = export_to_epair(ep, config=config)

# Save
epair_save(contents, "optimized_model.epair")
```

---

## epair File Structure

An `.epair` file is a ZIP archive compatible with the PT2 format:

```text
model.epair (ZIP structure)
├── extra/
│   ├── epair_manifest.json           ← epair metadata (new)
│   └── ep_json_with_ascend_ir.json   ← EP JSON with ascend_ir (new)
├── models/
│   └── model.json                    ← Standard EP JSON
└── data/
    ├── weights/
    │   ├── model_weights_config.json
    │   └── weight_0, weight_1, ...
    ├── constants/
    │   ├── model_constants_config.json
    │   └── tensor_0, tensor_1, ...
    └── model_sample_inputs.pt
```

### File Descriptions

| File | Description |
|------|------|
| `extra/epair_manifest.json` | Metadata: version, backend, statistics, CANN version |
| `extra/ep_json_with_ascend_ir.json` | Complete EP JSON annotated with the `ascend_ir` field |
| `models/model.json` | Standard EP JSON (without ascend_ir) |
| `data/weights/` | Model weight tensors |
| `data/constants/` | Constant tensors |
| `data/model_sample_inputs.pt` | Sample inputs |

### manifest.json Format

```json
{
  "version": 1,
  "backend_name": "ascend_ge",
  "cann_version": "9.2.0",
  "total_nodes": 20,
  "converted_nodes": 15
}
```

---

## ascend_ir Field Format

Each node object in the EP JSON may optionally contain an `ascend_ir` field:

### Successfully converted node

```json
{
  "target": "aten.add.Tensor",
  "inputs": [...],
  "outputs": [...],
  "ascend_ir": {
    "source": "converter",
    "ops": [
      {
        "name": "Add",
        "attr": [{"key": "alpha", "value": {"f": 1.0}}],
        "input": [{"name": "x", "type": "tensor"}, {"name": "y", "type": "tensor"}],
        "output": [{"name": "z", "type": "tensor"}]
      }
    ],
    "mapping": {
      "outputs": ["add_1:0"]
    }
  }
}
```

### Failed conversion node (Mock)

```json
{
  "target": "aten.custom_op",
  "ascend_ir": {
    "ops": [{"attr": [...]}]
  }
}
```

**Notes**:
- `source = "converter"`: normal GE conversion, the C++ side places it in the NPU subgraph
- `source = ""`: Mock node, the C++ side places it in the CPU segment

---

## Log Debugging

torch_delegate provides a complete logging system controlled by environment variables:

### Log Levels

| Level | Usage Scenario |
|------|----------|
| ERROR | Conversion failure termination, missing critical dependencies |
| WARNING | Node conversion failure creating a mock, CANN version retrieval failure |
| INFO | API entry/exit, conversion statistics |
| DEBUG | Configuration details, intermediate steps |

### Enabling DEBUG Logs

```bash
# Default: WARN level
python your_script.py

# Enable DEBUG level
TORCH_DELEGATE_DEBUG=1 python your_script.py
```

### Log Format

```text
[LEVEL] TORCH_DELEGATE(pid,python) timestamp [file:line] thread_id message
```

Example:
```text
[INFO] TORCH_DELEGATE(12345,python) 2026-04-23 10:30:15 [api.py:29] Exporting ExportedProgram to epair...
[WARNING] TORCH_DELEGATE(12345,python) 2026-04-23 10:30:16 [epair_graph.py:142] Failed to convert aten.custom_op: creating mock node
[INFO] TORCH_DELEGATE(12345,python) 2026-04-23 10:30:17 [api.py:46] Export complete: 15/20 nodes converted
```

---

## Relationship with the C++ delegate

### Workflow

```text
Python side → C++ side

1. Python: export_to_epair(ep)
   ↓ Generate EpairContents (annotated with ascend_ir)

2. Python: epair_save(contents, "model.epair")
   ↓ Save as a .epair file

3. C++: td::EpairModelLoader loader("model.epair")
   ↓ Load the .epair, parse ep_json_with_ascend_ir.json, and build the GE graph

4. C++: loader.CompileAndLoad(session_options, stream)
   ↓ Compile (including fallback custom operators) and load

5. C++: loader.RunModelWithStreamAsync(stream, inputs, outputs)
   ↓ Run inference
```

### C++ Loading Example

See `examples/export_and_load_example/main.cpp` for a complete runnable example.

```cpp
#include "ge/ge_api_v2.h"
#include "graph/graph.h"
#include "torch_delegate/epair_model_loader.h"
#include <map>
#include <vector>

// 1. Initialize GE
std::map<ge::AscendString, ge::AscendString> ge_options = {
    {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")}};
ge::GEInitializeV2(ge_options);

// 2. Load the epair file (generated by the Python side); builds, compiles, and loads
td::EpairModelLoader loader("model.epair");
std::map<ge::AscendString, ge::AscendString> session_options;
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
td::Status status = loader.CompileAndLoad(session_options, stream);
if (status != td::SUCCESS) {
    // Compile and load failed; report the error and return
    return FAILED;
}

// 3. Prepare inputs and run inference (outputs need no pre-allocation for dynamic shapes)
std::vector<gert::Tensor> inputs;   // filled according to the GE graph input descs
std::vector<gert::Tensor> outputs;
status = loader.RunModelWithStreamAsync(stream, inputs, outputs);
if (status != td::SUCCESS) {
    // Inference failed; report the error and return
    return FAILED;
}
// ...
```

---

## FAQ

### Q1 CANN version is unknown because torch_npu is not installed

**Symptom**: `cann_version` in the manifest is "unknown"

**Solution**:
1. Install torch_npu: `pip install torch_npu`
2. Or ignore this field; it does not affect epair file generation or loading

### Q2 Too many nodes fail to convert

**Symptom**: `converted_nodes` is much smaller than `total_nodes`

**Solution**:
1. Check whether the ops used by the model are in the TorchAir support list
2. Check the WARNING logs to locate the specific ops that failed
3. Mock nodes are placed in the CPU segment for execution and do not affect overall functionality

### Q3 epair file is too large

**Symptom**: the generated .epair file is too large

**Solution**:
1. Check whether weights and constants are correctly externalized
2. Use the weight compression option of `torch.export.save`
3. Avoid hardcoding large constant tensors in the model

### Q4 Execution fails after loading the epair

**Symptom**: `contents.ep.module()` on the Python side raises an error during execution

**Solution**:
1. Ensure the torch version of the loading environment matches that of the export environment
2. Check whether `ep_json_with_ascend_ir.json` is complete
3. Use DEBUG logs to inspect detailed error information

---

## Related Documents

- **[build.md](build.md)** - C++ delegate library build guide
- **[epair_format_design.md](../features/epair_format_design.md)** - Detailed design of the epair implementation

---

## References

- [PyTorch Export documentation](https://pytorch.org/docs/stable/export.html)
- [PT2 Archive format](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/pt2_archive.html)
- [TorchAir GE Converter](https://gitcode.com/Ascend/pytorch)
