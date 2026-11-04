# Quick Start: Export a Custom Model and Run It with C++

Starting from a custom `TinyMLP`, this document demonstrates how to export a paired
`model.pte` and `ascend.weights.package.ge_aot.ptd` in Python, and then write a C++ Runner
that loads the artifacts through the ExecuTorch `Module` and executes them on an Ascend NPU.

This document assumes you have already entered the prepared `executorch_npu` conda environment on the remote host,
and that the current directory is the root of the ExecuTorch-NPU repository. It does not cover environment installation, source builds,
full-network model adaptation and export, or performance tuning.

## Prepare the Directory

```bash
mkdir -p /tmp/cann-et-tiny
cd /tmp/cann-et-tiny
```

The final directory looks like this:

```text
/tmp/cann-et-tiny/
├── export_tiny.py
├── runner.cpp
├── CMakeLists.txt
├── artifacts/
│   ├── model.pte
│   ├── ascend.weights.package.ge_aot.ptd
│   └── input.bin
└── build/
    └── tiny_runner
```

## Export a Small Custom Model

Create `export_tiny.py`:

```python
from pathlib import Path

import torch
from torch import nn
from executorch.exir import to_edge_transform_and_lower

from executorch_npu import (
    AscendCompileOptions,
    AscendPartitioner,
    aten_edge_compile_config,
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


torch.manual_seed(0)
model = TinyMLP().eval()
example_input = torch.randn(1, 4)
exported_program = torch.export.export(model, (example_input,))

# Change to the SoC name of the actual target device.
options = AscendCompileOptions(
    method_name="forward",
    delegate_index=0,
    acl_soc_name="Ascend910_9362",
)
program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[AscendPartitioner(options.to_compile_specs())],
    compile_config=aten_edge_compile_config(),
).to_executorch()

output_dir = Path("artifacts")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "model.pte").write_bytes(program.buffer)
program.write_tensor_data_to_file(output_dir)
(output_dir / "input.bin").write_bytes(
    example_input.contiguous().numpy().tobytes()
)
print(f"wrote artifacts to {output_dir.resolve()}")
```

Run the export:

```bash
python export_tiny.py
ls -lh artifacts/model.pte \
  artifacts/ascend.weights.package.ge_aot.ptd \
  artifacts/input.bin
```

The PTE contains the ExecuTorch Program, Delegate metadata, and the OM; the PTD stores the external weights.
Both must come from the same export and cannot be mixed and matched.

## Write the C++ Runner

Create `runner.cpp`:

```cpp
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::from_blob;
using executorch::extension::module::Module;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::LoadBackendOptionsMap;

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return {};
  const auto size = stream.tellg();
  if (size <= 0) return {};
  stream.seekg(0);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!stream.read(reinterpret_cast<char*>(data.data()), size)) return {};
  return data;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.pte WEIGHTS.ptd INPUT.bin DEVICE_ID STATE_NAMESPACE\n";
    return 2;
  }

  executorch::runtime::runtime_init();
  BackendOptions<2> ascend_options;
  if (ascend_options.set_option("device_id", std::stoi(argv[4])) != Error::Ok ||
      ascend_options.set_option("state_namespace", argv[5]) != Error::Ok) {
    std::cerr << "failed to create Ascend backend options\n";
    return 1;
  }
  LoadBackendOptionsMap backend_options;
  if (backend_options.set_options("AscendBackend", ascend_options.view()) !=
      Error::Ok) {
    std::cerr << "failed to install Ascend backend options\n";
    return 1;
  }

  Module module(argv[1], argv[2]);
  const Error load_status = module.load(backend_options);
  if (load_status != Error::Ok) {
    std::cerr << "failed to load PTE/PTD: 0x" << std::hex
              << static_cast<uint32_t>(load_status) << '\n';
    return 1;
  }

  auto metadata = module.method_meta("forward");
  if (!metadata.ok() || metadata->num_inputs() != 1 ||
      metadata->num_outputs() != 1) {
    std::cerr << "runner expects one tensor input and one tensor output\n";
    return 1;
  }
  auto input_meta = metadata->input_tensor_meta(0);
  if (!input_meta.ok()) {
    std::cerr << "input 0 is not a tensor\n";
    return 1;
  }

  std::vector<uint8_t> input_data = read_file(argv[3]);
  if (input_data.size() != input_meta->nbytes()) {
    std::cerr << "input byte size does not match model metadata\n";
    return 1;
  }
  std::vector<executorch::aten::SizesType> input_shape;
  for (int32_t size : input_meta->sizes()) input_shape.push_back(size);
  auto input_tensor =
      from_blob(input_data.data(), input_shape, input_meta->scalar_type());

  auto outputs = module.forward(EValue(*input_tensor));
  if (!outputs.ok() || outputs->size() != 1 || !outputs->front().isTensor()) {
    std::cerr << "Ascend execution failed\n";
    return 1;
  }

  const auto& output = outputs->front().toTensor();
  if (output.scalar_type() != executorch::aten::ScalarType::Float) {
    std::cerr << "example runner expects a float32 output\n";
    return 1;
  }
  const auto* values = output.const_data_ptr<float>();
  const size_t count = output.nbytes() / sizeof(float);
  std::cout << "output:";
  for (size_t i = 0; i < count; ++i) std::cout << ' ' << values[i];
  std::cout << '\n';
  return 0;
}
```

The main path of the Runner is: initialize the Runtime, set the Ascend Backend options, load the
PTE/PTD, construct the input Tensor based on the Method metadata, and then call `forward`.

## Compile the Runner

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(tiny_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("$ENV{EXECUTORCH_NPU_SDK}/cmake/executorch_npu_backend.cmake")

add_executable(tiny_runner runner.cpp)
target_link_libraries(tiny_runner PRIVATE executorch_npu::backend)
```

Obtain the SDK path from the installed Python package, then compile:

```bash
export EXECUTORCH_ROOT="$(python -c \
  'import executorch; print(next(iter(executorch.__path__)))')"
export EXECUTORCH_NPU_SDK="$(python -c \
  'from executorch_npu import sdk_root; print(sdk_root())')"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$EXECUTORCH_ROOT" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build build -j
```

`executorch_npu::backend` transitively provides the whole-archive linking required for Backend registration,
the ExecuTorch Runtime, and the AscendCL and OpenSSL dependencies.

## Run the PTE/PTD

`DEVICE_ID` is the logical device number of the ACL Runtime. `STATE_NAMESPACE` is the state
isolation key of the Backend; even a stateless example must pass a non-empty value:

```bash
./build/tiny_runner \
  artifacts/model.pte \
  artifacts/ascend.weights.package.ge_aot.ptd \
  artifacts/input.bin \
  0 \
  tiny-quickstart
```

On success, two `float32` values are printed. The values depend on the weights and input used at export time;
being able to produce output indicates that the Runner has loaded the paired PTE/PTD and completed
NPU execution through the Ascend Backend.

## Switch to Your Own Model

- When replacing `TinyMLP`, also replace `example_input`; `torch.export` uses it to capture
  the graph and Tensor constraints.
- For multi-input models, iterate over the input metadata via `method_meta()` and construct
  an `EValue` for each Tensor.
- A production Runner should validate the number of inputs, shapes, dtypes, and byte sizes.
- Do not combine a PTE with a PTD from a different export. The Backend validates the Manifest,
  named-data keys, and weight digests.

For more complete APIs and error handling, see [C++ Runtime](runtime_en.md); for artifact association rules, see
[Artifact Protocol](protocol_en.md).
