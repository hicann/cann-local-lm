# 快速开始：导出自定义模型并用C++ 运行

本文从一个自定义 `TinyMLP` 出发，演示在Python中导出成对的
`model.pte` 和 `ascend.weights.package.ge_aot.ptd`，再编写C++ Runner
通过ExecuTorch `Module` 加载产物并在Ascend NPU上执行。

本文假设你已在远端进入准备好的 `executorch_npu` conda环境，
当前目录为ExecuTorch-NPU仓库根目录。不涉及环境安装、源码构建或
整网模型适配导出，或者性能调优。

## 准备目录

```bash
mkdir -p /tmp/cann-et-tiny
cd /tmp/cann-et-tiny
```

最终目录如下：

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

## 导出自定义小模型

创建 `export_tiny.py`：

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

# 改为实际运行设备的 SoC 名称。
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

执行导出：

```bash
python export_tiny.py
ls -lh artifacts/model.pte \
  artifacts/ascend.weights.package.ge_aot.ptd \
  artifacts/input.bin
```

PTE包含ExecuTorch Program、Delegate元数据和OM，PTD保存外置权重。
两者必须来自同一次导出，不能交叉替换。

## 编写C++ Runner

创建 `runner.cpp`：

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

Runner的主路径是：初始化Runtime、设置Ascend Backend选项、加载
PTE/PTD、根据Method元数据构造输入Tensor，然后调用 `forward`。

## 编译Runner

创建 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(tiny_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("$ENV{EXECUTORCH_NPU_SDK}/cmake/executorch_npu_backend.cmake")

add_executable(tiny_runner runner.cpp)
target_link_libraries(tiny_runner PRIVATE executorch_npu::backend)
```

从已安装的Python包获取SDK路径后编译：

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

`executorch_npu::backend` 会传递Backend注册所需的whole-archive链接、
ExecuTorch Runtime、AscendCL和OpenSSL依赖。

## 运行PTE/PTD

`DEVICE_ID` 是ACL Runtime的逻辑设备编号。`STATE_NAMESPACE` 是Backend的
状态隔离键，即使无状态示例也需要传入非空值：

```bash
./build/tiny_runner \
  artifacts/model.pte \
  artifacts/ascend.weights.package.ge_aot.ptd \
  artifacts/input.bin \
  0 \
  tiny-quickstart
```

成功时会输出两个 `float32` 值。数值取决于导出时的权重和输入；
能够输出结果即表明Runner已加载成对的PTE/PTD，并通过Ascend
Backend完成NPU执行。

## 改成自己的模型

- 替换 `TinyMLP` 时同时替换 `example_input`；`torch.export` 会以它捕获
  图和Tensor约束。
- 多输入模型应通过 `method_meta()` 遍历输入元数据，为每个Tensor
  构造 `EValue`。
- 业务Runner应校验输入数量、shape、dtype和字节数。
- 不要把PTE和另一次导出的PTD组合使用。Backend会校验Manifest、
  named-data key和权重摘要。

更完整的API和错误处理见 [C++ 运行](runtime.md)，产物关联规则见
[产物协议](protocol.md)。
