# Inplace Input NPU 验证

本目录提供完整的 NPU epair 导出、加载和精度验证流程。

## 目录结构

```
7_npu/
├── README.md           # 本文档
├── CMakeLists.txt      # C++ 构建配置
├── main.cpp            # 主程序 (C++ GE 加载和执行)
├── reference_data.h    # 参考数据头文件
├── reference_data.cpp  # 参考数据实现
├── export_epair.py     # Python epair 导出脚本
├── models.py           # 模型定义 (symlink 到上级目录)
├── run_all.sh          # 一键运行脚本
├── build/              # 构建目录 (脚本自动创建)
└── generated/          # epair 输出目录 (脚本自动创建)
    ├── epair_model_simple.epair
    ├── epair_model_chain.epair
    └ epair_model_param.epair
    └── epair_manifest_*.json
```

## 环境要求

1. **硬件**: Ascend NPU 设备
2. **软件**:
   - CANN 9.0.0 或更高版本
   - Python 3.11+ with torch, torch_npu
   - GCC 9+ with C++17 支持
   - CMake 3.14+

## 快速开始 (一键运行)

### Step 1: 设置环境变量

```bash
# 必须设置这两个环境变量
export TORCH_DELEGATE_ROOT=/mnt/workspace/File/torch-delegate
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
```

### Step 2: 运行一键脚本

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu
bash run_all.sh
```

脚本会自动完成:
- 构建 TorchDelegate
- 构建 C++ inplace_verify 可执行文件
- 导出三个模型的 epair 文件
- 在 NPU 上运行并验证精度

### 预期输出

```
[INFO] ============================================
[INFO] 验证结果汇总
[INFO] ============================================
  simple: PASS
  chain: PASS
  param: PASS
[INFO] ============================================
[INFO] 全部验证通过!
[INFO] ============================================
```

## 分步执行 (调试模式)

如果一键脚本失败或需要分步调试:

### Step 1: 构建 TorchDelegate

```bash
cd $TORCH_DELEGATE_ROOT

# 构建第三方依赖 (首次运行)
./build_third_party.sh

# 构建 TorchDelegate
./build.sh --release --cann_3rd_lib_path=$TORCH_DELEGATE_ROOT/third_party
```

### Step 2: 构建 C++ inplace_verify

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu

# 创建构建目录
mkdir -p build && cd build

# 配置 CMake
cmake .. \
    -DTorchDelegate_ROOT_DIR=$TORCH_DELEGATE_ROOT/build/release/install \
    -DCANN_ROOT=$ASCEND_HOME_PATH \
    -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")

# 编译
make -j$(nproc)
```

### Step 3: 导出 epair 文件

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu

# 安装 torch_delegate 包
pip install -e $TORCH_DELEGATE_ROOT

# 导出 epair
python3 export_epair.py
```

### Step 4: 运行 NPU 验证

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu/build

# 设置库路径
export LD_LIBRARY_PATH=$TORCH_DELEGATE_ROOT/build/release/install/lib:$ASCEND_HOME_PATH/lib64:$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$LD_LIBRARY_PATH

# 运行验证
./inplace_verify ../generated/epair_model_simple.epair simple
./inplace_verify ../generated/epair_model_chain.epair chain
./inplace_verify ../generated/epair_model_param.epair param
```

## 模型说明

| 模型 | 操作 | 参考公式 | 输入数量 |
|------|------|----------|----------|
| simple | `x.add_(y), x.mul_(scale)` | `out = (x + y) * 2.0` | 2 (x, y) |
| chain | `x.add_(1), x.mul_(2), x.div_(4)` | `out = ((x + 1) * 2) / 4` | 1 (x) |
| param | `x.add_(bias), x.mul_(scale)` | `out = (x + 1) * 2.0` | 1 (x) + weight |

## 移植到其他环境

### 方式一: 完整移植

将整个 `torch-delegate` 项目复制到目标环境:

```bash
# 1. 复制项目
scp -r /path/to/torch-delegate target_host:/destination/path/

# 2. 在目标环境设置变量
export TORCH_DELEGATE_ROOT=/destination/path/torch-delegate
export ASCEND_HOME_PATH=/path/to/cann

# 3. 运行脚本
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu
bash run_all.sh
```

### 方式二: 最小移植 (仅运行验证)

如果目标环境已构建好 TorchDelegate 和 epair 文件:

```bash
# 1. 复制必需文件
mkdir -p /destination/npu_verify
cd /path/to/torch-delegate/examples/inplace_input/7_npu

# 复制运行所需文件
cp -r main.cpp reference_data.h reference_data.cpp CMakeLists.txt /destination/npu_verify/
cp -r generated/*.epair /destination/npu_verify/epair/

# 2. 在目标环境构建 C++
cd /destination/npu_verify
mkdir build && cd build
cmake .. -DTorchDelegate_ROOT_DIR=/path/to/torch-delegate/build/release/install \
  -DCANN_ROOT=$ASCEND_HOME_PATH \
  -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
make -j$(nproc)

# 3. 设置库路径并运行
export LD_LIBRARY_PATH=/path/to/torch-delegate/build/release/install/lib:$ASCEND_HOME_PATH/lib64:$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$LD_LIBRARY_PATH

./inplace_verify ../epair/epair_model_simple.epair simple
./inplace_verify ../epair/epair_model_chain.epair chain
./inplace_verify ../epair/epair_model_param.epair param
```

## 常见问题

### Q: 找不到 libtorch_delegate_backend.so
**A**：确保 `LD_LIBRARY_PATH` 包含 TorchDelegate 的 lib 目录：
```bash
export LD_LIBRARY_PATH=$TORCH_DELEGATE_ROOT/build/release/install/lib:$LD_LIBRARY_PATH
```

### Q: 找不到 libc10.so
**A**：确保 `LD_LIBRARY_PATH` 包含 torch 的 lib 目录：
```bash
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$LD_LIBRARY_PATH
```

### Q: CMake 找不到 TorchDelegate
**A**：确保 `-DTorchDelegate_ROOT_DIR` 指向正确的 install 目录：
```bash
cmake .. -DTorchDelegate_ROOT_DIR=/path/to/torch-delegate/build/release/install
```

### Q: epair 加载失败 "manifest.json not found"
**A**: PyTorchStreamReader 要求所有文件在单一子目录，epair 格式已修复此问题。
如果遇到此问题，请确保使用最新版本的 `torch_delegate/api.py`。

### Q: NPU 设备不可用
**A**：检查 NPU 驱动和 CANN 安装：
```bash
# 检查 NPU 设备
npu-smi info

# 检查 CANN 版本
cat $ASCEND_HOME_PATH/version.info
```

## 相关文件路径

| 文件 | 位置 |
|------|------|
| TorchDelegate 源码 | `$TORCH_DELEGATE_ROOT/csrc/` |
| Python API | `$TORCH_DELEGATE_ROOT/torch_delegate/api.py` |
| epair 格式常量 | `$TORCH_DELEGATE_ROOT/csrc/common/epair_archive_constants.h` |
| C++ 加载流程 | `$TORCH_DELEGATE_ROOT/csrc/backend/delegate_session.cc` |
| 模型定义 | `$TORCH_DELEGATE_ROOT/examples/inplace_input/models.py` |
