# Inplace Input NPU Verification

English | [简体中文](README.md)

This directory provides the complete NPU flow for epair export, loading, and accuracy verification.

## Directory Layout

```text
7_npu/
├── README.md           # This document
├── CMakeLists.txt      # C++ build configuration
├── main.cpp            # Main program (C++ GE loading and execution)
├── reference_data.h    # Reference data header
├── reference_data.cpp  # Reference data implementation
├── export_epair.py     # Python epair export script
├── models.py           # Model definitions (symlink to the parent directory)
├── run_all.sh          # One-shot run script
├── build/              # Build directory (created by the script)
└── generated/          # epair output directory (created by the script)
    ├── epair_model_simple.epair
    ├── epair_model_chain.epair
    └ epair_model_param.epair
    └── epair_manifest_*.json
```

## Requirements

1. **Hardware**: Ascend NPU device
2. **Software**:
   - CANN 9.2.0 or later
   - Python 3.11+ with torch, torch_npu
   - GCC 9+ with C++17 support
   - CMake 3.14+

## Quick Start (one-shot run)

### Set environment variables

```bash
# These two environment variables are required
export TORCH_DELEGATE_ROOT=/mnt/workspace/File/torch-delegate
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.2.0
```

### Run the one-shot script

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu
bash run_all.sh
```

The script automatically:
- builds TorchDelegate
- builds the C++ `inplace_verify` executable
- exports the epair files for the three models
- runs on the NPU and verifies accuracy

### Expected output

```text
[INFO] ============================================
[INFO] Verification summary
[INFO] ============================================
  simple: PASS
  chain: PASS
  param: PASS
[INFO] ============================================
[INFO] All verifications passed!
[INFO] ============================================
```

## Step-by-Step (debug mode)

If the one-shot script fails or you need to debug step by step:

### Build TorchDelegate

```bash
cd $TORCH_DELEGATE_ROOT

# Build third-party dependencies (first run only)
./build_third_party.sh

# Build TorchDelegate
./build.sh --release --cann_3rd_lib_path=$TORCH_DELEGATE_ROOT/third_party
```

### Build the C++ inplace_verify

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu

# Create the build directory
mkdir -p build && cd build

# Configure CMake
cmake .. \
    -DTorchDelegate_ROOT_DIR=$TORCH_DELEGATE_ROOT/build/release/install \
    -DCANN_ROOT=$ASCEND_HOME_PATH \
    -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")

# Compile
make -j$(nproc)
```

### Export the epair files

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu

# Install the torch_delegate package
pip install -e $TORCH_DELEGATE_ROOT

# Export the epair
python3 export_epair.py
```

### Run the NPU verification

```bash
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu/build

# Set the library path
export LD_LIBRARY_PATH=$TORCH_DELEGATE_ROOT/build/release/install/lib:$ASCEND_HOME_PATH/lib64:$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$LD_LIBRARY_PATH

# Run the verification
./inplace_verify ../generated/epair_model_simple.epair simple
./inplace_verify ../generated/epair_model_chain.epair chain
./inplace_verify ../generated/epair_model_param.epair param
```

## Models

| Model | Operations | Reference formula | Number of inputs |
|------|------|----------|----------|
| simple | `x.add_(y), x.mul_(scale)` | `out = (x + y) * 2.0` | 2 (x, y) |
| chain | `x.add_(1), x.mul_(2), x.div_(4)` | `out = ((x + 1) * 2) / 4` | 1 (x) |
| param | `x.add_(bias), x.mul_(scale)` | `out = (x + 1) * 2.0` | 1 (x) + weight |

## Porting to Another Environment

### Option 1 Full port

Copy the entire `torch-delegate` project to the target environment:

```bash
# 1. Copy the project
scp -r /path/to/torch-delegate target_host:/destination/path/

# 2. Set the variables in the target environment
export TORCH_DELEGATE_ROOT=/destination/path/torch-delegate
export ASCEND_HOME_PATH=/path/to/cann

# 3. Run the script
cd $TORCH_DELEGATE_ROOT/examples/inplace_input/7_npu
bash run_all.sh
```

### Option 2 Minimal port (verification only)

If TorchDelegate and the epair files are already built in the target environment:

```bash
# 1. Copy the required files
mkdir -p /destination/npu_verify
cd /path/to/torch-delegate/examples/inplace_input/7_npu

# Copy the files needed for running
cp -r main.cpp reference_data.h reference_data.cpp CMakeLists.txt /destination/npu_verify/
cp -r generated/*.epair /destination/npu_verify/epair/

# 2. Build the C++ code in the target environment
cd /destination/npu_verify
mkdir build && cd build
cmake .. -DTorchDelegate_ROOT_DIR=/path/to/torch-delegate/build/release/install \
  -DCANN_ROOT=$ASCEND_HOME_PATH \
  -DTORCH_PKG_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
make -j$(nproc)

# 3. Set the library path and run
export LD_LIBRARY_PATH=/path/to/torch-delegate/build/release/install/lib:$ASCEND_HOME_PATH/lib64:$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$LD_LIBRARY_PATH

./inplace_verify ../epair/epair_model_simple.epair simple
./inplace_verify ../epair/epair_model_chain.epair chain
./inplace_verify ../epair/epair_model_param.epair param
```

## FAQ

### Q Cannot find libtorch_delegate_backend.so
**A**: Make sure `LD_LIBRARY_PATH` contains TorchDelegate's lib directory:
```bash
export LD_LIBRARY_PATH=$TORCH_DELEGATE_ROOT/build/release/install/lib:$LD_LIBRARY_PATH
```

### Q Cannot find libc10.so
**A**: Make sure `LD_LIBRARY_PATH` contains torch's lib directory:
```bash
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')"):$LD_LIBRARY_PATH
```

### Q CMake cannot find TorchDelegate
**A**: Make sure `-DTorchDelegate_ROOT_DIR` points to the correct install directory:
```bash
cmake .. -DTorchDelegate_ROOT_DIR=/path/to/torch-delegate/build/release/install
```

### Q epair load fails with "manifest.json not found"
**A**: PyTorchStreamReader requires all files to be in a single subdirectory; the epair format has been fixed for this.
If you still hit this issue, make sure you are using the latest `torch_delegate/api.py`.

### Q NPU device unavailable
**A**: Check the NPU driver and the CANN installation:
```bash
# Check NPU devices
npu-smi info

# Check the CANN version
cat $ASCEND_HOME_PATH/version.info
```

## Related File Paths

| File | Location |
|------|------|
| TorchDelegate source | `$TORCH_DELEGATE_ROOT/csrc/` |
| Python API | `$TORCH_DELEGATE_ROOT/torch_delegate/api.py` |
| epair format constants | `$TORCH_DELEGATE_ROOT/csrc/common/epair_archive_constants.h` |
| C++ loading flow | `$TORCH_DELEGATE_ROOT/csrc/backend/delegate_session.cc` |
| Model definitions | `$TORCH_DELEGATE_ROOT/examples/inplace_input/models.py` |
