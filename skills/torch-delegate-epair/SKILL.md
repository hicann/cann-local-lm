---
name: torch-delegate-epair
description: |
  torch-delegate-harmony 构建及 epair 导出技能。当用户提到以下场景时使用：
  - 构建 torch-delegate、torch_delegate、torch-delegate-harmony 项目
  - 安装/编译 torch_delegate Python whl 包
  - 导出/生成 epair 模型文件
  - 安装 torchair、torchair 依赖
  - 执行 build.sh、build_epair.md
  - 配置 CANN 环境、升级 CANN 版本
  - 模型导出为 .epair 格式供 C++ delegate 加载
version: 1.0.0
metadata:
  author: assistant
  category: build-deployment
allowedTools:
  - read
  - write
  - bash
  - task
  - question
  - websearch
  - webfetch
---

# torch-delegate-harmony 构建及 epair 导出技能

## 前置条件检查

执行任何操作前，必须检查以下环境：

```bash
# CANN 环境
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<未设置>}"
source /home/developer/Ascend/cann/set_env.sh 2>/dev/null || source ${ASCEND_HOME_PATH}/set_env.sh 2>/dev/null

# Python
python3 --version  # 需 >= 3.10

# 工具链
cmake --version | head -1  # 需 >= 3.18
gcc --version | head -1    # 需 >= 8 且 < 14

# torch（在 .venv 中）
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate 2>/dev/null
python3 -c "import torch; print(torch.__version__)"
```

## 步骤

### 1. 构建 torch-delegate-harmony

```bash
source /home/developer/Ascend/cann/set_env.sh

# 清理旧 CMake 缓存（CANN 版本变更时必须）
rm -rf /mnt/workspace/torch-delegate-harmony/build/release

# 一键构建
cd /mnt/workspace/torch-delegate-harmony && bash build.sh
```

产出物：
- `output/package/torch_delegate-<version>-linux-<arch>-release.run`
- `output/package/torch_delegate-<version>-py3-none-any.whl`

### 2. 安装 torch_delegate whl

```bash
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate
pip install /mnt/workspace/torch-delegate-harmony/output/package/torch_delegate-*-py3-none-any.whl
```

### 3. 安装 torchair 依赖

```bash
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate

# clone
git clone --recursive https://gitcode.com/Ascend/torchair.git /mnt/workspace/torchair

# configure（不指定 Ascend SDK 路径）
cd /mnt/workspace/torchair
TARGET_PYTHON_PATH=$(which python3) NO_ASCEND_SDK=1 bash configure

# build & install
bash build_and_install.sh $(dirname $(which python3))/../lib/python3.12/site-packages

# 安装缺失依赖
pip install packaging
```

### 4. 验证导入

```bash
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate
python3 -c "from torch_delegate import export_to_epair, epair_save, epair_load; print('OK')"
```

### 5. 导出 epair 文件

**无 NPU 设备时（stub 模式）：**

```bash
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
```

**有 NPU 设备时：**

```bash
source /home/developer/Ascend/cann/set_env.sh
source /mnt/workspace/torch-delegate-harmony/.venv/bin/activate
```

**执行导出：**

```python
import torch
import torch.nn as nn
from torch.export import export
from torch_delegate import export_to_epair, epair_save, epair_load

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    def forward(self, x):
        return torch.relu(self.linear(x))

model = SimpleModel()
example_inputs = (torch.randn(1, 10),)
ep = export(model, example_inputs)
contents = export_to_epair(ep)

epair_save(contents, "/mnt/workspace/torch-delegate-harmony/model.epair")

# 加载验证
contents2 = epair_load("/mnt/workspace/torch-delegate-harmony/model.epair")
outputs = contents2.ep.module()(example_inputs[0])
print(f"Output shape: {outputs.shape}")
```

## 常见问题

### CANN 版本过旧
- 检查: `cat ${ASCEND_HOME_PATH}/compiler/version.info`
- 下载每日构建: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master
- 安装: `chmod +x Ascend-cann-toolkit_*_linux-$(arch).run && ./Ascend-cann-toolkit_*_linux-$(arch).run --install --install-path=/home/developer/Ascend --quiet`

### ACL set device id failed
原因：无 NPU 硬件驱动。设置 stub 模式：
```bash
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
```

### torchair 导入失败
- 确保已 `pip install packaging`
- 确保 torch 已安装且版本 >= 2.6.0
