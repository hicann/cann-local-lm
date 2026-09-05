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
source ${CANN_SET_ENV} 2>/dev/null || source ${ASCEND_HOME_PATH}/set_env.sh 2>/dev/null

# Python
python3 --version  # 需 >= 3.10

# 工具链
cmake --version | head -1  # 需 >= 3.18
gcc --version | head -1    # 需 >= 8 且 < 14

# torch（在 .venv 中）
source ${TORCH_DELEGATE_HOME}/.venv/bin/activate 2>/dev/null
python3 -c "import torch; print(torch.__version__)"
```

## 步骤

### 1. 构建 torch-delegate-harmony
下载路径：https://gitcode.com/shengnan666/torch-delegate

```bash
source ${CANN_SET_ENV}

# 清理旧 CMake 缓存（CANN 版本变更时必须）
rm -rf ${TORCH_DELEGATE_HOME}/build/release

# 一键构建
cd ${TORCH_DELEGATE_HOME} && bash build.sh
```

产出物：
- `output/package/torch_delegate-<version>-linux-<arch>-release.run`
- `output/package/torch_delegate-<version>-py3-none-any.whl`

### 2. 安装 torch_delegate whl

```bash
source ${TORCH_DELEGATE_HOME}/.venv/bin/activate
pip install ${TORCH_DELEGATE_HOME}/output/package/torch_delegate-*-py3-none-any.whl
```

### 3. 安装 torchair 依赖

```bash
source ${TORCH_DELEGATE_HOME}/.venv/bin/activate

# clone
git clone --recursive https://gitcode.com/Ascend/torchair.git ${TORCHAIR_ROOT}

# configure（不指定 Ascend SDK 路径）
cd ${TORCHAIR_ROOT}
TARGET_PYTHON_PATH=$(which python3) NO_ASCEND_SDK=1 bash configure

# build & install
bash build_and_install.sh $(dirname $(which python3))/../lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages

# 手动复制 torchair Python 包到 venv（build_and_install.sh 可能未自动安装）
SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r ${TORCHAIR_ROOT}/build/dist/torchair ${SITE_PKG}/ 2>/dev/null || true
cp -r ${TORCHAIR_ROOT}/python/torchair ${SITE_PKG}/ 2>/dev/null || true

# 安装缺失依赖
pip install packaging
```

### 4. 验证导入

```bash
source ${CANN_SET_ENV}
source ${TORCH_DELEGATE_HOME}/.venv/bin/activate
export LD_LIBRARY_PATH=${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
python3 -c "from torch_delegate import export_to_epair, epair_save, epair_load; print('OK')"
```

### 5. 导出 epair 文件

**无 NPU 设备时（stub 模式）：**

```bash
source ${CANN_SET_ENV}
source ${TORCH_DELEGATE_HOME}/.venv/bin/activate
export LD_LIBRARY_PATH=${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
```

**有 NPU 设备时：**

```bash
source ${CANN_SET_ENV}
source ${TORCH_DELEGATE_HOME}/.venv/bin/activate
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

epair_save(contents, "${TORCH_DELEGATE_HOME}/model.epair")

# 加载验证
contents2 = epair_load("${TORCH_DELEGATE_HOME}/model.epair")
outputs = contents2.ep.module()(example_inputs[0])
print(f"Output shape: {outputs.shape}")
```

## 常见问题

### CANN 版本过旧
- 检查: `cat ${ASCEND_HOME_PATH}/compiler/version.info`
- 更新版本: https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260708000326093/
- 安装: `chmod +x Ascend-cann-toolkit_*_linux-$(arch).run && ./Ascend-cann-toolkit_*_linux-$(arch).run --install --install-path=<ascend_install_path> --quiet`

### ACL set device id failed
原因：无 NPU 硬件驱动。设置 stub 模式：
```bash
export LD_LIBRARY_PATH=${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
```

### torchair 导入失败
- 确保已 `pip install packaging`
- 确保 torch 已安装且版本 >= 2.6.0
