# 从源码构建ExecuTorch-NPU

本文只回答两件事：构建需要什么，以及如何把源码构建成包含Python AOT Backend和
C++ Backend SDK的平台wheel。模型导出见 [模型导出](export.md)，C++ 应用编译和
设备执行见 [C++ 运行](runtime.md)。

## 构建范围

默认命令：

```bash
bash build.sh
```

默认构建产生：

```text
output/package/executorch_npu-<version>-<python>-<abi>-<platform>.whl
build/release/install/include/executorch_npu/
build/release/install/lib/libexecutorch_npu_backend.a
build/release/install/lib/libexecutorch_npu_manifest.a
```

默认构建不生成Runner，也不执行模型导出。

## 构建依赖

### 工具依赖

| 依赖 | 要求 | 用途 |
|---|---|---|
| Python | 3.10或更高 | 打包和构建检查 |
| pip、setuptools、wheel、packaging | 当前环境可用 | 生成wheel |
| CMake | 3.24–3.x | 配置和编译C++ targets |
| C++ 编译器 | C++17；当前验证GCC 9.4 | 编译Backend |
| OpenSSL开发包 | 提供Crypto | Manifest SHA-256 |
| nlohmann_json | 仓库固定3.11.3 | Manifest JSON解析 |

### 软件依赖

| 依赖 | 用途 |
|---|---|
| ExecuTorch 1.5.0 full wheel | Backend头文件、`libexecutorch.so`、CMake package和版本信息 |
| CANN Toolkit开发文件 | ACL头文件；Runner/应用链接时还提供 `libascendcl.so` |

构建不依赖Torch、torchao、TorchAir、torch_npu、torch-delegate或NPU设备。
ExecuTorch必须是full wheel；minimal wheel缺少所需C++ SDK。

## 准备构建环境

建议为源码构建创建独立环境：

```bash
python3 -m venv /path/to/executorch_npu-build-env
source /path/to/executorch_npu-build-env/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  "setuptools>=68" \
  "wheel>=0.40" \
  "packaging>=23" \
  "cmake>=3.24,<4"
```

Backend构建只消费ExecuTorch wheel中的C++ 开发文件，因此使用：

```bash
python -m pip install --no-deps "executorch==1.5.0"
```

`--no-deps` 仅适用于这个隔离构建环境。模型导出环境不能照搬该命令。

加载已安装的CANN环境：

```bash
source /path/to/cann/set_env.sh
```

软件构建需要CANN开发文件，但不需要实际NPU。`build.sh` 使用
`ASCEND_HOME_PATH` 定位CANN，不要求用户逐个配置头文件路径。

## 执行构建

```bash
cd /path/to/executorch_npu
bash build.sh
```

Debug构建：

```bash
bash build.sh --debug
```

脚本执行顺序：

```text
检查 Python、CMake、CANN 和 ExecuTorch full wheel
→ 配置并编译 Backend 静态库
→ 执行 C++ 单元测试
→ 安装头文件和静态库到 build/<type>/install
→ 将 Python 包和 C++ SDK 打入平台 wheel
```

`build.sh` 不创建虚拟环境，也不修改当前环境中的依赖。

## 检查和安装产物

确认实际wheel文件名：

```bash
ls -1 output/package/executorch_npu-*.whl
```

将对应wheel安装到目标环境：

```bash
python -m pip install /path/to/executorch_npu-<version>-<python>-<abi>-<platform>.whl
```

只需要提取C++ SDK、不安装Python导出依赖时，可以在已经安装ExecuTorch 1.5
full wheel的环境中使用：

```bash
python -m pip install --no-deps /path/to/executorch_npu-*.whl
python -c "from executorch_npu import sdk_root; print(sdk_root())"
```

wheel中的 `compatibility.json` 会记录实际构建使用的ExecuTorch版本、commit、CANN、
Python和编译器。该文件由构建过程生成，不应手工修改。

## 构建失败时

常见构建错误见 [故障排查：源码构建](troubleshooting.md#源码构建)。版本范围见
[支持版本](supported_versions.md)。


## 开发质量检查

在已安装开发与示例依赖的环境中，从仓库根目录运行：

```bash
python -m pip install -e '.[test]'
ruff check python tests examples scripts
ruff format --check python tests examples scripts
python scripts/generate_manifest_schema.py --check
CANN_MANIFEST_TEST_BINARY="$PWD/build/release/executorch_npu_manifest_test" \
    python -m pytest tests/python -q
ctest --test-dir build/release --output-on-failure
```

安装了torch_npu、但不希望Torch自动导入设备扩展时，可设置
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`。模型测试仍会按需显式导入算子注册；
需要NPU的测试通过设备可用性判断执行或跳过。CPU eager缓存使用PyTorch
参考更新，CPU/FakeTensor导出保留原生NPU算子；融合GDN测试使用
BF16和128维state head。

本轮质量整理参考PyTorch 2.7.1的三个具体模式：
Dynamo在编译入口保护全局状态（`torch/_dynamo/convert_frame.py`），
后端显式维护初始化状态（`torch/cuda/__init__.py`），以及配置上下文在异常退出时恢复
（`torch/utils/_config_module.py`）。ExecuTorch-NPU使用自己的同步与状态管理，
不依赖这些PyTorch私有实现。


### Python风格

与参考的PyTorch 2.7.1配置一致，使用88列、双引号、导入分组及导入后的两行空行。
`requirements-dev.txt` 固定Ruff版本，检查入口见上文。

- 函数、方法、普通变量和可变模块状态使用 `snake_case`；内部辅助函数和内部状态加 `_` 前缀。
- 类使用 `CapWords`，常量使用 `UPPER_CASE`。保留上游Transformers的
  `Qwen3_5*` 模型命名，以维持配置、checkpoint和既有接口的一致性。
- 类管理有明确生命周期的状态或资源，例如 `OfflineCompiler`；无状态转换和校验使用函数。
- 公共及模块边界声明类型；异常转换使用明确的异常链；测试命名描述可观察行为。
- 可变的编译锁与会话实例分别命名为 `compile_lock`、`offline_compiler`。
