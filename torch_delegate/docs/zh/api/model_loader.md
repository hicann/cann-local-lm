# EpairModelLoader（Python API）<a id="section_python_epair_model_loader"></a>

## 概览

`torch_delegate.EpairModelLoader` 加载 `.epair` 归档并构建 GE 图，供 GE Offline_compile 使用。该类是 C++ `td::EpairModelLoader` 的 Python 封装（经 ctypes + C ABI），仅覆盖「取 GE 图」离线编译路径；执行态接口（`CompileAndLoad` / `RunModel` 等）请使用 [C++ API](epair_model_loader.md)。

**导入方式：**

```python
from torch_delegate import EpairModelLoader, EpairModelLoaderError
```

## 类 `EpairModelLoader`

```python
class torch_delegate.EpairModelLoader(epair_file)
```

构造时通过 C ABI 创建 C++ `td::EpairModelLoader` 并完成归档解析与 GE 图构建（不编译）；`get_ge_graph()` 每次返回独立持有的 GE 图副本。实例生命周期由本对象独占管理（构造 / `close()` / 析构三态），同一实例不承诺多线程并发安全。

### `__init__(epair_file)`

加载 backend 库、创建 C++ loader，并完成 `.epair` 解析与 GE 图构建（不编译）；失败抛异常。

**签名：**

```python
def __init__(self, epair_file: str | os.PathLike[str]) -> None: ...
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `epair_file` | `str \| os.PathLike[str]` | `.epair` 文件路径，必须非空 |

**异常：**

| 异常类型 | 触发条件 |
|---------|---------|
| `TypeError` | `epair_file` 不是 `str` 或 `os.PathLike[str]` |
| `ValueError` | `epair_file` 为空路径，或 C++ 侧返回 `INVALID_PARAM` |
| `EpairModelLoaderError` | backend 库加载失败（含老版本 whl 无 C ABI 的场景）、GE 未初始化、归档损坏/图构建失败（`FAILED`）、内部错误（`INTERNAL_ERROR`） |

> 构造 loader 前必须先完成 GE 初始化（`ge.offline_compile.build_initialize()`）；未初始化时 C ABI 通常返回 `FAILED`，表现为 `EpairModelLoaderError`。

异常消息包含状态码名与线程局部错误串，可明确定位失败阶段。

### `get_ge_graph()`

获取内部 GE 图的独立副本，可重复调用，副本间互不影响。

**签名：**

```python
def get_ge_graph(self) -> ge.graph.Graph: ...
```

**返回值：** `ge.graph.Graph` — CANN 随包发布的 GraphEngine Python API 图对象，可直接交由 `ge.offline_compile.build_model` 离线编译。

**异常：**

| 异常类型 | 触发条件 |
|---------|---------|
| `EpairModelLoaderError` | 实例已 `close` 后取图、GraphEngine Python API 不可用/空图创建失败、API 不提供原生句柄，或 C++ 侧图拷贝失败 |

### `close()`

显式销毁 C++ loader，释放 `.epair` 解析与 GE 图构建产物；重复调用幂等（no-op）。

**签名：**

```python
def close(self) -> None: ...
```

### `with` 上下文管理器

支持 `with` 语句：退出上下文时自动调用 `close()`；析构函数兜底调用 `close()`，保证资源无泄漏。

**签名：**

```python
def __enter__(self) -> EpairModelLoader: ...
def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
```

## 示例：初始化 GE 并离线编译

```python
import torch_delegate
from ge.offline_compile import (
    build_finalize,
    build_initialize,
    build_model,
    save_model,
)

build_initialize({"ge.socVersion": "Ascend910B", "ge.graphRunMode": "0"})
try:
    with torch_delegate.EpairModelLoader("model.epair") as loader:
        ge_graph = loader.get_ge_graph()
        model = build_model(ge_graph)
        save_model("./model", model)
finally:
    build_finalize()
```

**注意事项：**

- 示例给出可运行的最小流程：先 `build_initialize()` 完成 GE 初始化，再构造 `EpairModelLoader`；`ge.socVersion` 须与目标部署 SoC 一致，流程结束后调用 `build_finalize()` 释放 GE 资源
- `get_ge_graph()` 返回 CANN 随包发布的 GraphEngine Python API 对象（`ge.graph.Graph`，经空图创建与原生指针获取对接 backend C ABI）；该 API 随 CANN 安装于 `python/site-packages`，未安装 CANN 或 `ge` 模块不可导入时抛出 `EpairModelLoaderError` 并给出明确提示
- 同一实例不承诺多线程并发安全（与 C++ 侧约定一致）；错误串按线程隔离
- 旧版本 whl 未导出 C ABI 符号，调用时会在加载阶段抛出清晰的 `EpairModelLoaderError`，请重新安装匹配版本的 torch_delegate
