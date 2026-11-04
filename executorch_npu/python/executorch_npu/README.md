# `executorch_npu` Python package

简体中文 | [English](README_en.md)

该包作为ExecuTorch的CANN backend插件，在顶层保留三个接入点。
公共Python API由 `__init__.py` 统一导出，内部函数按编译、图改写、
产物构造和辅助能力分类。

## 目录层级

```text
executorch_npu/
├── __init__.py                  # 公共 Python API
├── backend.py                   # Backend 接入、多方法编排与 staging 生命周期
├── partitioner.py               # 分区接入
├── edge_compile_config.py       # Edge 导出配置接入
├── compiler/
│   ├── compile_spec.py          # 编译选项及 CompileSpec 编解码
│   ├── pipeline.py              # 单分区编译，返回 OM / 权重路径
│   ├── delegate_adapter.py      # ExportedProgram → EPAIR → GE，含兼容适配
│   └── offline.py               # GE 离线编译生命周期
├── passes/                     # 各 pass 独立成文件，保持图改写顺序
│   ├── edge_ops_pass.py
│   ├── static_split_pass.py
│   └── edge_copy_pass.py
├── artifacts/
│   ├── manifest.py              # Pydantic 协议模型及 ASCD 编码
│   ├── builder.py               # named data、完整 Manifest 和产物集合检查
│   ├── io_contract.py           # 图参数布局及 manifest I/O 构造
│   └── state.py                 # 状态声明、图关联及单图/跨方法状态规则
└── shared/
    ├── constants.py
    ├── symbolic_shape.py        # 编译和 manifest 共用的符号维度范围解析
    ├── diagnostics.py           # Edge、编译阶段及状态映射诊断
    └── sdk.py                   # 已安装 C++ SDK 的资源定位
```

## 主要流程

```text
ExecuTorch → AscendPartitioner → AscendBackend
                                    │
                                    ├─ compiler.compile_partition()
                                    │    passes → EPAIR → GE → OM + 权重文件
                                    │    artifacts.builder 注册产物并返回 Manifest
                                    │
                                    ├─ 多方法：check_manifest_set()
                                    │    delegate 身份检查 + 共享状态检查
                                    │
                                    └─ Manifest.to_bytes() → PreprocessResult
                                         ExecuTorch 写入 PTE / PTD
```

`backend.py` 拥有staging生命周期：编译产物由 `builder.py` 读取并注册为
named data后才释放临时目录。`compiler/` 不注册named data，也不构造Manifest。
EPAIR loader在离线编译前关闭；fallback禁用范围覆盖GE图提取和离线编译。

`artifacts/io_contract.py` 统一计算最终参数索引，状态映射消费同一份布局。
`artifacts/state.py` 区分导出时带来源信息的状态声明和委托后的最终状态绑定。
构造manifest时读取 `ExportedProgram` 属于产物映射职责，不需要经过额外的通用IR。

Pydantic类型及字段约束集中于 `manifest.py`；图关联、状态策略和产物集合规则
由各自的构造函数检查。跨方法检查完成后再编码，保持协议v1和现有产物语义。

## 公共API

推荐从包根导入，内部迁移不改变这些导入方式：

```python
from executorch_npu import (
    AscendBackend,
    AscendCompileOptions,
    AscendPartitioner,
    Manifest,
    aten_edge_compile_config,
    build_state_io_contract,
    sdk_root,
)
```

原 `executorch_npu.compile_spec`、`executorch_npu.manifest`、
`executorch_npu.sdk` 以及旧 `compiler` 内部路径的直接引用，应改为包根公共API
或上面列出的新模块路径。
