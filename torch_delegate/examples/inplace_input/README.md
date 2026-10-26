# Inplace Input 验证用例

验证带有 inplace 操作的模型在 `torch.export.save/load` 与 `torch_delegate` epair 两种导出方式下的行为一致性。

---

## 验证内容

| 验证项 | 说明 |
|-------|------|
| Inplace 输入修改 | 输入 tensor 被 inplace 修改的行为与 Eager 一致 |
| 输出精度 | 输出数值与 Eager 一致（误差 < 1e-5） |
| FX Graph 捕获 | 正确捕获 inplace 算子 |

---

## 目录结构

```
inplace_input/
├── models.py                 # 模型定义 (3种 inplace 模型)
├── utils.py                  # 工具函数
├── 1_export_fx_graph.py      # Phase 1: torch.export 导出 FX 图
├── 2_verify_export.py        # Phase 2: 验证 torch.export load (跨进程)
├── 3_aoti_compile.py         # Phase 3: AOTI 编译（默认跳过，需 gcc11+）
├── 4_verify_aoti/            # Phase 4: C++ AOTI 验证（默认跳过，需 gcc11+）
│   ├── main.cpp              # C++ AOTI 加载与执行
│   ├── verify_aoti.py        # Python 侧对比脚本（调用 C++ 二进制）
│   └── CMakeLists.txt        # 构建配置
├── 5_compile_fx_graph.py     # Phase 5: torch.compile FX 图对比观察
├── 7_npu/                    # Phase 7: NPU epair 导出与验证 (C++ + Python)
│   ├── export_epair.py       # epair 导出脚本
│   ├── main.cpp              # C++ GE 加载和执行
│   └── README.md             # NPU 验证详细文档
├── run.sh                    # 一键运行脚本
└── generated/                # 产物目录
```

---

## 环境要求

1. **torch 2.9.0+cpu**
2. **torch_npu** + **torchair**（NPU 阶段需要）
3. **torch-delegate** 已构建
4. **CANN 9.0.0+**（NPU 阶段需要）

---

## 快速开始

```bash
# 运行所有阶段
bash run.sh
```

`run.sh` 自动完成：
1. torch.export 导出并验证 FX 图（Phase 1-2）
2. torch.compile FX 图对比观察（Phase 5）
3. NPU epair 导出与 C++ GE 验证（Phase 7，需 NPU 环境）

> **注意**：Phase 3-4（AOTI 编译与验证）依赖 gcc11+ 高版本 GLIBCXX 库，默认跳过。如需在 gcc11+ 环境手动验证，将 `run.sh` 中 `SKIP_AOTI` 改为 `false`。

---

## 测试模型

| 模型 | Inplace 操作 |
|------|-------------|
| SimpleInplaceModel | `x.add_(y); x.mul_(scale)` |
| ChainInplaceModel | `x.add_(1); x.mul_(2); x.div_(4)` |
| ParamInplaceModel | `x.add_(bias); x.mul_(scale)` |

---

## 运行阶段

| Phase | 设备 | 说明 | 默认状态 |
|-------|------|------|---------|
| 1 | CPU | torch.export 导出并保存 FX 图 | 运行 |
| 2 | CPU | 跨进程验证 torch.export load | 运行 |
| 3 | CPU | AOTI 编译 | 跳过（需 gcc11+） |
| 4 | CPU | 跨进程验证 AOTI load | 跳过（需 gcc11+） |
| 5 | CPU | torch.compile FX 图对比观察 | 运行 |
| 7 | NPU | epair 导出与 C++ GE 验证（见 `7_npu/`） | 需 NPU 环境 |

---

## 产物命名

格式：`{method}_{content}_{model}.{ext}`

| 产物 | 说明 |
|------|------|
| `export_model_simple.pt2` | torch.export 导出的模型 |
| `export_fx_graph_simple.txt` | torch.export FX 图结构 |
| `dynamo_fx_graph_simple.txt` | Dynamo 捕获的 FX 图 |
| `epair_model_simple.epair` | torch_delegate epair 文件 |
| `epair_manifest_simple.json` | epair manifest 信息 |

---

## FX Graph 对比

| 方式 | Inplace 形式 |
|------|-------------|
| torch.export | `aten.add_.Tensor`, `aten.mul_.Tensor` |
| Dynamo | `call_method: add_`, `mul_` |

**关键发现**：torch.export 和 epair 保留原始 inplace 算子，运行结果与 Eager 模式一致。

---

## 判断标准

- **Inplace 修改**: `max|input_modified - eager_input_modified| < 1e-5`
- **输出精度**: `max|output - eager_output| < 1e-5`
