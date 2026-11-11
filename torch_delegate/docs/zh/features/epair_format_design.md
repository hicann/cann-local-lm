# Epair格式设计与实现

## API设计

### 三阶段API

torch-delegate提供三个公开API，分离转换和保存步骤：

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents:
    """
    转换 ExportedProgram 到 EpairContents

    Args:
        ep: ExportedProgram
        config: EpairConfig（可选），用于控制优化开关和算子分解白名单

    Returns:
        EpairContents: 包含 manifest、ep_json 和可执行的 ExportedProgram
    """

def epair_save(contents: EpairContents, f: str | os.PathLike) -> None:
    """
    保存 EpairContents 到 epair 文件

    Args:
        contents: EpairContents
        f: 输出文件路径
    """

def epair_load(f: str | os.PathLike) -> EpairContents:
    """
    加载 epair 文件（定义于 torch_delegate/epair/reader.py，由顶层包 re-export）

    Args:
        f: epair 文件路径

    Returns:
        EpairContents: 包含 manifest、ep_json 和可执行的 ExportedProgram
    """
```

### EpairConfig

```python
@dataclasses.dataclass
class EpairConfig:
    """epair 导出配置，控制图优化和算子分解行为"""
    remove_noop_ops: bool = True
    pattern_fusion_pass: bool = True
    whitelist_decompositions: set[Any] = dataclasses.field(
        default_factory=lambda: {
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.softmax.int,
            torch.ops.aten.log_softmax.int,
            torch.ops.aten.__and__.Tensor,
            torch.ops.aten.__or__.Tensor,
            torch.ops.aten.isfinite.default,
            torch.ops.aten.negative.default,
            torch.ops.aten.absolute.default,
            torch.ops.aten.relu6.default,
            torch.ops.aten.arccos.default,
            torch.ops.aten.arcsin.default,
            torch.ops.aten.arctan.default,
            torch.ops.aten.arccosh.default,
            torch.ops.aten.arcsinh.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.square.default,
        }
    )
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `remove_noop_ops` | `bool` | 移除无操作节点（如identity、reshape），默认 `True` |
| `pattern_fusion_pass` | `bool` | 启用模式融合优化，默认 `True` |
| `whitelist_decompositions` | `set[Any]` | 允许分解的算子集合，默认含18个aten算子；设为空 `set()` 可禁用所有分解 |

`whitelist_decompositions` 中的算子会与 `torch.export.default_decompositions()` 表取交集，仅交集内的算子才会被分解。这确保只有显式白名单中的算子才会被分解为更基础的算子。

### 使用示例

```python
from torch_delegate import export_to_epair, epair_save, epair_load
from torch.export import export

# 导出模型
ep = export(model, (inputs,))
contents = export_to_epair(ep)
epair_save(contents, "model.epair")

# 加载并执行
contents = epair_load("model.epair")
outputs = contents.ep.module()(inputs)

# 查看转换统计
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")
print(f"CANN version: {contents.manifest.cann_version}")
```

---

## 整体流程

```text
export_to_epair(ep, config) → EpairContents
  │ logger.info("Exporting ExportedProgram to epair...")
  ↓
patch_pytorch() → original_func
  │   替换 torch._higher_order_ops.auto_functionalize.do_auto_functionalize
  │   为 do_auto_functionalize_v2（幂等，_is_patched 守卫）
  │   logger.debug("PyTorch patched: do_auto_functionalize -> v2")
  ↓
TorchAirGEConverter(config)
  │   _create_compiler_config() → CompilerConfig
  │     · experimental_config.enable_view_optimize = False
  │     · experimental_config.remove_noop_ops = config.remove_noop_ops
  │     · experimental_config.pattern_fusion_pass = config.pattern_fusion_pass
  ↓
converter.convert_graph(ep) → optimized_ep
  │   decomps = {op: table[op] for op in config.whitelist_decompositions if op in default_decompositions()}
  │   _ge_graph_ascend = torchair.ge.ge_graph_ascend._optimize_and_convert(
  │       ep, config=compiler_config, custom_decompositions=decomps
  │   )
  │   _ascend_ir_map = {name: _convert_torchair_node_ir(ir) for name, ir in _ge_graph_ascend.ascend_ir_map.items()}
  │   │   _convert_torchair_node_ir: torchair _NodeIR → 本地 NodeIR（source, ops, mapping）
  │   logger.info("Conversion complete: {len(map)} nodes with ascend_ir")
  │   return _ge_graph_ascend.optimized
  ↓
api.py 后处理 node.meta["torch_fn"]
  │   for node in converted_ep.graph_module.graph.nodes:
  │       if node.op == "call_function":
  │           torch_fn = node.meta["torch_fn"]
  │           node.meta["torch_fn"] = torch_fn[:-1]  # 去掉最后一个元素（aten_ir 查找键）
  ↓
serialize(converted_ep) → artifact
  │   ep_json = json.loads(artifact.exported_program)
  ↓
EpairWriter(converter).write(ep_json) → (ep_json, total_nodes, converted_nodes)
  │   _inject_ascend_ir(ep_json):
  │     ascend_ir_map = converter.get_ascend_ir_map()
  │     geo_2_fxo: dict[str, str] = {}  # GE output name → FX output name，跨节点连线
  │     遍历 ep_json["graph_module"]["graph"]["nodes"]:
  │       _process_node(serial_node, ascend_ir_map, geo_2_fxo)
  │         · 解析 metadata.torch_fn（按 ";" 分割提取 aten_ir 查找键）
  │         · 处理 auto_functionalized_v2（从 inputs[0].arg.as_operator 取真实 op）
  │         · 查找 ascend_ir_map，获取 NodeIR
  │         · _get_ops_list: source="" 时按 type 匹配只返回 [{"attr": ...}]；否则返回完整 ops
  │         · _resolve_input_names: 用 geo_2_fxo 重写 GE input 引用为 FX name
  │         · _map_output_names: 展平 FX 输出，与 GE 输出逐一对应，跳过 CONST_* 和 None/None
  │         · 注入 serial_node["ascend_ir"] = {source, device, ops, mapping: {inputs: {...}, outputs: {...}}}
  │           device: "npu"（缺省）| "cpu"，导出期 Device 归属自动判定结果
  │           （有 PrivateUse1 kernel → npu，仅 CPU kernel → cpu，两者皆无 → 报错）
  │   logger.debug("Total graph nodes: {total}, converted: {converted}")
  ↓
writer._get_cann_version() → cann_version
  │   try: torch_npu.npu.utils.get_cann_version()
  │   except: logger.warning(...) → "unknown"
  ↓
EpairManifest(version=2, backend_name="ascend_ge",
              cann_version, total_nodes, converted_nodes, cpu_nodes)
  ↓
EpairContents(manifest, ep_json, converted_ep)
  │ logger.info("EpairContents created successfully")
  ↓
return contents
  │ finally: restore_pytorch(original_func)

---

epair_save(contents, f)
  │ logger.info("Saving epair to {f}")
  ├─ extra_files = {
  │     EPAIR_MANIFEST_FILENAME: json.dumps(asdict(manifest)),
  │     EPAIR_ANNOTATED_MODEL_FILENAME: json.dumps(ep_json)
  │   }
  │   # 常量来自 torch_delegate/epair/constants.py
  │   #   EPAIR_MANIFEST_FILENAME = "epair_manifest.json"
  │   #   EPAIR_ANNOTATED_MODEL_FILENAME = "ep_json_with_ascend_ir.json"
  └─ torch.export.save(ep, f, extra_files=extra_files)
  │   # PyTorch 会自动将 extra_files 放入归档的 extra/ 目录下
  │ logger.info("Epair saved successfully to {f}")

---

epair_load(f)   # 定义于 torch_delegate/epair/reader.py
  │ logger.info("Loading epair from {f}")
  ├─ torch.export.load(f, extra_files={EPAIR_MANIFEST_FILENAME: '', EPAIR_ANNOTATED_MODEL_FILENAME: ''})
  ├─ 解析 manifest, ep_json_with_ir
  │   logger.info("Epair loaded: backend=..., {converted}/{total} nodes")
  └─ return EpairContents(...)
```

---

## 数据结构

### NodeIR

节点级IR表示，记录单个FX节点的Ascend IR信息。定义于 `converter/base.py`：

```python
@dataclasses.dataclass
class NodeIR:
    source: str                          # 转换来源标识
                                         #   "converter": torchair 正常 GE 转换
                                         #   "":         fallback 节点（torchair 未支持）

    ops: list[dict[str, Any]]            # GE opdef 列表，每个元素包含：
                                         #   name, type, input, attr, inputDesc, outputDesc

    mapping: dict[str, Any] | None = None  # 输入/输出映射，结构为：
                                           #   {"inputs": {fx_name: ge_name, ...},
                                           #    "outputs": {fx_name: ge_name, ...}}
```

### EpairConfig

见[第一节](#epairconfig)。

### EpairManifest

epair文件的元数据，定义于 `epair/schema.py`：

```python
@dataclasses.dataclass
class EpairManifest:
    version: int             # 格式版本号，当前为 2（v1 为全 NPU 语义，向后兼容）
    backend_name: str        # 后端名称，固定 "ascend_ge"
    cann_version: str        # CANN 版本号，如 "9.2.0"；无法获取时为 "unknown"
    total_nodes: int         # EP JSON 中计算节点总数
    converted_nodes: int     # 成功转换（有 ascend_ir）的节点数
                             # converted_nodes < total_nodes 表示存在 fallback 节点
    cpu_nodes: int           # device="cpu" 的 fallback 节点数（v2 新增）
```

### EpairContents

epair的完整内容容器，定义于 `epair/schema.py`：

```python
@dataclasses.dataclass
class EpairContents:
    manifest: EpairManifest       # 元数据
    ep_json: dict[str, Any]       # 带 ascend_ir 的 EP JSON
    ep: ExportedProgram           # 可执行的 ExportedProgram
```

---

## ascend_ir格式

`ascend_ir` 是EP JSON中node对象上的可选字段，由 `EpairWriter` 注入。

### 正常转换节点（source = "converter"）

```json
{
  "target": "aten.add.Tensor",
  "inputs": [...],
  "outputs": [...],
  "metadata": {"torch_fn": "add_1"},
  "ascend_ir": {
    "source": "converter",
    "ops": [
      {
        "name": "add_1",
        "type": "Add",
        "input": ["x", "y"],
        "attr": {"alpha": {"f": 1.0}},
        "inputDesc": [
          {"dtype": "DT_FLOAT", "shape": {"dim": [2, 3]}, "layout": "FORMAT_ND", "deviceType": "ASCEND"}
        ],
        "outputDesc": [
          {"dtype": "DT_FLOAT", "shape": {"dim": [2, 3]}, "layout": "FORMAT_ND", "deviceType": "ASCEND"}
        ]
      }
    ],
    "mapping": {
      "inputs":  {"x": "add_1:0", "y": "add_1:1"},
      "outputs": {"add_1": "add_1:0"}
    }
  }
}
```

**mapping结构说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `mapping.inputs` | `dict[str, str]` | FX输入名 → GE输入名（`ge_node:port` 格式） |
| `mapping.outputs` | `dict[str, str]` | FX输出名 → GE输出名（`ge_node:port` 格式） |

### Fallback节点（source = ""）

torchair无法转换的节点，`source` 为空字符串。`EpairWriter._get_ops_list` 会按 `type` 字段匹配节点target，仅返回 `attr`：

```json
{
  "target": "aten.custom_op.default",
  "inputs": [...],
  "outputs": [...],
  "metadata": {"torch_fn": "custom_op_1"},
  "ascend_ir": {
    "source": "",
    "ops": [
      {
        "attr": {"key1": {"f": 1.0}, "key2": {"i": 2}}
      }
    ],
    "mapping": {
      "inputs":  {"input_0": "custom_op_1:0"},
      "outputs": {"output_0": "custom_op_1:0"}
    }
  }
}
```

### auto_functionalized_v2节点

PyTorch `auto_functionalized_v2` 格式的节点，writer从 `inputs[0].arg.as_operator` 提取真实op名用于查找 `ascend_ir_map`：

```json
{
  "target": "torch.ops.higher_order.auto_functionalized_v2",
  "inputs": [
    {"arg": {"as_operator": "aten.inplace_op.default"}},
    {"arg": {"as_tensor": {"name": "input_0"}}}
  ],
  "ascend_ir": {
    "source": "converter",
    "ops": [...],
    "mapping": {
      "inputs":  {...},
      "outputs": {...}
    }
  }
}
```

---

## PT2归档产物

### 文件结构

```text
model.epair (ZIP 结构，兼容 PT2 格式)
├── extra/
│   ├── epair_manifest.json               ← epair 元数据（新增）
│   └── ep_json_with_ascend_ir.json       ← 带 ascend_ir 的 EP JSON（新增）
├── models/
│   └── model.json                        ← 标准 EP JSON（不含 ascend_ir）
├── data/
│   ├── weights/
│   │   ├── model_weights_config.json
│   │   └── weight_0, weight_1, ...
│   ├── constants/
│   │   ├── model_constants_config.json
│   │   └── tensor_0, tensor_1, ...
│   └── model_sample_inputs.pt
└── byteorder                             ← 值为 "little"
```

> [!CAUTION]注意
> `extra/` 前缀由PyTorch的 `torch.export.save` 自动添加，所有 `extra_files` 参数中的文件都会被放入 `extra/` 目录。代码中常量（`epair/constants.py`）仅定义文件名，不含 `extra/` 前缀。

### 文件说明

| 文件 | 来源 | 说明 |
|------|------|------|
| `extra/epair_manifest.json` | torch-delegate | 元数据：版本、统计、CANN版本 |
| `extra/ep_json_with_ascend_ir.json` | torch-delegate | 带ascend_ir标注的完整EP JSON |
| `models/model.json` | torch.export.save | 标准EP JSON（权重引用） |
| `data/weights/` | torch.export.save | 模型权重张量 |
| `data/constants/` | torch.export.save | 常量张量 |
| `data/model_sample_inputs.pt` | torch.export.save | 示例输入 |

### manifest.json格式

```json
{
  "version": 1,
  "backend_name": "ascend_ge",
  "cann_version": "9.2.0",
  "total_nodes": 20,
  "converted_nodes": 15
}
```

---

## 场景处理

### 转换流程

图转换由 `TorchAirGEConverter.convert_graph()` 委托 `torchair.ge.ge_graph_ascend._optimize_and_convert()` 完成。torchair负责图优化、算子分解和FX → GE转换，返回 `_GeGraphAscend` 对象，其中包含：

- `optimized`：优化后的 `ExportedProgram`
- `ascend_ir_map`：`dict[str, _NodeIR]`，键为FX节点名，值为torchair内部 `_NodeIR`

```python
def convert_graph(self, ep: ExportedProgram) -> ExportedProgram:
    compiler_config = self._create_compiler_config()

    # 从白名单构建 decomps
    decomps = {}
    table = default_decompositions()
    for op in self._config.whitelist_decompositions:
        if op in table:
            decomps[op] = table[op]

    # 委托 torchair 执行转换
    self._ge_graph_ascend = _optimize_and_convert(
        ep, config=compiler_config, custom_decompositions=decomps
    )

    # torchair _NodeIR → 本地 NodeIR
    self._ascend_ir_map = {
        name: _convert_torchair_node_ir(ir)
        for name, ir in self._ge_graph_ascend.ascend_ir_map.items()
    }

    return self._ge_graph_ascend.optimized
```

### 场景标识

| source值 | 含义 | C++侧处理 |
|------------|------|------------|
| `"converter"` | torchair正常GE转换 | 划入NPU子图，GE编译执行 |
| `""`（空） | Fallback节点（torchair未支持） | 划入CPU段，fallback codegen生成自定义GE op |

### Fallback节点处理（EpairWriter侧）

`EpairWriter._get_ops_list` 对 `source=""` 的节点做特殊处理——按 `type` 字段匹配EP JSON中的target，仅返回 `attr` 字典：

```python
def _get_ops_list(self, node_ir: NodeIR, target: str) -> list[dict] | None:
    if len(node_ir.source) == 0:
        # source 为空：按 type 匹配，只返回 attr
        for ir in node_ir.ops:
            if target == ir.get("type"):
                return [{"attr": ir.get("attr", {})}]
        return None  # 未匹配到对应 op，跳过注入
    return node_ir.ops  # source 非空：返回完整 ops
```

---

## 关键技术点

### PyTorch Patch

在 `export_to_epair` 入口处patch `do_auto_functionalize` 为v2版本，确保 `ExportedProgram` 生成 `auto_functionalized_v2` 格式节点。patch实现于 `converter/patch_pytorch.py`，使用 `_is_patched` 标志保证幂等：

```python
def patch_pytorch():
    """幂等 patch，重复调用安全"""
    global _original_do_auto_functionalize, _is_patched
    if _is_patched:
        return _original_do_auto_functionalize
    _original_do_auto_functionalize = auto_func.do_auto_functionalize
    auto_func.do_auto_functionalize = do_auto_functionalize_patch
    _is_patched = True
    return _original_do_auto_functionalize

def restore_pytorch(original=None):
    """恢复原始函数，幂等"""
    global _is_patched
    if not _is_patched:
        return
    if original is not None:
        auto_func.do_auto_functionalize = original
    _is_patched = False
```

`export_to_epair` 在 `try/finally` 中调用，确保异常时也能恢复。

### 图转换委托

`TorchAirGEConverter` 不自行遍历FX图，而是委托torchair的 `_optimize_and_convert` API完成全部转换工作：

- **图优化**：`_optimize_fx`、模式融合、noop移除等由torchair内部完成
- **算子分解**：仅分解 `whitelist_decompositions` 与 `default_decompositions()` 交集内的算子
- **FX → GE转换**：由torchair内部完成，成功转换的节点 `source="converter"`，未支持的节点 `source=""`
- **结果获取**：`_GeGraphAscend.ascend_ir_map` 提供 `dict[str, _NodeIR]`，通过 `_convert_torchair_node_ir` 转为本地 `NodeIR`

### torch_fn元数据约定

torchair转换后，每个 `call_function` 节点的 `node.meta["torch_fn"]` 末尾被追加了 `aten_ir` 查找键。`api.py` 在序列化前去掉这个尾元素：

```python
for node in converted_ep.graph_module.graph.nodes:
    if node.op == "call_function":
        torch_fn = node.meta["torch_fn"]
        node.meta["torch_fn"] = torch_fn[:-1]
```

序列化后的EP JSON中，`metadata.torch_fn` 字符串格式为 `"<fx_name>;<aten_ir_key>"`。`EpairWriter._process_node` 按 `;` 分割提取 `aten_ir` 查找键，并修改 `metadata.torch_fn` 去掉 `;...` 后缀：

```python
parts = torch_fn.rsplit(";", 1)
if len(parts) > 1:
    torch_fn = parts[0]
    aten_ir = parts[1]
else:
    torch_fn = ""
    aten_ir = parts[0]
serial_node["metadata"]["torch_fn"] = torch_fn
```

### geo_2_fxo跨节点连线

`EpairWriter._inject_ascend_ir` 维护一个 `geo_2_fxo: dict[str, str]` 字典（GE output name → FX output name），用于跨节点连线：

- **输出注册**：`_map_output_names` 处理每个节点输出时，将 `ge_output → fx_output` 记入 `geo_2_fxo`
- **输入重写**：`_resolve_input_names` 遍历op的 `input` 列表，若某个GE input名出现在 `geo_2_fxo` 中，且对应的FX name属于当前节点的输入，则将其重写为FX name并记入 `mapping.inputs`

```python
def _resolve_input_names(self, ops_list, geo_2_fxo, fx_input_names) -> dict[str, str]:
    input_mapping: dict[str, str] = {}
    for op in ops_list:
        if "input" not in op:
            continue
        for idx, ge_input in enumerate(op["input"]):
            if ge_input in geo_2_fxo:
                fx_name = geo_2_fxo[ge_input]
                if fx_name not in fx_input_names:
                    continue
                input_mapping[fx_name] = ge_input
                op["input"][idx] = fx_name  # 重写为 FX name
    return input_mapping
```

### 输出映射提取

`_map_output_names` 将FX输出（可能为tensor、tensor list、sym int、scalar等多种类型）展平为一维列表，与GE输出逐一对应：

```python
def _map_output_names(self, fx_outputs, node_ir_mapping, geo_2_fxo) -> dict[str, str]:
    output_mapping: dict[str, str] = {}
    ge_outputs = node_ir_mapping["outputs"]

    # 展平 FX 输出（支持 as_tensor, as_tensors, as_sym_int, as_int, as_none 等）
    fx_flat = []
    for fx_output in fx_outputs:
        fx_flat.extend(_flatten_output(fx_output))

    # 数量必须匹配
    assert len(fx_flat) == len(ge_outputs)

    for fx_out, ge_out in zip(fx_flat, ge_outputs):
        if fx_out.startswith("CONST_"):    # 跳过常量占位符
            continue
        if fx_out == "None" and ge_out == "None":  # 跳过 None/None 对
            continue
        output_mapping[fx_out] = ge_out
        geo_2_fxo[ge_out] = fx_out         # 注册到跨节点连线表

    return output_mapping
```

`_flatten_output` 通过 `_OUTPUT_FLATTENERS` 分发表处理所有EP JSON输出类型，常量输出展平为 `"CONST_<type>"` 占位符（如 `"CONST_int"`、`"CONST_float"`）。

### ascend_ir注入

`_process_node` 是 `EpairWriter` 的核心方法，完整流程：

1. 读取 `metadata.torch_fn` 和 `target`
2. 处理 `auto_functionalized_v2`：从 `inputs[0].arg.as_operator` 取真实op名
3. 去掉 `target` 的 `torch.ops.` 前缀
4. 按 `;` 分割 `torch_fn` 提取 `aten_ir` 查找键，修改 `metadata.torch_fn`
5. 提取当前节点的FX输入名集合（`_extract_fx_input_names`）
6. 查找 `ascend_ir_map`，未找到则返回 `False`
7. `_get_ops_list` 获取ops列表（fallback节点只返回attr）
8. `_resolve_input_names` 解析输入映射并重写GE input引用
9. `_map_output_names` 建立输出映射并更新 `geo_2_fxo`
10. 注入 `serial_node["ascend_ir"] = {source, ops, mapping: {inputs, outputs}}`

---

## 日志系统

### 日志级别

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| ERROR | 转换失败终止、关键依赖缺失 | `logger.error(f"Graph conversion failed: {e}")` |
| WARNING | CANN版本获取失败、节点匹配失败 | `logger.warning(f"CANN version unavailable: {e}")` |
| INFO | API入口/出口、转换统计 | `logger.info(f"Export complete: 15/20 nodes converted")` |
| DEBUG | 配置详情、中间步骤、节点匹配 | `logger.debug(f"Config: remove_noop_ops=True")` |

### 日志格式

```text
[LEVEL] TORCH_DELEGATE(pid,executable) timestamp [file:line] thread_id message
```

示例：
```text
[INFO] TORCH_DELEGATE(12345,python) 2026-07-09 10:30:15.123 [api.py:60] 123 Exporting ExportedProgram to epair...
[DEBUG] TORCH_DELEGATE(12345,python) 2026-07-09 10:30:16.234 [writer.py:211] 123 Try to get key: add_1
[INFO] TORCH_DELEGATE(12345,python) 2026-07-09 10:30:17.001 [api.py:76] 123 Export complete: 15/20 nodes converted
```

### 使用方式

```python
from torch_delegate.core.utils import logger

logger.error("Critical error message")
logger.warning("Warning message")
logger.warning_once("Warning that only prints once")  # 防止重复（lru_cache）
logger.info("Info message")
logger.debug("Debug message")
```

### 环境变量控制

```bash
# 默认：WARN 级别
python your_script.py

# 启用 DEBUG 级别
TORCH_DELEGATE_DEBUG=1 python your_script.py
```

---

## 方案优势

| 优势 | 说明 |
|------|------|
| **三阶段API** | 分离转换和保存，用户可检查转换结果后再保存 |
| **source标识** | 通过source字段区分converter/fallback两种节点类型 |
| **完整opdef** | ops字段包含完整GE opdef信息（name, type, input, attr, inputDesc, outputDesc） |
| **输入输出映射** | mapping字段记录FX ↔ GE输入输出对应关系，支持跨节点连线 |
| **torchair委托** | 图转换委托torchair `_optimize_and_convert`，复用成熟的优化和转换逻辑 |
| **白名单分解** | `whitelist_decompositions` 精确控制算子分解范围 |
| **日志系统** | 完整日志记录，支持环境变量控制调试级别 |
| **CANN版本** | manifest记录CANN版本，便于版本一致性检查 |
| **兼容性** | 复用torch.export.save/load，兼容PT2工具链 |

---

## 与dynamo_export对比

| 方面 | dynamo_export (torchair) | export_to_epair (torch-delegate) |
|------|--------------------------|----------------------------------|
| **输入来源** | `torch.compile` → dynamo → gm | `ExportedProgram` → ep.graph_module |
| **图处理** | torchair内部优化 | torchair `_optimize_and_convert`（委托） |
| **错误处理** | 找不到converter报错 | torchair标记 `source=""`，C++侧fallback |
| **IR记录** | 后处理提取GE graph | torchair返回 `ascend_ir_map`，writer注入EP JSON |
| **输出** | `.air` 文件（二进制GE图） | `.epair` 文件（PT2归档+ ascend_ir） |
| **权重处理** | 自定义外部化 | PT2权重管理（torch.export.save） |
| **API设计** | 单一 `dynamo_export` | 三API：`export_to_epair` + `epair_save/load` |
| **日志系统** | torchair logger | torch_delegate logger（独立环境变量） |
| **运行时** | 仅CANN | 灵活（C++侧决定：NPU/CPU分区） |
| **CANN版本** | 不记录 | manifest中记录 |
| **算子分解** | torchair内部控制 | `whitelist_decompositions` 白名单控制 |

---

## 代码结构

```text
torch_delegate/
├── __init__.py                       ← 导出公开 API + 数据类
├── api.py                            ← export_to_epair, epair_save（epair_load 从 reader 导入）
├── core/
│   ├── __init__.py                   ← 导出 logger
│   └── utils.py                      ← 日志系统实现（_MillisecFormatter, warning_once）
├── converter/
│   ├── __init__.py                   ← 导出 NodeIR, NodeIRConverter, TorchAirGEConverter
│   ├── base.py                       ← NodeIR 数据类 + NodeIRConverter ABC
│   ├── torchair_converter.py         ← TorchAirGEConverter（委托 _optimize_and_convert）
│   └── patch_pytorch.py              ← do_auto_functionalize → v2 patch（幂等）
└── epair/
    ├── __init__.py                   ← 导出 schema/config/writer/reader
    ├── schema.py                     ← EpairManifest, EpairContents
    ├── config.py                     ← EpairConfig（3 字段）
    ├── constants.py                  ← EPAIR_MANIFEST_FILENAME, EPAIR_ANNOTATED_MODEL_FILENAME
    ├── writer.py                     ← EpairWriter（注入 ascend_ir，跨节点连线）
    └── reader.py                     ← epair_load 函数
```

### 各模块职责

| 模块 | 职责 |
|------|------|
| `api.py` | 顶层入口，组合converter + writer，管理patch生命周期 |
| `core/utils.py` | 日志系统，`TORCH_DELEGATE_DEBUG` 环境变量控制级别 |
| `converter/base.py` | `NodeIR` 数据类（单一数据源）+ `NodeIRConverter` 抽象基类 |
| `converter/torchair_converter.py` | 生产converter，委托torchair `_optimize_and_convert` |
| `converter/patch_pytorch.py` | 幂等patch `do_auto_functionalize` → v2 |
| `epair/schema.py` | `EpairManifest`、`EpairContents` 数据类 |
| `epair/config.py` | `EpairConfig` 配置类 |
| `epair/constants.py` | epair 文件名常量（不含 `extra/` 前缀） |
| `epair/writer.py` | `EpairWriter`，注入 ascend_ir 到 EP JSON |
| `epair/reader.py` | `epair_load` 函数，加载 .epair 文件 |

### ascend_ir.device 与 manifest v2（CPU/NPU 混合调度）

自格式 v2 起，每个节点的 `ascend_ir` 新增可选 `device` 字段，记录导出期
Device 归属自动判定结果：

```json
"ascend_ir": {
  "source": "",
  "device": "cpu",
  "ops": [],
  "mapping": {"inputs": {}, "outputs": {}}
}
```

- 判定规则（无配置项）：fallback 节点的 target 存在 `PrivateUse1` kernel →
  `"npu"`；仅有 `CPU` kernel 或仅注册 `CompositeImplicitAutograd`（分解实现，
  回调时传 CPU 张量）→ `"cpu"`；以上皆无 → 导出报错。
- 兼容性：`device` 缺失时按 `"npu"` 处理，等价于 v1 行为；C++ 侧对 manifest
  v1 保持全 NPU 语义。
- 严格校验开关：`EpairConfig(require_aten_on_npu=True)` 时，aten 命名空间算子
  被判到 CPU 执行会在导出阶段直接报错（完整命中清单），用于 CI / 性能基线看护。
- 归属可诊断性：导出结束打印完整归属报告（NPU/CPU 双侧清单，CPU 侧按
  namespace 分级告警）；C++ 加载期按 `device` 与 target namespace 复述 CPU
  节点分布，只告警不报错。归属判定记录仅在导出期内存中持有，**不随 epair
  落盘**（避免扩展 `extra/` 目录引入格式演进负担）。
- 加载期不做能力校验：不引入本地 CANN 版本/能力校验门禁；算子未注册、环境
  不支持 HostCpu 等能力问题由 GE 编译期报错自然暴露。

归属判定与混合调度的完整设计见
`docs/zh/features/cpu_npu_mixed_scheduling_hostcpu_20261027.md`。
