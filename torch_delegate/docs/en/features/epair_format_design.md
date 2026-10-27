# Epair Format Design and Implementation

## API Design

### Three-Stage API

torch-delegate provides three public APIs that separate the conversion and save steps:

```python
def export_to_epair(
    ep: ExportedProgram,
    config: EpairConfig = EpairConfig()
) -> EpairContents:
    """
    Convert an ExportedProgram to EpairContents

    Args:
        ep: ExportedProgram
        config: EpairConfig (optional), controls optimization switches and operator decomposition whitelist

    Returns:
        EpairContents: contains manifest, ep_json, and an executable ExportedProgram
    """

def epair_save(contents: EpairContents, f: str | os.PathLike) -> None:
    """
    Save EpairContents to an epair file

    Args:
        contents: EpairContents
        f: output file path
    """

def epair_load(f: str | os.PathLike) -> EpairContents:
    """
    Load an epair file (defined in torch_delegate/epair/reader.py, re-exported by the top-level package)

    Args:
        f: epair file path

    Returns:
        EpairContents: contains manifest, ep_json, and an executable ExportedProgram
    """
```

### EpairConfig

```python
@dataclasses.dataclass
class EpairConfig:
    """epair export configuration, controls graph optimization and operator decomposition behavior"""
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

| Field | Type | Description |
|------|------|------|
| `remove_noop_ops` | `bool` | Remove no-op nodes (such as identity, reshape); default `True` |
| `pattern_fusion_pass` | `bool` | Enable pattern fusion optimization; default `True` |
| `whitelist_decompositions` | `set[Any]` | Set of operators allowed to be decomposed; defaults to 18 aten operators; set to an empty `set()` to disable all decompositions |

Operators in `whitelist_decompositions` are intersected with the `torch.export.default_decompositions()` table, and only operators in the intersection are decomposed. This ensures that only operators explicitly whitelisted are decomposed into more basic operators.

### Usage Example

```python
from torch_delegate import export_to_epair, epair_save, epair_load
from torch.export import export

# Export the model
ep = export(model, (inputs,))
contents = export_to_epair(ep)
epair_save(contents, "model.epair")

# Load and execute
contents = epair_load("model.epair")
outputs = contents.ep.module()(inputs)

# Inspect conversion statistics
print(f"Converted {contents.manifest.converted_nodes}/{contents.manifest.total_nodes} nodes")
print(f"CANN version: {contents.manifest.cann_version}")
```

---

## Overall Flow

```text
export_to_epair(ep, config) → EpairContents
  │ logger.info("Exporting ExportedProgram to epair...")
  ↓
patch_pytorch() → original_func
  │   Replace torch._higher_order_ops.auto_functionalize.do_auto_functionalize
  │   with do_auto_functionalize_v2 (idempotent, guarded by _is_patched)
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
  │   │   _convert_torchair_node_ir: torchair _NodeIR → local NodeIR (source, ops, mapping)
  │   logger.info("Conversion complete: {len(map)} nodes with ascend_ir")
  │   return _ge_graph_ascend.optimized
  ↓
api.py post-processes node.meta["torch_fn"]
  │   for node in converted_ep.graph_module.graph.nodes:
  │       if node.op == "call_function":
  │           torch_fn = node.meta["torch_fn"]
  │           node.meta["torch_fn"] = torch_fn[:-1]  # remove the last element (aten_ir lookup key)
  ↓
serialize(converted_ep) → artifact
  │   ep_json = json.loads(artifact.exported_program)
  ↓
EpairWriter(converter).write(ep_json) → (ep_json, total_nodes, converted_nodes)
  │   _inject_ascend_ir(ep_json):
  │     ascend_ir_map = converter.get_ascend_ir_map()
  │     geo_2_fxo: dict[str, str] = {}  # GE output name → FX output name, cross-node wiring
  │     iterate ep_json["graph_module"]["graph"]["nodes"]:
  │       _process_node(serial_node, ascend_ir_map, geo_2_fxo)
  │         · parse metadata.torch_fn (split on ";" to extract the aten_ir lookup key)
  │         · handle auto_functionalized_v2 (take the real op from inputs[0].arg.as_operator)
  │         · look up ascend_ir_map to obtain NodeIR
  │         · _get_ops_list: when source="", match by type and return only [{"attr": ...}]; otherwise return full ops
  │         · _resolve_input_names: rewrite GE input references to FX names using geo_2_fxo
  │         · _map_output_names: flatten FX outputs and match them one-to-one with GE outputs, skipping CONST_* and None/None
  │         · inject serial_node["ascend_ir"] = {source, ops, mapping: {inputs: {...}, outputs: {...}}}
  │   logger.debug("Total graph nodes: {total}, converted: {converted}")
  ↓
writer._get_cann_version() → cann_version
  │   try: torch_npu.npu.utils.get_cann_version()
  │   except: logger.warning(...) → "unknown"
  ↓
EpairManifest(version=1, backend_name="ascend_ge",
              cann_version, total_nodes, converted_nodes)
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
  │   # Constants come from torch_delegate/epair/constants.py
  │   #   EPAIR_MANIFEST_FILENAME = "epair_manifest.json"
  │   #   EPAIR_ANNOTATED_MODEL_FILENAME = "ep_json_with_ascend_ir.json"
  └─ torch.export.save(ep, f, extra_files=extra_files)
  │   # PyTorch automatically places extra_files under the extra/ directory of the archive
  │ logger.info("Epair saved successfully to {f}")

---

epair_load(f)   # defined in torch_delegate/epair/reader.py
  │ logger.info("Loading epair from {f}")
  ├─ torch.export.load(f, extra_files={EPAIR_MANIFEST_FILENAME: '', EPAIR_ANNOTATED_MODEL_FILENAME: ''})
  ├─ parse manifest, ep_json_with_ir
  │   logger.info("Epair loaded: backend=..., {converted}/{total} nodes")
  └─ return EpairContents(...)
```

---

## Data Structures

### NodeIR

Node-level IR representation that records the Ascend IR information of a single FX node. Defined in `converter/base.py`:

```python
@dataclasses.dataclass
class NodeIR:
    source: str                          # Conversion origin identifier
                                         #   "converter": normal torchair GE conversion
                                         #   "":          fallback node (unsupported by torchair)

    ops: list[dict[str, Any]]            # List of GE opdefs; each element contains:
                                         #   name, type, input, attr, inputDesc, outputDesc

    mapping: dict[str, Any] | None = None  # Input/output mapping, structured as:
                                           #   {"inputs": {fx_name: ge_name, ...},
                                           #    "outputs": {fx_name: ge_name, ...}}
```

### EpairConfig

See [the first section](#epairconfig).

### EpairManifest

Metadata of the epair file, defined in `epair/schema.py`:

```python
@dataclasses.dataclass
class EpairManifest:
    version: int             # Format version number, currently 1
    backend_name: str        # Backend name, fixed to "ascend_ge"
    cann_version: str        # CANN version number, e.g. "9.2.0"; "unknown" when unavailable
    total_nodes: int         # Total number of compute nodes in the EP JSON
    converted_nodes: int     # Number of nodes successfully converted (with ascend_ir)
                             # converted_nodes < total_nodes means fallback nodes exist
```

### EpairContents

The complete content container of an epair, defined in `epair/schema.py`:

```python
@dataclasses.dataclass
class EpairContents:
    manifest: EpairManifest       # Metadata
    ep_json: dict[str, Any]       # EP JSON with ascend_ir
    ep: ExportedProgram           # Executable ExportedProgram
```

---

## ascend_ir Format

`ascend_ir` is an optional field on node objects in the EP JSON, injected by `EpairWriter`.

### Normally Converted Node (source = "converter")

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

**Mapping structure description:**

| Field | Type | Description |
|------|------|------|
| `mapping.inputs` | `dict[str, str]` | FX input name → GE input name (in `ge_node:port` format) |
| `mapping.outputs` | `dict[str, str]` | FX output name → GE output name (in `ge_node:port` format) |

### Fallback Node (source = "")

For nodes that torchair cannot convert, `source` is an empty string. `EpairWriter._get_ops_list` matches the node target by the `type` field and returns only `attr`:

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

### auto_functionalized_v2 Node

For nodes in PyTorch's `auto_functionalized_v2` format, the writer extracts the real op name from `inputs[0].arg.as_operator` to look up `ascend_ir_map`:

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

## PT2 Archive Artifact

### File Structure

```text
model.epair (ZIP structure, PT2-compatible)
├── extra/
│   ├── epair_manifest.json               ← epair metadata (added)
│   └── ep_json_with_ascend_ir.json       ← EP JSON with ascend_ir (added)
├── models/
│   └── model.json                        ← standard EP JSON (without ascend_ir)
├── data/
│   ├── weights/
│   │   ├── model_weights_config.json
│   │   └── weight_0, weight_1, ...
│   ├── constants/
│   │   ├── model_constants_config.json
│   │   └── tensor_0, tensor_1, ...
│   └── model_sample_inputs.pt
└── byteorder                             ← value is "little"
```

> [!CAUTION] Note
> The `extra/` prefix is added automatically by PyTorch's `torch.export.save`, and all files passed in `extra_files` are placed under the `extra/` directory. The constants in the code (`epair/constants.py`) define only filenames and do not include the `extra/` prefix.

### File Description

| File | Source | Description |
|------|------|------|
| `extra/epair_manifest.json` | torch-delegate | Metadata: version, statistics, CANN version |
| `extra/ep_json_with_ascend_ir.json` | torch-delegate | Complete EP JSON with ascend_ir annotations |
| `models/model.json` | torch.export.save | Standard EP JSON (weight references) |
| `data/weights/` | torch.export.save | Model weight tensors |
| `data/constants/` | torch.export.save | Constant tensors |
| `data/model_sample_inputs.pt` | torch.export.save | Sample inputs |

### manifest.json Format

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

## Scenario Handling

### Conversion Flow

Graph conversion is delegated by `TorchAirGEConverter.convert_graph()` to `torchair.ge.ge_graph_ascend._optimize_and_convert()`. torchair handles graph optimization, operator decomposition, and FX → GE conversion, returning a `_GeGraphAscend` object that contains:

- `optimized`: the optimized `ExportedProgram`
- `ascend_ir_map`: `dict[str, _NodeIR]`, keyed by FX node name, with torchair-internal `_NodeIR` values

```python
def convert_graph(self, ep: ExportedProgram) -> ExportedProgram:
    compiler_config = self._create_compiler_config()

    # Build decomps from the whitelist
    decomps = {}
    table = default_decompositions()
    for op in self._config.whitelist_decompositions:
        if op in table:
            decomps[op] = table[op]

    # Delegate conversion to torchair
    self._ge_graph_ascend = _optimize_and_convert(
        ep, config=compiler_config, custom_decompositions=decomps
    )

    # torchair _NodeIR → local NodeIR
    self._ascend_ir_map = {
        name: _convert_torchair_node_ir(ir)
        for name, ir in self._ge_graph_ascend.ascend_ir_map.items()
    }

    return self._ge_graph_ascend.optimized
```

### Scenario Identification

| source Value | Meaning | C++-side Handling |
|------------|------|------------|
| `"converter"` | Normal torchair GE conversion | Assigned to the NPU subgraph, compiled and executed by GE |
| `""` (empty) | Fallback node (unsupported by torchair) | Assigned to the CPU segment; fallback codegen generates a custom GE op |

### Fallback Node Handling (EpairWriter Side)

`EpairWriter._get_ops_list` handles `source=""` nodes specially — it matches the EP JSON target by the `type` field and returns only the `attr` dictionary:

```python
def _get_ops_list(self, node_ir: NodeIR, target: str) -> list[dict] | None:
    if len(node_ir.source) == 0:
        # source is empty: match by type, return only attr
        for ir in node_ir.ops:
            if target == ir.get("type"):
                return [{"attr": ir.get("attr", {})}]
        return None  # no matching op, skip injection
    return node_ir.ops  # source is non-empty: return full ops
```

---

## Key Technical Points

### PyTorch Patch

At the `export_to_epair` entry, `do_auto_functionalize` is patched to the v2 version, ensuring that the `ExportedProgram` generates `auto_functionalized_v2` format nodes. The patch is implemented in `converter/patch_pytorch.py` and uses the `_is_patched` flag to guarantee idempotence:

```python
def patch_pytorch():
    """Idempotent patch, safe to call repeatedly"""
    global _original_do_auto_functionalize, _is_patched
    if _is_patched:
        return _original_do_auto_functionalize
    _original_do_auto_functionalize = auto_func.do_auto_functionalize
    auto_func.do_auto_functionalize = do_auto_functionalize_patch
    _is_patched = True
    return _original_do_auto_functionalize

def restore_pytorch(original=None):
    """Restore the original function, idempotent"""
    global _is_patched
    if not _is_patched:
        return
    if original is not None:
        auto_func.do_auto_functionalize = original
    _is_patched = False
```

`export_to_epair` calls it inside `try/finally`, ensuring restoration even on exceptions.

### Graph Conversion Delegation

`TorchAirGEConverter` does not traverse the FX graph by itself; instead, it delegates all conversion work to torchair's `_optimize_and_convert` API:

- **Graph optimization**: `_optimize_fx`, pattern fusion, noop removal, etc. are performed internally by torchair
- **Operator decomposition**: only operators in the intersection of `whitelist_decompositions` and `default_decompositions()` are decomposed
- **FX → GE conversion**: performed internally by torchair; successfully converted nodes have `source="converter"` and unsupported nodes have `source=""`
- **Result retrieval**: `_GeGraphAscend.ascend_ir_map` provides `dict[str, _NodeIR]`, converted to the local `NodeIR` via `_convert_torchair_node_ir`

### torch_fn Metadata Convention

After torchair conversion, an `aten_ir` lookup key is appended to the end of `node.meta["torch_fn"]` for each `call_function` node. `api.py` removes this trailing element before serialization:

```python
for node in converted_ep.graph_module.graph.nodes:
    if node.op == "call_function":
        torch_fn = node.meta["torch_fn"]
        node.meta["torch_fn"] = torch_fn[:-1]
```

In the serialized EP JSON, the `metadata.torch_fn` string has the format `"<fx_name>;<aten_ir_key>"`. `EpairWriter._process_node` splits on `;` to extract the `aten_ir` lookup key and modifies `metadata.torch_fn` to remove the `;...` suffix:

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

### geo_2_fxo Cross-Node Wiring

`EpairWriter._inject_ascend_ir` maintains a `geo_2_fxo: dict[str, str]` dictionary (GE output name → FX output name) for cross-node wiring:

- **Output registration**: when `_map_output_names` processes each node's output, it records `ge_output → fx_output` into `geo_2_fxo`
- **Input rewriting**: `_resolve_input_names` iterates over the op's `input` list; if a GE input name appears in `geo_2_fxo` and the corresponding FX name belongs to the current node's inputs, it is rewritten to the FX name and recorded in `mapping.inputs`

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
                op["input"][idx] = fx_name  # rewrite to FX name
    return input_mapping
```

### Output Mapping Extraction

`_map_output_names` flattens FX outputs (which may be tensors, tensor lists, sym ints, scalars, etc.) into a one-dimensional list that maps one-to-one with GE outputs:

```python
def _map_output_names(self, fx_outputs, node_ir_mapping, geo_2_fxo) -> dict[str, str]:
    output_mapping: dict[str, str] = {}
    ge_outputs = node_ir_mapping["outputs"]

    # Flatten FX outputs (supports as_tensor, as_tensors, as_sym_int, as_int, as_none, etc.)
    fx_flat = []
    for fx_output in fx_outputs:
        fx_flat.extend(_flatten_output(fx_output))

    # Counts must match
    assert len(fx_flat) == len(ge_outputs)

    for fx_out, ge_out in zip(fx_flat, ge_outputs):
        if fx_out.startswith("CONST_"):    # Skip constant placeholders
            continue
        if fx_out == "None" and ge_out == "None":  # Skip None/None pairs
            continue
        output_mapping[fx_out] = ge_out
        geo_2_fxo[ge_out] = fx_out         # Register into the cross-node wiring table

    return output_mapping
```

`_flatten_output` handles all EP JSON output types via the `_OUTPUT_FLATTENERS` dispatch table, and constant outputs are flattened into `"CONST_<type>"` placeholders (such as `"CONST_int"`, `"CONST_float"`).

### ascend_ir Injection

`_process_node` is the core method of `EpairWriter`; the full flow is:

1. Read `metadata.torch_fn` and `target`
2. Handle `auto_functionalized_v2`: take the real op name from `inputs[0].arg.as_operator`
3. Strip the `torch.ops.` prefix from `target`
4. Split `torch_fn` on `;` to extract the `aten_ir` lookup key and modify `metadata.torch_fn`
5. Extract the current node's FX input name set (`_extract_fx_input_names`)
6. Look up `ascend_ir_map`; return `False` if not found
7. `_get_ops_list` obtains the ops list (fallback nodes return only attr)
8. `_resolve_input_names` resolves the input mapping and rewrites GE input references
9. `_map_output_names` builds the output mapping and updates `geo_2_fxo`
10. Inject `serial_node["ascend_ir"] = {source, ops, mapping: {inputs, outputs}}`

---

## Logging System

### Log Levels

| Level | Usage Scenario | Example |
|------|----------|------|
| ERROR | Conversion failure termination, missing critical dependency | `logger.error(f"Graph conversion failed: {e}")` |
| WARNING | CANN version retrieval failure, node matching failure | `logger.warning(f"CANN version unavailable: {e}")` |
| INFO | API entry/exit, conversion statistics | `logger.info(f"Export complete: 15/20 nodes converted")` |
| DEBUG | Configuration details, intermediate steps, node matching | `logger.debug(f"Config: remove_noop_ops=True")` |

### Log Format

```text
[LEVEL] TORCH_DELEGATE(pid,executable) timestamp [file:line] thread_id message
```

Example:
```text
[INFO] TORCH_DELEGATE(12345,python) 2026-07-09 10:30:15.123 [api.py:60] 123 Exporting ExportedProgram to epair...
[DEBUG] TORCH_DELEGATE(12345,python) 2026-07-09 10:30:16.234 [writer.py:211] 123 Try to get key: add_1
[INFO] TORCH_DELEGATE(12345,python) 2026-07-09 10:30:17.001 [api.py:76] 123 Export complete: 15/20 nodes converted
```

### Usage

```python
from torch_delegate.core.utils import logger

logger.error("Critical error message")
logger.warning("Warning message")
logger.warning_once("Warning that only prints once")  # Prevent repetition (lru_cache)
logger.info("Info message")
logger.debug("Debug message")
```

### Environment Variable Control

```bash
# Default: WARN level
python your_script.py

# Enable DEBUG level
TORCH_DELEGATE_DEBUG=1 python your_script.py
```

---

## Design Advantages

| Advantage | Description |
|------|------|
| **Three-stage API** | Separates conversion and saving so users can inspect conversion results before saving |
| **source identifier** | Distinguishes converter/fallback node types via the source field |
| **Complete opdef** | The ops field contains complete GE opdef information (name, type, input, attr, inputDesc, outputDesc) |
| **Input/output mapping** | The mapping field records FX ↔ GE input/output correspondences, supporting cross-node wiring |
| **torchair delegation** | Graph conversion is delegated to torchair `_optimize_and_convert`, reusing mature optimization and conversion logic |
| **Whitelist decomposition** | `whitelist_decompositions` precisely controls the scope of operator decomposition |
| **Logging system** | Complete logging, with environment-variable-controlled debug level |
| **CANN version** | The manifest records the CANN version for version consistency checks |
| **Compatibility** | Reuses torch.export.save/load, compatible with the PT2 toolchain |

---

## Comparison with dynamo_export

| Aspect | dynamo_export (torchair) | export_to_epair (torch-delegate) |
|------|--------------------------|----------------------------------|
| **Input source** | `torch.compile` → dynamo → gm | `ExportedProgram` → ep.graph_module |
| **Graph processing** | torchair internal optimization | torchair `_optimize_and_convert` (delegated) |
| **Error handling** | Error when no converter is found | torchair marks `source=""`, C++-side fallback |
| **IR recording** | Post-processing extracts GE graph | torchair returns `ascend_ir_map`, writer injects into EP JSON |
| **Output** | `.air` file (binary GE graph) | `.epair` file (PT2 archive + ascend_ir) |
| **Weight handling** | Custom externalization | PT2 weight management (torch.export.save) |
| **API design** | Single `dynamo_export` | Three APIs: `export_to_epair` + `epair_save/load` |
| **Logging system** | torchair logger | torch_delegate logger (independent environment variable) |
| **Runtime** | CANN only | Flexible (C++ side decides: NPU/CPU partitioning) |
| **CANN version** | Not recorded | Recorded in manifest |
| **Operator decomposition** | torchair internal control | `whitelist_decompositions` whitelist control |

---

## Code Structure

```text
torch_delegate/
├── __init__.py                       ← exports public API + data classes
├── api.py                            ← export_to_epair, epair_save (epair_load imported from reader)
├── core/
│   ├── __init__.py                   ← exports logger
│   └── utils.py                      ← logging system implementation (_MillisecFormatter, warning_once)
├── converter/
│   ├── __init__.py                   ← exports NodeIR, NodeIRConverter, TorchAirGEConverter
│   ├── base.py                       ← NodeIR data class + NodeIRConverter ABC
│   ├── torchair_converter.py         ← TorchAirGEConverter (delegates _optimize_and_convert)
│   └── patch_pytorch.py              ← do_auto_functionalize → v2 patch (idempotent)
└── epair/
    ├── __init__.py                   ← exports schema/config/writer/reader
    ├── schema.py                     ← EpairManifest, EpairContents
    ├── config.py                     ← EpairConfig (3 fields)
    ├── constants.py                  ← EPAIR_MANIFEST_FILENAME, EPAIR_ANNOTATED_MODEL_FILENAME
    ├── writer.py                     ← EpairWriter (injects ascend_ir, cross-node wiring)
    └── reader.py                     ← epair_load function
```

### Module Responsibilities

| Module | Responsibility |
|------|------|
| `api.py` | Top-level entry, composes converter + writer, manages the patch lifecycle |
| `core/utils.py` | Logging system, `TORCH_DELEGATE_DEBUG` environment variable controls the level |
| `converter/base.py` | `NodeIR` data class (single source of truth) + `NodeIRConverter` abstract base class |
| `converter/torchair_converter.py` | Production converter, delegates to torchair `_optimize_and_convert` |
| `converter/patch_pytorch.py` | Idempotent patch of `do_auto_functionalize` → v2 |
| `epair/schema.py` | `EpairManifest`, `EpairContents` data classes |
| `epair/config.py` | `EpairConfig` configuration class |
| `epair/constants.py` | epair filename constants (without the `extra/` prefix) |
| `epair/writer.py` | `EpairWriter`, injects ascend_ir into EP JSON |
| `epair/reader.py` | `epair_load` function, loads .epair files |
