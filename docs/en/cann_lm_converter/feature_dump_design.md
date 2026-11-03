# Dump Debug Capability

English | [简体中文](../../zh/cann_lm_converter/feature_dump_design.md)

> Per-op intermediate tensor dump at the FX node level, used for accuracy benchmark comparison and operator-level debugging before/after export and before/after quantization.

## Design Goal

The converter solidifies the HuggingFace eager model into an FX graph via `torch.export`, then converts it to epair IR / OM via GE. Debugging requires a dump that is **isomorphic to the exported graph**: the input and output of every FX node (including functional ops) can be written to disk, so that intermediate tensors before and after export can be compared op-by-op, precisely locating the source of drift.

A plain `register_forward_hook` can only capture `nn.Module`-level input/output, which does not cover functional ops (`add`/`mul`/`scale`/`reshape`/`transpose`/`getitem` etc. are nodes in the FX graph, not modules), and eager modules may be inlined/fused/reordered after export, so names no longer match. Therefore, an **FX Interpreter** is used.

## Tool

[`examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py`](../../../examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py)

The core is `DumpInterpreter` (inherits from `torch.fx.Interpreter`):

```python
class DumpInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.dump: dict[str, Any] = {}

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        self.dump[n.name] = result      # ← capture output of every FX node
        return result
```

For each FX node executed, `run_node` stores the output in `self.dump[node.name]`; after running the entire graph, it iterates over `gm.graph.nodes` and writes the output of each node (skipping placeholder/output) to disk as `.bin` via `.cpu().detach().contiguous()`.

## Capability Comparison

| Dimension | DumpInterpreter (this tool) | torch module hook |
|------|---------------------------|-------------------|
| **Granularity** | FX node level — every op (including functional) | module level — only `nn.Module` input/output |
| **Export required?** | Requires `torch.export` to solidify the FX graph first | Not required; runs directly in eager |
| **Coverage** | All ops (module + functional + control flow after expansion) | Only modules; misses functional ops |
| **Alignment with export graph** | dump nodes map 1:1 to epair IR / OM ops | eager modules may be inlined/fused after export; misaligned |
| **OOM control** | `--no-intermediates` can skip intermediates (store only input/output) | Can hook only selected modules; naturally controllable |
| **backward** | Not supported (forward only) | `register_full_backward_hook` can dump gradients |
| **meta** | Stores `node_name`/`op`/`target` (op type traceable) | Only module full path name |

Conclusion: export pipeline benchmark comparison must use an FX Interpreter (aligned with ops); eager debugging / backward dump uses hooks. The two are complementary.

## Usage

```bash
# gemma4 5-layer prefill (dump all intermediates)
source $PROJECT_ROOT/scripts/set_env_export.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
    --model-type gemma4 \
    --hf-path $PROJECT_ROOT/models/unsloth/gemma-4-E2B-it \
    --seq-len 128 --num-layers 5 --dtype float16 \
    --output-dir out/golden_gemma4_5l

# Dump only input/output (skip intermediates, save disk)
python .../gen_golden_with_dump.py ... --no-intermediates

# qwen3.5 5-layer
python .../gen_golden_with_dump.py \
    --model-type qwen3_5 \
    --hf-path $PROJECT_ROOT/models/Qwen/Qwen3.5-0.8B \
    --seq-len 8 --num-layers 5 --dtype float16 \
    --output-dir out/golden_qwen35_5l
```

## Artifact Layout

```
output-dir/
├── input_<idx>_<name>[_<j>].bin    ← Each input tensor (named by forward signature)
├── output_<idx>.bin                ← Each output tensor
├── node_<name>[_<j>].bin           ← Output of each FX node (skipped with --no-intermediates)
└── golden_meta.json                ← Meta info (names/shapes/dtypes/targets)
```

`golden_meta.json` records `file`/`shape`/`dtype` of each tensor + `node_name`/`op`/`target` of each node, making op type traceable and convenient for programmatic comparison.

## Key Implementation Details

- **bfloat16 → float32 upcast**: `numpy` does not support bf16; `_save_tensor` losslessly upcasts bf16 to fp32 for disk write (bf16→fp32 is exact and lossless); OM output is also upcast to fp32 for tolerance comparison.
- **Input names use forward signature**: Under `strict=False`, torch.export gives nested inputs (e.g. `past_key_values`) placeholder names derived from tree paths, which do not align with the top-level count of sample_inputs; using `inspect.signature(forward)` ensures 1:1 alignment.
- **Fixed seed**: `torch.manual_seed(seed)` fixes the `torch.randint` in `get_sample_inputs`, making the golden reproducible across runs (weights are real safetensors, unaffected by seed).
- **num_layers truncation**: `artifact.model.model.layers = layers[:N]` + `layer_types[:N]`, supporting fast generation of small-network benchmarks like 5-layer.

## Quantization Op Dump

`gen_golden_with_dump.py` supports `--quant-path` to load dopt GPTQ quantized weights, dumping the full FX node intermediate tensors of the fakequant (W4A8) model for per-op accuracy comparison before and after quantization.

### Usage

```bash
# System python + torchaudio shim (gemma4 multimodal requires stub)
source /home/developer/Ascend/cann/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=<shims>:$PYTHONPATH

# Quantized dump (W4A8 fakequant, per-op intermediate tensors)
python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
    --model-type gemma4 \
    --hf-path $PROJECT_ROOT/models/unsloth/gemma-4-E4B-it \
    --quant-path /path/to/dopt_output \
    --seq-len 8 --num-layers 3 --dtype float16 \
    --output-dir out/golden_gemma4_3l_quant

# fp16 baseline (without --quant-path, same model, same input)
python .../gen_golden_with_dump.py \
    --model-type gemma4 \
    --hf-path ... --seq-len 8 --num-layers 3 --dtype float16 \
    --output-dir out/golden_gemma4_3l_fp16

# Compare: fp16 golden vs quantized golden → per-op cosine/MSE, locate quantization loss source layer
```

### Quantization Integration Flow

`--quant-path` specifies the dopt output directory (containing `trained_quant_weight.pth` + `dopt_config.json`); the tool internally:

1. **Load model** → `load_model` → `wrapper` (same as fp16 path)
2. **Swap QLinear** → `_swap_linear_from_sd` (`nn.Linear` → `QLinear`, attaches `quant_op`)
3. **Re-materialize unpacked** → `materialize_int4_per_group_symmetric(packed=False)` produces `[K,N]` int8 (not nibble-packed `[K//2,N]`), because the CPU `DumpInterpreter` running `cann_local_lm::quant_matmul`'s CPU impl requires unpacked
4. **Load weights** → `load_state_dict(trained_quant_weight.pth)`
5. **set_quant_state** → `weight_state=True, input_state=True` (W4A8: weight int4 + activation int8 quantization)
6. **Skip embedding quantization** → `embed_tokens_per_layer [262144,10752]` fakequant intermediate tensor ~11GB, high CPU memory pressure; embedding is a lookup not a matmul, accuracy impact is small (`--no-skip-embed-quant` can disable the skip)
7. **torch.export** → `DumpInterpreter` → per FX node dump (including input/output of `cann_local_lm::quant_matmul` + `cann_local_lm::dynamic_quant`)

### Dumped Quantization Ops

In the FX graph, matmul is split into two `cann_local_lm::` custom ops (W4A8):

| FX node target | Op | Effect | Dumped tensors |
|---|---|---|---|
| `cann_local_lm.dynamic_quant.default` | Activation int8 quantization | `x → (x_q int8, x_scale fp32 per-token)` | Input x + output (x_q, x_scale) |
| `cann_local_lm.quant_matmul.default` | int4×int8 matmul | `(x_q, weight_q, weight_scale) → out` | Input (x_q, weight_q, weight_scale) + output out |

Each `quant_matmul_N` FX node corresponds to a QLinear (`q_proj` / `k_proj` / `mlp.gate_proj` etc.). The dumped `node_quant_matmul_N.bin` is that op's output, which can be compared with the corresponding node in the fp16 golden to locate quantization loss.

### golden_meta.json Quantization Info

```json
{
  "quant": {
    "quant_path": "/path/to/dopt_output",
    "scheme": "W4A8-s8s4",
    "n_quantized": 29,
    "weight_state": true,
    "input_state": true,
    "skip_embed_quant": true
  }
}
```

### Key Implementation Details (Quantization Dump Specific)

- **unpacked int8 vs nibble-packed**: `swap_from_dopt` by default produces nibble-packed int8 `[K//2,N]` (for export/NPU); but the CPU `DumpInterpreter` running `cann_local_lm::quant_matmul`'s CPU impl requires unpacked `[K,N]` (comment `quant_matmul.py:104` "int4 stored unpacked as int8 (signed)"). The tool manually re-materializes each QLinear's `weight_q` as unpacked (`materialize_int4_per_group_symmetric(packed=False)`).
- **CPU impl registration**: `_HAS_TORCH_NPU=False` (`_library.py:30`, does not import torch_npu) → `register_cpu_impl` registers the CPU impl (not no-op) → `cann_local_lm::quant_matmul` / `cann_local_lm::dynamic_quant` can run on CPU.
- **System python + shim**: gemma4 multimodal → transformers `import torchaudio` (system python's torchaudio version mismatch) → use `dopt_op_benchmark/shims/torchaudio` stub (auto-detected and enabled by `run.sh`). System python does not import torch_npu → `_HAS_TORCH_NPU=False` → CPU impl takes effect.
- **Embedding quantization skip**: After `set_quant_state(weight_state=True, input_state=True)`, iterate modules to set `embed` / `per_layer_model_projection`'s `quant_op.set_quant_state(False, False)` — these modules keep fp16 (no fakequant), avoiding the 11GB intermediate tensor of `embed_tokens_per_layer [262144,10752]`.

### hook vs DumpInterpreter Consistency Verification

[`verify_hook_vs_dump.py`](../../../examples/test_converter/dopt_op_benchmark/verify_hook_vs_dump.py) runs one forward pass on the same quantized model + same input using both methods:

- **hook**: `register_forward_hook` captures the output of 29 QLinear modules
- **DumpInterpreter**: FX Interpreter dumps per node (512 nodes, including `quant_matmul_N`)

Result: **29/29 QLinear module outputs all maxΔ=0 cos≥0.9999 PASS** — the two methods are completely equivalent, golden engineering correctness verified.

## Application Scenarios

1. **Pre/post export accuracy comparison**: eager golden (this tool) vs epair IR / OM output, per-op cosine/MSE
2. **Pre/post quantization accuracy comparison**: fp16 golden (without `--quant-path`) vs quantized golden (with `--quant-path`), per-op location of quantization loss source layer (see "Quantization Op Dump" above)
3. **Op-level debugging**: when a certain FX node output does not match the reference, directly inspect `node_<name>.bin`
4. **Regression benchmark**: re-run golden after code changes, compare for regressions
5. **Quantized simulation inference**: [`sim_inference.py`](../../../examples/test_converter/dopt_op_benchmark/sim_inference.py) loads the fakequant model → NPU per-token argmax text generation, observing post-quantization generation quality (single-card/dual-card mode)

## cann_lm_compile Compile-Time Dump

`cann_lm_compile` (`compile_main.cc`) converts epair → OM/.omc; compile-time dump relies on **built-in tap + GE environment variable export**. Refer to [kirinx90_compile_workarounds.md](./kirinx90_compile_workarounds.md) for the env logging style.

### ① Built-in tap (PrintStage stage printing)

`compile_main.cc` has 3-stage tap; stdout shows progress at runtime:

```
[1/3] Load epair and build GE Graph
[2/3] Extract GE Graph
[3/3] Compile graph
[PASS] aclgrphSaveModel succeeded, <name>.om generated
 All 3 stages passed
```

Source (`compile_main.cc:20` `PrintStage` + `:128/135/164/196`).

### ② GE graph dump + logs (export environment variables)

Export these env before running cann_lm_compile; GE dumps graphs + detailed logs at compile time:

```bash
# Source compile env before running (② OM compile, same as AGENTS.md "Runtime Environment Setup")
source scripts/set_env_cann.sh   # ②→0715 CANN

# ── GE graph dump (compile-time GE graph txt per stage, locate ops/fusion/shape) ──
export DUMP_GE_GRAPH=1          # 1=all graphs, 2=concise, 3=ops only; larger = more dump
export DUMP_GRAPH_LEVEL=1       # number of graphs to dump (1=first only, 2=all stages)

# ── GE detailed logs to stdout (locate "not registered" / missing ops / compile failure) ──
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0  # 0=DEBUG most detailed, 1=INFO, 2=WARNING, 3=ERROR
# docs: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/docs/zh/user_guides/atc_tools/references/dump_graph_details.md

# ── Run compile (cann_lm_compile via build_cpp.sh / build.sh cpp) ──
build/cann_lm_compile <epair_dir> --soc_version Ascend910_9362
# GE dump graphs land in ~/*.txt (compile dir), logs to stdout
```

> ⚠️ `DUMP_GE_GRAPH` produces large output (multiple txt per stage); `unset` promptly after locating issues to avoid disk flooding.
> `ASCEND_GLOBAL_LOG_LEVEL=0` logs are extremely detailed (including GE internal passes); set back to 2/3 after locating issues.

### ③ OM op list dump (atc --mode, after compilation)

OM is a compiled binary; use the CANN `atc` tool to parse it (**real CANN env, no stubs**):

```bash
source scripts/set_env_cann.sh   # ② OM compile env (same as compile)

# Summary info (seconds: soc_version / weights / tbe_kernels / stream_num etc.)
atc --mode=6 --om=<path>/decode.om

# Op list (export JSON then parse, see post-fusion op types + input_desc/output_desc/attr)
atc --mode=1 --om=<path>/decode.om --json=/tmp/opencode/decode_om
python3 -c "import json;from collections import Counter;ops=json.load(open('/tmp/opencode/decode_om'))['graph'][0]['op'];[print(v,k) for k,v in Counter(o['type'] for o in ops).most_common()]"
```

> OM can be several GB; `--mode=1` JSON is typically tens of MB; store parsing artifacts in `/tmp/opencode`, do not scatter in `examples/`.
> Empty `framework_type` is normal (epair goes through GE direct compile, not a standard framework).
> OM ops are the **final post-fusion** ops, 50%+ fewer than the IR stage (Reshape/Transpose fused into MatMul/RmsNorm).

### ④ Custom tap (add cout to print ops/shapes)

For deeper debugging, add taps in `compile_main.cc` (manual prints beyond PrintStage):

```cpp
// compile_main.cc: after Extract GE Graph stage, iterate GE graph to dump op list
auto graph = ...;  // GE ComputeGraph
std::cout << "[tap] ops=" << graph->GetAllNodesSize() << std::endl;
for (auto* node : graph->GetAllNodes()) {
    std::cout << "  " << node->GetType() << " " << node->GetName() << std::endl;
}
```

Rebuild with `bash scripts/build.sh cpp` + re-run after changes. **Confirm before modifying `cann_local_lm/`** (AGENTS.md convention).

### Debugging Scenarios

| Symptom | Which dump to use |
|------|-----------|
| Compile failure "op not registered" | ② `ASCEND_SLOG=1` + `LOG_LEVEL=0` view GE logs |
| Op fusion result unexpected | ② `DUMP_GE_GRAPH=1` view per-stage graphs + ③ `atc --mode=1` view post-fusion OM ops |
| OM weight/tbe size abnormal | ③ `atc --mode=6` view summary |
| Compile stuck at a stage | ① PrintStage see where it stopped + ② logs |
| Custom op not in graph | ④ add tap to iterate GE nodes |

## Related

- [export_design.md](./export/export_design.md) (full export flow + epair IR)
- [quant_design.md](./quant/quant_design.md) (quantization accuracy verification)
- [README.md](./README.md) (cross-module call chain + accuracy summary)
- [`gen_golden_with_dump.py`](../../../examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py) (tool source, supports `--quant-path` quantized dump)
- [`verify_hook_vs_dump.py`](../../../examples/test_converter/dopt_op_benchmark/verify_hook_vs_dump.py) (hook vs DumpInterpreter consistency verification)
- [`sim_inference.py`](../../../examples/test_converter/dopt_op_benchmark/sim_inference.py) (quantized simulation inference, single-card/dual-card)
