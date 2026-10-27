# Dump 维测能力

简体中文 | [English](../../en/cann_lm_converter/feature_dump_design.md)

> FX node 级逐算子中间张量 dump，用于导出前后 / 量化前后精度标杆比对与算子级排障。

## 设计目标

converter 把 HuggingFace eager 模型经 `torch.export` 固化为 FX 图，再 GE 转成 epair IR / OM。
维测需要一个**与导出图同构**的 dump：每个 FX node（含 functional op）的输入输出都能落盘，
这样导出前后的中间张量可逐算子比对，精确定位漂移来源。

普通 `register_forward_hook` 只能拿 `nn.Module` 级 input/output，覆盖不到 functional op
（`add`/`mul`/`scale`/`reshape`/`transpose`/`getitem` 等在 FX 图里是 node 不是 module），
且 eager module 在导出后可能被 inline/fuse/重排，名都对不上。故采用 **FX Interpreter**。

## 工具

[`examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py`](../../../examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py)

核心是 `DumpInterpreter`（继承 `torch.fx.Interpreter`）：

```python
class DumpInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.dump: dict[str, Any] = {}

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        self.dump[n.name] = result      # ← 每个 FX node 输出都 capture
        return result
```

每跑一个 FX node，`run_node` 把输出存进 `self.dump[node.name]`；跑完整图后遍历 `gm.graph.nodes`，
把每个 node（跳过 placeholder/output）的输出 `.cpu().detach().contiguous()` 落盘为 `.bin`。

## 能力对比

| 维度 | DumpInterpreter（本工具） | torch module hook |
|------|---------------------------|-------------------|
| **粒度** | FX node 级——每个 op（含 functional） | module 级——只 `nn.Module` input/output |
| **是否需 export** | 需 `torch.export` 先固化 FX 图 | 不需，eager 直接跑 |
| **覆盖度** | 全 op（module + functional + 控制流展开后） | 只 module，漏 functional op |
| **对齐导出图** | dump 的 node 1:1 对齐 epair IR / OM 算子 | eager module 导出后可能 inline/fuse，对不上 |
| **OOM 控制** | `--no-intermediates` 可跳过中间（只存 input/output） | 可只 hook 选定 module，天然可控 |
| **backward** | 不支持（只 forward） | `register_full_backward_hook` 可 dump 梯度 |
| **meta** | 存 `node_name`/`op`/`target`（可追溯 op 类型） | 只有 module 全路径名 |

结论：导出链路标杆比对必须用 FX Interpreter（对齐算子）；eager 调试 / backward dump 用 hook。两者互补。

## 用法

```bash
# gemma4 5-layer prefill（中间全 dump）
source $PROJECT_ROOT/scripts/set_env_export.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
    --model-type gemma4 \
    --hf-path $PROJECT_ROOT/models/unsloth/gemma-4-E2B-it \
    --seq-len 128 --num-layers 5 --dtype float16 \
    --output-dir out/golden_gemma4_5l

# 只 dump 输入输出（跳过中间，省盘）
python .../gen_golden_with_dump.py ... --no-intermediates

# qwen3.5 5-layer
python .../gen_golden_with_dump.py \
    --model-type qwen3_5 \
    --hf-path $PROJECT_ROOT/models/Qwen/Qwen3.5-0.8B \
    --seq-len 8 --num-layers 5 --dtype float16 \
    --output-dir out/golden_qwen35_5l
```

## 产物布局

```
output-dir/
├── input_<idx>_<name>[_<j>].bin    ← 每个 input tensor（按 forward 签名名）
├── output_<idx>.bin                ← 每个 output tensor
├── node_<name>[_<j>].bin           ← 每个 FX node 输出（--no-intermediates 跳过）
└── golden_meta.json                ← 元信息（names/shapes/dtypes/targets）
```

`golden_meta.json` 记录每个 tensor 的 `file`/`shape`/`dtype` + node 的 `node_name`/`op`/`target`，
可追溯算子类型，便于程序化比对。

## 关键实现细节

- **bfloat16 → float32 上转**：`numpy` 不支持 bf16，`_save_tensor` 把 bf16 无损上转 fp32 落盘
  （bf16→fp32 精确无损），比对时 OM 输出也上转 fp32 做容差比较。
- **input 名用 forward 签名**：`strict=False` 下 torch.export 给嵌套输入（如 `past_key_values`）
  的 placeholder 名是树路径衍生，与 sample_inputs 顶层数不对齐；用 `inspect.signature(forward)`
  保证 1:1 对齐。
- **固定 seed**：`torch.manual_seed(seed)` 固定 `get_sample_inputs` 的 `torch.randint`，
  使 golden 跨 run 可复现（权重是真实 safetensors，不受 seed 影响）。
- **num_layers 截断**：`artifact.model.model.layers = layers[:N]` + `layer_types[:N]`，
  支持 5-layer 等小网快速生成标杆。

## 量化算子 dump

`gen_golden_with_dump.py` 支持 `--quant-path` 加载 dopt GPTQ 量化权重，dump fakequant（W4A8）
模型的全 FX node 中间张量，用于量化前后逐算子精度比对。

### 用法

```bash
# 系统 python + torchaudio shim (gemma4 多模态需 stub)
source /home/developer/Ascend/cann/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=<shims>:$PYTHONPATH

# 量化 dump (W4A8 fakequant, 逐算子中间张量)
python examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py \
    --model-type gemma4 \
    --hf-path $PROJECT_ROOT/models/unsloth/gemma-4-E4B-it \
    --quant-path /path/to/dopt_output \
    --seq-len 8 --num-layers 3 --dtype float16 \
    --output-dir out/golden_gemma4_3l_quant

# fp16 基线 (无 --quant-path, 同模型同输入)
python .../gen_golden_with_dump.py \
    --model-type gemma4 \
    --hf-path ... --seq-len 8 --num-layers 3 --dtype float16 \
    --output-dir out/golden_gemma4_3l_fp16

# 对比: fp16 golden vs 量化 golden → 逐算子余弦/MSE, 定位量化损失来源 layer
```

### 量化接入流程

`--quant-path` 指定 dopt 输出目录（含 `trained_quant_weight.pth` + `dopt_config.json`），工具内部：

1. **加载模型** → `load_model` → `wrapper`（同 fp16 路径）
2. **swap QLinear** → `_swap_linear_from_sd`（`nn.Linear` → `QLinear`，附 `quant_op`）
3. **re-materialize unpacked** → `materialize_int4_per_group_symmetric(packed=False)` 产出 `[K,N]` int8（非 nibble-packed `[K//2,N]`），因为 CPU `DumpInterpreter` 跑 `cann_lm::quant_matmul` 的 CPU impl 要 unpacked
4. **load 权重** → `load_state_dict(trained_quant_weight.pth)`
5. **set_quant_state** → `weight_state=True, input_state=True`（W4A8: 权重 int4 + 激活 int8 量化）
6. **跳过 embedding 量化** → `embed_tokens_per_layer [262144,10752]` 的 fakequant 中间张量 ~11GB，CPU 内存压力大；embedding 是 lookup 不是 matmul，精度影响小（`--no-skip-embed-quant` 可关闭跳过）
7. **torch.export** → `DumpInterpreter` → 逐 FX node dump（含 `cann_lm::quant_matmul` + `cann_lm::dynamic_quant` 的输入输出）

### dump 的量化算子

FX 图里 matmul 被拆成两个 `cann_lm::` 自定义算子（W4A8）：

| FX node target | 算子 | 作用 | dump 的张量 |
|---|---|---|---|
| `cann_lm.dynamic_quant.default` | 激活 int8 量化 | `x → (x_q int8, x_scale fp32 per-token)` | 输入 x + 输出 (x_q, x_scale) |
| `cann_lm.quant_matmul.default` | int4×int8 matmul | `(x_q, weight_q, weight_scale) → out` | 输入 (x_q, weight_q, weight_scale) + 输出 out |

每个 `quant_matmul_N` FX node 对应一个 QLinear（`q_proj` / `k_proj` / `mlp.gate_proj` 等）。
dump 的 `node_quant_matmul_N.bin` 是该算子输出，可与 fp16 golden 的对应 node 比对，定位量化损失。

### golden_meta.json 量化信息

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

### 关键实现细节（量化 dump 特有）

- **unpacked int8 vs nibble-packed**：`swap_from_dopt` 默认产出 nibble-packed int8 `[K//2,N]`（给 export/NPU 用）；但 CPU `DumpInterpreter` 跑 `cann_lm::quant_matmul` 的 CPU impl 要 unpacked `[K,N]`（注释 `quant_matmul.py:104` "int4 stored unpacked as int8 (signed)"）。工具内手动 re-materialize 每个 QLinear 的 `weight_q` 为 unpacked（`materialize_int4_per_group_symmetric(packed=False)`）。
- **CPU impl 注册**：`_HAS_TORCH_NPU=False`（`_library.py:30`，不 import torch_npu）→ `register_cpu_impl` 注册 CPU impl（非 no-op）→ `cann_lm::quant_matmul` / `cann_lm::dynamic_quant` 在 CPU 上可跑。
- **系统 python + shim**：gemma4 多模态 → transformers `import torchaudio`（系统 python 的 torchaudio 版本错配）→ 用 `dopt_op_benchmark/shims/torchaudio` stub（`run.sh` 自动检测启用）。系统 python 不 import torch_npu → `_HAS_TORCH_NPU=False` → CPU impl 生效。
- **embedding 量化跳过**：`set_quant_state(weight_state=True, input_state=True)` 后，遍历 modules 把 `embed` / `per_layer_model_projection` 的 `quant_op.set_quant_state(False, False)` —— 这些模块保留 fp16（不做 fakequant），避免 `embed_tokens_per_layer [262144,10752]` 的 11GB 中间张量。

### hook vs DumpInterpreter 一致性验证

[`verify_hook_vs_dump.py`](../../../examples/test_converter/dopt_op_benchmark/verify_hook_vs_dump.py) 对同一量化模型 + 同一输入，用两种方式各跑一遍 forward：

- **hook**：`register_forward_hook` 抓 29 个 QLinear module 的 output
- **DumpInterpreter**：FX Interpreter 逐 node dump（512 nodes，含 `quant_matmul_N`）

结果：**29/29 QLinear module output 全部 maxΔ=0 cos≥0.9999 PASS**——两种方式完全等价，golden 工程正确性验证通过。

## 应用场景

1. **导出前后精度比对**：eager golden（本工具）vs epair IR / OM 输出，逐算子余弦/MSE
2. **量化前后精度比对**：fp16 golden（无 `--quant-path`）vs 量化 golden（有 `--quant-path`），逐算子定位量化损失来源 layer（见上方「量化算子 dump」）
3. **算子级排障**：某个 FX node 输出与 reference 不符时，直接查 `node_<name>.bin`
4. **回归标杆**：代码改动后重跑 golden，比对是否有回归
5. **量化仿真推理**：[`sim_inference.py`](../../../examples/test_converter/dopt_op_benchmark/sim_inference.py) 加载 fakequant 模型 → NPU 逐 token argmax 生成文本，看量化后生成质量（单卡/两卡模式）

## cann_lm_compile 编译期 dump

`cann_lm_compile`（`compile_main.cc`）把 epair → OM/.omc，编译期 dump 靠**内置 tap + GE 环境
变量 export** 两种方式。参考 [kirinx90_compile_workarounds.md](./kirinx90_compile_workarounds.md)
的 env 留档风格。

### ① 内置 tap（PrintStage 阶段打印）

`compile_main.cc` 已有 3 阶段 tap，跑时 stdout 可见进度：

```
[1/3] Load epair and build GE Graph
[2/3] Extract GE Graph
[3/3] Compile graph
[PASS] aclgrphSaveModel succeeded, <name>.om generated
 All 3 stages passed
```

源码（`compile_main.cc:20` 的 `PrintStage` + `:128/135/164/196`）。

### ② GE dump 图 + 日志（export 环境变量）

cann_lm_compile 跑前 export 这些 env，GE 编译时 dump 图 + 详细日志：

```bash
# 跑前 source 编译 env（② OM 编译，同 AGENTS.md「运行环境设置」）
source scripts/set_env_cann.sh   # ②→0715 CANN

# ── GE dump 图（编译期各阶段 GE 图落盘 txt，定位算子/融合/shape） ──
export DUMP_GE_GRAPH=1          # 1=全量图, 2=精简, 3=只 op; 越大 dump 越多
export DUMP_GRAPH_LEVEL=1       # dump 图个数（1=只首图, 2=全部阶段）

# ── GE 详细日志到 stdout（定位 "not registered" / 算子缺失 / 编译失败） ──
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0  # 0=DEBUG 最详, 1=INFO, 2=WARNING, 3=ERROR
# 文档: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/docs/zh/user_guides/atc_tools/references/dump_graph_details.md

# ── 跑编译（cann_lm_compile 经 build_cpp.sh / build.sh cpp 产出） ──
build/cann_lm_compile <epair_dir> --soc_version Ascend910_9362
# GE dump 图落 ~/*.txt（编译目录），日志到 stdout
```

> ⚠️ `DUMP_GE_GRAPH` 产出量大（每阶段多 txt），定位完及时 `unset` 避免刷盘。
> `ASCEND_GLOBAL_LOG_LEVEL=0` 日志极详（含 GE 内部 pass），定位完调回 2/3。

### ③ OM 算子列表 dump（atc --mode，编译后）

OM 是编译后二进制，用 CANN `atc` 工具解析（**真 CANN env，不含 stubs**）：

```bash
source scripts/set_env_cann.sh   # ② OM 编译 env（与编译同套）

# 概要信息（秒级：soc_version / 权重 / tbe_kernels / stream_num 等）
atc --mode=6 --om=<path>/decode.om

# 算子列表（导 JSON 再解析，看融合后算子类型 + input_desc/output_desc/attr）
atc --mode=1 --om=<path>/decode.om --json=/tmp/opencode/decode_om
python3 -c "import json;from collections import Counter;ops=json.load(open('/tmp/opencode/decode_om'))['graph'][0]['op'];[print(v,k) for k,v in Counter(o['type'] for o in ops).most_common()]"
```

> OM 可达数 GB，`--mode=1` JSON 通常几十 MB；解析产物存 `/tmp/opencode`，不散落 `examples/`。
> `framework_type` 为空属正常（epair 走 GE 直编，非标准框架）。
> OM 内算子是**编译融合后**的最终算子，比 IR 阶段少 50%+（Reshape/Transpose 被融合进 MatMul/RmsNorm）。

### ④ 自定义 tap（加 cout 打印算子/shape）

需更深定位时，在 `compile_main.cc` 加 tap（PrintStage 之外的手动 print）：

```cpp
// compile_main.cc: Extract GE Graph 阶段后, 遍历 GE 图 dump 算子列表
auto graph = ...;  // GE ComputeGraph
std::cout << "[tap] ops=" << graph->GetAllNodesSize() << std::endl;
for (auto* node : graph->GetAllNodes()) {
    std::cout << "  " << node->GetType() << " " << node->GetName() << std::endl;
}
```

改后重编 `bash scripts/build.sh cpp` + 重跑。**改 `cann_lm/` 前先找确认**（AGENTS.md 约定）。

### 排障场景

| 症状 | 用哪种 dump |
|------|-----------|
| 编译失败 "op not registered" | ② `ASCEND_SLOG=1` + `LOG_LEVEL=0` 看 GE 日志 |
| 算子融合结果不符预期 | ② `DUMP_GE_GRAPH=1` 看各阶段图 + ③ `atc --mode=1` 看融合后 OM 算子 |
| OM 权重/tbe 大小异常 | ③ `atc --mode=6` 看概要 |
| 编译卡某阶段 | ① PrintStage 看停哪 + ② 日志 |
| 自定义算子是否进图 | ④ 加 tap 遍历 GE node |

## 关联

- [export_design.md](./export/export_design.md)（导出全流程 + epair IR）
- [quant_design.md](./quant/quant_design.md)（量化精度验证）
- [README.md](./README.md)（跨模块调用链 + 精度汇总）
- [`gen_golden_with_dump.py`](../../../examples/test_converter/llm_models_run_on_cpu/gen_golden_with_dump.py)（工具源码，支持 `--quant-path` 量化 dump）
- [`verify_hook_vs_dump.py`](../../../examples/test_converter/dopt_op_benchmark/verify_hook_vs_dump.py)（hook vs DumpInterpreter 一致性验证）
- [`sim_inference.py`](../../../examples/test_converter/dopt_op_benchmark/sim_inference.py)（量化仿真推理，单卡/两卡）
