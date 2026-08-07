# 量化接入 Converter 设计文档

> 将 HuggingFace 模型经 GPTQ 权重量化后导出为含 s8s4 量化 matmul 算子的 epair 格式
>
> 量化方案：**W4A8 s8s4**（激活 s8 per-token 动态对称 + 权重 s4 per-group 对称）
>
> 算子实现：路径 C'（复用 CANN V4 整数 int8/int4）经实测不可达 s8s4 正确结果，转方案 A 自研 `quant_matmul_s8s4`（详见下文"方案 A"一节）
>
> 最后更新: 2026-08-06

## 目录

- [概述](#概述)
- [已有 dopt_lm 框架盘点](#已有-dopt_lm-框架盘点)
- [量化策略](#量化策略)
- [端到端流程](#端到端流程)
- [待实现模块](#待实现模块)
- [int4 物化的 CANN 约定（已定：V4 整数 int4）](#int4-物化的-cann-约定已定v4-整数-int4-per-group)
- [方案 A — 自研 quant_matmul_s8s4 算子规格](#方案-a--自研-quant_matmul_s8s4-算子规格)
- [遗留问题](#遗留问题)
- [环境与执行顺序](#环境与执行顺序)

---

## 概述

`cann_lm/converter/quant/` 下已内置一套华为 `dopt_lm`（GPTQ 风格权重 PTQ 框架，vendored），能在 **NPU/GPU** 上对 HF 模型做 GPTQ 权重量化，产出 per-layer `group_scale`（per-group 对称，offset=0）+ GPTQ 修正后的 fp16 权重。

本方案**复用 dopt 的 GPTQ 算法**，补两块缺口接入 converter 导出链路：
1. **物化（materialize）**：把 dopt 产出的 `(fp16 weight, group_scale)` 物化成真 int4 payload + scale，供 CANN `QuantBatchMatmulV4` 使用（dopt 当前只做 fakequant，权重留在 fp16）。
2. **导出用 QuantLinear**：`forward` 调 `cann_lm::quant_matmul`（而非 dopt 的 fakequant `F.linear`），使 `torch.export` 产出 `quant_matmul` 节点 → 经现有 `_quant_matmul_converter` → `ge.QuantBatchMatmulV4` IR。

### 能力边界

- **输入**：HF 模型目录 + 校准数据（`quant/calibration.json`）。
- **输出**：epair 模型目录，其 `ep_json_with_ascend_ir.json` 含 `QuantBatchMatmulV4` 节点，权重 bin 含物化的 int4（int8 存储）+ fp32 scale。
- **量化范围**：全部 `nn.Linear`（q/k/v/o/gate/up/down/per_layer_input_gate/per_layer_projection），**除 `embed_tokens` 与 `lm_head`**。
- **不包含**：真实 NPU 上的 OM 编译期 `QuantBatchMatmulV4` 算子实现注册（见 [待实现模块](#待实现模块) Track 0）；CPU-only 环境跑 GPTQ（GPTQ 需 NPU/GPU+校准，CPU 跑不了，见 [环境与执行顺序](#环境与执行顺序)）。

---

## 已有 dopt_lm 框架盘点

代码位于 `cann_lm/converter/quant/dopt_lm/`。关键事实：

### 入口与主流程

| 文件　　　　　　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| --------------------------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `opt_main.py`　　　　　　　　　| CLI 入口。两步：① 不存在 dopt_config 时 `generate_config` 生成脚手架并退出；② 调 `llm_quant_pipline` → `weight_quant_pipline`。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| `opt_funcs/eco_stage/train.py` | `weight_quant_pipline`：有 KD 走 `train`，否则走 `ptq_quant_pipline`。`ptq_quant_pipline` 调 `weight_quant_loop` 跑 GPTQ，再 `prepare_model`→`optimize_model` 把 `nn.Linear→QLinear`，注入 scale，存 `trained_quant_weight.pth`。 |
| `do_opt.py`　　　　　　　　　　| `optimize_model(model, config_file)`：按 `layer_strategy` 把目标 module wrap 成 `QLinear`/`QEmbedding`，`replace_module_by_names` 做模块替换。`set_quant_state`/`set_calibrate_state` 切换 fakequant 开关。　　　　　　　　　　　 |
| `builder.py`　　　　　　　　　 | `init_builder()` 注册 `modules` + `quantizer`（`Register` 模式）。　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|

### GPTQ 产出

`weight_quant_loop`（`ptq_algo_heissen/ptq_loop.py`）返回 `(quantizers, model.state_dict())`：
- `quantizers`：per-layer quantizer，带 `.group_scale`（per-channel × per-group，对称，offset=0）。
- `model.state_dict()`：GPTQ 修正后的 **fp16** 权重。

`ptq_quant_pipline` 随后：
1. `prepare_model` → `optimize_model` 把 HF 模型的 `nn.Linear` 换成 `QLinear`。
2. `set_quant_state(weight_state=True, input_state=False)` —— **只做权重量化，不量化激活**（dopt fakequant 路径）。
3. `load_state_dict(weight_q_weight)` + `weight_q_params_adapter_into_dopt`：把 `group_scale` 注入每个 `QLinear.quant_op.weight_quantizer.s`，`offset = torch.zeros_like(...)`（置零 → 对称）。
4. `torch.save(dopt_model.state_dict(), output_dir/trained_quant_weight.pth)`。

### 量化算子语义（关键）

- **`QLinear.forward`（`modules/qlinear.py`）是 fakequant**：`weight, x = self.quant_op(weight, x)`（量化后**反量化回 fp**），再 `F.linear(x, weight)`。`Quantizer.quantize`（`quant.py`）返回 `scale*(q-zero)`（dequant 值）。→ dopt 的 `QLinear` **不能直接用于导出 IR 替换**（导出图会是 `aten.mm/MatMul`，不是 `quant_matmul`）。
- **`Quantizer`（`quant.py`）**：`maxq = 2^bits - 1`（int4 → 15）；`sym=True` → `scale = (xmax-xmin)/maxq`，`zero = (maxq+1)/2`；`quantize(x) = clamp(round(x/scale)+zero, 0, maxq)`。但 adapter 把 offset 置零 → 实际 `q = round(x/scale)`（无 zero shift）。**这与物化成真 int4 时 signed/unsigned 的取定相关，见 [遗留问题](#遗留问题)**。

### dopt 与 converter 的差距

| 你的两步 | dopt 现状 | 缺口 |
|---|---|---|
| ① 权重量化→scale/offset→存权重文件 | ✅ GPTQ 算法（产出 group_scale + GPTQ fp16 权重，offset=0 对称） | ❌ 不物化真 int4 payload（state_dict 是 fp16 weight+scale）；❌ 跑在 HF `AutoModelForCausalLM` 非 cann_lm 模型；❌ 未接 converter 的 pack/bin |
| ② 导出前算子 IR 替换 | ✅ 有模块 swap 基础设施（`optimize_model→QLinear`） | ❌ `QLinear` 是 fakequant `F.linear`，**非 `cann_lm::quant_matmul`**，导出图是 `aten.mm/MatMul` 而非 `QuantBatchMatmulV4` |

---

## 量化策略

### 方案：W4A8 s8s4（对称）

| 侧 | 位宽 | 粒度 | 对称/非对称 | scale 来源 | offset |
|----|------|------|-------------|-----------|--------|
| 权重 (x2) | int4 (s4) | per-output-channel × per-group(group_size, 默认 256) | 对称 | GPTQ 离线求（`group_scale`） | 0（对称，不存） |
| 激活 (x1) | int8 (s8) | per-token 动态 | 对称 | 运行时从 x 算（`x_scale = x.abs().amax(-1,keepdim)/127`） | 0 |
| 输出 | fp16/bf16 | — | — | — | — |

对应 dopt `strategy.py` 的 **`Quant_act_weight_eco`**：
```python
"Quant_act_weight_eco": {
    "weight": {"blocksize": 128, "group_size": 256, "bit": 4, "per_channel": True, "weight_algo": "omni_group_min_max"},
    "input":  {"bit": 8, "per_channel": False, "input_algo": "dyn_s8", "unsigned_quant": False},
}
```
- `group_size` 可经 `"Quant_act_weight_eco;128"` 语法覆盖（`strategy.get_optimize_cfg`）。
- `dyn_s8` = 激活动态 per-token int8（signed）—— 对应 W4A8 的 A8 侧，运行时算 `pertoken_scale` 传给 `QuantBatchMatmulV4` 的 `x1_scale`。

### 量化范围与校准

- **范围**：全部 `nn.Linear`（`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/per_layer_input_gate/per_layer_projection`），**除 `embed_tokens` 与 `lm_head`**。
- **校准数据**：`cann_lm/converter/quant/calibration.json`（已存在，557KB）；`config.yaml` 默认 `cutoff_len=512`、`ptq_samples=128`。
- **GPTQ 跑在 HF 模型**：dopt 现成依赖 `transformers.AutoModelForCausalLM` + `accelerate` dispatch，不改 dopt；量化参数按层名 port 到 cann_lm 模型（两者命名一致：`model.layers.N.self_attn.q_proj` 等）。

### `cann_lm::quant_matmul` 算子签名（已存在）

定义于 `cann_lm/models/fused_ops/quant_matmul.py`（fake/meta only），GE 转换器 `_quant_matmul_converter.py` 映射到 `ge.QuantBatchMatmulV4`：
```
quant_matmul(Tensor x1, Tensor x2, Tensor scale, *,
             Tensor? offset=None, Tensor? pertoken_scale=None, Tensor? bias=None,
             int? output_dtype=None, int? x1_dtype=None, int? x2_dtype=None,
             int? pertoken_scale_dtype=None, int? scale_dtype=None,
             int[]? group_sizes=None, Tensor? y_scale=None,
             bool transpose_x1=False, bool transpose_x2=False) -> (Tensor y)
```
- `x2` = 物化的 int4 权重（int8 存储），`scale`/`offset` = 权重 scale/offset（对称 → offset=None/0）。
- `pertoken_scale` = 激活 per-token scale（runtime 输入，非权重）。
- `group_sizes=[m,n,k]` → converter 打包成单 int `(m<<32)|(n<<16)|k` 喂 `QuantBatchMatmulV4.group_size`。
- `output_dtype` 用 `_SCALAR_TYPE_TO_TORCH_DTYPE`（5=fp16, 7=bf16）。

---

## 端到端流程

`cann-lm-converter.py --quant`（**NPU 机**）：

```
converter --quant
  │
  ├─ A. dopt GPTQ（在 HF AutoModelForCausalLM 上，NPU）
  │    weight_quant_loop(hf_model, calibration.json, nsamples=128, seq_len=512, NPU)
  │      → quantizers{layer → .group_scale（per-channel×per-group，对称 offset=0）}
  │        state_dict{layer.weight = GPTQ 修正后 fp16 权重}
  │    （可直接用 dopt ptq_quant_pipline，产出 trained_quant_weight.pth）
  │
  ├─ B. converter 加载 cann_lm 自定义模型
  │    loader.load_model(hf_path) → ModelArtifact.model（cann_lm Gemma4ForCausalLM，已有）
  │
  ├─ C. 参数 port（HF→cann_lm 同名匹配）+ 物化
  │    for each Linear in cann_lm model:
  │      (w_fp16, group_scale) = fetch from dopt 产物（按 layer name）
  │      (weight_q:int8_packed_or_unpack, scale:fp32) = materialize_int4(w_fp16, group_scale, group_size)
  │
  ├─ D. 算子 IR 替换：cann_lm 模型 nn.Linear → 导出用 QuantLinear
  │    QuantLinear 持 weight_q/scale 为 buffer；
  │    forward(x): pertoken_scale = x.abs().amax(-1,keepdim)/127
  │               out = torch.ops.cann_lm.quant_matmul(
  │                   x, weight_q, scale, offset=None, pertoken_scale=pertoken_scale,
  │                   output_dtype=fp16, x2_dtype=<int4 or int8>, scale_dtype=fp32,
  │                   pertoken_scale_dtype=fp32, group_sizes=[1,1,group_size])
  │
  └─ E. save_epair（已有链路，无需改 pack）
       torch.export(wrapper) → 图含 cann_lm::quant_matmul 节点（非 aten.mm）
       → register_ge_converters（已含 _quant_matmul_converter）
       → torch_delegate.export_to_epair → ge.QuantBatchMatmulV4 IR
       → consolidated_weights_save：weight_q(int8)/scale(fp32) 进 bin（pack 通用）
       → C++ ge_graph_builder.PreparePlaceholders 把 weight_q/scale 作 Const 节点喂 QuantBatchMatmulV4 的 x2/x2_scale 输入；pertoken_scale 是 runtime 输入
```

### 权重落盘（无需改 pack）

`epair_utils.pack_weights` 是**通用**的——序列化 state_dict 里任意 tensor（int8 weight_q + fp32 scale 自动进 bin，bf16→int16 view 已处理）。`weight_info.json` 按 name 记 offset/size。**量化权重无需改打包逻辑**。

### C++ 侧（已通用，需补算子注册）

- `ge_graph_builder.PreparePlaceholders` 通用喂权重常量到任意算子输入（`ReadTensorData`→Const 节点）→ `QuantBatchMatmulV4` 的 `x2`/`x2_scale`/`x2_offset` 自动喂入。
- `ge_weight_loader` 支持 int8/uint8/fp32 等 dtype（`ElementSizeForPt2Dtype` / `Pt2DtypeFromGeString`）。
- **缺口**：`examples/export_llm_model/reg_register.h` 未注册 `QuantBatchMatmulV4`（只有 `MatMul`/`BatchMatMul`）→ OM 编译报 `UNMAPPED_OPERATOR`。见 [待实现模块](#待实现模块) Track 0。

---

## 待实现模块

### Track 1 — 物化 + 导出 QuantLinear（Python 侧）  ✅ 已完成

- **`materialize.py`**（`cann_lm/converter/quant/materialize.py`）：
  ```python
  def materialize_int4(weight_fp16, group_scale, group_size, *, signed=True, packed=True):
      """fp16 GPTQ weight + per-group scale → (weight_q:int8, scale:fp32)。
      q = round(weight / scale)  # scale broadcast per-group
      # signed: clamp q to [-8, 7]；unsigned: clamp to [0, 15]  ← 遗留，见下
      # packed: 2 int4 / byte (int8)；unpack: 1 int4 / int8  ← 遗留
      return weight_q_int8, scale_fp32
  ```
  - **占位实现**：先 `unpack + signed[-8,7]` 跑通 Python 流程；CANN 对齐（packed/signed）放遗留。
- **`QuantLinear`**（`cann_lm/models/fused_ops/quant_linear.py` 或 `quant/`）：持 `weight_q`/`weight_scale` buffer，forward 调 `cann_lm::quant_matmul`（签名见上）。**可选 `offset` 参数**：非 None 时存为 `weight_offset` buffer、forward 转置后传给 `npu_quant_matmul`（GE converter `x2_offset`/CPU eager `w_fp+=offset` 已就绪）；对称时传 None（dormant plumbing，非对称 offset 语义见遗留 2）。
- **`swap_to_quant_linear(model, scope)`**：遍历目标 `nn.Linear`（除 embed/lm_head），按名 port 物化的 int4+scale，替换为 `QuantLinear`。接入 `loader.load_model` 后、`save_epair` 前（`--quant` 触发）。
- **`swap_to_quant_linear_from_dopt` 现读 dopt 产物的 `bit`/`unsigned_quant`/`offset`**（`swap._read_dopt_quant_meta`）：bit→`materialize nbits`、offset→`QuantLinear.weight_offset`（全 0 归一化为 None）、unsigned_quant 校验=0（非 0 报错，unsigned int4 仍遗留）。dopt 产物 key 实测：`{name}.quant_op.weight_quantizer.{bit,int64 标量}/{unsigned_quant,fp32 标量}/{offset,fp32 [out*groups]同 s}`。

### Track 2 — CPU eager impl（数值回归）  ✅ 已完成

`cann_lm/models/fused_ops/quant_matmul.py` 加 `@torch.library.impl("cann_lm::quant_matmul", "cpu")`（仿 `add_rms_norm` 模式）：
- 手动：`x1_q = round(x1/pertoken_scale).to(int8)`；weight_q 解包 int4→fp16（per-group 反量化 `w_fp16 = unpack(weight_q) * scale`）；`out = (x1_q.float() * pertoken_scale) @ w_fp16`；cast `output_dtype`。
- 精度参考用（不必快）。回归：cann_lm 模型 swap 后 eager forward vs fp16 baseline，比 logits maxdiff + argmax 命中率。

### Track 3 — CLI/config 接入  ✅ 已完成

- `cann-lm-converter.py` 加 `--quant`（触发 A→E）、`--quant-group-size`（默认 256）、`--quant-scope`（默认 all-no-embed-lmhead）。
- `load_model` 后若 `--quant`：调 dopt GPTQ（A）→ 物化（C）→ swap QuantLinear（D）→ 正常 `save_epair`（E）。
- `model_config.json` 记量化元信息（scheme=W4A8-s8s4, group_size, scope, packing）。

### Track 0 — C++ 前置

`examples/export_llm_model/reg_register.h` + `examples/export_and_load_epair/reg_register.h` 加 `REG_OP(QuantBatchMatmulV3)` + `REG_OP(QuantBatchMatmulV4)` + `REG_OP(DynamicQuant)`（对齐 `auto_generated_ge_raw_ops.py` 的 CANN REG_OP 定义）。**已完成**：两文件各加 3 算子，REG_OP/OP_END_FACTORY_REG 配对平衡（75/75 & 44/44）。注意 V3=整数 int4/int8、V4=MXFP（见下节遗留）——两者都注册，留待 converter 选型。

---

## int4 物化的 CANN 约定（已定：V4 整数 int4 per-group）

Track 0 注册算子时核对 CANN REG_OP（`torchair/.../auto_generated_ge_raw_ops.py`）发现 V4 的 REG_OP 定义列的是 MXFP dtype（DT_FLOAT8_E4M3FN / DT_FLOAT4_E2M1），曾据此误判"V4 = MXFP only，非整数 int4"。

**实测推翻此结论**：本机 CANN 9.2.0 的预编译 kernel 目录 `opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_nn/quant_batch_matmul_v4/` 下有 **7 个预编译 .o**，其中含整数 int4 组合（命名约定 `QuantBatchMatmulV4_{x1_fmt}_{x2_fmt}_{x1_dtype}_{x2_dtype}_{out_dtype}_{variant}`）：

| 预编译 kernel | x1(act) | x2(weight) | out | 方案 | tilingKey |
|---|---|---|---|---|---|
| `ND_ND_int4_int4_{bf16,fp16}` | int4 | int4 | bf16/fp16 | A4W4 | 1040 |
| **`ND_ND_int8_int4_{bf16,fp16}`** | **int8** | **int4** | bf16/fp16 | **A8W4 = W4A8** | **24** |
| `ND_ND_int8_int8_bf16` | int8 | int8 | bf16 | A8W8 = W8A8（已验证跑通）| 529 |
| `ND_NZ_int8_int4_{bf16,fp16}` | int8 | int4(NZ) | bf16/fp16 | W4A8 (x2 NZ 格式) | 276 |

→ **V4 同时支持整数 int4（`int8_int4` = W4A8）和 mxfp4**，REG_OP 只列了 mxfp 分支但 runtime kernel 覆盖整数 dtype。**V3（per-channel 无 group_size）不再必要**。

**当前实现的对齐状态**：`_quant_matmul_converter` 把 `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4`，`materialize_int4` 产出**整数 int4（int8 unpack 存）**。x1=int8（DynamicQuant 产）、out=fp16 命中 `ND_ND_int8_int4_fp16`（tilingKey=24）。**proto（`cann/ops-nn` 官方 `quant_batch_matmul_v4_proto.h`）暴露两个 converter 须对齐点**：

| 项 | proto 要求（A8W4） | converter 现状 | 对齐 |
|---|---|---|---|
| x2(weight) dtype | `DT_INT4`（proto `INPUT(x2, {DT_FLOAT4_E2M1, DT_INT4, DT_INT8})` 明列 int4 for A8W4）| 不设 `x2_dtype` → x2 留 `DT_INT8`（unpack 存） | ❌ 须设 `x2_dtype` 触发 `ge.Bitcast`→`DT_INT4` |
| transpose_x2 | proto 注释："Currently, it should **always be true**"；true 时 x2 shape `(n,k)`、x2_scale `(n, k/group)` | 硬编码 `transpose_x2=False`，预转置 x2=`[K,N]`、scale=`[K//group,N]` | ❌ 可能须改 True（不预转置） |
| x1(act) dtype | `DT_INT8` | DynamicQuant 产 int8 | ✅ |
| x2_scale dtype | `DT_FLOAT32`/`DT_UINT64` | fp32 | ✅ |
| x1_scale(pertoken) | `DT_FLOAT32` shape `(m,)` | fp32 `(m,)` | ✅ |
| compute_type attr | -1(auto, 整数 int4 走 int8_int4 kernel) / 4(f8f4, mxfp) | 默认 -1 | ✅ |
| dtype attr | 1=fp16, 27=bf16 | output_dtype=5(fp16)→映射 | 需验映射 |

→ converter 的 `transpose_x2=False` + 不设 `x2_dtype` 可能就是 OM 编译没命中 `int8_int4` 预编译 kernel 的原因。**待实测确认**：(a) x2 用 `DT_INT4`（unpack int8 存 or int4pack `(k,n//8) int32`）+ `transpose_x2=True` 是否命中 tilingKey=24 预编译 kernel；(b) unpack int8 存储是否被 DT_INT4 接受（还是须 int4pack）。

`DynamicQuant`（act int8）正确匹配（已注册、QuantLinear 已用 `npu_dynamic_quant`→`custom_op("DynamicQuant")`）。

### 路径选型（已定：C'）

| 路径 | 选用 op | 改动 | 取舍 |
|---|---|---|---|
| **C'. V4 整数 int4 per-group（已选定）** | `QuantBatchMatmulV4` | 命中 `ND_ND_int8_int4` 预编译 kernel；确认 x2 存储布局（unpack vs int4pack）+ 可能设 `x2_dtype` 触发 Bitcast 到 DT_INT4 | 匹配真实 CANN 整数 int4 kernel，支持 per-group，无需 V3/无需 mxfp4 转换；预编译 kernel 跳过 LxFusion 重编 |
| ~~A. V3 整数 per-channel~~ | — | — | 弃用：V4 已有 int8_int4 kernel，V3 per-channel 粒度粗没必要 |
| ~~B. V4 mxfp4~~ | — | — | 弃用：V4 已支持整数 int4，无需转 mxfp4 |
| ~~C. 维持现状（旧结论）~~ | — | — | 旧结论"V4=mxfp4 不符、真 NPU OM 编译失败"已被预编译 kernel 实测推翻 |

**物化层已验证无 bug（diag）**：dopt 产物用 reshape `[out, in//group]` + signed `[-8,7]` + offset=0 物化，重建 vs GPTQ 权重 cosine=1.0000（与 dopt 一致）。物化的整数 int4 直接喂 V4 的 `int8_int4` kernel（路径 C'）。

---

## V4 NPU 验证状态（本机 Ascend910_9362 / ascend910_93 族）

`npu_quant_matmul`（= `aclnnQuantMatmulV5` = QuantBatchMatmulV4 eager 接口）在本机 NPU 实跑验证：

| 模式 | 状态 | 详情 |
|---|---|---|
| **W8A8（int8+int8）K-C** | ✅ **跑通，maxdiff≈0** | x1=INT8, x2=INT8, pertoken_scale=fp32(m,), scale=fp32(n,) per-channel → fp16。与 ref `(x1@(q*scale).T)*pertoken` 逐位一致（maxdiff=5.96e-08）。验证脚本 `examples/verify_quant_matmul_v4_w8a8.py`。证明 V4 算子 + scale/pertoken 方法论在本机可用。 |
| **W4A8 int4（A8W4 K-G）** | ⚠️ op 能执行，结果错 | int4pack(`npu_convert_weight_to_int4pack`→`(k,n//8) int32`) + UINT64 scale(`npu_trans_quant_param`) + group[0,0,256] + k=256 对齐 + pertoken(m,1) + yOffset。op 不报错但输出与 ref 偏差大，且 `(ref-y0)/pertoken` **非 per-channel 常量**（x1 相关）→ op 内部 int4 反量化 ≠ fp32 `q*scale`，是 packing 交错排布 + UINT64 fixed-point scale + yOffset 三者语义未对齐。 |

### A8W4 int4 的关键工具（已验证可用）
- `torch_npu.npu_convert_weight_to_int4pack(int32_kn)` → `(k, n//8) int32`（每 int32 存 8 个 int4，交错排布）。DT_INT4=29。
- `torch.ops.npu.npu_trans_quant_param(fp32_scale_1d)` → int64（UINT64 deqScale，fp32→定点）。
- `torch.ops.npu.npu_quant_matmul`（V5 aclnn）A8W4 K-G 要求：x2Scale=UINT64/INT64、yOffset=FLOAT32(n,) 非空、k 256 对齐、group_sizes=[0,0,256]。

---

## 遗留问题

1. ~~**int4 CANN 算子选型（核心遗留，已具体化）**~~ — **[已解决，选 C']** 见上节"int4 物化的 CANN 约定"。实测 CANN 9.2.0 预编译 kernel 目录含 `QuantBatchMatmulV4_ND_ND_int8_int4_{bf16,fp16}_high_performance.o`（tilingKey=24），即 **V4 支持整数 int4 W4A8**（x1=int8, x2=int4），推翻旧结论"V4=MXFP only"。**已选路径 C'：V4 整数 int4 per-group**（无需 V3、无需 mxfp4 转换）。剩唯一未知：V4 DT_INT4 期望的 x2 存储布局（unpack int8 vs int4pack `(k,n//8) int32`），需实测。
2. **dopt 物化数学对齐**：dopt `Quantizer` 的 `maxq=15` 无符号+zero-point 与 adapter 置零 offset 的组合；diag 已确认用 reshape`[out,in//group]`+signed`[-8,7]`+offset=0 物化，重建 vs GPTQ 权重 cos=1.0000（物化层无 bug）。物化的整数 int4 直接喂 V4 的 `int8_int4` kernel（路径 C'，同 1）。
   - **dopt 产物 bit/unsigned_quant/offset 现已可读+驱动**（`swap._read_dopt_quant_meta`）：实测 dopt `model.safetensors` 每个 QLinear `{name}.quant_op.weight_quantizer.{bit,unsigned_quant,offset}` 都在（bit=int64 标量=4、unsigned_quant=fp32 标量=0.0、offset=fp32 扁平`[out*groups]`同`s`布局）。`swap_to_quant_linear_from_dopt` 现读 bit→`materialize nbits`、读 offset→`QuantLinear.weight_offset` buffer（对称时全 0 归一化为 None，dormant plumbing，下游 GE converter `x2_offset`/CPU eager `w_fp+=offset` 已就绪）、读 unsigned_quant 校验必须=0（unsigned int4 不支持，非 0 报错，见下条）。E2E 验证 qwen3-0.6B 196 层 swap + forward 跑通。**unsigned_quant 仍遗留**：非 0（unsigned int4 `[0,15]`）触及 CANN int4 signed/unsigned 编码约定（见遗留 1 的 V4 选型），当前直接报错拒绝。
   - **offset 反量化语义未对齐（未来非对称才触发）**：dopt `QuantizerFuncIntOffset` 反量化 `w_fp=(q-offset)*scale`；当前 CPU eager kernel `_npu_quant_matmul_cpu` 用 `w_fp=q*scale+offset`（公式不符）且 offset 未按 group `repeat_interleave` 扩维。对称 offset=None 时不触发；非对称支持需先对齐 op 的 offset 语义+扩维。
3. **CPU-only 适配**：converter 侧 GPTQ 跑不了（需 NPU/GPU+校准）。**`--quant-path` 已作兜底**：离线机跑 dopt GPTQ 产出（`quant_output/model.safetensors` 或 `trained_quant_weight.pth`），converter 加载它只做物化+swap+导出（不跑 GPTQ）。dopt `opt_main` 在本机 NPU 上跑 GPTQ 曾静默卡住（step A 接入 converter 待解）。
4. **`reg_register.h` 算子注册**——**已完成**（Track 0）：两文件加 `QuantBatchMatmulV3`+`QuantBatchMatmulV4`+`DynamicQuant`，配对平衡。op 定义须对齐目标真 NPU CANN 版本（ATTR 名/OUTPUT 个数/dtype 枚举如 DT_INT4/DT_FLOAT4_E2M1 是否定义）。
5. **OM 编译验证（新遗留）**——见下节"OM 编译调试记录"。

---

## OM 编译调试记录（W4A8 qwen3-0.6B 2 层）

在 NPU（Ascend910_9362 / CANN 9.2.0）上对 quant epair 做 `aclgrphBuildModel`（OM 编译），逐步调试：

### OM 编译步骤与状态

| 步骤 | 状态 | 修复 |
|------|------|------|
| Step 1: EpairModelLoader | ✅ PASS | — |
| Step 2: GetGEGraph | ✅ PASS（经多轮修复） | 见下"修复清单" |
| Step 3: aclgrphBuildModel (TBE) | ❌ FAIL | `LxFusion: Unsupported platform socVersion[Ascend910_9362]` |

### 修复清单（Step 2 GetGEGraph 从 FAIL → PASS）

1. **重建 binary**：旧 binary 在 `reg_register.h` 加 V4 前构建 → 无 QuantBatchMatmulV4 注册 → C++ parser 创建未知 op → heap corruption（`corrupted size vs. prev_size`）。重建后 Step 2 PASS。

2. **`is_cann_compat` monkey-patch**（`ge_converter.py`）：stub GE 的 `_torchair.check_cann_compat` 拒绝 DynamicQuant（`OperatorFactory find optype DynamicQuant failed`）。Patch `is_cann_compat = lambda *a, **k: ""` 绕过 → torchair 原生 `custom/npu_dynamic_quant.py` converter（用 `ge.DynamicQuant` 全 IR）加载并注册 → ep_json 含完整 IR。

3. **删掉手写 DynamicQuant converter**：之前用 `custom_op("DynamicQuant")`（ir=None → 无 IR → TBE 编译器无法选 kernel）。改为依赖 torchair 原生 converter（`ge.DynamicQuant` 全 IR）。

4. **`_quant_matmul_converter` 用 `ge.QuantBatchMatmulV4`（全 IR）**：`custom_op("QuantBatchMatmulV4")` 只 3 输入 → V4→V3 融合输入映射丢失 → `Assert (scale) != nullptr`。`ge.QuantBatchMatmulV4` 全 IR（10 输入含 None 占位）→ 融合正确映射。

5. **`_strip_none_input_descs`（`epair_utils.py`）**：QBMM 的 OPTIONAL_INPUT bias/y_scale/offset 等 None inputDesc → C++ parser 创建 GE 输入（dtype=None, shape=[]）→ `InferShape: dimensions a(8) and b(2048) must be same`（bias 维度不匹配）。后处理：只对 QBMM strip `dtype=None AND input ref=""` 的 inputDesc + 对应 input 引用（不影响其他 op 如 Cast/Mul）。

6. **dummy bias**（`quant_linear.py`）：QBMM `InferNDimWithBias` 不处理 bias=nullptr → 即使 strip 去掉 bias inputDesc，REG_OP 定义 bias 为 OPTIONAL_INPUT → InferShape 仍处理 → 报错。QuantLinear.forward 传 `dummy_bias = torch.zeros(n_out)` → bias 非空 → InferShape 通过。

7. **DynamicQuant fake meta scale shape `(m,)`**（非 `(m,1)`）：V4 的 x1_scale 期望 1D `(m,)`（per-token），2D `(m,1)` 导致 InferShape 不匹配。

### Step 3 TBE 编译失败根因

```
[WARNING] LxFusionFinalize: "Unsupported platform, socVersion[Ascend910_9362]"
[ERROR] ProcessAllFailedCompileTasks: failed to recompile single op[QuantBatchMatmulV3_0_QuantBatchMatmulV4_*]
```

- V4→V3 融合后，TBE 尝试用 LxFusion 实时编译 V3 kernel → **LxFusion 不支持 Ascend910_9362** → 4/12 个 QBMM 编译失败（8 个用预编译 binary 通过）。
- 这不是代码 bug——**CANN 9.2.0 的 LxFusion 编译器不支持本机芯片 Ascend910_9362**。
- **根因新解**（见上节"int4 物化的 CANN 约定"）：当时 converter 导出的 V4 节点 dtype/format 组合**没命中任何预编译 kernel**（7 个预编译 .o 见上节表），触发 LxFusion 重编（910_9362 不支持）→ 失败。**若让 dtype 组合命中 `ND_ND_int8_int4` 预编译 kernel（x1=int8, x2=int4, out=fp16/bf16，tilingKey=24），runtime 直接用预编译 binary，跳过 LxFusion**，OM 编译应能过。这是路径 C' 的关键依据。
- 备选：更新 CANN 到支持 Ascend910_9362 的版本（让 LxFusion 也能用）。

### 关键 ep_json 修复效果对照

| 修复 | ep_json QBMM inputDesc | InferShape 结果 |
|------|----------------------|----------------|
| 无修复（custom_op） | 3 个（x1,x2,x2_scale），无 IR | TBE: Assert scale!=nullptr |
| ge.QuantBatchMatmulV4（全 IR，不 strip） | 10 个（含 None bias dtype=None, shape=[]） | InferShape: dimensions a(8) b(2048) |
| ge.QuantBatchMatmulV4 + strip None | 4 个（x1,x2,x1_scale,x2_scale） | 同上（bias REG_OP 仍处理） |
| ge.QuantBatchMatmulV4 + strip + dummy bias | 5 个（x1,x2,bias(DT_FLOAT16),x1_scale,x2_scale） | ✅ PASS → TBE: LxFusion unsupported |

### 已验证的工具链（OM 编译路径）

- ✅ `npu_convert_weight_to_int4pack` → `(k, n//8) int32`（顺序打包，非交错——0x11111111 = 8×int4=1）
- ✅ `npu_trans_quant_param` → int64（但 bit-reinterpret `fp32.view(int32).to(int64)` 更接近正确值）
- ✅ `npu_quant_matmul` eager W8A8 K-C（int8+int8+fp32 scale+fp32 pertoken → fp16）maxdiff=0
- ✅ `npu_quant_matmul` eager A8W4 K-G（int8+int32+UINT64 scale+yOffset+group[0,0,256]）能执行，但 K-G kernel 把 x1 当 int4 → 真 INT8 x1 被错误拆包
- ✅ torchair `custom/npu_dynamic_quant.py` converter（pertoken → `ge.DynamicQuant` 全 IR）
- ✅ `_strip_none_input_descs` 只对 QBMM strip None+空引用的 inputDesc
- ✅ `is_cann_compat` monkey-patch 让 stub GE 不拒绝 DynamicQuant/Bitcast 等
- ❌ TBE LxFusion 不支持 Ascend910_9362（CANN 9.2.0 版本限制）

### 环境

- **GPTQ（A 步）需 NPU/GPU**：dopt 依赖 `transformers.AutoModelForCausalLM` + `accelerate` dispatch + `autocast`/`GradScaler`，`DEVICE=cuda/npu`。当前 dev 机虽有 NPU，可跑；纯 CPU 机跑不了 GPTQ（见遗留 3）。
- converter 导出（B–E 步）本身 CPU 可跑（torch.export + pack + cpu eager 回归），不依赖 NPU。
- **OM 编译需 NPU + CANN**：`aclgrphBuildModel` 需真 NPU 环境（CANN toolkit + device）。本机 Ascend910_9362 + CANN 9.2.0 的 **LxFusion 不支持 910_9362**，但**预编译 kernel 不走 LxFusion**——只要 converter 导出的 V4 dtype 组合命中预编译 kernel（如 `ND_ND_int8_int4`，tilingKey=24），runtime 直接用预编译 binary，OM 编译应能过（见"int4 物化的 CANN 约定"路径 C'）。
- **stub GE 限制**：converter 导出在 stub 环境（venv + torchair stubs）上，`check_cann_compat` 拒绝 DynamicQuant/Bitcast 等 → 需 `is_cann_compat` monkey-patch 绕过（见 OM 调试记录修复 #2）。

### 建议执行顺序（当前进度）

1. ~~**Track 1（物化+QuantLinear）+ Track 2（cpu eager impl）**~~ —— **已完成**：`materialize_int4`/`materialize_int4_with_scale` + `QuantLinear`（W4A8/W4A16）+ `npu_quant_matmul`/`npu_dynamic_quant` cpu impl。qwen3-0.6B eager 回归（W4A8 cos 0.826、物化 diag cos 1.0000）。
2. ~~**Track 3（CLI 接入）**~~ —— **已完成**：`cann-lm-converter.py` 加 `--quant-path`/`--quant-group-size`/`--quant-act`；`load_model` 后调 `swap_to_quant_linear_from_dopt`（port dopt 产物→物化→swap）；`model_config.json` 写 `quant_info`。qwen3-0.6B 端到端跑通，ep_json 含 `QuantBatchMatmulV4`+`DynamicQuant` IR。
3. ~~**Track 0（C++ reg）**~~ —— **已完成**：两 `reg_register.h` 加 `QuantBatchMatmulV3`+`QuantBatchMatmulV4`+`DynamicQuant`（75/75 & 44/44 配对）。
4. ~~**OM 编译调试**~~ —— **InferShape PASS / TBE FAIL**：经 7 轮修复（binary 重建 + is_cann_compat patch + strip None + dummy bias + ge.QuantBatchMatmulV4 全 IR + DynamicQuant 用 torchair 原生 converter + fake meta scale shape），Step 1-2 PASS，Step 3 卡在 `LxFusion: Unsupported platform socVersion[Ascend910_9362]`（CANN 9.2.0 版本限制，非代码 bug）。
5. ~~**int4 CANN 算子选型**~~ —— **[已解决，选 C']**：实测 CANN 9.2.0 预编译 kernel 目录含 `QuantBatchMatmulV4_ND_ND_int8_int4_{bf16,fp16}`（V4 支持整数 int4 W4A8），推翻旧"V4=MXFP only"结论。选路径 C'（V4 整数 int4 per-group，命中预编译 kernel 跳过 LxFusion）。
6. **对齐 x2 存储布局命中预编译 kernel** ← **下一步**：
   - 实测 V4 `int8_int4` kernel 期望的 x2(weight) 存储布局：unpack int8（1 int4/int8）还是 int4pack `(k,n//8) int32`。
   - 若需 int4pack：物化后/导出侧调 `npu_convert_weight_to_int4pack`；若接受 unpack：直接喂。
   - 必要时设 `x2_dtype` 触发 Bitcast 到 DT_INT4，确保 runtime 选到 tilingKey=24 的 `ND_ND_int8_int4` kernel。
   - 重跑 OM 编译验证命中预编译 kernel（跳过 LxFusion）。
7. **dopt GPTQ step A 接入 converter**（NPU）+ 真 NPU OM 编译验证（需命中预编译 kernel 或更新 CANN）。

---

## 方案 A — 自研 quant_matmul_s8s4 算子规格

> 路径 C'（复用 CANN V4 整数 int8/int4）经实测不可达 s8s4 正确结果：
> - eager 路径 `npu_quant_matmul` int8/int4 结果错——x2_scale 要求 **uint64 deqScale（FP19 定点 + int9 offset 位打包）**，非 fp32 `q*scale` 语义（`trans_quant_param_v2.h:23-24,270-304`、预编译 kernel JSON `x2_scale dtype=uint64`）；
> - OM 编译路径经 V4→V3 融合 pass 降级（int8/int4 既非 MxA8W4 亦非 A8W8GB，见附录 A.2.2 `MeetRequirements`），而 V3 无 `int8_int4` 预编译 kernel → LxFusion 重编 → 910_9362 不支持（见上节"OM 编译调试记录"）。
>
> 故转方案 A：自研 AscendC kernel `quant_matmul_s8s4`，用 **fp32 scale** 做 `q*scale` 反量化，绕开 CANN V4 的 uint64/FP19 与 V4→V3 融合降级。本节给定算子输入输出与计算语义，供实现方对齐；**kernel 内部流水为实现方自由**。

### 计算语义（s8s4 对称，offset=0）

记激活 `x1` `[b, m, k]`、权重 `x2` `[n, k]`（nn.Linear 自然朝向）、group_size `g`（`k%g==0`）：

```
y[b,m,n] = Σ_k ( x1_q[b,m,k] · x1_scale[b,m] ) · ( x2_q[n,k] · x2_scale[n, k//g] ) + bias[n]
```

- `x1_q` int8（per-token 动态对称，每行一个 scale）、`x2_q` int4（per-channel×per-group 对称，每 (n,group) 一个 scale）。
- `x1_scale[b,m]` 是 per-token，**可提到 k 求和之外**（kernel 里后乘即可）。
- `x2_scale[n, k//g]` 是 per-group，**在 k 求和之内**（每组 g 个元素共用一个，反量化时按组广播，无法提到求和之外）。

### 推荐算子签名

| 方向 | 名称 | dtype | shape | 语义/来源 |
|---|---|---|---|---|
| in | `x1` | int8 | `[b, m, k]` | 来自 DynamicQuant 的已量化激活 |
| in | `x1_scale` | fp32 | `[b, m]` | per-token scale，来自 DynamicQuant |
| in | `x2` | int4（packed） | `[n, k]` 逻辑 ≈ `[n, k//2]` int8 物理 | nn.Linear 权重物化后的 int4 |
| in | `x2_scale` | fp32 | `[n, k//g]` | per-channel×per-group，来自 GPTQ |
| in（opt） | `bias` | 同输出 | `[n]` | 融合 add，省一个节点 |
| attr | `group_size` | int | — | K 轴分组 g |
| attr | `output_dtype` | int | — | 1=fp16 / 27=bf16 |
| attr | `transpose_x1` | bool | — | 默认 False（激活恒不转） |
| attr | `transpose_x2` | bool | — | 默认 True（x2 给 `[n,k]`，kernel 内部转置喂 cube，对齐 CANN 约定） |
| out | `y` | fp16/bf16 | `[b, m, n]` | — |

### 硬需求 vs 设计选择

> 经验教训：勿把"现状数据格式"抬成"算子对齐目标"。下表区分给定 spec（不可改）与实现方设计选择（可调）。

- **硬需求（s8s4 spec，不可改）**：`x1=int8 per-token 对称`、`x2=int4 per-channel×per-group 对称`、输出 fp16/bf16、offset=0。
- **设计选择（推荐，实现方可改）**：
  1. **激活量化拆出 op 外**：DynamicQuant 独立（已可用、已注册 `ge.DynamicQuant`），本 op 只做"已量化 int8×int4 → fp16"。权衡：若要更少图节点可把 per-token max/div 融进 op，但 cube+vector 混合融合更复杂。
  2. **scale 用 fp32**：精度参考 W8A8 已验证路径（`examples/verify_quant_matmul_v4_w8a8.py` maxdiff≈0 用 fp32 pertoken）；x2_scale 沿用 dopt/GPTQ 产出的 fp32。fp16 占带宽更小但有精度风险，实现方可据 kernel 实测定。
  3. **x2 nibble-packed `[n, k//2] int8`**：比 unpack 省 2x 带宽。kernel 内部 L1/L0B 的具体布局是实现方自由（CANN 用 NZ/int4pack，不必照搬）。
  4. **transpose_x2=True**：让 converter 不必预转置权重（更干净），kernel 在 L1 load 时处理转置。
  5. **不带 offset 输入**：对称=0；非对称未来再加 dormant 字段。

### kernel 内部流水参考（非 I/O 约束）

权重侧 per-group scale 在 k 求和之内，故 kernel 须在 cube 累加前把 int4 权重按组反量化。推荐流水（供实现方参考）：

1. L1/L0B load 时 int4→int8 unpack，按组取 `x2_scale` 广播；
2. cube 做 **int8×int8 → int32** 累加（命中 W8A8 已验证的 int8 cube 路径，规避 FP19/uint64）；
3. fixpipe 阶段乘 `x1_scale`（per-row，可后乘）+ `bias` → cast fp16/bf16。

此流水让 s8s4 数值落在"int8 cube + fp32 scale 反量化"语义上，是绕开 CANN V4 失败点的关键。

### 与现有 converter 链路的接续（当前流程 vs 方案 A 的差距）

当前导出流程是"路径 C' 形态"（`QuantLinear` 预转置权重 + `npu::npu_quant_matmul` → `ge.QuantBatchMatmulV4`），与方案 A 签名有 **7 处不匹配**，按下表改动。**已匹配项**（无需改）：DynamicQuant 拆出、`x1_scale`/`x2_scale` dtype=fp32、`offset=None`（对称）、`output_dtype` 语义。

| # | 文件:行 | 当前（路径 C' 形态） | 改为（对齐方案 A） |
|---|---|---|---|
| 1 | `materialize.py:65` | 产出 unpack int8 `[out,in]` | nibble-packed int8 `[n,k//2]` |
| 2 | `quant_linear.py:73-74` | 预转置 `w2/sc=...transpose()` | 删预转置，权重/scale 自然朝向 `[n,k]`/`[n,k//g]` |
| 3 | `quant_linear.py:101` | `group_sizes=[1,1,g]`（V4 的 48 位 m/n/k 编码） | `group_size=g`（单 int attr） |
| 4 | `quant_linear.py:90-94` | 恒传 dummy `zeros[n_out]`（V4 InferShape workaround） | 源 Linear 无 bias 时传 None |
| 5 | `quant_linear.py:95,103` | `npu::npu_quant_matmul` + `transpose_x2=False` | `cann_lm::quant_matmul_s8s4` + `transpose_x2=True` |
| 6 | `_quant_matmul_converter.py:109-124` | 映射到 `ge.QuantBatchMatmulV4`（`transpose_x2` 硬编码 False、不设 `x2_dtype`） | 改映射到本 op 的自定义 GE op/graph |
| 7 | `reg_register.h`（`examples/export_llm_model/` + `examples/export_and_load_epair/` 两文件） | 注册 `QuantBatchMatmulV4` | 注册新 op（替换/并存 V4） |

**无需改**：`ge_graph_builder.PreparePlaceholders` 通用喂权重常量到任意算子输入（weight_q/scale 自动进 bin、作 Const 喂 x2/x2_scale）；`pack_weights` 通用序列化（int8/fp32 自动落盘）；`swap_to_quant_linear_from_dopt` 的 dopt 产物 port 逻辑复用，仅下游 buffer 朝向/packing 跟随 #1#2 调整。

> 注：上文"建议执行顺序"第 5 项（"已解决，选 C'"）与第 6 项（"对齐 x2 存储布局命中预编译 kernel"）已被本节方案 A 取代——C' 路径不可达，不再追求命中 V4 预编译 kernel。

---

## 附录 A — CANN QuantBatchMatmulV3 / V4 op_graph 源码分析

> 源码位置：`cann/ops-nn/matmul/quant_batch_matmul_v3/op_graph/` 与 `cann/ops-nn/matmul/quant_batch_matmul_v4/op_graph/`
>
> 本附录记录对 CANN 官方 V3/V4 op_graph 层代码的逐文件分析，作为 converter 选型与对齐的依据。

### A.1 QuantBatchMatmulV3 op_graph 结构

```
quant_batch_matmul_v3/op_graph/
├── CMakeLists.txt                                      构建脚本
├── quant_batch_matmul_v3_proto.h                       算子原型注册
├── quant_batch_matmul_v3_op_graph.cpp                  OpSelectFormat 实现
├── quant_batch_matmul_v3_fallback.cpp                  Host 侧 fallback 执行
└── fusion_pass/
    ├── quant_batch_matmul_v3_transpose_fusion_pass.{h,cpp}  Transpose 融合 pass
    └── z_z_matmul_t_qbmmv3_fusion_pass.{h,cpp}             MatMul→QBMMV3 融合 pass
```

#### A.1.1 算子原型（quant_batch_matmul_v3_proto.h）

通过 `REG_OP(QuantBatchMatmulV3)` 注册，定义：

- **输入**：`x1`、`x2`（必选）；`scale`、`offset`、`bias`、`pertoken_scale`（可选）
  - x1/x2 类型：int8, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1
  - scale 类型：uint64, float32, int64, bfloat16, float8_e8m0
- **属性**：
  - `dtype`（必选 Int）：输出类型（0=float32, 1=float16, 2=int8, 3=int32, 27=bfloat16）
  - `transpose_x1`/`transpose_x2`（Bool，默认 false）
  - `group_size`（Int，默认 0）：48 位编码（k/n/m 各 16 位）
- **输出**：`y`（float16/int8/bfloat16/int32/float32）

proto 注释详细说明了各平台（Atlas Inference、A2/A3、Ascend 950）支持的 dtype 组合与 quant 模式表。

#### A.1.2 OpSelectFormat（quant_batch_matmul_v3_op_graph.cpp）

根据 `x2` 是否为常量输入，动态生成 `OpSelectFormat` 结果（JSON 描述的 dtype/format 组合表）：

- `listWeightNZ`：x2 为常量（常量折叠场景），weight 只保留 NZ 格式
- `listQbmmV3`：非常量场景，与原始算子信息库一致

`QbmmV3OpCheckHelper` 构造函数根据 `context->IsConstInput("x2")` 选择列表；`OpSelectFormat` 用 lambda 抽取每个张量的 dtype/format/unknownshapeFormat 列表，组装成 `TensorSpec`，用 `nlohmann::json` 序列化为字符串返回。

通过 `IMPL_OP(QuantBatchMatmulV3).OpSelectFormat(QbmmV3OpSelectFormat)` 注册到 optiling。

#### A.1.3 Host 侧 Fallback（quant_batch_matmul_v3_fallback.cpp）

当算子在设备侧无法直接执行时，通过 `IMPL_OP(QuantBatchMatmulV3).OpExecuteFunc(...)` 注册 Host 回退函数 `QuantBatchMatmulV3ExecuteFunc`，根据输入 dtype/shape 路由到 `aclnnQuantMatmulV5` 或 `aclnnQuantMatmulV4`：

**V5 路径**：当满足以下任一条件时调用 `aclnnQuantMatmulV5`：
- x1 dtype ∈ {FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1}（v5 输入类型）
- 或为 "A8W8GB" 场景：x1/x2=int8、y=bf16、pertoken_scale/scale=float32、bias 为 null 或 float，且 x1 与 pertoken_scale、x2 与 scale 的存储维度数一致

V5 调用还会读取 `group_size` 属性并传入，且补齐 yScale/x1Offset/yOffset 为 nullptr。

**V4 路径**：其余情况调用 `aclnnQuantMatmulV4`。

属性索引硬编码：`transpose_x1=1`、`transpose_x2=2`、`group_size=3`。

#### A.1.4 Transpose 融合 Pass（quant_batch_matmul_v3_transpose_fusion_pass）

**目标**：删除 `QuantBatchMatmulV3` 之前的 `Transpose`/`TransposeD`/`Reshape`（以及 Bitcast 模式下的同类节点），并翻转 `transpose_x1`/`transpose_x2` 属性。

两个 Pass 类：
- `QuantBatchMatmulV3TransposeFusionPass`：常规 pass
- `QuantBatchMatmulV3TransposeLimitFusionPass`：继承前者，在"禁止关闭融合规则"场景下兜底，允许在内轴 > 65535 而外轴 ≤ 65535 时强制融合

主要流程：
1. 版本检查（仅 GE compiler ≥ 9.1.0 才生效）
2. 前置检查：校验是 QBMMV3 节点、平台能力（supportL12btBf16、supportMmadS8S4）、dtype 合法性、shape 维度 ≥ 2、是否动态 shape、是否 Bitcast 模式
3. 查找待删节点：根据内轴/外轴与 65535 的关系、平台是否支持 l12bt-bf16、x2 是否 NZ，决定 isDelTransx1/isDelTransx2；取出 x1/x2 上的 Transpose 节点；进一步尝试取 scale 上的等价 Transpose/Reshape（对 mx 量化的 FLOAT8_E8M0 scale 做了特殊等价判定）
4. 执行融合：翻转 transpose 属性、绕过 Transpose 重连上游节点、上报融合、删节点

#### A.1.5 MatMul→QBMMV3 融合 Pass（z_z_matmul_t_qbmmv3_fusion_pass）

**目标**：把 `MatMul / MatMulV2 / MatMulV3 / BatchMatMul / BatchMatMulV2 / BatchMatMulV3` 转换成 `QuantBatchMatmulV3`，用一个常量 `scale=1.0`（`0x3F800000`）当作"伪量化"输入，从而复用 QBMMV3 的高性能实现。

模式与构造：`Patterns()` 为 2/3/4 输入构造 6 类共 18 个匹配 pattern。

准入条件：GE compiler 版本 ≥ 9.0.0；排除 hifloat8；Ascend950 平台还排除 `int8+int8→int32` 组合。

替换流程：读取原节点的转置属性映射到 QBMMV3 的 `transpose_x1/transpose_x2`；创建常量 scale（uint64 1.0，shape={1}）；调用 `QuantBatchMatmulV3(...)` 生成新节点；更新输入/输出 desc；复制其他属性。

### A.2 QuantBatchMatmulV4 op_graph 结构

```
quant_batch_matmul_v4/op_graph/
├── CMakeLists.txt                                      构建脚本
├── quant_batch_matmul_v4_proto.h                       算子原型注册
└── fusion_pass/
    ├── quant_batch_matmul_v4_to_v3_fusion_pass.{h,cpp}     V4 → V3 转换 pass
    └── quant_batch_matmul_v4_transpose_fusion_pass.{h,cpp}  Transpose 融合 pass
```

相比 V3 的 op_graph，V4 这里**只包含算子原型和两个融合 pass**，没有 `OpSelectFormat` 和 `fallback` 实现——因为 V4 在 fallback 时会被转成 V3。

#### A.2.1 算子原型（quant_batch_matmul_v4_proto.h）

通过 `REG_OP(QuantBatchMatmulV4)` 注册，注释定位为"antiquant + matmul 的融合算子，支持 A8W4 / MxA8W4 / A8W8 / A4W4"。

**输入（10 个，2 必选 + 8 可选）**：

| Index | 名称 | 必选 | 类型 |
|---|---|---|---|
| 0 | `x1` | 是 | float8_e5m2 / float8_e4m3fn / int8 / int4 |
| 1 | `x2` | 是 | float4_e2m1 / int4 / int8 |
| 2 | `bias` | 否 | bfloat16 / float16 / float32 |
| 3 | `x1_scale` | 否 | float16 / bfloat16 / float8_e8m0 / float32 |
| 4 | `x2_scale` | 否 | float16 / bfloat16 / float8_e8m0 / uint64 / float32 |
| 5 | `y_scale` | 否 | uint64 |
| 6 | `x1_offset` | 否 | float16 / bfloat16（not supported yet） |
| 7 | `x2_offset` | 否 | float16 / bfloat16 |
| 8 | `y_offset` | 否 | float16 / bfloat16 / float32（not supported yet） |
| 9 | `x2_table` | 否 | int8（not supported yet） |

**属性**：
- `dtype`（必选 Int）：输出类型，仅支持 1(float16) 和 27(bfloat16)
- `compute_type`（Int，默认 -1）：-1(auto) 或 4(f8f4)
- `transpose_x1`（Bool，默认 false）：注释说明"目前应当总是 false"
- `transpose_x2`（Bool，默认 false）：注释说明"目前应当总是 true"
- `group_size`（Int，默认 **-1**）：注意 V3 默认 0，V4 默认 -1

**与 V3 的关键差异**：V4 把 scale 拆成了 `x1_scale`/`x2_scale`/`y_scale`/`x1_offset`/`x2_offset`/`y_offset`/`x2_table` 共 8 个可选输入，对应"反量化 + matmul"融合语义。

#### A.2.2 V4→V3 转换 Pass（quant_batch_matmul_v4_to_v3_fusion_pass）

**目标**：当 V4 在当前平台/场景下无需独立实现时，把它重写为 V3。

**索引映射**（V4_V3_INDEX_MAP）：

| V4 输入 | V4 idx | → | V3 输入 | V3 idx |
|---|---|---|---|---|
| x1 | 0 | → | x1 | 0 |
| x2 | 1 | → | x2 | 1 |
| bias | 2 | → | bias | 4 |
| x1_scale | 3 | → | pertoken_scale | 5 |
| x2_scale | 4 | → | scale | 2 |
| x2_offset | 7 | → | offset | 3 |

V4 的 `y_scale / x1_offset / y_offset / x2_table` 在转 V3 时被丢弃。

**Patterns**：构造 9 个 pattern（optionalInputNumber 从 0 到 8），匹配 V4 节点不同可选输入数目的实际形态。

**MeetRequirements**——决定是否需要把 V4 转 V3。两个"无需转换"的条件，满足任一则 `return false`（不做融合）：

1. **V4 原生支持（MxA8W4）**：
   ```
   x1 == FLOAT8_E4M3FN && (x2 == FLOAT4_E2M1 || x2 == FLOAT) && (y == BF16 || y == FLOAT16)
   ```
   即 MxA8W4/MxA8W8FP 场景，V4 本身能跑，不需要降级。

2. **A8W8 G-B 量化**：当 x1/x2 scale 都存在，且
   ```
   x1=int8, x2=int8, x1_scale=float32, x2_scale=float32, y=bf16
   && x1 与 x1_scale 维度数相同 && x2 与 x2_scale 维度数相同
   ```
   即 perblock/pergroup 量化，V4 本身能跑。

否则返回 true，进入 `Replacement` 把 V4 重写为 V3。

**Replacement**：用 `EsGraphBuilder` 建新子图；遍历 `V4_V3_INDEX_MAP`，对每个 V4 实际存在的输入在 V3 侧按序创建输入端口；读取 V4 属性（`group_size` 为 -1 时改为 0）；调用 `es::QuantBatchMatmulV3(...)` 生成新节点；按映射表回写各输入 desc 和输出 desc。

#### A.2.3 V4 Transpose 融合 Pass（quant_batch_matmul_v4_transpose_fusion_pass）

**目标**：删除 `QuantBatchMatmulV4` 的 `x2` 与 `x2_scale` 输入前的 `Transpose` / `Reshape`（含动态 shape 的 `Pack-Gather-Shape` 链），并翻转 `transpose_x2` 属性。

与 V3 transpose pass 的差异：
- V3 pass 同时处理 x1/x2/scale/pertoken_scale 四个候选；V4 pass **只处理 x2 和 x2_scale**
- V4 pass 仅支持 `x1=FLOAT8_E4M3FN`、`x2 ∈ {FLOAT4_E2M1, FLOAT}`、`y ∈ {BF16, FLOAT16}`（即 MxA8W4 场景）
- V4 pass 额外处理了**动态 shape 下的 `Reshape(Shape→Gather→Pack)` 链**

### A.3 V4 与 V3 的整体关系

| 维度 | V3 op_graph | V4 op_graph |
|---|---|---|
| 算子原型 | scale/offset/pertoken_scale 3 量化输入 | 拆成 x1_scale/x2_scale/y_scale/x1_offset/x2_offset/y_offset/x2_table 8 个可选 |
| OpSelectFormat | 有（按 weight 是否常量分两表） | **无**（直接用算子信息库） |
| Fallback | 有，路由到 aclnnQuantMatmulV4/V5 | **无**（V4 通过 V4→V3 pass 走 V3 的 fallback） |
| Transpose 融合 | x1/x2/x2_scale/pertoken_scale 四候选，Bitcast 模式，平台 l12bt/mmad 能力判定 | 仅 x2/x2_scale 两候选，仅 MxA8W4 dtype，额外处理 Pack-Gather-Shape 动态链 |
| MatMul→V3 融合 | 有（z_z_matmul_t_qbmmv3，scale=1.0） | 无对应 |
| V4→V3 转换 | 不适用 | 有（核心 pass，按 dtype/场景决定是否降级） |

**整体设计**：V4 是更新的"反量化+matmul 融合"算子，原生支持 MxA8W4/A8W8GB 等新场景；当场景超出当前平台 V4 实现能力时，由 `QuantBatchMatmulV4ToV3FusionPass` 降级到 V3 走老路径。

**两个 V4 pass 的互斥性**：`QuantBatchMatmulV4TransposeFusionPass` 的合法 dtype 恰好等于 to_v3 pass 里 `isV4Supported` 的条件。满足 MxA8W4 → transpose 融合 pass 生效，to_v3 pass 不生效；不满足 MxA8W4 且非 A8W8GB → to_v3 pass 生效，transpose 融合 pass 不生效。

### A.4 MxA8W4 与 A8W8GB 场景详解

这两个名字是量化场景的缩写，规则是 `<scale-mode>A<xbits>W<ybits>`：

- **A8 / W4 / W8**：Activation 是 8-bit、Weight 是 4-bit 或 8-bit。这里的"8/4"指的是数据位宽，不限定 int 还是 float——float8_e4m3fn 算 8-bit，float4_e2m1 算 4-bit。
- **Mx 前缀**：Microscaling 量化模式，使用低比特浮点（float8/float4）做激活和权重、用 float8_e8m0 做 scale，scale 按 k 轴 32 元素一组（对应 V3 proto 里 `mx quant` 表，group_size `[1,1,32]`）。
- **GB 后缀**：Group-Block，即 pergroup-perblock 或 perblock-perblock 量化（V3 proto 里 group_size `[1,128,128]` / `[128,128,128]`）。区别于 perchannel/pertensor（scale 是 1D），GB 模式 scale 的形状与数据同维数。

| 场景 | x1 | x2 | scale | 输出 |
|---|---|---|---|---|
| MxA8W4 | float8_e4m3fn（8b float） | float4_e2m1（4b float） | x1_scale/x2_scale 一般是 float8_e8m0 | bf16/fp16 |
| A8W8GB | int8 | int8 | float32（与 x1/x2 同维数） | bf16 |

**A8W8GB 的判定关键**：scale 的维数必须和数据相同。
- perchannel/pertensor 时 scale 是 1D（如 `(n,)`），与 x2 的 2D 不匹配 → 不算 GB
- perblock/pergroup 时 scale 是 2D（如 `(ceil(k/128), ceil(n/128))`），与 x2 同是 2D → 命中 GB

**完整路径**：

| 场景 | 图算子 | 最终 opapi |
|---|---|---|
| MxA8W4 / A8W8GB | **保留 V4**（不融合） | V4 自己的 tiling/kernel |
| 其他场景 | V4 → V3（to_v3 pass） | V3 fallback → `aclnnQuantMatmulV4`（多数情况） |

### A.5 V3 的 perchannel 量化支持

V3 支持 perchannel 量化。proto 里明确列了两种含 perchannel 的量化模式（针对 Ascend 950）：

命名规则是 `<x1-side>-<x2-side>`，对应 proto 里 x1 用 `pertoken_scale`、x2 用 `scale`：
- `pertensor-perchannel`：x1=pertensor（或 null），x2=perchannel
- `pertoken-perchannel`：x1=pertoken，x2=perchannel

**perchannel 时 scale 的形状**（`quant_batch_matmul_v3_proto.h:59-62`）：
```
When the quant mode of x2 is perchannel or pertensor, the shape is 1D (t,),
with t equal to 1 or n, where n is the same as that of x2.
```
- `t = n`：真正的 perchannel（每通道一个 scale）
- `t = 1`：实际是 pertensor

**与 V4 的关系**：V4 的 A8W8GB 判定（`quant_batch_matmul_v4_to_v3_fusion_pass.cpp:127-128`）要求 scale 维数与数据相同，这**排除了 perchannel**——因为 perchannel 时 scale 是 1D，而 x2 是 2D/3D，维数不等。所以 A8W8 的 perchannel/pertensor 场景下，V4 会被转成 V3 来执行。

| A8W8 量化模式 | scale 形状 | V4 处理方式 |
|---|---|---|
| pertensor | 1D, t=1 | V4→V3 转换（走 V3 路径） |
| perchannel | 1D, t=n | V4→V3 转换（走 V3 路径） |
| perblock/pergroup | 2D/3D（同维数） | 保留 V4（A8W8GB） |

### A.6 pertoken / pergroup / perchannel / perblock 概念

用一句话概括：**这几个"per-X"就是回答"多少个元素共用一个 scale（缩放因子）"的问题。颗粒度从粗到细排列。**

假设激活矩阵 x1 形状是 `(m, k)`（m 行 k 列），权重矩阵 x2 形状是 `(n, k)`。

```
pertensor  —— 整个矩阵共用 1 个 scale
┌────────────────────┐
│  全部用 S0          │   → 1 个 scale
└────────────────────┘

pertoken   —— 每一行（每个 token）一个 scale
┌────────────────────┐
│  S0 S0 S0 S0 S0 S0 S0 S0 │  ← 第 0 行用 S0
│  S1 S1 S1 S1 S1 S1 S1 S1 │  ← 第 1 行用 S1
│  S2 S2 S2 S2 S2 S2 S2 S2 │
│  S3 S3 S3 S3 S3 S3 S3 S3 │
└────────────────────┘   → m 个 scale

perchannel —— 每一列（每个 output channel）一个 scale
┌────────────────────┐
│ S0 S1 S2 S3 S4 S5 S6 S7│  ← 第 0 列用 S0, 第 1 列用 S1...
│ S0 S1 S2 S3 S4 S5 S6 S7│
│ S0 S1 S2 S3 S4 S5 S6 S7│
│ S0 S1 S2 S3 S4 S5 S6 S7│
└────────────────────┘   → k 个 scale（或 n 个，取决于谁被当"channel"）

pergroup   —— 沿某一轴每连续 group_size（如 128）个元素一个 scale
┌────────────────────┐
│ S0 S0 S0 S0 S1 S1 S1 S1│  ← 每 4 个一组（示意，实际是 128）
│ S2 S2 S2 S2 S3 S3 S3 S3│
│ S4 S4 S4 S4 S5 S5 S5 S5│
│ S6 S6 S6 S6 S7 S7 S7 S7│
└────────────────────┘   → m × ceil(k/group) 个 scale

perblock   —— 切成 2D 小块，每块一个 scale
┌───────┬───────┐
│  S0   │  S1   │   ← 左上 4×4 块用 S0, 右上 4×4 块用 S1
├───────┼───────┤
│  S2   │  S3   │
└───────┴───────┘   → ceil(m/gs) × ceil(k/gs) 个 scale
```

| 模式 | 谁共用一个 scale | scale 数量 | 颗粒度 |
|---|---|---|---|
| **pertensor** | 整个张量 | 1 | 最粗 |
| **perchannel** | 同一通道（一整列/一整行） | n 或 k | 中 |
| **pertoken** | 同一行（同一个 token） | m | 中 |
| **perblock** | 一个 2D 小块（如 128×128） | ceil(m/128)×ceil(k/128) | 较细 |
| **pergroup** | 沿某轴连续一维的一段（如每 128 个） | m×ceil(k/128) | 最细 |

**在 V3 里谁用哪个**：
- **x1（激活）** 用 `pertoken_scale`：可选 pertoken / pertensor / perblock / pergroup
- **x2（权重）** 用 `scale`：可选 perchannel / pertensor / perblock

V3 权重侧没有"pergroup"模式——最细就是 **perblock**（按 K 和 N 两个轴都分组）。如果只想沿 K 轴分组、N 轴不分组，V3 没有现成模式支持。

### A.7 pergroup 的判定机制

pergroup 是通过**两个东西组合判定**的：scale 张量的**形状** + 算子的 **group_size 属性**。没有任何显式的"quant_mode"枚举字段。

#### 判定依据 1：pertoken_scale 的形状

对比 x1（激活）侧四种模式的 pertoken_scale 形状（`quant_batch_matmul_v3_proto.h:77-83`）：

| 模式 | pertoken_scale 形状 | 关键特征 |
|---|---|---|
| pertensor | 1D `(1,)` | 只有一个值 |
| pertoken | 1D `(m,)` | 每行一个值 |
| **pergroup** | 3D `(batch, m, ceil(k/128))` | **m 是完整的，k 被分组** |
| perblock | 3D `(batch, ceil(m/128), ceil(k/128))` | m 和 k 都被分组 |

pergroup 和 perblock 的**唯一形状差异**：

```
pergroup :  (batch, m          , ceil(k/128))   ← m 没除 128
perblock :  (batch, ceil(m/128), ceil(k/128))   ← m 也除了 128
```

#### 判定依据 2：group_size 属性

`group_size` 是一个 48 位编码的 int 属性（`quant_batch_matmul_v3_proto.h:93-105`）：

```
bits  0-15  → group_size_k   （沿 K 轴分组大小）
bits 16-31  → group_size_n   （沿 N 轴分组大小）
bits 32-47  → group_size_m   （沿 M 轴分组大小）
```

pergroup vs perblock 由 **group_size_m** 决定：

| group_size_m | 含义 | x1 模式 |
|---|---|---|
| **1** | M 轴不分组（每组 1 个 m → 等价于不分组） | **pergroup** |
| **128** | M 轴每 128 个一组 | perblock |

支持表（`:231-243`）也印证：

```
pergroup-perblock : [group_size_m, group_size_n, group_size_k] = [1,   128, 128]
perblock-perblock : [group_size_m, group_size_n, group_size_k] = [128, 128, 128]
```

两者 x2 侧都是 perblock（n=128, k=128），**唯一区别就是 group_size_m 是 1 还是 128**。

#### 判定依据 3：dtype 限制

pergroup-perblock 模式还要求特定 dtype（`:212-218`）：

```
| x1                       | x2                       | pertoken_scale | scale  |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32        | float32 |
| hifloat8                  | hifloat8                 | float32        | float32 |
```

#### 总结：pergroup 是怎么被"看出来"的

当图编译器看到一个 V3 节点时，它通过下面这套组合判断 x1 是不是 pergroup：

```
pertoken_scale 是 3D && 第 2 维 == m（完整，没除 128）&& 第 3 维 == ceil(k/128)
    && group_size 属性解出 group_size_m == 1
    && dtype ∈ {float8_e4m3fn, float8_e5m2, hifloat8} + float32 scale
        → 判定为 pergroup
```

没有任何一个字段直接写着"pergroup"，全靠**形状结构 + group_size 编码 + dtype** 三者联合推断。

### A.8 W4A8 pergroup + pertoken 场景的可行性分析

如果权重使用 W4A8 的 pergroup 量化、激活使用 pertoken 量化，需要澄清以下关键点：

#### 问题 1："pergroup for weight" 在 V3 里不是独立模式

V3 对 x2（weight）侧只支持三种量化模式：
- pertensor：scale 1D `(1,)`
- perchannel：scale 1D `(n,)`
- perblock：scale `(batch, ceil(k/128), ceil(n/128))`

而"pergroup"在 V3 里是 **x1（activation）侧**的模式。"weight pergroup（沿 K 轴每 128 个一组）"最接近的是 x2 的 **perblock**（但 perblock 同时按 K 和 N 分组）。V3 不支持只按 K 分组、不按 N 分组的 x2 量化。

#### 问题 2：int8/int4（W4A8）的 dtype 组合不被支持

V3 和 V4 的支持表中，x1 和 x2 的 dtype 必须**成对相同**：

| 平台 | int8 行 | int4 行 |
|---|---|---|
| A2/A3 | int8/int8 | int4/int4 |
| 950 | int8/int8 | 无 int4 |

**没有 int8/int4 这一行**。V4 虽然注册了 `x1=DT_INT8` 和 `x2=DT_INT4`，但 proto 注释里 A8W4 场景的 x2 是 `float4_e2m1`（float4），不是 int4；int4 只在 A4W4（int4/int4）里出现。

#### 可行的替代方案

| 方案 | x1 dtype | x2 dtype | 量化模式 | 平台 | 说明 |
|---|---|---|---|---|---|
| A. MxA8W4（V4 原生） | float8_e4m3fn | float4_e2m1 | Mx（float8_e8m0 scale） | 950 | x1_scale/x2_scale 是 float8_e8m0，group_size [1,1,32] |
| B. A4W4 pertoken-perchannel（V3） | int4 | int4 | pertoken + perchannel | A2/A3 | x1 必须也是 int4，scale 1D |
| C. int8/int4 pergroup | — | — | — | — | **当前 V3/V4 均不支持** |

> **注意**：本项目的 converter 已选定路径 C'（V4 整数 int4 per-group），实测 CANN 9.2.0 预编译 kernel 目录含 `QuantBatchMatmulV4_ND_ND_int8_int4_{bf16,fp16}`（tilingKey=24），即 V4 runtime kernel 覆盖了整数 int8/int4 组合。proto 的 REG_OP dtype 列表与 runtime kernel 的覆盖范围不完全一致，proto 只列了 mxfp 分支但 runtime kernel 覆盖整数 dtype。详见上文"int4 物化的 CANN 约定"一节。
