# 量化接入 Converter 设计文档

> 将 HuggingFace 模型经 GPTQ 权重量化后导出为含 `QuantBatchMatmulV4` 算子的 epair 格式
>
> 量化方案：**W4A8 s8s4**（激活 s8 per-token 动态对称 + 权重 s4 per-group 对称）
>
> 最后更新: 2026-08-03

## 目录

- [概述](#概述)
- [已有 dopt_lm 框架盘点](#已有-dopt_lm-框架盘点)
- [量化策略](#量化策略)
- [端到端流程](#端到端流程)
- [待实现模块](#待实现模块)
- [int4 物化的 CANN 约定（已定：V4 整数 int4）](#int4-物化的-cann-约定已定v4-整数-int4-per-group)
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

| 文件 | 作用 |
|------|------|
| `opt_main.py` | CLI 入口。两步：① 不存在 dopt_config 时 `generate_config` 生成脚手架并退出；② 调 `llm_quant_pipline` → `weight_quant_pipline`。 |
| `opt_funcs/eco_stage/train.py` | `weight_quant_pipline`：有 KD 走 `train`，否则走 `ptq_quant_pipline`。`ptq_quant_pipline` 调 `weight_quant_loop` 跑 GPTQ，再 `prepare_model`→`optimize_model` 把 `nn.Linear→QLinear`，注入 scale，存 `trained_quant_weight.pth`。 |
| `do_opt.py` | `optimize_model(model, config_file)`：按 `layer_strategy` 把目标 module wrap 成 `QLinear`/`QEmbedding`，`replace_module_by_names` 做模块替换。`set_quant_state`/`set_calibrate_state` 切换 fakequant 开关。 |
| `builder.py` | `init_builder()` 注册 `modules` + `quantizer`（`Register` 模式）。 |

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
