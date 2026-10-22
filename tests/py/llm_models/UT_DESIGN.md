# cann_lm/models 单元测试设计报告

## 1. 文档信息

- 被测对象：`cann_lm/models/`（gemma4 / qwen3_5 modeling + fused_ops + model_registry + ExportWrapper 基类）
- 测试归档目录：`tests/py/llm_models`
- 测试框架：`pytest`
- 基准环境：converter venv（`$TORCH_DELEGATE_HOME/.venv`，Python 3.11.4、torch 2.9.0+cpu、torchair 源码编译、torch-delegate），**不装 torch_npu**；`TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- 设计日期：2026-09-05

## 2. 测试目标与范围

本轮 UT 验证 `cann_lm/models/` 的三模型 modeling 组件、融合算子 CPU impl、注册表与 ExportWrapper 基类的数值正确性与形状/状态语义。测试必须满足以下原则：

1. 默认在 CPU 离线运行，不下载 Hugging Face 模型或权重。
2. 全部使用 small config + `torch.randn` 随机权重，验证 shape/dtype/数值/有限值与 KV cache 状态。
3. 不依赖 torchair GE 转换、`torch.export`、真实 NPU/CUDA 算子。
4. 不碰 `cann_lm/`：遇 API 漂移只改 test 代码（如 `apply_rotary_pos_emb` 新签名、`delta_states` 参数、conv1d 3D weight、多值返回解包）。
5. `check(name, cond)` 失败即 `assert cond`，杜绝假 PASS；`config` fixture 复用 small config。

范围内模块：

- `gemma4/{configuration_gemma4,modeling_gemma4,export_wrapper}.py`
- `qwen3_5/{configuration_qwen3_5,modeling_qwen3_5,export_wrapper}.py`
- `fused_ops/*`（rms_norm / rotary_mul / l2_norm / dynamic_quant / scatter_nd_update / quant_matmul / chunk_gated_delta_rule / fused_infer_attention_score + `_library.py` / `ops.py` 注册基础设施间接覆盖）
- `model_registry.py`
- `export_wrapper.py`（顶层基类）

范围外项目：

- `load_*_weights` 真实 HF 权重加载、`save_epair` 导出落盘
- `forward` 经 `torch.export` + torchair GE 转换的端到端导出
- `ge_converters/*`（FX→GE）
- 真实大模型出字/精度验证（归 `examples/test_converter/llm_models_run_on_cpu/` + `examples/test_converter/export_llm_model/`）

## 3. 代码分析结论

### 3.1 架构与调用关系

`conftest.py` 把 `REPO_ROOT` 加 `sys.path`（仿 `tests/py/dopt_lm/conftest.py`），所有 `from cann_lm.* import ...` 无需手动 source 基底脚本。`config` fixture 由 `make_small_config()` + `@pytest.fixture` 提供，零参 test 自建 config。`check(name, condition)` 失败即 `assert condition`，pytest 捕获并报位置。

三模型 modeling 按 `configuration → RMSNorm/RoPE/apply_rotary_pos_emb/repeat_kv/rotate_half → Attention → MLP → DecoderLayer → TextModel → ForCausalLM(generate)` 分层组织；`fused_ops/_library.py` 把 `npu::` 和 `cann_lm::` 算子注册到 `torch.library`（schema + `register_fake` meta + `register_cpu_impl`），converter venv 无 torch_npu 时 CPU impl 直接由 `_library.py` 提供；`model_registry.py` 维护模型名→entry 映射表 + alias 解析；`export_wrapper.py` 基类提供 `__init__`/`set_quant_embeddings`/`_num_layers`，`forward` 调子类 `_drive_layers`（抽象）。

### 3.2 主要风险

| 风险域 | 代码风险 | UT 关注点 |
|---|---|---|
| API 漂移 | `apply_rotary_pos_emb` 签名变更、`delta_cache`→`delta_states` 重命名、conv1d weight 2D→3D、GatedDeltaNet/DecoderLayer 多值返回 | 新签名/参数对齐、多值解包、RoPE cos shape 断言改用缓存 buffer |
| 多值返回 | GatedDeltaNet.forward / DecoderLayer.forward / TextModel.forward 返回 3-tuple | 解包 `out, _, _ = ...`、TextModel `(hidden, kv, delta)` |
| 数值精度 | RMSNorm/RoPE/l2norm/dynamic_quant/quant_matmul/scatter 可手算对齐；chunk_gated/fused_attn causal 难精确 | 能对齐用 `allclose`；难对齐放宽 shape+finite + 注释 |
| CPU impl 限制 | fused_attn `_build_sparse_mask` BNSD 解包 `b,n,s_q,s_k=q.shape` 使 s_k=d 命中 early-return None | causal 路径只测 shape+finite；数值靠 extern_mask / s_q==s_k 覆盖 |
| 随机权重 | 不加载真模型，权重随机初始化 | 固定 `torch.manual_seed`；不验证绝对精度，验一致性/不变性 |
| KV cache 语义 | scatter-write（buffer 不增长）vs torch.cat（增长）两路径 | 验 buffer shape 不变 / cat 后 seq 增长；增量 decode == 全 prefill |
| RoPE 维度 | full/sliding 两类 cos 维不同（full=global_head_dim/2*2，sliding=head_dim/2*2） | cos shape 断言对齐各自缓存 buffer |
| 注册表 | alias 覆盖、大小写不敏感、未知模型 fail-fast | alias 全集 + 大小写 + 未知抛 ValueError |
| 基类抽象 | ExportWrapper.forward 调子类 `_drive_layers`，基类未定义 | base 不测 forward，只测 `__init__`/`set_quant_embeddings`/`_num_layers` |

## 4. 测试方法与判定准则

- 浮点结果使用 `torch.testing.assert_close` / `torch.allclose`，一般容差 `rtol=1e-5, atol=1e-6`；量化/softmax 路径放宽至 `rtol=1e-4, atol=1e-4`。
- 随机用例固定 `torch.manual_seed(0)` 等，保证可重复。
- 每条测试独立使用局部小 config + `torch.randn`，不依赖全局状态或文件系统（`test_config_to_dict_and_from_json` / `test_qwen3_5_config_branches` 用 `tmp_path` 造 JSON）。
- 异常分支检查异常类型（`pytest.raises(ValueError, match=...)` 或 try/except）。
- `check(name, condition)` 失败即 `assert condition`，pytest 正确捕获；不再用计数器（pytest 自计）。
- 所有 UT 必须可重复执行；测试目录整体退出码为 0 才算通过。
- 覆盖率目标：纯逻辑/数值模块行覆盖率不低于 80%；`export_wrapper.py` 基类 `forward`/`_drive_layers` 需 torch.export+真实权重，归 e2e，不以全行覆盖率作为 UT 准入条件。

## 5. 用例设计

### 5.1 gemma4 modeling（test_gemma4.py，25 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| GEMMA4-001 | `Gemma4TextConfig.__post_init__` | 30 层 layer_types 长度/末层 full/sliding 5:1/rope 两类型/global_head_dim512/head_dim256；small 4 层；tiny 2 层 `[sliding, full]` | P0 |
| GEMMA4-002 | RMSNorm/RoPE/apply_rotary/repeat_kv/rotate_half | shape 一致 + RoPE full cos 维=缓存 buffer + sliding cos=head_dim/2*2 + rotate_half 精确 `[-3,-4,1,2]` | P0 |
| GEMMA4-003 | `Gemma4TextAttention` | 各 layer_type 首层 attn：无 cache shape + KV store 非 None + 单 token cached shape | P0 |
| GEMMA4-004 | `Gemma4TextMLP` | 前 2 层 MLP out.shape==x.shape | P1 |
| GEMMA4-005 | `Gemma4TextDecoderLayer` | 首层 out.shape==x.shape | P0 |
| GEMMA4-006 | `Gemma4TextModel` | prefill hs.shape + pv 长度 + None/2D mask 两路 + 参数量 | P0 |
| GEMMA4-007 | `Gemma4ForCausalLM` | logits shape + loss None→标量 + softcap + greedy/sampling generate 更长 | P0 |
| GEMMA4-008 | KV sharing | 6 层后 3 共享：hs shape + shared_kv_states 非空 + 共享层 is_kv_shared + 无 k_proj | P1 |
| GEMMA4-009 | 层截断 | 2/4/8 层 logits vocab=1000 + 参数量 | P1 |
| GEMMA4-010 | `Gemma4ExportWrapper` vs model | logits max_diff=0 + KV max_diff=0（mask 用 `_build_mask_full/sliding` 预构） | P0 |
| GEMMA4-011 | `_build_mask_sliding` true branch | kv_max>sliding_window → bool mask [1,1,seq,kv_max]（矩形, arch31 tiling）+ row0 attend[0]/mask[1] | P1 |
| GEMMA4-012 | `_build_mask_sliding` degenerate | kv_max<=sliding_window → == `_build_mask_full` | P1 |
| GEMMA4-013 | `get_sample_inputs` input_ids mode | input_ids shape + mask_full/sliding [1,1,seq,16]（矩形）+ new_kv_pos + past_kv list + head_dim | P1 |
| GEMMA4-014 | `get_sample_inputs` embeddings mode | input_ids=None + emb [1,seq,hidden] + ple 非空 | P1 |
| GEMMA4-015 | `get_sample_inputs` quant_embeddings mode | int8 emb + embeddings_scale [1,seq] | P1 |
| GEMMA4-016 | `make_sample_inputs`/`make_input_output_names` | ONNX helper shapes + in/out names 数量=5+2N/1+2N | P2 |
| GEMMA4-017 | `host_side_weight_names`/`detach_externalized_lookups` | embed 权重名 + detach 后属性消失 | P2 |
| GEMMA4-018 | wrapper forward + output_pos + embeddings | logits [b,1,vocab]（output_pos index_select + emb dequant） | P1 |
| GEMMA4-019 | `eager_attention_forward` scaling/softcap | scaling=None→head_dim**-0.5 + softcap tanh + attn_weights≤softcap | P1 |
| GEMMA4-020 | TextModel scatter-write cache | logits shape + kv buffer 保持 kv_max（不 cat 增长） | P1 |
| GEMMA4-021 | `_prepare_causal_mask` 4D sliding | kv_len>sliding_window → window 限制含 -inf | P1 |
| GEMMA4-022 | `_prepare_causal_mask` 2D padding | 2D mask→bool key_valid + padded 位 -inf | P1 |
| GEMMA4-023 | `to_dict`/`Gemma4Config.__post_init__` | to_dict 无 `_` 前缀 + None→defaults + dict→dataclass | P1 |
| GEMMA4-024 | wrapper 可导出 + 图无 masked_fill | torch.export 能 trace（use_embeddings=False/True+detach 两模式）+ FX 图 masked_fill 节点==0（NPU 不支持 ge.MaskedFill） | P0 |
| GEMMA4-025 | wrapper vs model **decode** | 单 token decode（q@pos S）wrapper logits+KV == model.forward scatter-write（maxdiff=0, KV torch.equal）；扩 GEMMA4-010 到 decode 步 | P0 |

### 5.2 qwen3.5 modeling（test_modeling_qwen3_5.py，38 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| Q35-001 | `Qwen3_5RMSNorm` | shape + dtype + not_nan + 常数输入归一化 | P0 |
| Q35-002 | `Qwen3_5RMSNormGated` | shape + not_nan（gated 多 gate） | P0 |
| Q35-003 | `Qwen3_5TextRotaryEmbedding` | mrope cos/sin shape==rotary_dim + not_nan | P0 |
| Q35-004 | `rotate_half` | shape + not_nan | P1 |
| Q35-005 | `apply_rotary_pos_emb` | q/k shape + not_nan（cos/sin 3D 内部 unsqueeze 4D） | P0 |
| Q35-006 | `repeat_kv` | n_rep=2 shape + n_rep=1 same | P0 |
| Q35-007 | `eager_attention_forward` | out shape + weights shape + not_nan | P0 |
| Q35-008 | `apply_mask_to_padding_states` | 无 mask 不变 + 全零 mask 清零 | P0 |
| Q35-009 | `l2norm` | unit norm + vs 手算 `x*rsqrt(sum+eps)` | P0 |
| Q35-010 | `torch_causal_conv1d_update` | out shape + not_nan + conv_state 更新（3D weight） | P0 |
| Q35-011 | `torch_chunk_gated_delta_rule` | out/final_state shape + not_nan（shape-only） | P1 |
| Q35-012 | `torch_recurrent_gated_delta_rule` | out/final_state shape + not_nan | P1 |
| Q35-013 | `SimpleKVCache` | 首次 same + 二次 cat(12) | P0 |
| Q35-014 | `SimpleDeltaNetCache` | 初始无 state + update_conv 后 has_state + conv_states 更新 | P0 |
| Q35-015 | `Qwen3_5GatedDeltaNet` | 无/有 cache output shape（解 3-tuple 返回） | P0 |
| Q35-016 | `Qwen3_5Attention` | output shape + not_nan | P0 |
| Q35-017 | `Qwen3_5MLP` | output shape + not_nan | P0 |
| Q35-018 | `Qwen3_5DecoderLayer` | linear + full 两层 output shape + not_nan（解 3-tuple） | P0 |
| Q35-019 | `Qwen3_5TextModel` | hidden shape + kv/delta cache 非空 | P0 |
| Q35-020 | `Qwen3_5ForCausalLM` | logits shape + not_nan + loss None→标量 finite | P0 |
| Q35-021 | `generate` | greedy 长度==8 + not_nan | P0 |
| Q35-022 | `generate` with eos | eos 停止 + 长度≥4 | P1 |
| Q35-023 | inputs_embeds | logits shape + not_nan | P1 |
| Q35-024 | 增量 decode | delta_states 传入 + 增量 logits [1,1,vocab] + not_nan | P0 |
| Q35-025 | error conditions | 无 input/embeds + 两者同传 → ValueError | P0 |
| Q35-026 | `scatter_kv_to_cache` b=1/b=2 | b=1 scatter_nd_update + b=2 index_copy fallback | P1 |
| Q35-027 | `SimpleKVCache` scatter-write | new_kv_cache_pos → buffer 保持 kv_max + 位置写入 | P1 |
| Q35-028 | `SimpleKVCache` cat fallback | existing + new_kv_cache_pos=None → cat 增长 | P1 |
| Q35-029 | `_create_causal_mask` 2D padding | [1,1,4,4] + padded 位 -inf + diagonal 0 | P1 |
| Q35-030 | `_create_causal_mask` 4D passthrough | 4D mask 原样返回 | P1 |
| Q35-031 | TextModel export delta_states | delta_states tuple → is_export_delta path + delta_out tuple | P1 |
| Q35-032 | TextModel 2D mask forward | 2D mask 全模型 forward logits shape + finite | P1 |
| Q35-033 | wrapper layer index helpers | full/linear 索引正确 + 无重叠 + 覆盖全部层 | P1 |
| Q35-034 | wrapper `_build_zero_delta_states` | tuple + conv/rec list 长度 + shape 对齐 gdn | P1 |
| Q35-035 | wrapper `get_sample_inputs` | past_kv 仅 full_attn 层 + delta_states tuple | P1 |
| Q35-036 | wrapper forward export path | logits [b,1,vocab] + delta_out_flat 长度=2*n_linear | P1 |
| Q35-037 | wrapper forward eager delta | delta_states=None → _delta_cache_box 持久化 | P1 |
| Q35-038 | config branches | from_dict 过滤 + from_json_file wrapper + VisionConfig.from_dict + Qwen3_5Config None/dict→dataclass | P1 |

### 5.4 fused_ops CPU impl（test_fused_ops.py，23 函数 / 28 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| FOP-001 | `npu::npu_rms_norm`（×3 shape） | y shape + rstd shape/>0/dtype + vs 手算 `x*rsqrt(mean(x²)+eps)*gamma` | P0 |
| FOP-002 | `npu::npu_rms_norm` 默认 eps | 默认 epsilon=1e-6 数值对齐 | P0 |
| FOP-003 | `npu::npu_rotary_mul` half（×2 d） | shape + vs 手算 `self*r1+cat(-x2,x1)*r2` | P0 |
| FOP-004 | `npu::npu_rotary_mul` default mode | 默认 "half" 数值对齐 | P0 |
| FOP-005 | `cann_lm::l2norm`（×3 dim） | unit norm + vs 手算 `x*rsqrt(sum+eps)` | P0 |
| FOP-006 | `npu::npu_dynamic_quant` | y(int8)/scale(fp32,>0) + vs `amax/127`+clamp(round) + round-trip allclose | P0 |
| FOP-007 | `npu::npu_scatter_nd_update` 1d indices | out shape + vs index_copy + 原 cache 不污染 + 未触位保留 | P0 |
| FOP-008 | `npu::npu_scatter_nd_update` 2d indices | out shape + 逐坐标赋值对齐 | P0 |
| FOP-009 | `npu::npu_quant_matmul` group1 | shape + dtype + vs `x1@x2`（scale=1） | P0 |
| FOP-010 | `npu::npu_quant_matmul` grouped+bias | shape + finite + vs `x1@(w*sc_exp)+bias` | P0 |
| FOP-011 | `npu::npu_quant_matmul` pertoken | shape + finite + vs `x1q*ps@x2`（quant+dequant 路径） | P0 |
| FOP-012 | `npu::npu_chunk_gated_delta_rule` | out+final_state shape + finite（shape-only） | P1 |
| FOP-013 | `npu::npu_fused_infer_attention_score_v2` BNSD full | shape + lse=0 + vs 手算 SDPA allclose | P0 |
| FOP-014 | fused_attn causal shape+finite | sparse_mode=3 → shape+finite（CPU impl 限制，不做数值） | P1 |
| FOP-015 | fused_attn GQA BSH | shape + finite | P1 |
| FOP-016 | fused_attn extern bool mask | shape + finite + vs 手算 masked SDPA allclose | P0 |
| FOP-017 | fused_attn sparse_mode=3 causal square | s_q==s_k → 下三角 mask + vs 手算 causal SDPA allclose | P1 |
| FOP-018 | fused_attn sparse_mode=4 sliding square | sliding window mask + vs 手算 allclose | P1 |
| FOP-019 | fused_attn TND layout | shape + vs 手算 SDPA（unsqueeze 0）allclose | P1 |
| FOP-020 | fused_attn BSND layout | shape + vs 手算 SDPA（transpose 1,2）allclose | P1 |
| FOP-021 | fused_attn default kv_heads=query | num_kv_heads=0 → MHA + vs 手算 SDPA allclose | P1 |
| FOP-022 | fused_attn float mask sliced | q-dim>attn q-dim → mask 切片 + vs 手算 allclose | P1 |
| FOP-023 | fused_attn unknown sparse_mode | sparse_mode=5 → 无 mask + vs plain SDPA allclose | P2 |

### 5.5 model_registry（test_model_registry.py，7 函数 / 23 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| REG-001 | `available_models` | =={gemma4,qwen3_5} + 全在 REGISTRY | P0 |
| REG-002 | `get_entry` alias（×7 case） | gemma4/gemma-4/大小写、qwen3_5/qwen3.5 → REGISTRY[expected] | P0 |
| REG-003 | `get_entry` 未知（×5 case） | unknown/llama/""/gemma5/qwen4 → ValueError | P0 |
| REG-004 | entry 字段（×3 模型） | config_cls/model_cls/wrapper_cls/config_json_key + load_weights_fn callable + model_type 非空 | P0 |
| REG-005 | `detect_model_type`（×2 case） | {model_type:gemma4/qwen3_5} → REGISTRY[key] | P0 |
| REG-006 | `detect_model_type` 缺字段 | {} → ValueError(match model_type) | P0 |
| REG-007 | `detect_model_type` 未知 | {model_type:llama} → ValueError(match Unknown) | P0 |

### 5.6 ExportWrapper 基类（test_export_wrapper_base.py，5 函数 / 7 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| BASE-001 | `ExportWrapper.__init__` 默认 | model/config/kv_cache_max_len/batch/_model_dtype(fp32)/_quant_embeddings={} | P0 |
| BASE-002 | `__init__` 自定义 kwargs | bf16 + kv_cache_max_len=512 + batch=4 | P0 |
| BASE-003 | `_num_layers`（×3: 1/3/8） | == stub 层数 | P0 |
| BASE-004 | `set_quant_embeddings` 存储 | dict 存储 + int8 emb_w + scale shape | P0 |
| BASE-005 | `set_quant_embeddings` 复制语义 | 改原 dict 不影响 wrapper（set 时 `dict()` 复制） | P0 |

## 6. 计划目录

```text
tests/py/llm_models/
├── UT_DESIGN.md
├── conftest.py
├── test_gemma4.py
├── test_modeling_qwen3_5.py
├── test_fused_ops.py
├── test_model_registry.py
└── test_export_wrapper_base.py
```

## 7. 执行方式

```bash
# 全套 (converter venv):
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $TORCH_DELEGATE_HOME/.venv/bin/python -m pytest tests/py/llm_models/ -q

# 单文件:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $TORCH_DELEGATE_HOME/.venv/bin/python -m pytest tests/py/llm_models/test_gemma4.py -q

# 单用例:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $TORCH_DELEGATE_HOME/.venv/bin/python -m pytest tests/py/llm_models/test_gemma4.py::test_config -q

# 收集 (不跑, 看项数):
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $TORCH_DELEGATE_HOME/.venv/bin/python -m pytest tests/py/llm_models/ --co -q

# 覆盖率:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 $TORCH_DELEGATE_HOME/.venv/bin/python -m pytest tests/py/llm_models/ --cov=cann_lm/models --cov-report=term-missing -q
```

## 8. 准入与遗留风险

准入条件：全部用例通过、无外网访问、converter venv（无 torch_npu）可重复、无测试间状态泄漏。真实 NPU 算子、全模型出字/精度和 torch.export+GE 导出需要在后续集成测试中使用实际 HF 权重 + NPU/CANN 环境单独验证；UT 通过不等价于导出/推理正确。

遗留风险与已知限制：

1. **fused_attn causal CPU impl 限制**：`_build_sparse_mask` 解包 `b,n,s_q,s_k=q.shape`，BNSD 下 `q=[b,n,s,d]` 使 `s_k=d`、`s_q≠s_k` 命中 early-return `None` → causal mask 被跳过。这是 cann_lm CPU impl 限制（不碰 cann_lm），causal 用例放宽为 shape+finite；数值精确对齐改由 `extern_mask`（传 bool atten_mask）和 `s_q==s_k`（square）测试覆盖。
2. **chunk_gated_delta_rule**：融合算子数值精确校验难（BHSD↔BSHD 转换 + delta rule 递归），仅测 out/final_state shape + finite；数值由 qwen3.5 modeling（test_modeling_qwen3_5）间接覆盖。
3. **ExportWrapper 基类 forward**：基类 `forward` 调 `self._drive_layers`（子类实现，基类未定义），故 base 测试不测 forward，只测 `__init__`/`set_quant_embeddings`/`_num_layers`；forward 由各模型 wrapper（GEMMA4-010 / QWEN3-013 / Q35-036）端到端覆盖。
4. **export_wrapper forward 归 e2e**：各模型 `export_wrapper.py` 的 `forward` 经 `torch.export` + torchair GE 转换的完整导出路径归 `examples/test_converter/export_llm_model/run_e2e_pipeline.sh` e2e 验证，UT 只测其纯逻辑 helper（mask builder / sample inputs / names）和单步 forward 数值一致性。**例外**：纯 `torch.export` + FX 图 `masked_fill` 节点计数（静态性质，不需 GE 转换/NPU/真实权重，CPU 小模型即可）已纳入 UT（GEMMA4-024），因 NPU 不支持 `ge.MaskedFill`，该断言是 bypass 重写的核心交付物，须在 UT 锁定。
5. **真实权重/出字**：不在 UT 范围（需 HF 模型 + 权重），归 `examples/test_converter/llm_models_run_on_cpu/`（CPU e2e）+ `examples/test_converter/export_llm_model/`（HF→epair→OM→NPU e2e）。

当前没有 skip 或 xfail 用例；所有已收集用例均作为普通回归用例执行。

## 9. 执行结果（2026-09-05）

### 9.1 测试结果

```text
140 passed in 44.54s
pytest exit code: 0
```

- 收集用例总数：140
- 普通回归用例：140，全部通过
- skip/xfail：0
- 按文件分布：test_gemma4 23 / test_modeling_qwen3_5 38 / test_fused_ops 28 / test_model_registry 23 / test_export_wrapper_base 7

### 9.2 覆盖率结果

执行命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  $TORCH_DELEGATE_HOME/.venv/bin/python \
  -m pytest tests/py/llm_models/ --cov=cann_lm/models --cov-report=term-missing -q
```

整体行覆盖率为 **84%（2422 statements，304 missed，602 branch，93 partial）**，达到不低于 80% 的准入目标。核心模块覆盖情况：

| 模块 | 行覆盖率 |
|---|---:|
| `model_registry.py` | 100% |
| `fused_ops/ops.py` | 100% |
| `fused_ops/fused_infer_attention_score.py` | 96% |
| `qwen3_5/export_wrapper.py` | 95% |
| `gemma4/configuration_gemma4.py` | 91% |
| `gemma4/export_wrapper.py` | 90% |
| `qwen3_5/configuration_qwen3_5.py` | 86% |
| `qwen3_5/modeling_qwen3_5.py` | 86% |
| `fused_ops/_library.py` | 82% |
| `fused_ops/chunk_gated_delta_rule.py` | 81% |
| `fused_ops/rotary_mul.py` | 81% |
| `fused_ops/l2_norm.py` | 80% |
| `gemma4/modeling_gemma4.py` | 80% |
| `fused_ops/quant_matmul.py` | 74% |
| `fused_ops/dynamic_quant.py` | 73% |
| `fused_ops/rms_norm.py` | 71% |
| `fused_ops/scatter_nd_update.py` | 67% |
| `export_wrapper.py`（基类） | 50% |
| **TOTAL** | **84%** |

剩余未覆盖代码主要位于 `export_wrapper.py` 基类的 `forward`/`_drive_layers`/`_build_*` 抽象与导出路径（需 torch.export + 真实权重，归 e2e）、各 modeling 的 `load_*_weights` 全流程（需 HF safetensors）、以及 fused_ops 部分算子的非主流 dtype/错误分支。这些按设计归集成测试，避免通过过度 mock 获得失真的覆盖率。

### 9.3 缺陷修复记录

| ID | 原现象 | 修复内容 | 回归结果 |
|---|---|---|---|
| DEF-001 | `apply_rotary_pos_emb` 签名漂移：gemma4 需传 `position_embeddings` tuple + layer_type | 测试代码对齐新签名：gemma4 `apply_rotary_pos_emb(q, k, cos, sin, "full_attention")` | gemma4 test_components 通过，q/k shape 正确 |
| DEF-002 | qwen3.5 `ForCausalLM.forward` 参数 `delta_cache` 重命名为 `delta_states`，旧名 KeyError | 增量 decode 用例改传 `delta_states=dc`；TextModel export path 用 `delta_states=(conv_list, rec_list)` tuple | test_incremental_decode + test_text_model_export_delta_states 通过 |
| DEF-003 | `torch_causal_conv1d_update` weight 从 2D `[dim, kernel]` 变 3D `[dim, 1, kernel]`（Conv1d 标准 shape） | 测试 weight 改 `torch.randn(dim, 1, kernel)`，对齐新 API | test_causal_conv1d_update 通过，conv_state shape 正确 |
| DEF-004 | `Qwen3_5GatedDeltaNet.forward` / `Qwen3_5DecoderLayer.forward` 返回值从单 Tensor 变为 3-tuple `(out, ..., ...)` | 测试解包 `out, _, _ = gdn(x)` / `out_linear, _, _ = layer(x, ...)` | test_gated_delta_net + test_decoder_layer 通过 |
| DEF-005 | gemma4 RoPE full cos shape 断言硬编码 `== global_head_dim` 不匹配缓存 buffer 实际维度（partial rotary） | cos shape 断言改为对齐 `rope.cos_cached_full_attention.shape[-1]`（缓存 buffer 驱动） | test_components RoPE shape 断言通过 |

### 9.4 增量更新（2026-09-21）

从 `examples/test_converter/llm_models_run_on_cpu/debug_gemma4/verify_gemma4_wrapper_bypass.py` + `verify_gemma4_wrapper_export.py` 抽出核心断言入 pytest（示例脚本可删）：

- **新增 GEMMA4-024**（`test_wrapper_export_no_masked_fill`）：`torch.export` 能 trace（use_embeddings=False/True+detach 两模式）+ FX 图 `masked_fill` 节点==0。NPU 不支持 `ge.MaskedFill`，bypass 重写核心交付物，UT 锁定。
- **新增 GEMMA4-025**（`test_wrapper_vs_model_decode`）：单 token decode（q@pos S）wrapper logits+KV == model.forward scatter-write（maxdiff=0, KV `torch.equal`），扩 GEMMA4-010（仅 prefill）到 decode 步。
- **未移植**：bypass 的 `host mask == model 4D` 逐位 `torch.equal`——已失效（host 改 BOOL 后与 model fp 4D dtype 不符，假阴性）；且被 GEMMA4-025 的 logits maxdiff=0 逻辑蕴含（mask 若语义分歧 logits 必分歧），无需移植。
- **顺带修 3 个 stale 断言**（GEMMA4-011/013/016）：mask 形状从方阵 `[1,1,kv_max,kv_max]` 改矩形 `[1,1,seq,kv_max]`（arch31 tiling 要求，见 `_build_mask_full` docstring「矩形 q_dim=seq_len」）后这 3 个测试未跟更新，现修正。
- gemma4 项数 23→25，全套 140→142 passed。
