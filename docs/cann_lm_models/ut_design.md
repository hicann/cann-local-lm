# UT 设计：cann_lm/models 单元测试 (tests/llm_models/)

> 状态: 已落地 | 最后核对: 2026-09-05 | 全套 99 passed (converter venv, 无 torch_npu)

## 背景

`cann_lm/models/` 下三模型（gemma4 / qwen3 / qwen3_5）+ 融合算子（fused_ops/）+ 注册表 + 基类 wrapper 是 converter 导出与 NPU 推理的数值基础。此前单元测试散落在 `examples/test_converter/llm_models_run_on_cpu/debug_*/verify_*.py` 的 `--test-only` 模式里（与 e2e 推理耦合，自定义 `check()` runner 不抛异常导致 pytest 假 PASS），且 fused_ops / model_registry / 基类 wrapper 无任何测试。

本设计把单元测试统一抽到 `tests/llm_models/`，参考 `tests/dopt_lm/` 范式（pytest 风格 + `@pytest.mark.parametrize` + `assert`），实现 `cann_lm/models/` **全覆盖**，CI/本地均可 `pytest` 直接跑。

## 测试架构

```
tests/llm_models/
├── conftest.py                  # sys.path→REPO_ROOT; pytest 无需手动 source env 即可 import cann_lm
├── test_gemma4.py               # gemma4 modeling (10 项)
├── test_qwen3.py                # qwen3 modeling (13 项, 从 verify_qwen3.py --test-only 抽出)
├── test_modeling_qwen3_5.py     # qwen3.5 modeling (25 项)
├── test_fused_ops.py            # fused_ops 8 算子 CPU impl (21 项)
├── test_model_registry.py       # model_registry 注册表 (23 项)
└── test_export_wrapper_base.py  # 顶层 ExportWrapper 基类 (7 项)
```

- **`conftest.py`** 把 `REPO_ROOT` 加 `sys.path`（仿 `tests/dopt_lm/conftest.py`），所有 `from cann_lm.* import ...` 无需 source 基底脚本；与 `tests/dopt_lm/` 并列，pytest 发现 `tests/` 全树。
- **pytest 风格**：顶层 `test_*` 函数（零参或收 `config` fixture），`@pytest.mark.parametrize` 覆盖 shape/dtype/alias/边界；`check(name, cond)` 改 `assert cond, f"{name} failed"`（失败即 raise，pytest 正确捕获，杜绝假 PASS）。
- **与 examples 的边界**：`examples/test_converter/llm_models_run_on_cpu/debug_*/verify_*.py` 保留 e2e（真实权重 + 出字），其 `--test-only` 单元测试已抽到 `tests/llm_models/`，verify 脚本不再自带单元测试（gemma4 已去 `--test-only`；qwen3 保留但 canonical 在 tests/）。

## 覆盖矩阵

| cann_lm/models 文件 | 测试文件 | 项数 | 覆盖要点 |
|---|---|---|---|
| `gemma4/{configuration,modeling,export_wrapper}.py` | test_gemma4.py | 10 | config 默认值 / RMSNorm/RoPE/apply_rotary/repeat_kv/rotate_half / attention(含 KV cache) / MLP / DecoderLayer / TextModel / ForCausalLM(logits+loss+softcap+generate) / KV sharing / 层截断 / wrapper vs model(max_diff=0.0) |
| `qwen3/{configuration,modeling,export_wrapper}.py` | test_qwen3.py | 13 | config / RMSNorm / RoPE / MLP / attention+layer / forward shape+tie / 确定性 / causal 不变性 / KV cache 一致性 / greedy==argmax / loss / 截断 / wrapper vs model |
| `qwen3_5/{configuration,modeling,export_wrapper}.py` | test_modeling_qwen3_5.py | 25 | RMSNorm/GatedRMSNorm / RoPE(mrope) / rotate_half / apply_rotary_pos_emb / l2norm / repeat_kv / mask_to_padding / causal_conv1d_update / chunk+recurrent gated_delta_rule / KV cache / DeltaNet cache / GatedDeltaNet / Attention / DecoderLayer / TextModel / ForCausalLM / generate / inputs_embeds / 增量 decode / 错误处理 |
| `fused_ops/*`（10 文件） | test_fused_ops.py | 21 | 8 算子 CPU impl（见下「fused_ops 测试要点」）；`_library.py`/`ops.py` 注册基础设施间接覆盖 |
| `model_registry.py` | test_model_registry.py | 23 | `available_models` / `_ALIASES` 解析（gemma4/gemma-4/大小写、qwen3/qwen-3、qwen3.5/qwen3_5）/ `get_entry` 未知抛 ValueError / entry 字段（config_cls/model_cls/wrapper_cls/load_weights_fn/config_json_key）×3 模型 / `detect_model_type`（gemma4/qwen3/qwen3_5 真实 HF model_type）/ 缺字段+未知抛错 |
| `export_wrapper.py`（基类） | test_export_wrapper_base.py | 7 | `__init__` 属性（kv_cache_max_len/batch/_model_dtype/_quant_embeddings）/ 自定义 kwargs / `_num_layers`×{1,3,8} / `set_quant_embeddings` 存储 + 复制语义（改原 dict 不影响 wrapper） |

## 环境要求

- **converter venv**（`/mnt/workspace/torch-delegate-harmony/.venv`，torch 2.9.0+cpu + torchair + torch_delegate + transformers），**不装 torch_npu**。
- **必须用 venv python**，不能用系统 python（系统 python 加载 torch_npu → `npu::` 算子走 torch_npu dispatcher，CPU 上报 `NotImplementedError`；venv 无 torch_npu，`_library.py` 自定义 schema+fake+CPU impl 生效）。
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（阻止 torch autoload npu backend）。
- 跑前可 `source scripts/set_env_export.sh`（设 CONVERTER_PYTHON/PYTHONPATH/stubs LD），但 `conftest.py` 已加 sys.path，pytest 裸跑也能 import cann_lm；仅需 venv 的 torch/torchair 可 import。

## 设计原则与约定

1. **不碰 `cann_lm/`** — 测试只读模型代码，遇 API 漂移只改 test 代码（如 `apply_rotary_pos_emb` 新签名、`delta_states` 参数、conv1d 3D weight、GatedDeltaNet/DecoderLayer 多值返回）。cann_lm 改动需单独授权。
2. **`check()=assert`** — `check(name, cond)` 失败即 `assert cond`，pytest 捕获并报位置；不再用计数器（pytest 自计）。
3. **`config` fixture** — small config 由 `make_small_config()` + `@pytest.fixture` 提供，带 `config` 参数的 test_* 复用；零参 test（test_config/test_truncation/test_kv_sharing）自建 config。
4. **数值 vs 手算 allclose** — 能精确对齐的算子（rms_norm/rotary_mul/l2norm/dynamic_quant/scatter_nd_update/quant_matmul/attention）写手算参考用 `torch.allclose`；精确难对齐的（chunk_gated_delta_rule/fused_infer_attention causal 路径）放宽到 shape + `torch.isfinite().all()`，注释说明原因。
5. **随机权重，无外部依赖** — 全部用 small config + `torch.randn` 随机权重，不加载真实 HF 模型/权重，快（全套 ~38s）且无机器路径。
6. **parametrize 覆盖维度** — shape/dtype/边界（如 model_registry 的 alias 大小写、ExportWrapper 的层数 1/3/8、fused_ops 的 dim）用 `@pytest.mark.parametrize` 隔离单变量。

## fused_ops 测试要点

`cann_lm/models/fused_ops/_library.py` 把 `npu::` 和 `cann_lm::` 算子注册到 `torch.library`（schema + `register_fake` meta + `register_cpu_impl`）。在 venv（`_STUB_ENV = "libacl_stub" in LD_PRELOAD` 为 False）下 CPU impl 已注册，直接 `torch.ops.npu.npu_rms_norm(...)` 拿 CPU 实现。

| 算子（调用） | 测试 | 数值对齐方式 |
|---|---|---|
| `npu::npu_rms_norm` | shape×3 + rstd>0 + 数值 | vs 手算 `x * rsqrt(mean(x², -1, keepdim) + eps) * gamma` |
| `npu::npu_rotary_mul` | shape×2 + 数值 | vs 手算 `self*r1 + cat(-x2, x1)*r2`（half 模式，对齐 CPU impl） |
| `cann_lm::l2norm` | shape×3 dim + unit norm + 数值 | vs 手算 `x * rsqrt(sum(x², dim, keepdim) + eps)` |
| `npu::npu_dynamic_quant` | y(int8)/scale(fp32) + round-trip | vs `amax/127` scale + `clamp(round(x/s), -128, 127)`；`y*scale` 反量化 allclose |
| `npu::npu_scatter_nd_update` | 1d + 2d indices + out-of-place | vs 手算 `index_copy_(2, ...)` / 逐坐标赋值；原 cache 不被污染 |
| `npu::npu_quant_matmul` | group1 / grouped+bias / pertoken | vs `x1 @ (w*scale_exp) + bias`（含 repeat_interleave、pertoken 量化路径） |
| `npu::npu_chunk_gated_delta_rule` | out + final_state shape + finite | shape-only（数值精确难，qwen3.5 modeling 间接覆盖） |
| `npu::npu_fused_infer_attention_score_v2` | full/causal/GQA-BSH/extern mask | full + extern bool mask 两路 vs 手算 SDPA allclose；causal 路径 shape+finite（CPU impl `_build_sparse_mask` BNSD 解包限制，见已知限制） |

> `_library.py` / `ops.py` 是注册基础设施（定义 namespace + `safe_define` + `register_cpu_impl` helper），由上述算子 import 触发注册，不单独写测试（文件顶部注释说明）。

## 运行方式

```bash
# 全套 (converter venv):
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/ -q

# 单文件:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/test_fused_ops.py -q

# 单用例:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/test_model_registry.py::test_get_entry_resolves_aliases -q

# 收集 (不跑, 看项数):
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/ --co -q
```

## 已知限制

1. **`fused_infer_attention_score_v2` causal 路径**：CPU impl `_build_sparse_mask` 解包 `b,n,s_q,s_k = q.shape`，BNSD 下 `q=[b,n,s,d]` 使 `s_k=d`、`s_q≠s_k` 命中 early-return `None` → causal mask 被跳过。这是 cann_lm CPU impl 限制（不碰 cann_lm），测试中 causal 用例放宽为 shape+finite；数值精确对齐改由 `extern_mask`（传 bool causal atten_mask）测试覆盖。
2. **`chunk_gated_delta_rule`**：融合算子数值精确校验难（BHSD↔BSHD 转换 + delta rule 递归），仅测 out/final_state shape + finite；数值由 qwen3.5 modeling（test_modeling_qwen3_5）间接覆盖。
3. **`ExportWrapper` 基类 forward**：基类 `forward` 调 `self._drive_layers`（子类实现，基类未定义），故 base 测试不测 forward，只测 `__init__`/`set_quant_embeddings`/`_num_layers`；forward 由各模型 wrapper（test_gemma4 `test_wrapper_vs_model` 等）端到端覆盖。
4. **真实权重/出字**：不在 UT 范围（需 HF 模型 + venv/系统 python），归 `examples/test_converter/llm_models_run_on_cpu/`（CPU e2e）+ `examples/test_converter/export_llm_model/`（HF→epair→OM→NPU e2e，`run_e2e_pipeline.sh`）。

## 维护指南

- **新增模型**：在 `cann_lm/models/<new>/` 加 modeling/configuration/export_wrapper 后，于 `model_registry.py` REGISTRY 注册；测试侧加 `tests/llm_models/test_<new>.py`（仿 test_gemma4.py：config fixture + 各组件 test_*），并在 `test_model_registry.py` 补 entry 字段 + alias parametrize。
- **新增融合算子**：在 `cann_lm/models/fused_ops/<op>.py` 加 schema+fake+CPU impl 后，于 `test_fused_ops.py` 加 `test_<op>_*`（shape + 数值 vs 手算 allclose，或难对齐则 shape+finite + 注释）。
- **API 漂移**：cann_lm modeling 演进导致 test 失败时，**只改 test 代码**对齐新签名（如多值返回解包、参数重命名），不动 cann_lm/；commit message 注明漂移点。
- **环境**：测试依赖 venv 的 torch/torchair/torch_delegate + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；CI 用 `bash scripts/build.sh deps` 装 venv 后跑 `pytest tests/llm_models/`。
