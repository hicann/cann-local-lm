# UT 设计：cann_lm 单元测试 (tests/llm_models/ + tests/converter/)

> 状态: 已落地 | 最后核对: 2026-09-05 | 全套 192 passed (llm_models 99 + converter 93, converter venv, 无 torch_npu)

## 背景

`cann_lm/` 分两大测试域：
- **`cann_lm/models/`**（三模型 gemma4/qwen3/qwen3_5 + 融合算子 fused_ops + 注册表 + 基类 wrapper）— converter 导出与 NPU 推理的数值基础。此前单元测试散落在 `examples/test_converter/llm_models_run_on_cpu/debug_*/verify_*.py` 的 `--test-only` 模式里（与 e2e 耦合，自定义 `check()` runner 不抛异常导致 pytest 假 PASS），fused_ops/registry/基类 wrapper 无测试。
- **`cann_lm/converter/`**（loader/saver/swap/quant_swap materialize/epair_utils/CLI + quant/dopt_lm）— 导出工具链。`tests/dopt_lm/` 已覆盖 dopt_lm/* + swap/dopt_wrapper，但 loader/saver/epair_utils/quant_swap/CLI 无测试。

本设计把单元测试统一抽到 `tests/llm_models/` + `tests/converter/`，参考 `tests/dopt_lm/` 范式（pytest + `@pytest.mark.parametrize` + `assert` + `tmp_path`），实现 `cann_lm/` **全覆盖**（torchair/GE/真实权重/网络依赖的函数跳过 + 注释，归 e2e）。

## 测试架构 (tests/llm_models/)


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

## converter 测试 (tests/converter/)

```
tests/converter/
├── conftest.py            # sys.path→REPO_ROOT; import cann_lm.converter.* 无需手动 source env
├── test_materialize.py    # quant_swap/materialize.py int4/int8 权重量化 (15 项, 纯 tensor ops)
├── test_epair_utils.py    # export/epair_utils.py 权重打包/切分/落盘/ep_json 后处理 (21 项)
├── test_loader.py         # loader.py ModelArtifact/build_config/collect_tokenizer_files (12 项)
├── test_swap.py           # swap.py _resolve_parent/_infer_group_size/_read_dopt_quant_meta/_read_quant_param_map (22 项)
├── test_saver.py          # saver.py save_model_config/copy_tokenizer_files (7 项)
└── test_cli.py            # cann_lm_converter.py CLI argparse (16 项)
```

`tests/dopt_lm/` 已覆盖 `quant/dopt_lm/*`（vendored dopt 包）+ `swap.swap` + `quant/dopt_wrapper.py`，不重复；`tests/converter/` 补齐 converter 其余可测函数。

### 覆盖矩阵

| cann_lm/converter 文件 | 测试文件 | 项数 | 覆盖 | 跳过（归 e2e/需重依赖） |
|---|---|---|---|---|
| `export/quant_swap/materialize.py` | test_materialize.py | 15 | transdata↔restore_scale round-trip（mantissa 低 13bit 清零）/ pack_int4 nibble 布局+round-trip / int4 per-group 量化（shape+clamp(-7,7)+fakequant 精确重建）/ int8 per-channel（shape+dequant allclose+dopt fakequant 精确）/ 错误处理 | — |
| `export/epair_utils.py` | test_epair_utils.py | 21 | ConsolidatedWeights / `_pack_weights_to_dir`（单文件+按 max_file_size 切分+bf16→int16+weight_scale→int64 transdata）/ `consolidated_weights_save` 落盘结构 / `_strip_none_input_descs` 清理 / `inline_fused_attn_seq_constants`+`inline_weight_scale_dtype` 原地改写 | 需真实 ExportedProgram/torchair（用 `_FakeContents` stub 绕开） |
| `loader.py` | test_loader.py | 12 | ModelArtifact 字段 / `build_config`（qwen3/gemma4/qwen3_5 嵌套 text_config + post_init + fallback）/ `collect_tokenizer_files`（有/缺标准/legacy 回退/空目录） | `load_model` 全流程（需真实 HF 权重+safetensors） |
| `swap.py` | test_swap.py | 22 | `_resolve_parent`（深层树/单段/索引）/ `_infer_group_size`（pergroup/perchannel/拒绝 pertensor/非整除）/ `_read_dopt_quant_meta`+`_read_quant_param_map`（tmp_path 造假 dopt 产物解析）/ `load_dopt_state_dict`（.pth/safetensors/剥离 language_model.） | `swap_from_dopt` 全流程（需完整 dopt artifacts + 真实 model） |
| `saver.py` | test_saver.py | 7 | `save_model_config`（to_dict/dataclass asdict/__dict__ 三路 + use_embeddings/quant_embeddings/quant_info）/ `copy_tokenizer_files`（复制全部/空列表） | `save_epair`/`save_epair_prefill_decode`（需 torch.export wrapper/ep_module + torchair GE） |
| `cann_lm_converter.py` | test_cli.py | 16 | `parse_args` --help/缺 required/非法 choices（dtype/mode/quant-method）/quant-path∩quant-method 互斥/合法 Namespace+默认值 | `convert()` 全流程（需真实 HF + torchair GE） |
| `quant/dopt_lm/*` + `quant/dopt_wrapper.py` + `swap.swap` | tests/dopt_lm/（既有） | — | dopt 量化全链路 | — |

### 已知限制（converter）

1. **torchair/GE 依赖**：`saver.save_epair*`（torch.export+GE 转换）、`ge_converters/*`（FX→GE）、`cann_lm_converter.convert()` 全流程需 torchair stubs + 真实导出，归 `examples/test_converter/export_llm_model/run_e2e_pipeline.sh` e2e 验证，不进 UT。
2. **真实 HF 权重**：`loader.load_model` 全流程、`swap.swap_from_dopt` 全流程需 safetensors/完整 dopt artifacts，UT 只测其**解析子函数**（build_config/collect_tokenizer_files/_read_dopt_*）。
3. **网络**：`quant/download_dataset.py` 需联网，不测。
4. **NPU/dopt subprocess**：`quant/auto_quant.py` spawn dopt GPTQ subprocess 需 NPU，不测（归 `examples/test_converter/dopt_quantization_st/`）。
5. **stub 绕开**：epair_utils 测试用 `_FakeEP`/`_FakeGraphSignature`/`_FakeContents` stub 绕开真实 ExportedProgram（符合 converter 设备无关约定，CPU 可跑）。
6. **int4 clamp 范围**：`materialize` 用 `clamp(-qmax, qmax)=clamp(-7,7)`（对称量化排除 -8 边界），测试据此断言（非 [-8,7]）。

## 运行方式

```bash
# 全套 (converter venv):
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/ tests/converter/ -q

# llm_models 全套:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/ -q

# converter 全套:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/converter/ -q

# 单文件:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/converter/test_materialize.py -q

# 单用例:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/converter/test_materialize.py::test_pack_int4_roundtrip -q

# 收集 (不跑, 看项数):
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/llm_models/ tests/converter/ --co -q
```

 ## 已知限制（llm_models）

 1. **`fused_infer_attention_score_v2` causal 路径**：CPU impl `_build_sparse_mask` 解包 `b,n,s_q,s_k = q.shape`，BNSD 下 `q=[b,n,s,d]` 使 `s_k=d`、`s_q≠s_k` 命中 early-return `None` → causal mask 被跳过。这是 cann_lm CPU impl 限制（不碰 cann_lm），测试中 causal 用例放宽为 shape+finite；数值精确对齐改由 `extern_mask`（传 bool causal atten_mask）测试覆盖。
 2. **`chunk_gated_delta_rule`**：融合算子数值精确校验难（BHSD↔BSHD 转换 + delta rule 递归），仅测 out/final_state shape + finite；数值由 qwen3.5 modeling（test_modeling_qwen3_5）间接覆盖。
 3. **`ExportWrapper` 基类 forward**：基类 `forward` 调 `self._drive_layers`（子类实现，基类未定义），故 base 测试不测 forward，只测 `__init__`/`set_quant_embeddings`/`_num_layers`；forward 由各模型 wrapper（test_gemma4 `test_wrapper_vs_model` 等）端到端覆盖。
 4. **真实权重/出字**：不在 UT 范围（需 HF 模型 + venv/系统 python），归 `examples/test_converter/llm_models_run_on_cpu/`（CPU e2e）+ `examples/test_converter/export_llm_model/`（HF→epair→OM→NPU e2e，`run_e2e_pipeline.sh`）。

 ## 维护指南

 - **新增模型**：在 `cann_lm/models/<new>/` 加 modeling/configuration/export_wrapper 后，于 `model_registry.py` REGISTRY 注册；测试侧加 `tests/llm_models/test_<new>.py`（仿 test_gemma4.py：config fixture + 各组件 test_*），并在 `test_model_registry.py` 补 entry 字段 + alias parametrize。
 - **新增融合算子**：在 `cann_lm/models/fused_ops/<op>.py` 加 schema+fake+CPU impl 后，于 `test_fused_ops.py` 加 `test_<op>_*`（shape + 数值 vs 手算 allclose，或难对齐则 shape+finite + 注释）。
 - **新增 converter 函数**：在 `cann_lm/converter/<x>.py` 加纯逻辑/解析/打包函数后，于 `tests/converter/test_<x>.py` 补测（纯 tensor ops 数值 allclose、解析函数用 tmp_path 造假产物、CLI argparse）；需 torchair/GE/真实权重/网络的函数跳过 + 注释（归 e2e）。
 - **API 漂移**：cann_lm 演进导致 test 失败时，**只改 test 代码**对齐新签名/行为（如多值返回解包、参数重命名、materialize clamp 范围），不动 cann_lm/；commit message 注明漂移点。
 - **环境**：测试依赖 venv 的 torch/torchair/torch_delegate + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；CI 用 `bash scripts/build.sh deps` 装 venv 后跑 `pytest tests/llm_models/ tests/converter/`。

## C++ 测试 (tests/c++/epair_archive/)

测 `cann_lm/converter/export/csrc/` 的 C++ backend（epair 加载/解析/权重打包/aten 映射/value registry）。GTest + CMake，link `torch_delegate_backend` + CANN，**ASAN 强制**。

### 架构

```
tests/c++/epair_archive/
├── CMakeLists.txt              # 顶层 CMakeLists add_subdirectory (BUILD_TESTS=ON 默认); helper add_cann_lm_test
├── gen_fixtures.py             # 生成 .epair fixtures (test_consolidated_weights/epair_archive/epair_model_parser 用)
├── test_consolidated_weights.cc  # 需 fixtures: GeWeightLoader 权重加载/索引/meta
├── test_epair_archive.cc         # 需 fixtures: epair 目录结构 (model.json/weights.bin/weight_info)
├── test_epair_model_parser.cc    # 需 fixtures: EpairModelParser 注解解析 (symint/value refs/tensor meta)
├── test_base64.cc                # 纯单测: base64 解码 (空串/二进制/非法/长度)
├── test_value_registry.cc        # 纯单测: ValueRegistry placeholder/producer/alias/重复/未知
├── test_aten_schema_resolver.cc  # 纯单测: AtenSchemaResolver aten/自定义命名空间/inplace/默认 overload
└── test_aten_tensor_desc_mapper.cc  # 纯单测: aten→GE dtype/layout 映射 (float/half/int64/bool/unsupported)
```

- **顶层 `CMakeLists.txt`** 第 31-34 行 `if(BUILD_TESTS) add_subdirectory(tests/c++/epair_archive) endif()`，默认 ON。
- `add_cann_lm_test(name src [NEEDS_FIXTURES])` helper：建 exe + include csrc + link backend+CANN+GTest + `delegate_apply_common_flags`（走 `DELEGATE_COMMON_FLAGS`，`-Wall -Wextra -Werror`）+ rpath-link + 注册 ctest（`LD_LIBRARY_PATH=CANN_LIB_DIRS`）。
- fixtures 由 `gen_fixtures.py`（custom target `generate_fixtures`）产到 `${BINARY_DIR}/fixtures`，3 个 NEEDS_FIXTURES 测试 add_dependencies 它；`FIXTURES_DIR` 宏注入。

### 覆盖矩阵（62 用例 / 7 exe，全 ASAN PASS）

| csrc 模块 | 测试文件 | 用例数 | 覆盖要点 |
|---|---|---|---|
| `AtenSchemaResolver` | test_aten_schema_resolver.cc | 10 | aten/自定义命名空间解析、inplace op、默认 overload→空、无点 target 拒绝、MakeFallbackOpType |
| `aten_tensor_desc_mapper` | test_aten_tensor_desc_mapper.cc | 18 | pt2→GE dtype（float/half/int64/bool/unsupported）、layout（strided/unsupported）、GE dtype string（float/int64/bool）、layout string（ND/NCHW/NHWC） |
| `base64` | test_base64.cc | 6 | 解码纯串/二进制、空串、padding、非法 payload/长度拒绝 |
| `GeWeightLoader`（consolidated weights） | test_consolidated_weights.cc | 9 | tensor holder 加载、缺 name/meta/index 报错、binary values、is_parameter、record offset/size、GetTensorMeta、HasTensor |
| `EpairArchiveDirectory` | test_epair_archive.cc | 5 | model.json/weights.bin/weight_info 存在+非空、缺目录 fail |
| `EpairModelParser` | test_epair_model_parser.cc | 7 | 注解 fixture 解析、symint user input、value refs 标量类型、拒绝 auto_functionalized v1 节点、tensor meta as-expr、scalar value maps、scalar fallback 节点 inputs |
| `ValueRegistry` | test_value_registry.cc | 7 | placeholder/producer output 注册解析、重复 key 拒绝、alias（含自动补 port0）、未知返 null、链式 alias |

### 环境与运行

- **② 真 CANN env**（`source scripts/set_env_cann.sh`，无 stubs），需 `ASCEND_HOME_PATH` + `build/libtorch_delegate_backend.so`。
- **ASAN 强制**（兑现 AGENTS.md）：`bash scripts/build.sh cpp --asan`（= `build_cpp.sh --asan`，重配 build/ `ENABLE_ASAN=ON` + 编 backend/cann_lm_compile/tests 全部）→ `ctest --test-dir build`。
- 零警告：`-Wall -Wextra -Werror`（经 `DELEGATE_COMMON_FLAGS`，`delegate_apply_common_flags`），PR 不许引入新警告；GE 头 `ge_error_codes.h` deprecated 提示属 CANN 侧，非本仓。
- 裸跑（ctest 无需 LD_PRELOAD，ASAN flags 含 `-static-libasan`）。

### 已知限制（C++）

1. **fixtures 依赖**：3 个 NEEDS_FIXTURES 测试需 `gen_fixtures.py` 产 .epair（纯 Python + torchair，converter venv 跑）；缺 fixtures 则这 3 个失败，4 个纯单测仍可跑。
2. **CANN 重库依赖**：link graph/metadef/ascendcl 等，需 ② 真 CANN env（非 stubs）；与 Python UT（converter venv + stubs）env 互斥，分开跑。
3. **不测 OM 编译/运行**：epair→OM 编译（`cann_lm_compile`）+ NPU acl 运行归 e2e（`examples/test_converter/export_llm_model/run_e2e_pipeline.sh`），UT 只测 csrc 的 epair 加载/解析/映射层。

## 提交前 C++ 本地验证（强制）

修改 `cann_lm/converter/export/csrc/`（或 `tests/c++/epair_archive/`）后，commit 前必须：
```bash
bash scripts/build.sh cpp --asan && ctest --test-dir build --output-on-failure
```
全 7 exe PASS + ASAN 无内存错误才提交；失败只改 test/csrc（csrc 改动需授权），禁带错提交。

## 用例级设计（per-case）

每个测试函数的设计：场景/输入要点 + 期望断言 + 边界·原理。parametrize 变体合并为一行（标注 case 数）。

### llm_models

#### test_gemma4.py（gemma4 modeling, 10 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_config | 30 层默认 config + small 4 层 + tiny 2 层 | layer_types 长度/末层 full/sliding 5:1/rope 两类型/global_head_dim512/head_dim256 | 验 config 默认值 + 层数→layer_types 生成规则 |
| test_components | RMSNorm/RoPE/apply_rotary/repeat_kv/rotate_half 小张量 | shape 一致 + RoPE cos 维=缓存 buffer + rotate_half 数值精确 | RoPE full 返回部分 rotary_dim（cos≠global_head_dim），用缓存 buffer 对齐 |
| test_attention | 各 layer_type 首层 attn：无 cache + 存 KV + 单 token cached | shape 一致 + KV cache store 非 None + cached shape | 验 sliding/full 两类 + prefill/decode 两阶段 |
| test_mlp | 前 2 层 MLP | out.shape==x.shape | 组件 smoke |
| test_decoder_layer | 首层 DecoderLayer | out.shape==x.shape | 组件 smoke |
| test_text_model | TextModel prefill（无 cache/2D mask） | hs.shape + past_key_values 长度 + masked shape | 验 attention_mask=None/2D 两路（None 走 _prepare_causal_mask past_len 分支） |
| test_causal_lm | ForCausalLM logits/loss/softcap/generate(sampling) | logits shape + loss 标量 + generate 更长 | 验 final_logit_softcapping + greedy/sampling |
| test_kv_sharing | 6 层后 3 共享 KV | hs shape + shared_kv_states 非空 + 共享层 is_kv_shared + 无 k_proj | 验 KV sharing 配置生效（共享层无 k/v proj） |
| test_truncation | 2/4/8 层截断 | logits vocab 维=1000 + 参数量 | 验 --num-layers 截断 forward |
| test_wrapper_vs_model | wrapper 单步 vs 原始 model（mask_full+sliding） | logits max_diff=0 + KV max_diff=0 | wrapper 是 model 薄封装；mask 用 _build_mask_full/sliding helper 预构 |

#### test_qwen3.py（qwen3 modeling, 13 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_config | 默认 config | layer_types/字段默认 | config smoke |
| test_rmsnorm | RMSNorm | 数值 vs 手算 | 组件 |
| test_rope | RoPE | cos/sin shape + 数值 | 组件 |
| test_mlp | MLP | shape | 组件 |
| test_attention_and_layer | attn + DecoderLayer | shape | 组件 |
| test_forward_shape_and_tie | forward + tie_word_embeddings | logits shape + tie 生效 | 验 weight tying |
| test_determinism | 两次 forward | logits allclose | 确定性（无随机） |
| test_causal_invariance | 打乱 prompt 顺序 | 各 token logits 不受后续影响 | 验 causal mask |
| test_kv_cache_consistency | prefill+decode 用 KV | cached vs 非cached 一致 | 验 KV cache 复用 |
| test_greedy_equals_argmax | greedy 首 token | == argmax(forward[:,-1]) | 验 greedy 等价 argmax |
| test_loss | labels | loss 标量 + 有限 | loss 计算 |
| test_truncation | 截断层数 | forward shape | 截断 smoke |
| test_wrapper_vs_model | wrapper vs model | max_diff=0 | 薄封装一致性 |

#### test_modeling_qwen3_5.py（qwen3.5 modeling, 25 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_rms_norm / _gated | RMSNorm / GatedRMSNorm | shape + not_nan + 数值 | 组件（gated 多一个 gate） |
| test_rotary_embedding | mrope cos/sin | shape + 数值 | qwen3.5 mrope |
| test_rotate_half | rotate_half([-1..-4,1..4]) | ==[-3,-4,1,2] | 精确数值 |
| test_apply_rotary_pos_emb | apply_rotary | q/k shape + not_nan | 验 RoPE 应用 |
| test_repeat_kv | repeat_kv | shape + n_rep=1 same | GQA repeat |
| test_eager_attention_forward | eager attn | shape + not_nan | attention 组件 |
| test_apply_mask_to_padding_states | mask | 无 mask 不变 + 零 mask 清零 | padding 处理 |
| test_l2norm | l2norm | unit norm + 数值 | 组件 |
| test_causal_conv1d_update | conv1d update（3D weight） | shape + not_nan + state 更新 | 修了 weight 2D→3D API 漂移 |
| test_chunk_gated_delta_rule | chunk kernel | out/final_state shape + not_nan | 融合算子 shape-only |
| test_recurrent_gated_delta_rule | recurrent kernel | shape + not_nan | 递归路径 |
| test_simple_kv_cache | SimpleKVCache | 首次 same + 二次 cat(12) | KV cache |
| test_simple_delta_net_cache | DeltaNetCache | has_state + conv 更新 | delta cache |
| test_gated_delta_net | GatedDeltaNet（无/有 cache） | output shape（解 3-tuple 返回） | 修了多值返回 API 漂移 |
| test_attention / _mlp / _decoder_layer | full attn + mlp + layer | shape | 组件（DecoderLayer 解 3-tuple） |
| test_text_model | TextModel（无/有 cache） | hidden + kv/delta cache 非空 | 全模型 forward |
| test_causal_lm | ForCausalLM | logits + loss + generate | LM smoke |
| test_generate / _with_eos | greedy + EOS | 长度 + EOS 停 | 生成 + 提前停 |
| test_model_with_inputs_embeds | inputs_embeds | logits shape | embeddings 入口 |
| test_incremental_decode | prefill+decode（delta_states） | 增量 logits shape | 修了 delta_cache→delta_states 漂移 |
| test_error_conditions | 无 input_ids/inputs_embeds | ValueError | fail-fast |

#### test_fused_ops.py（fused_ops CPU impl, 21 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_npu_rms_norm（×3 shape + 默认eps） | rms_norm(x, gamma, eps) | y.shape + rstd shape/>0 + vs 手算 allclose | 数值对齐 `x*rsqrt(mean(x²)+eps)*gamma` |
| test_npu_rms_norm_default_eps | 默认 epsilon=1e-6 | 数值 | 默认参数 |
| test_npu_rotary_mul_half（×2 d）/ _default_mode | rotary_mul(self,r1,r2,half) | shape + vs 手算 `self*r1+cat(-x2,x1)*r2` | half 模式数值对齐 CPU impl |
| test_cann_lm_l2norm_unit_norm_and_values（×3 dim） | cann_lm::l2norm | 沿 dim L2=1 + vs 手算 | cann_lm 命名空间 + 多 dim |
| test_npu_dynamic_quant_shape_dtype_and_roundtrip | dynamic_quant(input) | y(int8)/scale(fp32,>0) + y*scale allclose | vs `amax/127` scale + clamp(round)；反量化 round-trip |
| test_npu_scatter_nd_update_1d/2d_indices | scatter_nd_update | out.shape + vs 手算 index_copy/逐坐标 + 原 cache 不污染 | KV scatter-write，1d/2d 两路 |
| test_npu_quant_matmul_group1/grouped_pertoken | quant_matmul（3 配置） | out shape + finite + vs `x1@(w*scale_exp)+bias` | group1/grouped+bias/pertoken 三量化路径 |
| test_npu_chunk_gated_delta_rule_shape_finite | chunk_gated_delta_rule | out+final_state shape + finite | 数值难对齐，shape-only（qwen3.5 间接覆盖） |
| test_npu_fused_infer_attention_score_v2_bnsd_full / _causal_shape_finite / _gqa_bsh / _extern_mask | fused attn（4 配置） | out shape + finite；full/extern_mask vs 手算 SDPA allclose | causal 放宽（CPU impl _build_sparse_mask 限制），数值靠 extern_mask |

#### test_model_registry.py（model_registry, 23 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_available_models_covers_three_models | available_models | =={gemma4,qwen3,qwen3_5} | 注册表完整性 |
| test_get_entry_resolves_aliases（9 case） | gemma4/gemma-4/大小写、qwen3/qwen-3、qwen3_5/qwen3.5 | == REGISTRY[expected] | _ALIASES + 大小写不敏感 |
| test_get_entry_unknown_raises（5 case） | unknown/llama/""/gemma5/qwen4 | ValueError | 未知模型 fail-fast |
| test_entry_fields（3 模型） | 各 entry | config_cls/model_cls/wrapper_cls/config_json_key + load_weights_fn callable | 字段正确性（gemma4/qwen3_5 text_config，qwen3 None） |
| test_detect_model_type_from_hf_config（3 case） | {"model_type":gemma4/qwen3/qwen3_5} | == REGISTRY[key] | 真实 HF config 顶层 model_type |
| test_detect_model_type_missing_field_raises | {} | ValueError | 缺 model_type |
| test_detect_model_type_unknown_raises | {"model_type":"llama"} | ValueError | 未知 |

#### test_export_wrapper_base.py（基类, 7 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_init_attributes_defaults | ExportWrapper(stub, 2048, 1) | kv_cache_max_len/batch/_model_dtype(fp32)/_quant_embeddings={} | 默认属性 |
| test_init_custom_kwargs | bf16 stub, 512, 4 | 自定义值 + dtype=bf16 | 自定义 kwargs |
| test_num_layers_matches_stub（1/3/8） | _num_layers | == stub 层数 | 层数读取（parametrize 1/3/8） |
| test_set_quant_embeddings_stores_dict | set_quant_embeddings | dict 存储 + int8/scale shape | quant-embeddings 注入 |
| test_set_quant_embeddings_is_copy_not_ref | 改原 dict | wrapper 不受影响 | set 时 dict() 复制（隔离） |

### converter

#### test_materialize.py（quant_swap, 15 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_transdata_scale_dtype_shape | transdata_scale(s) | int64 + 同 shape | dtype/shape |
| test_transdata_restore_roundtrip_zeroes_mantissa_low13 | restore(transdata(s)) | 低 13 mantissa bit=0 + allclose | round-trip 清低 13 位（存储格式） |
| test_transdata_scale_masks_high32_of_int64 | transdata | 高 32 bit=0 + mask 外位=0 | int64 高位清零 |
| test_pack_int4_layout_known_values | [1,-1,7,-8] | byte0=0xF1, byte1=0x87（&0xFF） | nibble 布局（int8 有符号，用 &0FF 比无符号） |
| test_pack_int4_roundtrip | 随机 int4 | unpack==原 | pack/unpack 互逆（4-bit 符号扩展） |
| test_pack_int4_odd_last_dim_raises | 奇数末维 | ValueError | 偶数约束 |
| test_materialize_int4_shape_unpacked_and_packed | w+s（packed 开/关） | q shape [out,in]/[out,in//2] | packed vs unpacked |
| test_materialize_int4_quant_dequant_exact_fakequant | w=q0*s（q0∈[-7,7]） | q==q0 + dequant==w | fakequant 精确重建（无 clamp） |
| test_materialize_int4_clamp_beyond_range | w/s 超 7 | clamp(-7,7) | 对称量化排除 -8 边界 |
| test_materialize_int4_errors（2 case） | in 不整除/shape 不符 | ValueError | 参数校验 |
| test_materialize_int8_shape_and_range | int8 量化 | shape + [-128,127] | 范围 |
| test_materialize_int8_dequant_allclose | 随机 w | dequant allclose | 量化误差 |
| test_materialize_int8_dopt_fakequant_exact_roundtrip | w=q0*scale | q==q0 | dopt fakequant 精确 |
| test_materialize_int8_scale_shape_error | scale shape 错 | ValueError | 校验 |

#### test_epair_utils.py（epair_utils, 21 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_consolidated_weights_dataclass_fields | ConsolidatedWeights | 字段 | dataclass |
| test_pack_weights_to_dir_single_file | 小权重 | 单 weights_0.bin + info | 不切分 |
| test_pack_weights_to_dir_splits_on_max_file_size | 小 max_file_size | 多分片 + 总大小 | 切分逻辑 |
| test_pack_weights_to_dir_single_tensor_exceeds_limit_not_split | 单 tensor 超限 | 独立文件不切 | 单 tensor 不可切 |
| test_pack_weights_to_dir_bfloat16_viewed_as_int16 | bf16 权重 | int16 view | bf16→int16 打包 |
| test_pack_weights_to_dir_weight_scale_transdata_to_int64 | weight_scale | int64 transdata | scale 存储格式 |
| test_consolidated_weights_save_writes_all_files | save（tmp_path） | weights_*.bin+info+ep_json | 落盘结构 |
| test_consolidated_weights_save_skip_weights_only_ep_json | save_weights=False | 仅 ep_json | 跳权重 |
| test_consolidated_weights_save_strips_none_input_descs_in_ep_json | save 联动 strip | None desc 清理 | save+strip 联动 |
| test_strip_none_input_descs_removes_none_dtype_and_empty_ref | ep_json | 删 None dtype+空 ref | 清理 |
| test_strip_none_input_descs_keeps_nonempty_ref_even_if_dtype_none | ep_json | 保留非空 ref | 不误删 |
| test_strip_none_input_descs_ignores_non_quant_ops | 非 QuantBatchMatmulV4 | 跳过 | 仅清量化 op |
| test_strip_none_input_descs_skips_when_length_mismatch | 长度不符 | 跳过 | 防越界 |
| test_pack_weights_writes_to_cwd_and_returns_none | 遗留 pack_weights | 写 cwd + 返 None | 遗留函数 |
| test_pack_weights_splits_on_max_file_size | 遗留 + 切分 | 切分 | — |
| test_inline_fused_attn_seq_constants_inlines_qlen_and_kvlen | ep_json（tmp） | 内联 qlen+kvlen | 常量内联 |
| test_inline_fused_attn_seq_constants_skips_empty_kvlen | 空 kvlen | 跳过 | 边界 |
| test_inline_fused_attn_seq_constants_skips_op_without_qlen | 无 qlen | 整体跳 | 边界 |
| test_inline_fused_attn_seq_constants_kvlen_defaults_to_qlen | 无 kvlen | =qlen | 默认 |
| test_inline_weight_scale_dtype_relabeled_int64 | dtype7→5 | relabel | dtype 标注 |
| test_inline_weight_scale_dtype_no_tensor_values | 无 tensor_values | 跳过 | 边界 |

#### test_loader.py（loader, 12 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_model_artifact_fields / _quant_info_defaults_none | ModelArtifact | 字段 + quant_info 默认 None | dataclass |
| test_build_config_qwen3_filters_unknown_keys | qwen3 hf_config | 过滤未知 key | 字段过滤 |
| test_build_config_qwen3_uses_defaults_for_missing | 缺字段 | 用默认 | 默认值 |
| test_build_config_gemma4_reads_nested_text_config / _post_init_ran / _falls_back_to_top_level | gemma4 嵌套/缺 key | text_config 读取 + post_init + fallback | 嵌套 config + fallback |
| test_build_config_qwen3_5_reads_nested_text_config | qwen3.5 嵌套 | text_config | 嵌套 |
| test_collect_tokenizer_files_with_tokenizer_json / _missing_some_standard / _fallback_to_legacy / _empty_dir | tmp_path 假 HF 目录 | tokenizer 文件列表（有/缺/legacy/空） | 4 路径覆盖 |

#### test_swap.py（swap, 22 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_resolve_parent_returns_parent_and_leaf / _single_segment / _index_path | nn.Module 树 | (parent, leaf) 正确 | 深层/单段/索引路径 |
| test_infer_group_size_pergroup / _perchannel | scale 1D | group_size | 两模式 |
| test_infer_group_size_rejects_pertensor / _non_1d / _not_divisible_by_out / _in_not_divisible_by_group / _numel_below_out | 非法 | ValueError | 5 种错误路径 |
| test_read_dopt_quant_meta_fallback_when_buffers_absent / _reads_buffers_and_reshapes_offset | tmp dopt | fallback/读取+reshape | dopt meta 解析 |
| test_read_quant_param_map_none / _dir_without_config / _unknown_strategy / _invalid_suffix / _layer_field_overrides | tmp dopt | 空/跳过/覆盖 | param map 解析 + 边缘 |
| test_load_dopt_state_dict_pth / _strips_language_model_prefix / _dir_resolves_model_safetensors / _missing_raises / _not_found_raises | .pth/safetensors/dir | 加载/剥离前缀/解析/报错 | 5 路径 |

#### test_saver.py（saver, 7 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_save_model_config_uses_to_dict / _dataclass_asdict / _dict_fallback | stub artifact（3 config 路径） | model_config.json 内容 | 3 config 序列化路径 |
| test_save_model_config_use_embeddings_flag | use_embeddings | flag 写入 | flag 传递 |
| test_save_model_config_writes_quant_info_when_present | quant_info | 写入 | 量化信息 |
| test_copy_tokenizer_files_copies_all / _empty_list_no_error | tmp 文件列表 | 复制全部/空不报错 | 文件复制 |

#### test_cli.py（cann_lm_converter CLI, 16 项）

| 测试函数 | 场景/输入 | 期望断言 | 边界·原理 |
|---|---|---|---|
| test_help_exits_zero_and_lists_key_args | --help | exit 0 + 含关键参数 | help smoke |
| test_missing_required_hf_path/output_dir_exits_nonzero | 缺 required | exit 2 | required 校验 |
| test_invalid_dtype/mode/quant_method_exits_nonzero | 非法 choices | exit 2 | choices 校验 |
| test_quant_path_and_quant_method_mutually_exclusive | 两都给 | exit 2 | 互斥组 |
| test_valid_minimal_args_namespace_and_defaults | 合法最小 | Namespace + 默认值 | 解析正确 |
| test_valid_dtype_choices / _no_use_embeddings / _num_layers / _single_mode / _quant_path_alone / _quant_method_gptq_alone | 各参数单用 | 解析 | choices/flag/类型/单用 |
