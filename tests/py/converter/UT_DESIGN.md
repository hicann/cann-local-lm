# converter 单元测试设计报告

## 1. 文档信息

- 被测对象：`cann_lm/converter/`（不含 `quant/dopt_lm/` 与 `quant/dopt_wrapper.py`，二者归 `tests/py/dopt_lm/`）
- 测试归档目录：`tests/converter`
- 测试框架：`pytest`
- 基准环境：Python 3.11、converter venv（torch 2.9.0+cpu + torchair + torch-delegate，不装 `torch_npu`），`TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- 设计日期：2026-09-05

## 2. 测试目标与范围

本轮 UT 验证 `cann_lm/converter/` 中**可隔离的纯逻辑/解析/打包函数**：权重量化 materialize、epair 权重打包切分与 ep_json 后处理、loader 配置构造与 tokenizer 收集、swap 的 dopt 产物解析与 Linear/Embedding 替换、saver 配置落盘与文件复制、QuantLinear 构造、CLI argparse、weight_format 布局变换。测试必须满足以下原则：

1. 默认在 CPU 离线运行，`tmp_path` 造假 HF/dopt 产物，不下载 Hugging Face 模型或数据集。
2. 数值算法使用小尺寸确定性 Tensor，验证数值、形状、dtype 与 `torch.equal`/`allclose` 精确重建。
3. `torchair`/GE/真实 `ExportedProgram`/真实权重/网络依赖的函数跳过（归 e2e），不混入 UT。
4. CLI argparse 用临时替换 `sys.argv` + `SystemExit` 验证，不跑真实 `convert()`。
5. 覆盖率目标：纯逻辑/数值模块行覆盖率不低于 80%；需 torchair/GE/真实权重的执行体以 orchestration 分支覆盖为主，不以全行覆盖率作为 UT 准入条件。

范围内模块：

- `export/quant_swap/materialize.py`、`export/quant_swap/quant_linear.py`、`export/weight_format/layout.py`、`export/weight_format/rules.py`
- `export/epair_utils.py`
- `loader.py`、`swap.py`、`saver.py`
- `cann_lm_converter.py`（仅 `parse_args`/`_DTYPE_MAP`）

范围外项目：

- `export/ge_converters/*`（FX→GE converter，需 torchair stubs + 真实导出，归 `examples/test_converter/export_llm_model/run_e2e_pipeline.sh`）
- `saver.save_epair` / `saver.save_epair_prefill_decode`（需 `torch.export` 的 wrapper/ep_module + torchair GE 转换）
- `loader.load_model` 全流程（需真实 HF 权重 + safetensors + model_cls 实例化）
- `swap.swap_from_dopt` 全流程需完整 dopt artifacts + 真实 model（UT 只测其解析子函数）
- `quant/auto_quant.py`（spawn dopt GPTQ subprocess 需 NPU）、`quant/download_dataset.py`（需联网）、`quant_entry.py`、`cann_lm_converter.convert()` 全流程
- `quant/dopt_lm/*` 与 `quant/dopt_wrapper.py`（已由 `tests/py/dopt_lm/` 覆盖，不重复）

## 3. 代码分析结论

### 3.1 架构与调用关系

`cann_lm_converter.convert()` 是入口，`parse_args` 解析 CLI（argparse + 互斥组 + choices + required），`convert()` 装载 HF 模型（`loader.load_model` → `build_config` 从 `hf_config` + `ModelEntry` 构造 config，过滤未知字段 / 读 `config_json_key` 嵌套 / fallback 顶层 / `__post_init__`）、`collect_tokenizer_files` 收集 tokenizer 文件；`swap_from_dopt` 把 dopt 产物（`.pth`/`model.safetensors` + `dopt_config.json`）解析为 `state_dict`，`_read_dopt_quant_meta`/`_read_quant_param_map` 读取量化元信息，`_swap_linear_from_sd` 把 `nn.Linear` 替换为 `QuantLinear`（nibble-packed `weight_q` + `weight_scale`），`swap_embeddings_from_dopt` 把 `nn.Embedding` 物化为 int8+scale；`materialize_int4/int8` 做权重量化（`transdata_scale`↔`restore_scale` 存储格式 + `pack_int4` nibble 布局 + `clamp`）；`epair_utils._pack_weights_to_dir` 按 `max_file_size` 切分权重落盘（bf16→int16 view / weight_scale→int64 transdata），`consolidated_weights_save` 写 `weights_*.bin` + `weight_info.json` + ep_json，`_strip_none_input_descs`/`inline_fused_attn_seq_constants`/`inline_weight_scale_dtype` 原地改写 ep_json；`saver.save_model_config` 写 `model_config.json`，`copy_tokenizer_files` 复制 tokenizer 文件。

### 3.2 主要风险

| 风险域 | 代码风险 | UT 关注点 |
|---|---|---|
| int4 nibble 布局 | int8 有符号，负数 nibble 经符号扩展得错误字节；末维须偶数 | `pack_int4` 已知值 + round-trip + `& 0xFF` + 奇数末维拒绝 |
| int4 clamp 范围 | 对称量化排除 -8 边界，clamp 范围为 `(-7, 7)` 而非 `[-8, 7]` | clamp 超界值断言 `q.tolist()==[[7,-7,5,-5]]` |
| scale 存储格式 | `transdata_scale` 把 fp32 bit-pattern 装进 int64 低 32 位并 mask 掉高 32 位 + mantissa 低 13 位；`restore_scale` 逆变换 | round-trip 低 13 mantissa bit=0 + 高 32 bit=0 + 数值 allclose |
| fakequant 重建 | 随机权重 round 误差无法精确断言；dopt 存 `w=q*scale` float，`round(w/s)=q` 精确 | 构造 `w=q0*scale`，`torch.equal(q, q0)` 精确重建 |
| epair 打包切分 | `max_file_size` 切分边界；单 tensor 超限不跨文件切分；bf16→int16 / weight_scale→int64 字节数 | 单文件/多分片/单 tensor 超限/bf16/weight_scale 五路径 |
| ep_json 后处理 | `QuantBatchMatmulV4` 的 None `inputDesc` 须删；`inputDesc`/`input` 长度不一致时不改；非量化 op 不处理 | 删 None+空 ref / 保留非空 ref / 忽略非量化 op / 长度不符跳过 |
| inline 常量 | `FusedInferAttentionScore` 的 qlen/kvlen ref 空 → 跳过；kvlen 默认=qlen | 内联 qlen+kvlen / 空 kvlen 跳 / 无 qlen 整体跳 / 默认值 |
| dopt meta 解析 | `bit`/`unsigned_quant`/`offset` 键缺失须 fallback；`offset` 须 reshape 为 `[out, groups]` | fallback 默认值 / 读取+reshape |
| param_map 解析 | 未知 strategy / 非法 `;suffix` 跳过；layer 字段覆盖默认 | None / 无 config / 未知 strategy / 非法 suffix / 覆盖 |
| dopt 产物加载 | `.pth` / 目录查 `model.safetensors` / VLM `model.language_model.` 前缀剥离 | .pth / safetensors / 剥离前缀 / 缺文件报错 |
| swap Linear | `unsigned_quant!=0` / 非零 offset 拒绝；`embed_tokens` 名 skip；param_map group_size 须整除且 scale numel 匹配 | happy path / skip / param_map 覆盖 / 整除错误 / scale 不符 / 拒绝 unsigned / 拒绝 offset |
| QuantLinear 构造 | `weight_q` nibble-packed `[k//2, n]`（GE 解 nibble），CPU impl 期望 unpacked，forward 不在 UT 范围 | `__init__` buffer 注册 + act_quant 校验 + dtype 映射 |
| config 构造 | `config_json_key` 嵌套（gemma4/qwen3_5）/ fallback 顶层 / `__post_init__` | 字段过滤 / 默认值 / 嵌套 / fallback / post_init |
| tokenizer 收集 | 有 `tokenizer.json` → 标准三件套；无 → 追加 optional legacy | 有/缺/legacy/空目录 |
| config 序列化 | `to_dict` / dataclass `asdict` / `__dict__` 三路径 | 三路径 + use_embeddings/quant_info flag |
| CLI argparse | required / choices / 互斥组 | 缺 required / 非法 choices / 互斥 / Namespace+默认值 |
| weight_format | nd2nz 须 2D + 对齐（或 pad）；resolve_format 按规则 | round-trip / dtype / 非 2D 拒绝 / 不对齐拒绝 / 规则匹配 / env-gated 集成 |

## 4. 测试方法与判定准则

- 浮点结果使用 `torch.allclose`（`atol=1e-3, rtol=1e-2` 放宽 scale 存储格式的精度损失）；fakequant 精确重建用 `torch.equal`（`maxdiff=0`）。
- 随机用例固定数值范围（`torch.randint` / `torch.randn` 小尺寸），保证可重复。
- 每条测试独立使用 `tmp_path`、`monkeypatch`、局部假对象（`_FakeEP`/`_FakeContents`/`_FakeGraphSignature` stub 绕开真实 `ExportedProgram`），避免环境变量及全局状态污染。
- 异常分支同时检查异常类型（`ValueError`/`NotImplementedError`/`FileNotFoundError`/`SystemExit`）和关键消息（`match=`）。
- CLI 测试用临时替换 `sys.argv` 调 `parse_args`，argparse `--help`/缺 required/非法 choices/互斥组触发 `SystemExit`（code 0/2）。
- 所有 UT 必须可重复执行；测试目录整体退出码为 0 才算通过。
- 覆盖率目标：纯逻辑/数值模块（`materialize`/`swap`/`weight_format`/`quant_linear`/`loader`）行覆盖率不低于 80%；`epair_utils`/`saver`/`cann_lm_converter` 的 torchair/GE/真实权重执行体以分支覆盖为主，不以全行覆盖率作为准入条件。

## 5. 用例设计

### 5.1 materialize（test_materialize.py，15 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| MAT-001 | `transdata_scale` | fp32 scale → int64，dtype/shape 正确 | P0 |
| MAT-002 | `transdata_scale`↔`restore_scale` | round-trip 后低 13 mantissa bit=0 + 高 32 bit=0 + 数值 allclose | P0 |
| MAT-003 | `transdata_scale` | int64 高 32 位被 mask 清零，mask 外位全 0 | P1 |
| MAT-004 | `pack_int4` | [1,-1,7,-8] → byte0=0xF1, byte1=0x87（`& 0xFF` 取无符号字节） | P0 |
| MAT-005 | `pack_int4` | 随机 int4 pack/unpack round-trip 精确重建 | P0 |
| MAT-006 | `pack_int4` | 奇数末维抛 `ValueError` | P1 |
| MAT-007 | `materialize_int4_per_group_symmetric` | packed 开/关 q shape 分别为 `[out,in]`/`[out,in//2]` | P0 |
| MAT-008 | `materialize_int4_per_group_symmetric` | fakequant `w=q0*s`（q0∈[-7,7]）→ `torch.equal(q, q0)` 精确重建 | P0 |
| MAT-009 | `materialize_int4_per_group_symmetric` | w/s 超 7 → clamp(-7,7)，断言 `[[7,-7,5,-5]]` | P0 |
| MAT-010 | `materialize_int4_per_group_symmetric` | in 不整除 group / scale shape 不符 → `ValueError`（2 case） | P1 |
| MAT-011 | `materialize_int8_per_channel_symmetric` | q shape/dtype + 范围 [-128,127] | P0 |
| MAT-012 | `materialize_int8_per_channel_symmetric` | 随机 w dequant allclose | P0 |
| MAT-013 | `materialize_int8_per_channel_symmetric` | dopt fakequant `w=q0*scale` → `torch.equal(q, q0)` 精确重建 | P0 |
| MAT-014 | `materialize_int8_per_channel_symmetric` | scale shape 错（应 `[vocab]` 非 `[hidden]`）→ `ValueError` | P1 |

### 5.2 epair_utils（test_epair_utils.py，21 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| EPU-001 | `ConsolidatedWeights` | dataclass 字段 bin_files/weights_index 正确 | P1 |
| EPU-002 | `_pack_weights_to_dir` | 小权重不切分，单 `weights_0.bin` + index offset/size | P0 |
| EPU-003 | `_pack_weights_to_dir` | 小 max_file_size → 多分片，各文件大小正确 | P0 |
| EPU-004 | `_pack_weights_to_dir` | 单 tensor 超 max_file_size → 占独立文件不切分 | P0 |
| EPU-005 | `_pack_weights_to_dir` | bf16 → int16 view 落盘，字节数=elem*2 | P1 |
| EPU-006 | `_pack_weights_to_dir` | weight_scale fp32 → int64 transdata，字节数=elem*8 | P1 |
| EPU-007 | `consolidated_weights_save` | 落盘 weights_*.bin + weight_info.json + ep.json 三件套 | P0 |
| EPU-008 | `consolidated_weights_save` | `save_weights=False` → 仅 ep.json，跳权重 | P1 |
| EPU-009 | `consolidated_weights_save` | save 联动 `_strip_none_input_descs` 清理 ep_json | P1 |
| EPU-010 | `_strip_none_input_descs` | 删 None dtype + 空 ref input | P0 |
| EPU-011 | `_strip_none_input_descs` | dtype=None 但 ref 非空 → 保留 | P0 |
| EPU-012 | `_strip_none_input_descs` | 非 `QuantBatchMatmulV4` op（Cast）→ 跳过 | P1 |
| EPU-013 | `_strip_none_input_descs` | inputDesc/input 长度不符 → 不修改 | P1 |
| EPU-014 | `pack_weights`（遗留） | 写 cwd + 返回 None | P2 |
| EPU-015 | `pack_weights`（遗留） | 小 max_file_size → 切分 | P2 |
| EPU-016 | `inline_fused_attn_seq_constants` | 内联 qlen+kvlen，追加 2 个 Const op（DT_INT64） | P0 |
| EPU-017 | `inline_fused_attn_seq_constants` | 空 kvlen ref → 跳过 kvlen，只追加 qlen | P0 |
| EPU-018 | `inline_fused_attn_seq_constants` | 空 qlen ref → 整体跳过，无 Const 追加 | P1 |
| EPU-019 | `inline_fused_attn_seq_constants` | kvlen_value=None → 默认=qlen，两 Const 值相等 | P1 |
| EPU-020 | `inline_weight_scale_dtype` | weight_scale dtype 7(fp32)→5(int64) relabel | P0 |
| EPU-021 | `inline_weight_scale_dtype` | 无 tensor_values → 返回 0 | P2 |

### 5.3 loader（test_loader.py，12 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| LDR-001 | `ModelArtifact` | dataclass 字段集合完整 | P1 |
| LDR-002 | `ModelArtifact` | `quant_info` 默认 None | P1 |
| LDR-005 | `build_config` | gemma4 读嵌套 `text_config` | P0 |
| LDR-006 | `build_config` | gemma4 `__post_init__` 运行（layer_types 末层 full + rope） | P0 |
| LDR-007 | `build_config` | gemma4 `text_config` 缺失 → fallback 顶层 hf_config | P0 |
| LDR-008 | `build_config` | qwen3_5 读嵌套 `text_config` | P1 |
| LDR-009 | `collect_tokenizer_files` | 有 tokenizer.json → 只收标准三件套 | P0 |
| LDR-010 | `collect_tokenizer_files` | 缺部分标准文件 → 只收存在的 | P1 |
| LDR-011 | `collect_tokenizer_files` | 无 tokenizer.json → 追加 optional legacy（tokenizer.model/vocab.json） | P0 |
| LDR-012 | `collect_tokenizer_files` | 空目录 → 返回 `[]` | P2 |

### 5.4 swap（test_swap.py，34 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| SWP-001 | `_resolve_parent` | 深层树 `model.layers.0.self_attn.q_proj` → (parent, "q_proj") | P0 |
| SWP-002 | `_resolve_parent` | 单段路径 `proj` → (self, "proj") | P1 |
| SWP-003 | `_resolve_parent` | 索引路径 `model.layers.0` → parent=ModuleList, name="0" | P1 |
| SWP-004 | `_infer_group_size` | pergroup：numel=8,out=4 → (4, 2) | P0 |
| SWP-005 | `_infer_group_size` | perchannel：numel=out → (in, 1) | P0 |
| SWP-006 | `_infer_group_size` | pertensor → `ValueError` | P0 |
| SWP-007 | `_infer_group_size` | 非 1D scale → `ValueError` | P1 |
| SWP-008 | `_infer_group_size` | numel 不被 out 整除 → `ValueError` | P1 |
| SWP-009 | `_infer_group_size` | in 不被推断 group_size 整除 → `ValueError` | P1 |
| SWP-010 | `_infer_group_size` | 空 tensor（numel<out）→ `ValueError` | P1 |
| SWP-011 | `_read_dopt_quant_meta` | sd 无 bit/unsigned/offset 键 → fallback (4, 0.0, None) | P0 |
| SWP-012 | `_read_dopt_quant_meta` | 读 buffers + offset reshape 为 `[out, groups]` | P0 |
| SWP-013 | `_read_quant_param_map` | None 路径 → `{}` | P1 |
| SWP-014 | `_read_quant_param_map` | 目录有 safetensors 无 dopt_config.json → `{}` | P1 |
| SWP-015 | `_read_quant_param_map` | 未知 strategy → 跳过 | P0 |
| SWP-016 | `_read_quant_param_map` | 非法 `;suffix` → 跳过 | P0 |
| SWP-017 | `_read_quant_param_map` | layer 字段覆盖默认 bit/blocksize/group_size | P0 |
| SWP-018 | `load_dopt_state_dict` | .pth 加载 + 内容一致 | P0 |
| SWP-019 | `load_dopt_state_dict` | VLM `model.language_model.` 前缀剥离 | P0 |
| SWP-020 | `load_dopt_state_dict` | 目录 → 解析 `model.safetensors` | P0 |
| SWP-021 | `load_dopt_state_dict` | 目录无 safetensors → `FileNotFoundError` | P1 |
| SWP-022 | `load_dopt_state_dict` | 路径不存在 → `FileNotFoundError` | P1 |
| SWP-023 | `_swap_linear_from_sd` | happy path：nn.Linear→QuantLinear + scheme/group_size/buffer shape | P0 |
| SWP-024 | `_swap_linear_from_sd` | param_map group_size/bit/blocksize/per_channel/weight_algo 透传 | P0 |
| SWP-025 | `_swap_linear_from_sd` | param_map group_size 不整除 in → `ValueError` | P0 |
| SWP-026 | `_swap_linear_from_sd` | param_map group_size 与 scale numel 不符 → `ValueError` | P0 |
| SWP-027 | `_swap_linear_from_sd` | 无 dopt key 的 Linear → skip 不替换 | P0 |
| SWP-028 | `_swap_linear_from_sd` | `embed_tokens` 名匹配 DEFAULT_SKIP → 不替换 | P1 |
| SWP-029 | `_swap_linear_from_sd` | `unsigned_quant=1.0` → `ValueError` | P0 |
| SWP-030 | `_swap_linear_from_sd` | 非零 offset → `NotImplementedError` | P0 |
| SWP-031 | `swap_embeddings_from_dopt` | int8 materialize + scale shape `[vocab]` | P0 |
| SWP-032 | `swap_embeddings_from_dopt` | 无 scale key → 返回 `{}` | P1 |
| SWP-033 | `swap_embeddings_from_dopt` | 非 nn.Embedding → 跳过 | P1 |
| SWP-034 | `swap_from_dopt` | 端到端 stub：safetensors+dopt_config → Linear swap + Embedding 提取 | P0 |

### 5.5 saver（test_saver.py，7 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| SAV-001 | `save_model_config` | `to_dict` config 路径 + hf_config + flags + 无 quant_info | P0 |
| SAV-002 | `save_model_config` | dataclass `asdict` config 路径 | P0 |
| SAV-003 | `save_model_config` | `__dict__` fallback config 路径 | P1 |
| SAV-004 | `save_model_config` | `use_embeddings`/`quant_embeddings` flag 写入 | P1 |
| SAV-005 | `save_model_config` | `quant_info` 存在时写入 | P0 |
| SAV-006 | `copy_tokenizer_files` | 复制全部文件 + 内容一致 | P0 |
| SAV-007 | `copy_tokenizer_files` | 空列表不报错、不写文件 | P2 |

### 5.6 quant_linear（test_quant_linear.py，5 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| QLN-001 | `_OUT_DTYPE_CODE` | fp16→5 / bf16→7 / fp32→6 映射 | P1 |
| QLN-002 | `QuantLinear.__init__` | 无 bias：weight_q `[k//2,n]` int8 + weight_scale `[k//g,n]` fp32 + group_size/act_quant | P0 |
| QLN-003 | `QuantLinear.__init__` | 有 bias：bias buffer 注册 `[n]` | P0 |
| QLN-004 | `QuantLinear.__init__` | `act_quant=False` → `NotImplementedError`（W4A8 s8s4 only） | P0 |
| QLN-005 | `QuantLinear.__init__` | bf16 bias → 按 weight_scale.dtype(fp32) 注册 | P1 |

### 5.7 cli（test_cli.py，16 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| CLI-001 | `parse_args` | `--help` exit 0 + 含关键参数名 | P2 |
| CLI-002 | `parse_args` | 缺 `--hf-path` → exit 2 | P0 |
| CLI-003 | `parse_args` | 缺 `--output-dir` → exit 2 | P0 |
| CLI-004 | `parse_args` | 非法 `--dtype` → exit 2 | P0 |
| CLI-005 | `parse_args` | 非法 `--mode` → exit 2 | P0 |
| CLI-006 | `parse_args` | `--quant-path` ∩ `--quant-method` 互斥 → exit 2 | P0 |
| CLI-007 | `parse_args` | 非法 `--quant-method` → exit 2 | P1 |
| CLI-008 | `parse_args` | 合法最小参数 → Namespace + 默认值（dtype/mode/use_embeddings/kv_cache_max_len/seq_len） | P0 |
| CLI-009 | `parse_args` | 各合法 `--dtype` choice 解析正确（3 case） | P1 |
| CLI-010 | `parse_args` | `--no-use-embeddings` → use_embeddings=False | P2 |
| CLI-011 | `parse_args` | `--num-layers` 解析为 int | P1 |
| CLI-012 | `parse_args` | `--mode single` | P1 |
| CLI-013 | `parse_args` | `--quant-path` 单用 accepted | P1 |
| CLI-014 | `parse_args` | `--quant-method gptq` 单用 accepted | P1 |

### 5.8 weight_format（test_weight_format.py，13 项）

| ID | 被测单元 | 场景与预期 | 优先级 |
|---|---|---|---|
| WF-001 | `nd2nz` | 对齐 128×128 → `[8,8,16,16]` + round-trip 精确重建 | P0 |
| WF-002 | `nd2nz` | 不对齐 130×130 + pad → `[9,9,16,16]` + round-trip + pad 区零 | P0 |
| WF-003 | `nd2nz` | dtype 保持（fp16/int8/int32） | P1 |
| WF-004 | `nd2nz` | bf16 经 int16 view → int16 + shape 正确 | P1 |
| WF-005 | `nd2nz` | 非 2D → `ValueError` | P0 |
| WF-006 | `nd2nz` | 不对齐且 `pad=False` → `ValueError` | P0 |
| WF-007 | `load_rules` | shipped rules：default ND + MatMul 规则有 NZ | P0 |
| WF-008 | `resolve_format` | `*_proj.weight` 非量化 → NZ + block=16 | P0 |
| WF-009 | `resolve_format` | `*_proj.weight` 量化 → ND（保持） | P0 |
| WF-010 | `resolve_format` | 非 proj 名（norm.weight）→ ND | P1 |
| WF-011 | `_pack_weights_to_dir`（env off 集成） | env off → 全 ND | P0 |
| WF-012 | `_pack_weights_to_dir`（env off 集成） | env off → 字节与无变换一致 | P0 |
| WF-013 | `_pack_weights_to_dir`（env on 集成） | env on → proj.weight bf16→NZ，字节 ≠ ND，bin 以 NZ 开头 | P0 |

## 6. 计划目录

```text
tests/py/converter/
├── UT_DESIGN.md
├── conftest.py
├── test_materialize.py
├── test_epair_utils.py
├── test_loader.py
├── test_swap.py
├── test_saver.py
├── test_quant_linear.py
├── test_cli.py
└── test_weight_format.py
```

## 7. 执行方式

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/converter -q
```

单文件 / 单用例：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/py/converter/test_materialize.py -q
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python -m pytest tests/py/converter/test_materialize.py::test_pack_int4_roundtrip -q
```

覆盖率：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python \
  -m pytest tests/converter --cov=cann_lm/converter --cov-report=term-missing -q
```

## 8. 准入与遗留风险

准入条件：全部用例通过、无外网访问、converter venv（CPU + stubs）可重复、无测试间状态泄漏、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`。torchair GE 转换、真实 HF→epair→OM→NPU 端到端、dopt GPTQ NPU 量化和真实出字需要在后续集成测试中使用实际模型/权重/设备单独验证；UT 通过不等价于导出/推理正确。

遗留风险与已知限制：

1. **torchair/GE 依赖**：`saver.save_epair*`（`torch.export`+GE 转换）、`export/ge_converters/*`（FX→GE）、`cann_lm_converter.convert()` 全流程需 torchair stubs + 真实导出，归 `examples/test_converter/export_llm_model/run_e2e_pipeline.sh` e2e 验证，不进 UT（对应模块覆盖率 0%）。
2. **真实 HF 权重**：`loader.load_model` 全流程、`swap.swap_from_dopt` 全流程需 safetensors/完整 dopt artifacts，UT 只测其解析子函数（`build_config`/`collect_tokenizer_files`/`_read_dopt_*`/`load_dopt_state_dict`）。
3. **网络 / NPU subprocess**：`quant/download_dataset.py`（联网）、`quant/auto_quant.py`（spawn dopt GPTQ 需 NPU）不测（归 `examples/test_converter/dopt_quantization_st/`）。
4. **stub 绕开**：epair_utils 测试用 `_FakeEP`/`_FakeGraphSignature`/`_FakeContents` stub 绕开真实 `ExportedProgram`（符合 converter 设备无关约定，CPU 可跑），不验证真实 export 图结构。
5. **QuantLinear forward**：`QuantLinear` 存 nibble-packed `weight_q [k//2,n]`（GE export 路径，GE 内部解 nibble），CPU impl `_npu_quant_matmul_cpu` 期望 unpacked `[K,N]`，格式不匹配 → forward 走 `torch.export`→GE（e2e，`run_e2e_pipeline.sh` 验证），UT 仅测 `__init__` 可测面。
6. **int4 clamp 范围**：`materialize` 用 `clamp(-qmax, qmax)=clamp(-7,7)`（对称量化排除 -8 边界），测试据此断言（非 `[-8,7]`）。

当前没有 skip 或 xfail 用例；所有已收集用例均作为普通回归用例执行。

## 9. 执行结果（2026-09-05）

### 9.1 测试结果

```text
123 passed in 1.43s
pytest exit code: 0
```

- 收集用例总数：123
- 普通回归用例：123，全部通过
- skip/xfail：0

123 条用例按文件分布：`test_materialize.py` 15 / `test_epair_utils.py` 21 / `test_loader.py` 12 / `test_swap.py` 34 / `test_saver.py` 7 / `test_quant_linear.py` 5 / `test_cli.py` 16 / `test_weight_format.py` 13。运行期间无 warning 影响退出码（converter venv 不装 `torch_npu`，无设备自动探测提示）。

### 9.2 覆盖率结果

执行命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /mnt/workspace/torch-delegate-harmony/.venv/bin/python \
  -m pytest tests/converter --cov=cann_lm/converter --cov-report=term -q
```

整体行覆盖率为 **44%（1239 statements，707 missed）**。覆盖率含 `cann_lm/converter/` 全量模块（含按设计归 e2e 的 torchair/GE/网络/NPU 模块，均为 0%）；可测纯逻辑模块覆盖情况：

| 模块 | 行覆盖率 |
|---|---:|
| `export/quant_swap/materialize.py` | 96% |
| `export/quant_swap/quant_linear.py` | 86% |
| `export/weight_format/layout.py` | 100% |
| `export/weight_format/rules.py` | 100% |
| `swap.py` | 99% |
| `loader.py` | 82% |
| `export/epair_utils.py` | 75% |
| `saver.py` | 25% |
| `cann_lm_converter.py` | 28% |

e2e 排除模块（按设计归集成测试，0% 非 UT 缺陷）：`export/ge_converters/*`（7 文件，FX→GE 需 torchair）、`quant/auto_quant.py`（NPU subprocess）、`quant/dopt_wrapper.py`（由 `tests/py/dopt_lm/` 覆盖）、`quant/download_dataset.py`（联网）、`quant_entry.py`、`__main__.py`。

剩余未覆盖代码主要位于 `saver.py` 的 `save_epair`/`save_epair_prefill_decode`（需 `torch.export` wrapper/ep_module + torchair GE 转换）、`cann_lm_converter.py` 的 `convert()` 全流程（需真实 HF + torchair GE）、`epair_utils.py` 的真实 `ExportedProgram` 路径（UT 用 stub 绕开）。真实权重加载、GE 算子转换、端到端导出/出字需要实际模型语义与设备，留给集成测试，避免通过过度 mock 获得失真的覆盖率。

### 9.3 缺陷修复记录

| ID | 原现象 | 修复内容 | 回归结果 |
|---|---|---|---|
| DEF-001 | `test_pack_int4_layout_known_values` 直接比较 int8 字节值，负数 nibble（-1/-8）经符号扩展得到错误字节 | 用 `& 0xFF` 取无符号字节值再比较（byte0=`0xF1`、byte1=`0x87`），注释标注"int8 有符号，用 &0xFF 比无符号字节" | `test_pack_int4_layout_known_values` 通过，nibble 布局精确校验（低/高 nibble 各 4-bit 符号扩展） |
| DEF-002 | int4 对称量化 clamp 范围误用 `[-8, 7]`（含 -8 边界），clamp 用例数值不符 | 改为 `clamp(-7, 7)`（对称量化排除 -8 边界），`test_materialize_int4_clamp_beyond_range` 断言 `q.tolist()==[[7,-7,5,-5]]` | clamp 用例 + fakequant 精确重建用例（`MAT-008`/`MAT-009`）通过 |
| DEF-003 | int4/int8 fakequant 重建用例用随机权重，round 误差导致无法 `torch.equal` 精确断言 | 改用 `w=q0*scale` 构造（dopt 存储格式，weight 存的已是 `q*scale` float），`round(w/s)=q` 精确，`torch.equal(q, q0)` 断言 `maxdiff=0` | `test_materialize_int4_quant_dequant_exact_fakequant` + `test_materialize_int8_dopt_fakequant_exact_roundtrip` 通过 |
