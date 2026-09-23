# 从 Python 源码改到 OM 出字：自验证全流程

> 适用于 cann_lm 仓库 gemma4 / qwen3.5 模型。记录从改 Python 源码（fused_ops / modeling / ge_converter / reg_register.h）到最终 OM 在 NPU 上跑通出字的完整验证链路。

---

## 一、总览：7 个阶段

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  阶段 0  改 Python 源码 (fused_ops / modeling / ge_converter / reg_register.h)     │
│    ↓                                                                               │
│  阶段 1  schema + fake + CPU eager 注册验证 (import 探测)                           │
│    ↓                                                                               │
│  阶段 2  torch.export 导出 epair (ep_json_with_ascend_ir.json + weights_0.bin)      │
│    ↓                                                                               │
│  阶段 3  epair IR 算子清单统计 (确认新算子出现 / 旧算子消除)                         │
│    ↓                                                                               │
│  阶段 4  C++ OM 编译 (reg_register.h + aclgrphBuildModel)                           │
│    ↓                                                                               │
│  阶段 5  eager wrapper NPU 出字验证 (确认 fused 逻辑正确)                           │
│    ↓                                                                               │
│  阶段 6  OM 出字验证 (prefill + decode OM 在 NPU 跑通)                              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

每一步的 **PASS 条件** 和 **常见失败原因** 见下文。

---

## 二、环境准备

### 2.1 导出 / IR 验证环境（converter venv，CPU + torchair stubs）

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/mnt/workspace/torchair/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_DELEGATE_HOME=$TORCH_DELEGATE_HOME
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
VENV=$TORCH_DELEGATE_HOME/.venv
```

> 关键：stubs 必须用 torchair 编译生成的 `/mnt/workspace/torchair/build/stubs`（ABI 匹配）。
> 不能用真实 CANN lib64（会 `aclrtSetDevice → 107001` 无设备）。

### 2.2 OM 编译环境（真 CANN，NPU 在机）

```bash
source /home/developer/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}   # 不带 stubs
export SOC_VERSION=Ascend910_9362
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}
export GE_FUSION_SWITCH_FILE=/tmp/fusion_autoub_off.cfg   # 关 AUB, 让元素算子独立可读
```

> AUB cfg（一次性创建）：
> ```bash
> echo '{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"}}}' > /tmp/fusion_autoub_off.cfg
> ```

### 2.3 NPU eager + OM 出字环境（系统 Python + torch_npu）

```bash
source /home/developer/Ascend/cann/set_env.sh
export PYTHONPATH=/home/developer/Ascend/cann-9.2.0/python.site-packages:$PROJECT_ROOT:${PYTHONPATH:-}
# 系统 Python: $DOPT_PYTHON (含 torch_npu + acl + transformers)
```

---

## 三、各阶段详解

### 阶段 0：改 Python 源码

#### 涉及文件 & 职责

| 文件 | 职责 | 改什么 |
|---|---|---|
| `cann_lm/models/fused_ops/<op_name>.py` | 算子 schema + fake + CPU eager | 新建或修改：`NPU_LIB.define(...)` + `register_fake` + `register_cpu_impl` |
| `cann_lm/models/fused_ops/ops.py` | 聚合 import | 加 `from . import <op_name>` |
| `cann_lm/models/fused_ops/_library.py` | NPU_LIB / register_cpu_impl 工具 | 一般不动（已处理 stubs 环境） |
| `cann_lm/models/gemma4/modeling_gemma4.py` | gemma4 模型 forward | 改 attention / RoPE / KV cache 调用 |
| `cann_lm/models/qwen3_5/modeling_qwen3_5.py` | qwen3.5 模型 forward | 同上 |
| `cann_lm/converter/export/ge_converters/ge_converter.py` | torchair GE converter | 加 `@register_fx_node_ge_converter(torch.ops.npu.xxx.default)` + 调 `ge.Xxx(...)` |
| `cann_lm/converter/saver.py` | epair 落盘 + config | 改 `whitelist_decompositions` / `EpairConfig` |
| `examples/test_converter/export_llm_model/om_npu_verify/reg_register.h` | C++ REG_OP 注册 | 加 `REG_OP(Xxx).INPUT(...).ATTR(...).OP_END_FACTORY_REG(Xxx)` |

#### 关键规则

1. **schema 命名空间**：cann_lm 用 `npu` 命名空间（与图灵一致），不用 `air`（torchair 自有，stubs 环境会冲突）
2. **fake impl 必须**：`torch.export` trace 时调 fake 提供 shape 推断，没 fake → trace 失败
3. **CPU eager 可选**：eager 数值验证用，不进 OM 图；stubs 环境（`LD_PRELOAD` 含 `libacl_stub`）自动跳过（`register_cpu_impl` decorator）
4. **in-place 版本**：如果 torchair `_optimize_pass` 的 re-inplace 会访问 `npu_xxx_`（带下划线），必须同时注册 in-place schema + fake + CPU eager
5. **GE converter 参数全传**：`ge.FusedInferAttentionScore(...)` 等 GE 算子的 Python 包装要求所有参数（即使 None），否则 `inspect.signature.bind` 报 `missing a required argument`
6. **参数名映射**：Python schema 的 `num_query_heads` → GE IR 的 `num_heads`；`softmax_scale` → `scale`；`actual_seq_qlen` → `actual_seq_lengths` 等
7. **REG_OP 的 dtype**：用 `DT_BF16`（本机 CANN `graph/types.h`），不用 `DT_BFLOAT16`（ops-transformer 仓库的原文名，本机不认）

---

### 阶段 1：schema + fake + CPU eager 注册验证

```bash
$VENV/bin/python3 -c "
import torch
from cann_lm.models.fused_ops import ops
op = getattr(torch.ops.npu, 'npu_xxx', None)
print('schema:', op is not None)
if op:
    print('_ge_converter:', hasattr(op.default, '_ge_converter'))
    conv = getattr(op.default, '_ge_converter', None)
    if conv: print('converter:', conv.__name__, conv.__module__)
"
```

**PASS 条件**：
- `schema: True`（cann_lm 注册成功）
- `_ge_converter: True`（register_ge_converters 后，converter 注册成功）
- `converter` 的 `__module__` 是 `cann_lm.converter.export.ge_converter`（不是 `torchair...`，否则被 torchair 的 air converter 覆盖）

**常见失败**：
- `schema: False` → `NPU_LIB` 为 None（`_HAS_TORCH_NPU=True`，真实 torch_npu 环境，schema 由 torch_npu 提供）
- `_ge_converter: False` → 没调 `register_ge_converters()`，或 converter 注册失败
- `converter.__module__` 是 `torchair...` → torchair 的 air converter 覆盖了 cann_lm 的 npu converter（需在 `register_ge_converters()` 最后强制重新注册 cann_lm converter）

#### CPU eager 数值验证（可选但推荐）

```bash
$VENV/bin/python3 -c "
import torch
from cann_lm.models.fused_ops import ops
# 构造简单输入, 跑 CPU eager, 确认 shape + 数值合理
q = torch.randn(1, 4, 8, 64, dtype=torch.float32)
out = torch.ops.npu.npu_xxx(q, ...)
print('output shape:', out.shape)
"
```

---

### 阶段 2：torch.export 导出 epair

#### single 模式（快速验证，seq_len=8）

```bash
cd examples/test_converter/export_llm_model/cann_lm_cli_export
$VENV/bin/python3 $PROJECT_ROOT/cann_lm/converter/cann_lm_converter.py \
    --hf-path /mnt/workspace/data/models/gemma-4-E2B-it \
    --output-dir epair_gemma4_test \
    --dtype bfloat16 --seq-len 8 --num-layers 5
```

> `--num-layers 5`：截断到 5 层加速（只验证 IR，不出字）。

#### prefill-decode 模式（完整 35 层，出字验证）

```bash
cd examples/test_converter/export_llm_model/om_npu_verify
$VENV/bin/python3 $PROJECT_ROOT/cann_lm/converter/cann_lm_converter.py \
    --hf-path /mnt/workspace/data/models/gemma-4-E2B-it \
    --output-dir epair_gemma4_pd \
    --dtype bfloat16 --mode prefill-decode --seq-len 128 --use-embeddings
```

> `--use-embeddings`：gemma4 必须（embed_tokens + PLE 外置）；qwen3.5 不需要。

**PASS 条件**：
- `Done. Output: epair_xxx`
- 产物含 `ep_json_with_ascend_ir.json`（IR 图）+ `weights_0.bin`（权重）+ `weight_info.json`（索引）

**常见失败**：
- `NotImplementedError: ge_converter is not implemented` → 某个 `aten.xxx` 没 converter，需在 `ge_converter.py` 加或加到 `whitelist_decompositions`
- `TypeError: missing a required argument` → GE converter 参数没全传（GE 算子包装要求所有参数）
- `AttributeError: 'npu' object has no attribute 'npu_xxx_'` → in-place 版本没注册，补 in-place schema
- `Exception: _optimize_pass failed` → torchair re-inplace pass 访问 in-place op，补 in-place schema

---

### 阶段 3：epair IR 算子清单统计

```bash
python3 -c "
import json
from collections import Counter
with open('epair_xxx/ep_json_with_ascend_ir.json') as f: d=json.load(f)
def find_ir(obj):
    r=[]
    if isinstance(obj,dict):
        if 'ascend_ir' in obj: r.append(obj['ascend_ir'])
        for v in obj.values(): r.extend(find_ir(v))
    elif isinstance(obj,list):
        for v in obj: r.extend(find_ir(v))
    return r
ops=Counter()
for ir in find_ir(d):
    if isinstance(ir,dict) and 'ops' in ir:
        for op in ir['ops']:
            if isinstance(op,dict) and 'type' in op: ops[op['type']]+=1
for t,c in ops.most_common():
    print(f'  {t:30s} {c}')
"
```

**PASS 条件**：
- 新算子出现（如 `FusedInferAttentionScore: 5`）
- 旧算子消除（如 `BatchMatMul: 0`、`SoftmaxV2: 0`）

**常见失败**：
- 新算子没出现 → converter 没注册或没调 `register_ge_converters()`
- 旧算子还在 → 没替换成功（forward 没改调用）

---

### 阶段 4：C++ OM 编译

#### 4.1 构建 C++ 编译器（一次性，改 reg_register.h 后重编）

```bash
cd examples/test_converter/export_llm_model/om_npu_verify
export TORCH_PKG_DIR=$VENV/lib/python3.11/site-packages/torch
cmake -S .. -B ../build -DCANN_ROOT="$ASCEND_HOME_PATH" -DTORCH_PKG_DIR="$TORCH_PKG_DIR"
cmake --build ../build -j4
```

> 改 `reg_register.h` 后只需 `cmake --build ../build -j4` 重编（main.cpp+CMakeLists 已上移到上级 `export_llm_model/`）。

#### 4.2 编译 OM

```bash
# single OM (NUM_LAYERS=5 快速验证)
../build/cann_lm_compile \
    ../cann_lm_cli_export/epair_gemma4_test/ep_json_with_ascend_ir.json \
    ../cann_lm_cli_export/epair_gemma4_test/weight_info.json \
    gemma4_test

# prefill + decode OM (完整 35 层, 出字用)
../build/cann_lm_compile epair_gemma4_pd/prefill_ep_json_with_ascend_ir.json epair_gemma4_pd/weight_info.json /tmp/opencode/gemma4_pd/prefill
../build/cann_lm_compile epair_gemma4_pd/decode_ep_json_with_ascend_ir.json  epair_gemma4_pd/weight_info.json /tmp/opencode/gemma4_pd/decode
ln -sf prefill_linux_aarch64.om prefill.om
ln -sf decode_linux_aarch64.om decode.om
```

**PASS 条件**：
- `[PASS] GE Graph extracted successfully`（算子被 GE OperatorFactory 识别）
- `[PASS] aclgrphBuildModel succeeded`（编译成功）
- `[PASS] aclgrphSaveModel succeeded`（OM 保存成功）
- `.om` 文件生成（几 GB）

**常见失败**：
| 错误 | 原因 | 修复 |
|---|---|---|
| `Op type Xxx is not registered in OperatorFactory` | `reg_register.h` 没 `REG_OP(Xxx)` | 加 `REG_OP` |
| `data type DT_INT32 of input [actual_seq_lengths] is not supported` | 参数 dtype 不对 | 改 int64 |
| `atten_mask has incorrect dtype DT_FLOAT16` | mask dtype 不对 | 转 bool |
| `atten_mask has incorrect shape [1,1,8,2048]` | mask shape 不对（要 `[kv_len,kv_len]` 方阵） | pad q 维到 kv_len |
| `GetGEGraph failed` | stubs 的 libge_runner 不支持（用了 stubs 环境） | 用真 CANN（去掉 stubs LD_LIBRARY_PATH） |
| `aclgrphBuildModel failed, status: 1343225857` | 参数无效（mask shape / dtype / layout / kernel 不支持） | 开 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 看具体 reason |
| `aclgrphSaveModel failed` | GE finalize 堆损坏（OM 可能已保存） | 检查 `.om` 文件是否生成 |

#### 4.3 OM 算子清单（atc dump）

```bash
atc --mode=1 --om=gemma4_test.om --json=/tmp/om.json
python3 -c "
import json
from collections import Counter
with open('/tmp/om.json') as f: d=json.load(f)
g = d.get('graph', [])
ops_list = g[0].get('op', []) if g and isinstance(g[0], dict) else []
types = Counter(op.get('type','?') for op in ops_list if isinstance(op, dict))
for t,c in types.most_common():
    print(f'  {t:30s} {c}')
"
```

> 注意：关 AUB 后 OM 可能把算子封装到 `PartitionedCall` 子图，`atc --mode=1` 不展开。用阶段 3 的 epair IR 看完整算子清单。

---

### 阶段 5：eager wrapper NPU 数值验证

> 确认 fused 算子逻辑正确（NPU eager），不经过 OM 编译。用 wrapper（scatter-write buffer）跑 eager 单次 forward，与 OM logits 对比（原 eager 出字脚本 `run_gemma4_eager_gen.py` 已移除，eager 验证改为单 forward 数值对比）。

```bash
GEMMA4_OM=/path/to/gemma4.om \
$DOPT_PYTHON examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_om.py
```

> `run_gemma4_om.py`：用 `wrapper.get_sample_inputs` + `_build_mask` 构造输入，`wrapper(*sample_npu)` 跑 eager forward（NPU），再用同一份 CPU sample 跑 OM（acl），对比 OM logits vs eager logits。

**PASS 条件**：
- `OM vs eager logits maxdiff < 1.0` → `MATCH (OM 运行机制正确)`
- eager top5 token 合理（非全 `<pad>`）

**常见失败**：
| 输出 | 原因 | 修复 |
|---|---|---|
| 全 `<pad>`（tok=0） | attention 输出全 0（mask 语义反 / actual_seq_kvlen 错 / 输出 layout 错） | 逐步排查（见下方调试技巧） |
| 重复 `<turn|>` + `<eos>` | attention 数值部分错（mask shape / sparse_mode / 输出 transpose） | 同上 |
| `RuntimeError: execute ret=500002` | OM execute 失败（mask shape / 参数组合不支持） | 看具体错误 |

#### 调试技巧：对比 fused vs eager attention 输出

在 `Attention.forward` 里加 debug（`FUSED_DEBUG=1` 时打印 layer0 的 fused vs eager diff）：

```python
import os as _os
if _os.environ.get("FUSED_DEBUG") and self.layer_idx == 0:
    with torch.no_grad():
        _attn_eager, _ = eager_attention_forward(
            self, query_states, key_states, value_states,
            attention_mask, scaling=self.scaling,
        )
        _diff = (attn_output.float() - _attn_eager.float()).abs().max().item()
        print(f"[debug] layer0 fused vs eager diff: {_diff:.4e}")
```

> 如果 diff 大 → fused 逻辑错；如果 diff 小但出字错 → 问题在别处（mask / KV cache / layout）。

---

### 阶段 6：OM 出字验证

```bash
$DOPT_PYTHON examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_gen.py "Hello, my name is"
```

> `run_gemma4_gen.py`：用 `acl.mdl.execute` 跑 prefill OM + decode OM，prefill + decode loop 出字。

**PASS 条件**：
- 输出与阶段 5（eager wrapper）一致
- 不重复 `<turn|>` / `<pad>`

**常见失败**：
- OM 出字 ≠ eager 出字 → OM 编译优化改变了算子语义（fusion pass），或 OM 权重不对
- `RuntimeError: execute ret=500002` → OM execute 失败，看具体错误

---

## 四、验证检查清单（Checklist）

| # | 验证项 | 命令 / 方法 | PASS 条件 |
|---|---|---|---|
| 1 | schema 注册 | `import ops; getattr(torch.ops.npu, 'xxx', None) is not None` | `True` |
| 2 | converter 注册 | `hasattr(op.default, '_ge_converter')` | `True`，`__module__` 是 `cann_lm...` |
| 3 | CPU eager | `torch.ops.npu.xxx(简单输入)` | output shape 正确 |
| 4 | epair 导出 | `cann_lm_converter.py --num-layers 5` | `Done. Output: epair_xxx` |
| 5 | IR 含新算子 | python3 统计 ascend_ir ops | 新算子 >0，旧算子 =0 |
| 6 | OM 编译 | `../build/cann_lm_compile ...` | `BuildModel succeeded` + `.om` 生成 |
| 7 | eager 数值对比 | `run_gemma4_om.py` | `OM vs eager maxdiff < 1.0 MATCH` |
| 8 | OM 出字 | `run_gemma4_gen.py "prompt"` | 输出合理文本 |

> 阶段 1-3 用 converter venv（CPU + stubs）；阶段 4 用真 CANN（NPU）；阶段 5-6 用系统 Python（torch_npu + acl）。

---

## 五、常见问题排查

### 5.1 torchair 的 air converter 覆盖 cann_lm 的 npu converter

**现象**：`npu._ge_converter.__module__` 是 `torchair...`，导出时报 `import torch_npu` 失败。

**原因**：torchair `custom/` import 时，如果 npu schema 存在（cann_lm 注册），torchair 的 air converter 可能被错误注册到 npu target。

**修复**：在 `register_ge_converters()` 最后，强制重新注册 cann_lm 的 npu converter：

```python
register_fx_node_ge_converter(torch.ops.npu.npu_fused_infer_attention_score_v2.default)(_convert_fused_infer_attention_score_v2)
```

### 5.2 aten.xxx 没 converter（NotImplementedError）

**现象**：`NotImplementedError: torch.ops.aten.xxx.default ge_converter is not implemented!`

**修复**：
1. 在 `ge_converter.py` 加 `@register_fx_node_ge_converter(torch.ops.aten.xxx.default)` converter
2. 或在 `saver.py` 的 `whitelist_decompositions` 加该 op（让它被 decomposition 分解，但前提是该 op 在 `torch.default_decompositions()` 表里）

### 5.3 GE 算子参数缺失（missing a required argument）

**现象**：`TypeError: missing a required argument: 'dequant_scale1'`

**原因**：GE 算子的 Python 包装（`auto_generated_ge_raw_ops.py`）要求所有参数，即使 None。

**修复**：converter 里传全参数（None 给不用），参考 torchair v2 converter 的完整调用。

### 5.4 mask dtype / shape 不匹配

**现象**：
- `atten_mask has incorrect dtype DT_FLOAT16. It should be int8, uint8 or bool`
- `atten_mask has incorrect shape [1,1,8,2048]. must be [2048,2048]...`

**修复**：
- dtype：float mask → `.bool()`
- shape：`[*, q_len, kv_len]` → pad q 维到 `kv_len`（`_pad_mask_to_kv_square`），融合算子用 `actual_seq_qlen` 切片前 q_len 行

### 5.5 actual_seq_kvlen 用 buffer 大小（attend 到垃圾）

**现象**：出字全 `<pad>`（attention attend 到 KV buffer 未初始化部分）

**原因**：`actual_seq_kvlen = key.shape[2]`（buffer 大小 2048），但实际有效 kv 只有 seq_len（8/128）

**修复**：`actual_seq_kvlen = new_kv_cache_pos.max() + 1`（实际有效 kv），需注册 `ReduceMax` 的 `REG_OP`

### 5.6 fused 输出 layout 不匹配 reshape（n/s 维错位）

**现象**：出字不正确（重复 `<turn|>` / `<pad>`），但 eager wrapper 和 OM 一致

**原因**：融合算子输出 BNSD `[b,n,s,d]`，但 `Attention.forward` 的 `reshape(*input_shape, -1)` 假设 BSND `[b,s,n,d]`（eager 输出 transpose 后是 BSND），n/s 维错位

**修复**：fused 输出加 `.transpose(1, 2).contiguous()`（BNSD → BSND）

### 5.7 npu_scatter_nd_update 的 indices scatter dim 0（不是 dim 2）

**现象**：KV cache scatter-write 数值错（出字不正确）

**原因**：`npu_scatter_nd_update(cache, indices=[seq,1], updates)` 的 indices scatter dim 0，cann_lm 的 cache `[b,n,kv_max,d]`（BNSD）要 scatter dim 2，需 transpose（开销大于 index_copy）

**修复**：回退到 `index_copy`（InplaceUpdate + Transpose），或改 cache layout 成 BKND（跟图灵一致）

---

## 六、涉及的文件 & 工具清单

| 类别 | 文件 / 工具 | 作用 |
|---|---|---|
| 算子定义 | `cann_lm/models/fused_ops/*.py` | schema + fake + CPU eager |
| 算子聚合 | `cann_lm/models/fused_ops/ops.py` | import 触发注册 |
| stubs 工具 | `cann_lm/models/fused_ops/_library.py` | NPU_LIB / register_cpu_impl（stubs 环境跳过 CPU impl） |
| 模型 forward | `cann_lm/models/gemma4/modeling_gemma4.py` | attention / RoPE / KV cache 调用 |
| 模型 forward | `cann_lm/models/qwen3_5/modeling_qwen3_5.py` | 同上 |
| GE converter | `cann_lm/converter/export/ge_converters/ge_converter.py` | `@register_fx_node_ge_converter` + `ge.Xxx(...)` |
| epair 落盘 | `cann_lm/converter/saver.py` | `whitelist_decompositions` / `EpairConfig` |
| epair 序列化 | `cann_lm/converter/export/epair_utils.py` | `consolidated_weights_save` |
| 导出 CLI | `cann_lm/converter/cann_lm_converter.py` | `--hf-path --mode --use-embeddings --num-layers` |
| C++ REG_OP | `examples/test_converter/export_llm_model/om_npu_verify/reg_register.h` | 算子注册到 GE OperatorFactory |
| C++ OM 编译器 | `examples/test_converter/export_llm_model/main.cpp` + `CMakeLists.txt` | `EpairModelLoader` + `aclgrphBuildModel` |
| eager 数值对比 | `examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_om.py` | eager NPU 单次 forward（wrapper）作 OM logits 参考基准（maxdiff）；原 `run_gemma4_eager_gen.py`（eager 出字）已移除，未合并 |
| OM 出字 | `examples/test_converter/export_llm_model/om_npu_verify/run_gemma4_gen.py` | `acl.mdl.execute` prefill+decode OM 出字 |
| OM dump | `atc --mode=1 --om=xxx.om --json=xxx.json` | OM 算子清单 |
| IR 统计 | python3 解析 `ep_json_with_ascend_ir.json` | epair IR 算子清单 |
| 设计文档 | `docs/cann_lm_models/fused_ops/fused_ops.md` | 融合算子落地状态 + 遗留问题（原 compositte_op，现合并入 fused_ops.md §4/§5） |

---

## 七、已验证的改动案例（参考）

### 7.1 RoPE 切换：npu_rotary_mul → npu_apply_rotary_pos_emb（full 层配对）

- 新建 `fused_ops/apply_rotary_pos_emb.py`（schema + fake + CPU eager）
- `ge_converter.py` 无需加（torchair 自带 `custom/apply_rotary_pos_emb.py` converter，但 schema 不存在时 warning 跳过，需 `importlib.reload` 触发）
- `reg_register.h` 加 `REG_OP(ApplyRotaryPosEmb)`（来源 ops-transformer 仓库 proto.h，`DT_BFLOAT16` → `DT_BF16`）
- 出字验证：`"Hello! It's nice to meet you. What is your name? 😊"` ✅

### 7.2 Attention 替换：eager → npu_fused_infer_attention_score_v2

- 新建 `fused_ops/fused_infer_attention_score.py`（schema + fake + CPU eager，简化版 16 参数）
- `ge_converter.py` 加 `_convert_fused_infer_attention_score_v2`（调 `ge.FusedInferAttentionScore`，40+ 参数全传）
- `reg_register.h` 加 `REG_OP(FusedInferAttentionScore)` + `REG_OP(ReduceMax)`（`new_kv_cache_pos.max()` 算 actual_kv_len）
- mask 适配：float → bool，`[q_len, kv_len]` → pad 到 `[kv_len, kv_len]` 方阵
- actual_seq_kvlen：`new_kv_cache_pos.max() + 1`（不是 `key.shape[2]` buffer 大小）
- 输出 transpose：`BNSD → BSND`（`.transpose(1,2).contiguous()`）
- 出字验证：`"Hello! It's nice to meet you. What is your name? 😊"` ✅

### 7.3 KV cache scatter-write 替换：index_copy → npu_scatter_nd_update（失败回退）

- 新建 `fused_ops/scatter_nd_update.py`（schema + fake + CPU eager + in-place）
- `reg_register.h` 加 `REG_OP(ScatterNdUpdate)`
- gemma4 IR 替换成功（`ScatterNdUpdate:10`），但出字失败（cache BNSD 要额外 transpose）
- **回退** gemma4 + qwen3.5 到 `index_copy`（BNSD layout 不适合 npu_scatter_nd_update）
- 保留 schema + converter + REG_OP（供后续 PagedAttention 或 cache 改 BKND 用）
