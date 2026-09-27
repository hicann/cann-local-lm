# op_npu_verify — 算子级 NPU 验证 (④ 算子路径)

把 dopt 单个算子 (weight+scale) → materialize (按 target_soc) → 1-op QuantLinear 模型 → export epair → OM → NPU 跑 → 余弦对比 (Tier 0 W4A8 + dopt fakequant W4A16 golden)。

是 VERIFICATION.md 的 ④ 算子级落地：NPU `QuantBatchMatmulV4` 真内核输出 vs reference，余弦判据（同语义 cos ≥0.999 / 跨语义 cos ≥0.99）。

## 流程 (run.sh 编排骨 3 env)

```
[1/4] export epair   (converter env ①, source set_env_export.sh)  — build_epair.py
[2/4] compile OM     (CANN env ②, source set_env_cann.sh)         — compile_om.sh (cann_lm_compile)
[3/4] run NPU        (env ③, system python + torch_npu)           — run_npu.py (acl, desc 路径)
[4/4] compare (CPU)  余弦                                          — verify_op.py
```

## 用法

```bash
export DOPT_QUANT_DIR=<dopt 量化输出目录>   # 含 trained_quant_weight.pth
# 先构建 cann_lm_compile (一次性): bash scripts/build.sh cpp
bash examples/test_converter/dopt_op_benchmark/op_npu_verify/run.sh model.layers.0.self_attn.q_proj Ascend910B
```

- `Ascend910B`：int8 nibble 路径（本机 Ascend910 可跑）。
- `KirinX90`：int4pack int32 路径（需 kirinx90 硬件/模拟器，本机跑不了 OM）。

## 文件

| 文件 | env | 作用 |
|---|---|---|
| `build_epair.py` | ① converter | 从 dopt 算子 weight+scale materialize (按 SoC) → 1-op QuantLinear → torch.export → epair |
| `compile_om.sh` | ② CANN | `cann_lm_compile` epair → `.om` (QuantBatchMatmulV4) |
| `run_npu.py` | ③ NPU | acl 加载 OM (desc 路径) → 跑 golden input_act → 存 npu_output.pt |
| `verify_op.py` | CPU | 三方余弦对比：Tier0(W4A8) vs dopt golden(W4A16) [sanity] + NPU vs Tier0 [④ 同语义 ≥0.999] + NPU vs dopt [④ 跨语义 ≥0.99] |
| `run.sh` | 编排 | 3 env 顺序 + compare |

## 状态

- ✅ `verify_op.py` (CPU 对比 harness) 已跑通：Tier0(W4A8) vs dopt golden(W4A16) cos=0.9996 (跨语义 sanity OK)。
- ⏳ `build_epair.py`/`compile_om.sh`/`run_npu.py`/`run.sh` 已搭好结构，**待实跑**：需先 `bash scripts/build.sh cpp` 建 cann_lm_compile + 3 env + 真实 OM（`run_npu.py` 的 acl buffer 读取是草稿，需真实 OM 验证）。
- ⏳ KirinX90 int4pack 路径：本机无 kirinx90 hw，OM 跑不了；Ascend910 (int8 nibble) 可跑通方法。

判据 + 方法论见 `../VERIFICATION.md`（④ + 信任分级）。
