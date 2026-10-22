# 快速入门

## 版本配套

<!-- TODO: 版本配套信息待补充。本项目源码会跟随 CANN 软件版本发布，关于 CANN 软件版本与本项目标签的对应关系请参阅 release 仓库中的相应版本说明。当前配套版本（CANN toolkit 9.2.0 / torch 2.9.0+cpu / torch-delegate 0.1.0 / torchair 0.1 / torch_npu 2.7.1.post4）及标签信息（2026_09_22 / support_qwen3）待正式发布后补充。 -->

## 环境要求

### 前置条件（不本仓构建，见 [构建指南](build_guide.md)）
- **CANN toolkit**：source 其 `set_env.sh` 设 `ASCEND_HOME_PATH`（外部预置，不自动 source）
- **torchair + torch-delegate**：上游仓（源码构建或 whl 直装两种方式），产出 P1 venv（torch 2.9.0+cpu + torchair + torch_delegate）。torch_npu 仅量化需 NPU 加速时装（与 converter CPU-torch venv 冲突，须独立 python 环境）

### 运行 env（两套，复用/分离）
```bash
source scripts/set_env.sh              # converter(①) + compile(②) 统一 env (真 CANN, 无 stubs/无 0715, 幂等 LD, cann_lm 自定位 PYTHONPATH)
source scripts/set_env_cann.sh --npu   # ③ NPU verify-runtime (真 CANN + DEVICE=npu; CANN 轻量化工具 GPTQ / NPU 出字 / omc 验证)
# set_env_export.sh 为 back-compat 薄 wrapper (→ set_env.sh)
```

端到端构建用顶层 `bash scripts/build.sh <subcommand>`：`check`/`deps`(装 cann_lm) /`third_party`(自建三方库) /`cpp`(.so+cann_lm_compile) /`tests`(ASAN 测试) /`wheel`/`package`/`all`。omc 编译属测试/示例，不在 build.sh（直接跑 `examples/.../run_compile.sh`）。详见 `bash scripts/build.sh --help` 与 [构建指南](build_guide.md)。

## 1. 导出 epair（无量化）

```bash
source scripts/set_env.sh
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3.5-0.8B \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 8
```

## 2. 量化导出（一键式 GPTQ，需 NPU 加速时装 torch_npu）

```bash
source scripts/set_env.sh
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 128 \
    --quant-method gptq        # 触发 dopt subprocess (DOPT_PYTHON, ③ NPU env 需 torch_npu)
```

## 3. 消费 dopt 产物（离线量化）

```bash
source scripts/set_env.sh
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model \
    --output-dir /path/to/output \
    --quant-path /path/to/dopt_output
```

## 4. omc 编译 + NPU 出字

```bash
# omc 编译 (统一 env; 950DT CANN 机加 export SOC_VERSION=Ascend950DT_9575)
source scripts/set_env.sh
build/cann_lm_compile \
    output/prefill_ep_json_with_ascend_ir.json output/weight_info.json output/prefill

# NPU 出字 (③ env)
source scripts/set_env_cann.sh --npu
python examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen_emb.py "Hello, my name is"
```

> 完整端到端（① 导出 epair + ② omc 编译 + ③ NPU 出字）见 [构建指南「用例：HF → epair → omc」](build_guide.md#用例hf--epair--om-端到端gemma4)，或一键跑 `examples/test_converter/export_llm_model/run_e2e_pipeline.sh`。

## 量化方案

**W4A8**（权重 int4 per-group 对称 + 激活 int8 per-token 动态对称）

- 权重侧：轻量化工具 GPTQ 离线求 per-group scale，`materialize_int4_per_group_symmetric` 物化 int4
- 激活侧：运行时 `cann_lm::dynamic_quant`（镜像 `npu::npu_dynamic_quant`）→ int8 per-token
- 算子：`cann_lm::quant_matmul` → CANN 原生 `ge.QuantBatchMatmulV4`（路径 C'，dummy bias workaround 规避 V4→V3 融合降级）

## 端到端测试

```bash
cd examples/test_converter/export_llm_model/cann_lm_cli_export
bash run_pd6_cases.sh 1   # qwen3.5 无量化
bash run_pd6_cases.sh 4   # gemma4 量化 (--quant-path)
bash run_pd6_cases.sh all # 全部 6 用例
```
