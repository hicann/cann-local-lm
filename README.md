# CANN-LM

将 HuggingFace 模型导出为 epair 格式，供 C++ 后端加载编译 OM 并在 Ascend NPU 上推理。

## 支持模型

| 模型　　| 架构特点　　　　　　　　　　　　　　　　　　　　　　 |
| ---------| ------------------------------------------------------|
| gemma4　| KV 共享 + sliding window + PLE (Per-Layer Embedding) |
| qwen3　 | 标准 GQA　　　　　　　　　　　　　　　　　　　　　　 |
| qwen3.5 | 混合架构 (full attention + linear attention)　　　　 |

## 目录结构

```
CANN-LM_5883/
├── cann_lm/                  # 核心源码
│   ├── converter/            # 导出工具链 (loader → wrapper → torch.export → epair)
│   ├── engine/               # 推理引擎 (epair → OM 运行)
│   └── models/               # 模型定义 (gemma4/qwen3/qwen3_5 + fused_ops 算子)
├── docs/                     # 设计文档 (cli/export/quant 三模块)
├── examples/                 # 端到端用例 (OM 编译 + NPU 出字验证)
├── scripts/                  # 辅助脚本 (set_env.sh 统一环境)
├── tests/                    # 单元测试
├── ci/                       # CI/CD
└── cmake/                    # CMake 公共模块
```

### 各目录介绍

| 目录　　　　　　　　　　　　 | 职责　　　　　　　　　　　 | 关键文件　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ------------------------------| ----------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **cann_lm/converter/**　　　 | HF 模型 → epair 导出工具链 | `cann_lm_converter.py`(CLI 入口) / `loader.py`(加载) / `saver.py`(保存) / `quant/`(量化:swap/materialize/auto_quant/dopt_wrapper) / `export/`(ge_converter + C++ csrc) |
| **cann_lm/engine/**　　　　　| epair → OM 推理引擎　　　　| `api/`(Python 接口) / `core/`(核心逻辑) / `backend/`(C++ 后端) / `pipeline/`(推理流水线)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **cann_lm/models/**　　　　　| 三个模型定义 + 自定义算子　| `gemma4/`(Gemma4ForCausalLM + export_wrapper) / `qwen3/` / `qwen3_5/` / `fused_ops/`(cann_lm:: + npu:: 算子 schema+fake-meta)　　　　　　　　　　　　　　　　　　　　　|
| **docs/cann_lm_converter/**　| converter 设计文档　　　　 | `cli/cli_design.md` / `export/export_design.md` / `quant/quant_design.md` / `converter_design.md`(总览)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **examples/test_converter/** | 端到端测试用例　　　　　　 | `export_llm_model/`(C++ OM 编译 + om_npu_verify 出字验证) / `cann_lm_cli_export/run_pd6_cases.sh`(6 用例)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **scripts/**　　　　　　　　 | 辅助脚本　　　　　　　　　 | `set_env.sh`(统一环境:converter/npu/om) / `build_wheel.sh`(打包)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **tests/**　　　　　　　　　 | 单元测试　　　　　　　　　 | `build_and_test.sh` / `c++/`(C++ ASAN 测试)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **ci/**　　　　　　　　　　　| CI/CD　　　　　　　　　　　| 流水线定义、构建脚本、镜像配置　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **cmake/**　　　　　　　　　 | CMake 共享　　　　　　　　 | `DELEGATE_COMMON_FLAGS`(编译/链接选项统一管理)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

## 环境要求

三套独立环境，互不叠加：

```bash
source scripts/set_env.sh converter   # ① 导出 epair (CPU venv + torchair stubs)
source scripts/set_env.sh npu         # ② GPTQ 量化 / NPU 出字 (系统 Python + torch_npu)
source scripts/set_env.sh om          # ③ OM 编译 (C++ 二进制 + SOC + fusion switch)
```

## 快速开始

### 1. 导出 epair（无量化）

```bash
source scripts/set_env.sh converter
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3-0.6B \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 8
```

### 2. 量化导出（一键式 GPTQ）

```bash
source scripts/set_env.sh converter
DEVICE=npu python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 128 \
    --quant-method gptq
```

### 3. 消费 dopt 产物（离线量化）

```bash
source scripts/set_env.sh converter
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model \
    --output-dir /path/to/output \
    --quant-path /path/to/dopt_output
```

### 4. OM 编译 + NPU 出字

```bash
# OM 编译 (③ 环境)
source scripts/set_env.sh om
./examples/test_converter/export_llm_model/build/export_llm_model \
    output/prefill_ep_json_with_ascend_ir.json output/weight_info.json output/prefill

# NPU 出字 (② 环境)
source scripts/set_env.sh npu
python examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen_emb.py "Hello, my name is"
```

## 量化方案

**W4A8 s8s4**（权重 int4 per-group 对称 + 激活 int8 per-token 动态对称）

- 权重侧：dopt GPTQ 离线求 per-group scale，`materialize_int4_per_group_symmetric` 物化 int4
- 激活侧：运行时 `npu_dynamic_quant` → int8 per-token
- 算子：自研 `cann_lm::quant_matmul_s8s4`（fp32 scale 反量化，绕开 CANN V4 限制）

## 端到端测试

```bash
cd examples/test_converter/export_llm_model/cann_lm_cli_export
bash run_pd6_cases.sh 1   # qwen3.5 无量化
bash run_pd6_cases.sh 4   # gemma4 量化 (--quant-path)
bash run_pd6_cases.sh all # 全部 6 用例
```

## 文档

| 文档 | 内容 |
|------|------|
| [CLI 设计](docs/cann_lm_converter/cli/cli_design.md) | CLI 入口 / convert() / 命令行参数 / 调用链 |
| [导出设计](docs/cann_lm_converter/export/export_design.md) | loader / saver / ExportWrapper / C++ 加载 |
| [量化设计](docs/cann_lm_converter/quant/quant_design.md) | W4A8 方案 / dopt GPTQ / swap / 算子规格 |
| [converter 总览](docs/cann_lm_converter/converter_design.md) | 跨模块调用链 / 模型差异 / 精度汇总 |
