# CANN-LM

将 HuggingFace 模型导出为 epair 格式，供 C++ 后端加载编译 OM 并在 Ascend NPU 上推理。

## 支持模型

| 模型　　| 架构特点　　　　　　　　　　　　　　　　　　　　　　 |
| ---------| ------------------------------------------------------|
| gemma4　| KV 共享 + sliding window + PLE (Per-Layer Embedding) |
| qwen3.5 | 混合架构 (full attention + linear attention)　　　　 |

## 目录结构

```
CANN-LM_5883/
├── cann_lm/                  # 核心源码
│   ├── converter/            # 导出工具链 (loader → wrapper → torch.export → epair)
│   ├── engine/               # 推理引擎 (epair → OM 运行)
│   └── models/               # 模型定义 (gemma4/qwen3_5 + fused_ops 算子)
├── docs/                     # 设计文档 (cli/export/quant 三模块)
├── examples/                 # 端到端用例 (OM 编译 + NPU 出字验证)
├── scripts/                  # 构建脚本 (build.sh + build.conf + set_env*.sh + cann_lm_compile/)
├── tests/                    # 单元测试 (c++/ + py/)
├── ci/                       # CI/CD
└── cmake/                    # CMake 公共模块 (modules/ + third_party/)
```

### 各目录介绍

| 目录　　　　　　　　　　　　 | 职责　　　　　　　　　　　 | 关键文件　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ------------------------------| ----------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **cann_lm/converter/**　　　 | HF 模型 → epair 导出工具链 | `cann_lm_converter.py`(CLI 入口) / `loader.py`(加载) / `saver.py`(保存) / `quant/`(量化:swap/materialize/auto_quant/dopt_wrapper) / `export/`(ge_converter + C++ csrc) |
| **cann_lm/engine/**　　　　　| epair → OM 推理引擎　　　　| `api/`(Python 接口) / `core/`(核心逻辑) / `backend/`(C++ 后端) / `pipeline/`(推理流水线)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **cann_lm/models/**　　　　　| 两个模型定义 + 自定义算子　| `gemma4/`(Gemma4ForCausalLM + export_wrapper) / `qwen3_5/` / `fused_ops/`(cann_lm:: + npu:: 算子 schema+fake-meta)　　　　　　　　　　　　　　　　　　　　　|
| **docs/cann_lm_converter/**　| converter 设计文档　　　　 | `cli/cli_design.md` / `export/export_design.md` / `quant/quant_design.md` / `converter_design.md`(总览)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **examples/test_converter/** | 端到端测试用例　　　　　　 | `export_llm_model/`(C++ OM 编译 + om_npu_verify 出字验证) / `cann_lm_cli_export/run_pd6_cases.sh`(6 用例)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| **scripts/**　　　　　　　　 | 构建脚本　　　　　　　　　 | `build.sh`(顶层 check/deps/third_party/cpp/tests/wheel/package/all) / `build.conf`(版本+git源+自定位派生路径) / `set_env.sh`(converter+compile 统一 env, 真 CANN 无 stubs) / `set_env_cann.sh`(③ NPU) / `_env_common.sh`(公共) / `cann_lm_compile/build_cpp.sh`(C++ .so+cann_lm_compile) / `build_test.sh`(ASAN 测试) / `build_third_party.sh`(自建 json/gtest/makeself) / `cann_lm_{converter,engine}/build_wheel.sh`(whl 打包)　 |
| **tests/**　　　　　　　　　 | 单元测试　　　　　　　　　 | `c++/`(C++ 测试; 跑: `bash scripts/build.sh tests` → ASAN 构建+ctest, 独立 build_tests/) / `py/`(pytest)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **ci/**　　　　　　　　　　　| CI/CD　　　　　　　　　　　| 流水线定义、构建脚本、镜像配置　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| **cmake/**　　　　　　　　　 | CMake 共享　　　　　　　　 | `modules/`(DelegateCommonFlags + Find*.cmake + TorchDelegateCannCommon) / `third_party/`(ExternalProject 下载 json/gtest/makeself)　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |

## 环境要求

### 前置条件（不本仓构建，见 [构建指南](docs/build_guide.md)）
- **CANN toolkit**：source 其 `set_env.sh` 设 `ASCEND_HOME_PATH`（外部预置，不自动 source）
- **torchair + torch-delegate**：上游仓（源码构建或 whl 直装两种方式），产出 P1 venv（torch 2.9.0+cpu + torchair + torch_delegate）。torch_npu 仅量化需 NPU 加速时装（与 converter CPU-torch venv 冲突，须独立 python 环境）

### 运行 env（两套，复用/分离）
```bash
source scripts/set_env.sh              # converter(①) + compile(②) 统一 env (真 CANN, 无 stubs/无 0715, 幂等 LD, cann_lm 自定位 PYTHONPATH)
source scripts/set_env_cann.sh --npu   # ③ NPU verify-runtime (真 CANN + DEVICE=npu; dopt GPTQ / NPU 出字 / OM 验证)
# set_env_export.sh 为 back-compat 薄 wrapper (→ set_env.sh)
```

端到端构建用顶层 `bash scripts/build.sh <subcommand>`：`check`/`deps`(装 cann_lm) /`third_party`(自建三方库) /`cpp`(.so+cann_lm_compile) /`tests`(ASAN 测试) /`wheel`/`package`/`all`。OM 编译属测试/示例，不在 build.sh（直接跑 `examples/.../run_compile.sh`）。详见 `bash scripts/build.sh --help` 与 [构建指南](docs/build_guide.md)。

## 快速开始

前置：source CANN set_env.sh + 上游就绪（见 [构建指南](docs/build_guide.md)），然后 `source scripts/set_env.sh`。

### 1. 导出 epair（无量化）

```bash
source scripts/set_env.sh
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/qwen3.5-0.8B \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 8
```

### 2. 量化导出（一键式 GPTQ，需 NPU 加速时装 torch_npu）

```bash
source scripts/set_env.sh
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model \
    --output-dir /path/to/output \
    --dtype float16 --seq-len 128 \
    --quant-method gptq        # 触发 dopt subprocess (DOPT_PYTHON, ③ NPU env 需 torch_npu)
```

### 3. 消费 dopt 产物（离线量化）

```bash
source scripts/set_env.sh
python cann_lm/converter/cann_lm_converter.py \
    --hf-path /path/to/model \
    --output-dir /path/to/output \
    --quant-path /path/to/dopt_output
```

### 4. OM 编译 + NPU 出字

```bash
# OM 编译 (统一 env; 950DT CANN 机加 export SOC_VERSION=Ascend950DT_9575)
source scripts/set_env.sh
build/cann_lm_compile \
    output/prefill_ep_json_with_ascend_ir.json output/weight_info.json output/prefill

# NPU 出字 (③ env)
source scripts/set_env_cann.sh --npu
python examples/test_converter/export_llm_model/om_npu_verify/run_qwen3_5_gen_emb.py "Hello, my name is"
```

> 完整端到端（① 导出 epair + ② OM 编译 + ③ NPU 出字）见 [构建指南「用例：HF → epair → OM」](docs/build_guide.md#用例hf--epair--om-端到端gemma4)，或一键跑 `examples/test_converter/export_llm_model/run_e2e_pipeline.sh`。

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
| [构建指南](docs/build_guide.md) | 前置条件 / 两种装法 / 统一 env / build.sh subcommand / 参数说明 / HF→epair→OM 用例 |
| [CLI 设计](docs/cann_lm_converter/cli/cli_design.md) | CLI 入口 / convert() / 命令行参数 / 调用链 |
| [导出设计](docs/cann_lm_converter/export/export_design.md) | loader / saver / ExportWrapper / C++ 加载 |
| [量化设计](docs/cann_lm_converter/quant/quant_design.md) | W4A8 方案 / dopt GPTQ / swap / 算子规格 |
| [converter 总览](docs/cann_lm_converter/converter_design.md) | 跨模块调用链 / 模型差异 / 精度汇总 |
