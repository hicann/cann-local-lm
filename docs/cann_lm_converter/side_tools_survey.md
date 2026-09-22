# 端侧 LLM 工具 CLI 与文档调研

> 调研 6 个端侧/转换 LLM 工具的 CLI 入口 / 命令风格 / 文档结构 / 命令参考 / 能力,对比 cann-lm-converter。
> 代码 clone 在 `/mnt/workspace/side_tools/`(shallow --depth 1,跳 LFS)。
>
> 调研日期: 2026-09-01 | 关联: [CLI-013](./cleanup_notes.md#cli-013-todo端侧-llm-工具-cli-与文档提供方式调研)

## 目录

- [调研对象](#调研对象)
- [1. mlx-lm(Apple,Python+MLX)](#1-mlx-lmapplepythonmlx)
- [2. ollama(Go)](#2-ollamago)
- [3. executorch(PyTorch 端侧)](#3-executorchpytorch-端侧)
- [4. mlc-llm(MLC)](#4-mlc-llmmlc)
- [5. litert-lm(Google,Bazel)](#5-litert-lmgooglebazel)
- [6. coremltools(苹果)](#6-coremltools苹果)
- [对比表](#对比表6-工具--cann-lm)
- [cann-lm-converter 对比 + 启示](#cann-lm-converter-对比--启示)
- [关键差异:导出工具 vs 端侧运行时](#关键差异导出工具-vs-端侧运行时)

---

## 调研对象

| 工具 | 定位 | 语言/栈 | 代码路径 |
|------|------|---------|----------|
| mlx-lm | 全链端侧运行时(转换+量化+推理+serve+微调) | Python+MLX(Apple Silicon) | `/mnt/workspace/side_tools/mlx-lm/` |
| ollama | 全链端侧运行时(运行+serve+pull) | Go(cobra) | `/mnt/workspace/side_tools/ollama/` |
| executorch | 全链端侧部署引擎(导出+量化+推理+serve) | C++/Python(PyTorch) | `/mnt/workspace/side_tools/executorch/` |
| mlc-llm | 全链端侧部署引擎(compile+serve+多平台 SDK) | C++/Python(TVM) | `/mnt/workspace/side_tools/mlc-llm/` |
| litert-lm | 消费侧编排(消费预导出 .litertlm,推理+serve) | Python(click,Bazel) | `/mnt/workspace/side_tools/litert-lm/` |
| coremltools | 转换工具(PyTorch/TF→CoreML) | Python(C API) | `/mnt/workspace/side_tools/coremltools/` |

---

## 1. mlx-lm(Apple,Python+MLX)

### CLI 入口

**setup.py `console_scripts` 18 个**(点分多 entry + 主 dispatcher):

| console_script | 指向 |
|---|---|
| `mlx_lm` | `mlx_lm.cli:main`(主调度器) |
| `mlx_lm.generate`/`.chat`/`.convert`/`.server`/`.cache_prompt`/`.benchmark`/`.evaluate`/`.perplexity` | `mlx_lm.<cmd>:main` |
| `mlx_lm.lora`/`.fuse` | `mlx_lm.<cmd>:main`(微调) |
| `mlx_lm.awq`/`.dwq`/`.dynamic_quant`/`.gptq` | `mlx_lm.quant.<cmd>:main`(学习量化) |
| `mlx_lm.manage`/`.upload`/`.share` | `mlx_lm.<cmd>:main`(模型管理) |

`mlx_lm/cli.py`(53 行)是**手写子命令分发器**:`sys.argv.pop(1)` 取子命令 + `importlib.import_module(f"mlx_lm.{subcommand}")` 动态加载 + 调 `submodule.main()`。**非** argparse subparsers/typer/click。

### 命令风格

**双轨制**(每命令两种形式,行为一致):
- 点分多 entry:`mlx_lm.generate --prompt "..."`
- 子命令:`mlx_lm generate --prompt "..."`(经 cli.py 分发)

README 主推点分形式。

### 17 个命令清单

| 命令 | 能力 | 模块 |
|------|------|------|
| generate | 一次性推理 | `mlx_lm/generate.py` |
| chat | 交互式 REPL | `mlx_lm/chat.py` |
| convert | HF→MLX 转换+基础量化+上传 | `mlx_lm/convert.py` |
| server | OpenAI 兼容 HTTP API | `mlx_lm/server.py` |
| cache_prompt | 长上下文 prompt 缓存 | `mlx_lm/cache_prompt.py` |
| benchmark | tok/sec+内存基准 | `mlx_lm/benchmark.py` |
| evaluate | MMLU Pro 评测(lm-eval) | `mlx_lm/evaluate.py` |
| perplexity | 困惑度 | `mlx_lm/perplexity.py` |
| lora | LoRA/QLoRA/DoRA/full 微调 | `mlx_lm/lora.py` + `tuner/` |
| fuse | 融合 adapter+GGUF 导出 | `mlx_lm/fuse.py` |
| awq/dwq/dynamic_quant/gptq | 4 种学习量化 | `mlx_lm/quant/` |
| manage | HF 缓存扫描/删除 | `mlx_lm/manage.py` |
| upload | 上传 HF Hub | `mlx_lm/upload.py` |
| share | 本地分享端点 | `mlx_lm/share.py` |

### 文档结构

**无独立 docs 站点**(无 mkdocs/sphinx)。两层:
- 顶层 `README.md`(290 行):Quick Start / Python API / Streaming / Sampling / Command Line / Long Prompts / Supported Models / Large Models
- `mlx_lm/` 下按特性分文件(非按命令):
  - `SERVER.md`(server OpenAI API 字段表)
  - `LORA.md`(lora+fuse 微调/数据格式/GGUF)
  - `LEARNED_QUANTS.md`(awq/dwq/dynamic_quant/gptq 对比/调参)
  - `BENCHMARKS.md`(benchmark/evaluate)
  - `MANAGE.md`(manage)
- `mlx_lm/examples/`(9 文件:generate_response.py/chat.py/batch_generate/sharded/tool_use 等)

### 命令参考

**`--help` 为权威**(argparse `help=`),README 反复引导 "`mlx_lm.<cmd> --help`"。关键命令有独立 `.md` 含**手写参数表**(SERVER.md Request/Response Fields)。**无自动生成参数表**。全部 18 个 `main()` 用标准库 `argparse`,无第三方 CLI 框架。

### 主要能力

| 能力 | 实现 |
|------|------|
| 转换 | `convert`(HF→MLX,`-q` 基础量化,`--upload-repo`) |
| 量化 | 基础(`convert -q`)+ 4 种学习量化(awq/dwq/dynamic_quant/gptq)+ KV cache 量化(`--kv-bits`) |
| 推理 | `generate`(batch)/`chat` REPL/`cache_prompt`/推测解码(`--draft-model`)/旋转 KV(`--max-kv-size`)/prefill 分步 |
| serve(OpenAI API) | `server`(port 8080,`/v1/chat/completions`,streaming/logprobs/draft_model) |
| 微调 | `lora`(LoRA/QLoRA/DoRA/full,YAML config,wandb)+ `fuse`(融合+GGUF) |
| 评测 | `benchmark`/`evaluate`(MMLU Pro)/`perplexity` |
| 管理 | `manage`/`upload`/`share` |
| GGUF | `fuse --export-gguf`(Mistral/Llama fp16) |
| 分布式 | `mx.distributed`(`examples/sharded_generate.py`) |
| 工具调用 | `mlx_lm/tool_parsers/`(13 parser:pythonic/gemma4/glm47 等) |

### 关键文件

- `setup.py`(18 console_scripts)
- `mlx_lm/__main__.py`(`python -m mlx_lm`)
- `mlx_lm/cli.py`(主 dispatcher)
- `mlx_lm/__init__.py`(Python API:load/generate/stream_generate/convert)
- `mlx_lm/{generate,chat,convert,server,cache_prompt,benchmark,evaluate,perplexity,lora,fuse,manage,upload,share}.py`
- `mlx_lm/quant/{awq,dwq,dynamic_quant,gptq}.py`
- `mlx_lm/tuner/`(lora/dora/trainer/datasets)
- `mlx_lm/models/`(50+ 模型)
- `mlx_lm/tool_parsers/`(13 parser)
- `README.md` + `mlx_lm/{SERVER,LORA,LEARNED_QUANTS,BENCHMARKS,MANAGE}.md`
- `mlx_lm/examples/`(9 文件)

### 参数表 + 代码细节（深入调研 2026-09-01）

**代码细节**：
- `setup.py:52-72` console_scripts 18 个：1 个 dispatcher `mlx_lm = mlx_lm.cli:main` + 17 个点分 entry 直指各模块 `main`（4 个 quant 命令 `mlx_lm.awq/dwq/dynamic_quant/gptq` 指向 `mlx_lm.quant.<cmd>:main`）
- `cli.py:7-53` 主 dispatcher：`sys.argv.pop(1)` 取子命令（修改 argv 透传剩余）+ `importlib.import_module(f"mlx_lm.{subcommand}")` 按需动态加载 + `submodule.main()`；`subpackages` dict 把 4 个 quant 命令映射到 `quant` 子包；`--version`/`-h`/`--help` 特判；未命中 raise ValueError。**不用** argparse subparsers/click/typer
- `__main__.py:3-5`：`python -m mlx_lm` → `cli.main()`
- **双轨**：点分 entry（`mlx_lm.generate`）直达模块 main vs 子命令（`mlx_lm generate`）经 dispatcher，都调同一 `submodule.main()`，行为一致；README 主推点分
- **单命令 main**：`setup_arg_parser()`（argparse 定义独立函数，可复用）+ `main()` 调 `parser.parse_args()`（generate.py 示例）

#### `mlx_lm generate`（29 参数）

模型/加载：`--model` `--trust-remote-code` `--adapter-path` `--quantize-activations`/`-qa`

输入/prompt：`--prompt`/`-p`（`-` 读 stdin）`--system-prompt` `--prefill-response` `--extra-eos-token`(nargs+) `--ignore-chat-template` `--use-default-chat-template` `--chat-template-config`(JSON)

生成/采样：`--max-tokens`/`-m` `--temp` `--top-p` `--min-p` `--top-k` `--xtc-probability` `--xtc-threshold` `--min-tokens-to-keep` `--seed`

KV cache：`--max-kv-size`(旋转 KV) `--kv-bits` `--kv-group-size`(默认64) `--quantized-kv-start` `--prompt-cache-file` `--prefill-step-size`

推测解码：`--draft-model` `--num-draft-tokens`(默认3)

输出：`--verbose`(str2bool 默认True)

#### `mlx_lm convert`（11 参数，HF→MLX 转换+量化+上传）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--hf-path`/`--model` | str | — | HF 模型路径/repo（必填） |
| `--mlx-path` | str | mlx_model | MLX 保存路径 |
| `--upload-repo` | str | — | 上传 HF repo |
| `--trust-remote-code` | flag | False | 信任 remote code |
| `-q`/`--quantize` | flag | — | 生成量化模型 |
| `--q-group-size` | int | None | 量化 group size |
| `--q-bits` | int | None | 每权重位数 |
| `--q-mode` | str | affine | 量化模式：affine/mxfp4/nvfp4/mxfp8 |
| `--quant-predicate` | str | — | 混合位宽 recipe（choices=QUANT_RECIPES） |
| `-d`/`--dequantize` | flag | False | 反量化 |
| `--dtype` | str | None | 非量化参数类型（choices=MODEL_CONVERSION_DTYPES） |

#### `mlx_lm.awq`（11 参数，AWQ 学习量化）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--model`/`-m` | str | mlx-community/Qwen2.5-7B-Instruct-bf16 | 模型 |
| `--mlx-path` | str | mlx_model | 保存路径 |
| `--bits` | int | 4 | 权重位数 |
| `--group-size` | int | 64 | 权重 group size |
| `--embed-bits` | int | 4 | embedding 位数 |
| `--embed-group-size` | int | 32 | embedding group size |
| `--num-samples` | int | 128 | 校准样本数 |
| `--sequence-length` | int | 512 | 校准序列长度 |
| `--n-grid` | int | 20 | AWQ 搜索网格 |
| `--seed` | int | 123 | 随机种子 |
| `--trust-remote-code` | flag | — | 信任 remote code |

#### `mlx_lm.gptq`（10 参数，GPTQ 学习量化）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--model`/`-m` | str | <hf_model_id> | 模型 |
| `--mlx-path` | str | mlx_model | 保存路径 |
| `--bits` | int | 4 | GPTQ 层位数 |
| `--group-size` | int | 64 | GPTQ 层 group size |
| `--fallback-bits` | int | 6 | **非 GPTQ 层**位数 |
| `--fallback-group-size` | int | 64 | 非 GPTQ 层 group size |
| `--num-samples` | int | -1 | 校准样本数（-1 全部） |
| `--sequence-length` | int | 512 | 校准序列长度 |
| `--seed` | int | 123 | 随机种子 |
| `--trust-remote-code` | flag | — | 信任 remote code |

**awq vs gptq**：共有 8 个（`--model`/`--mlx-path`/`--bits`/`--group-size`/`--num-samples`/`--sequence-length`/`--seed`/`--trust-remote-code`）；awq 独有 `--embed-bits`/`--embed-group-size`（embedding 单独量化）+ `--n-grid`（搜索网格）；gptq 独有 `--fallback-bits`/`--fallback-group-size`（非 GPTQ 层 fallback 量化，混合位宽）。

**与 cann-lm-converter 对比**：mlx-lm awq/gptq 是独立量化命令（参数直接控 bits/group/embed/fallback），产 MLX 量化模型；cann-lm 把量化合到 `--quant-method gptq`（spawn dopt subprocess），量化参数（W4A8/group128/A8）由 dopt_config `Quant_INT8-4_ECO` strategy 决定，CLI 不暴露。mlx-lm 灵活（任意 bits/group），cann-lm 固定（W4A8 strategy）。

---

## 2. ollama(Go)

### CLI 入口

**Go binary(cobra)**,无 console_scripts:
- `main.go`(13 行):`cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))`
- `NewCLI()` 在 `cmd/cmd.go:2286`(注册所有子命令,`rootCmd.AddCommand(...)` 一次)
- 构建:`cmake -B build . && cmake --build build`(含 C++ native payload)或 `go build .`(纯 Go)

### 命令风格

**统一 `ollama <cmd>`(cobra)**,13 子命令 + launch 集成 + 独立 bench:

| 子命令 | Use | Handler | 关键 flag |
|--------|-----|---------|-----------|
| `serve`(alias `start`) | serve | RunServer | 20+ `OLLAMA_*` env(appendEnvDocs) |
| `create` | create MODEL | CreateHandler | `-f/--file`,`-q/--quantize`,`--experimental` |
| `show` | show MODEL | ShowHandler | `--license/--modelfile/--parameters/--system` |
| `run` | run MODEL [PROMPT] | RunHandler | `--keepalive`,`--format`,`--think`,`--truncate` |
| `stop` | stop MODEL | StopHandler | — |
| `pull`/`push` | pull/push MODEL | Pull/PushHandler | `--insecure` |
| `signin`/`signout`(hidden,alias login/logout) | signin/signout | Signin/SignoutHandler | — |
| `list`(alias `ls`)/`ps` | list/ps | List/ListRunningHandler | — |
| `cp`/`rm` | cp/rm | Copy/DeleteHandler | — |
| `runner`/`gpu-discover`(hidden) | runner/gpu-discover | — | 透传 |
| `launch` | launch [INTEGRATION] | LaunchIntegration | `--model`,`--config`,`--yes` |

`launch` 集成 18+ 应用(claude/codex/codex_app/opencode/copilot/droid/deepseek_harness/cline/kimi/muse/pi/poolside/qwen/vscode/hermes/openclaw/omp/claude_desktop),每集成一个 `cmd/launch/<name>.go`。

独立 binary:`bench`(`cmd/bench/`,Go flag,非 cobra)、`runner`(`cmd/runner/`,llama.cpp runner 子进程)。

### 文档结构

- 仓库 `README.md`(356 行):Download / Get started / Coding / AI assistant / Chat / REST API / Python/JS / Backends / Documentation / Community Integrations(超长)
- `docs/`(Mintlify 驱动,部署 docs.ollama.com):
  - `docs.json`(站点配置,3 tab:Guide/Integrations/API Reference)
  - `cli.mdx`(CLI Reference,示例式)
  - `api.md`(54KB REST API)+ `openapi.yaml`(58KB)
  - `modelfile.mdx`(Modelfile Reference)
  - `docs/api/`(7 mdx:introduction/authentication/streaming/usage/errors/openai-compat/anthropic-compat)
  - `docs/capabilities/`(7 mdx:embeddings/streaming/structured-outputs/thinking/tool-calling/vision/web-search)
  - `docs/integrations/`(27 mdx:claude-code/codex/opencode 等)
  - `development.md`/`quickstart.mdx`/`faq.mdx`/`gpu.mdx`/`cloud.mdx` 等

### 命令参考

三层互补:
1. `--help`(cobra 自动):`serve` 追加 20+ `OLLAMA_*` env(`OLLAMA_HOST`/`OLLAMA_DEBUG`/`OLLAMA_CONTEXT_LENGTH`/`OLLAMA_KEEP_ALIVE`/`OLLAMA_NUM_PARALLEL`/`OLLAMA_FLASH_ATTENTION` 等)
2. `docs.ollama.com/cli`(`docs/cli.mdx`):示例驱动(### Run a model → 代码块)
3. README 内联(`ollama run gemma4`/`ollama launch claude`/`curl` 等)

### 主要能力

| 能力 | 实现 |
|------|------|
| 运行 | `ollama run MODEL [PROMPT]`(交互式 REPL,`cmd/interactive.go`) |
| serve(REST API) | `ollama serve`(OpenAI 兼容 `/v1/*` + Anthropic 兼容) |
| pull/push | registry 模型 |
| create(Modelfile) | `ollama create -f Modelfile`(`--quantize`/`--experimental`) |
| 嵌入 | `ollama run embeddinggemma`/`nomic-embed-text`(`--truncate`/`--dimensions`) |
| 多模态 | `ollama run gemma4 "What's in this image? /path.png"` |
| thinking | `run --think true|high|medium|low`/`--hidethinking` |
| 集成启动 | `ollama launch [INTEGRATION]`(Claude Code/Codex/OpenCode/Copilot/Droid 等) |
| cloud 模型 | `:cloud` 后缀 |
| Python/JS SDK | `ollama-python`/`ollama-js` |

### 关键文件

- `main.go`(Go binary 入口)
- `cmd/cmd.go`(`NewCLI()` + 13 Handler)
- `cmd/launch/`(`launch.go` + 18 集成 `*.go` + `registry.go`)
- `cmd/runner/main.go`(llama.cpp runner 子进程)
- `cmd/bench/bench.go`(独立基准)
- `cmd/interactive.go`(run REPL)/`cmd/agent_tui.go`(root 无参 TUI)
- `api/`(REST server)/`openai/`/`anthropic/`(兼容层)
- `README.md` + `docs/`(Mintlify 站)

---

## 3. executorch(PyTorch 端侧)

### CLI 入口

**pyproject.toml `[project.scripts]` 仅 `flatc`**(FlatBuffers 编译器),无顶层 dispatcher:
```
[project.scripts]
flatc = "executorch.data.bin:flatc"
```
所有工具以 **`python -m <module.path>`** 独立运行,无 `executorch <subcommand>` 聚合入口,无 `__main__.py`。每个工具自带 `argparse` + `if __name__ == "__main__": main()`。

C++ binary 分散构建(`examples/models/llama/main.cpp`/`extension/llm/runner/`),通过 `CMakeLists.txt` + `CMakePresets.json` + `Makefile` 构建到 `cmake-out/`。

### 命令风格

**混合:flat 单脚本 + Hydra 点分配置**:

| 风格 | 工具 | 说明 |
|------|------|------|
| `python -m module` + argparse | inspector_cli/diff_pte/size_analysis_tool/llm_server/install_executorch | 每个一个 ArgumentParser,无 subparsers |
| Hydra 点分 + YAML | `export_llm`(主 LLM 导出) | `base.model_class=llama3` `quantization.qmode=8da4w` `--config xxx.yaml` |
| Makefile target + CMake preset | C++ runner | `make llama-cpu`/`make llama-cuda-debug` |

主要"命令"(独立模块):

| 命令 | 入口 | 风格 |
|------|------|------|
| export(LLM) | `python -m executorch.extension.llm.export.export_llm` | Hydra + `--config` |
| export(Llama legacy) | `python -m executorch.examples.models.llama.export_llama` | argparse + Hydra 双模 |
| serve | `python -m executorch.examples.llm_server.python.server` | argparse(FastAPI) |
| finetune | `python examples/llm_pte_finetuning/runner.py` | argparse + torchtune YAML |
| inspect | `python -m executorch.devtools.inspector.inspector_cli` | argparse |
| diff pte | `python -m executorch.devtools.pte_tool.diff_pte` | 位置参数 |
| size analysis | `python -m executorch.devtools.size_analysis_tool.size_analysis_tool` | argparse |
| install | `python install_executorch.py` | argparse |
| flatc(唯一 console_script) | `flatc` | — |

无 `convert`/`run` 顶层动词。"导出"=export,"运行"=runner(C++)/server,"转换"通过 export_llm 内部 7 步(load→quantize→export→partition→.pte)。

### 文档结构

**Sphinx 为主**(`docs/source/`)+ README + 各目录 README + `.claude`/`.wiki`:

```
README.md                    # 根(Quick Start 3 步 + 平台表 + 模型列表)
docs/
├── README.md                # Sphinx 构建说明
├── Makefile                 # sphinx
  source/
    ├── index.md             # toctree(intro/quick-start/edge/backends/llm/advanced/tools/api)
    ├── *.md (60+ 文件, flat) # 按 section 组织
    ├── llm/                 # LLM 专题(11 篇:export-llm/run-with-c-plus-plus/run-on-android 等)
    ├── backends/            # 各 backend 文档
    └── tutorials_source/   # 教程
examples/                    # 端到端用例(按主题分)
    ├── models/              # 模型动物园(60+:llama/gemma3/whisper/stable_diffusion 等)
    ├── portable/ selective_build/ devtools/ demo-apps/
    ├── xnnpack/ arm/ qualcomm/ samsung/ cadence/ apple/ mediatek/ nxp/
    └── llm_server/ llm_pte_finetuning/
.claude/  .wiki/             # skills 参考 + tribal knowledge(2200+ issue 综合)
```

### 命令参考

**无统一命令参考页**,分散:
1. `--help`(argparse 自动)
2. 各工具 README(`extension/llm/export/README.md` Hydra 用法 + YAML;`examples/llm_server/python/README.md` server flag;`examples/llm_pte_finetuning/README.md` 微调;`extension/llm/runner/README.md` 33KB C++ runner API)
3. `docs/source/llm/export-llm.md`(LLM 导出教程式)
4. `Makefile help` target(C++ runner 构建 target)

参数表风格:Hydra 工具用 YAML + 点分 CLI 覆盖(schema 在 `LlmConfig` dataclass);普通工具经典 argparse;LLM server kebab-case flag。

### 主要能力

| 能力 | 实现 |
|------|------|
| 导出(torch.export→edge→.pte) | `extension/llm/export/`(export_llm.py / builder.py LLMEdgeManager / export_passes.py / partitioner_lib.py);7 步:实例化→载权→transform/quantize→torch.export→pt2e→partition→.pte |
| 量化 | `extension/llm/export/quantize.py`(双路径:TorchAO source-level `8da4w`/`int8` + pt2e graph-level QNN/CoreML/Vulkan) |
| 推理(C++ runtime) | `extension/llm/runner/`(TextLLMRunner/MultiModalRunner + pybindings `Runtime.get()`;KV cache/attention sink/dynamic shape) |
| serve(OpenAI 兼容) | `examples/llm_server/`(FastAPI 控制面 + C++ worker JSONL;`/v1/chat/completions` 流式 + 工具调用 Hermes/Qwen XML) |
| 微调 | `examples/llm_pte_finetuning/`(torchtune LoRA,先 .pte checkpoint→训练,CPU/fp32) |
| 模型动物园 | `examples/models/`(60+:llama/gemma3/phi_4_mini/llava/whisper/stable_diffusion 等) |
| DevTools | `devtools/`(inspector/etdump/etrecord/pte_tool/size_analysis/visualization/bundled_program) |
| 多 backend | `backends/`(xnnpack/coreml/qnn/mediatek/samsung/arm/cadence/nxp/vulkan/openvino/cuda/metal/mlx) |
| 跨平台 SDK | `extension/apple/`/`extension/android/` + Swift Package |

### 关键文件

- `pyproject.toml`(`[project.scripts]` 仅 flatc)
- `setup.py`(130KB 动态依赖 + wheel)
- `CMakeLists.txt`(63KB)/`CMakePresets.json`/`Makefile`
- `install_executorch.py`(安装 CLI)
- `extension/llm/export/export_llm.py`(Hydra 主)/`builder.py`(LLMEdgeManager)/`config/llm_config.py`(LlmConfig schema)
- `extension/llm/runner/`(C++ runtime + pybindings + README 33KB)
- `examples/llm_server/python/server.py`(FastAPI serve)
- `examples/llm_pte_finetuning/`(微调)
- `devtools/`(inspector_cli/diff_pte/size_analysis)
- `docs/source/`(Sphinx 60+ md)+ `examples/models/`(60+ 模型)

---

## 4. mlc-llm(MLC)

### CLI 入口

**无 `[project.scripts]`**(`pyproject.toml` 全文搜索 `project.scripts`/`console_scripts`/`entry_points` 均无)。`mlc_llm` 命令来自 **alias**:
```bash
export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
alias mlc_llm="python -m mlc_llm"
```
文档兜底:"command not found: mlc_llm, try `python -m mlc_llm --help`"。

真正入口:`python/mlc_llm/__main__.py`(67 行)— **手写 argv[1] 分发**:`ArgumentParser` 解析 `sys.argv[1:2]` 取 subcommand(choices 硬编码 8 个),`if/elif` 延迟 import `mlc_llm.cli.<name>` 调 `main(sys.argv[2:])`。**非** typer/click/argparse subparsers。

C++ `CMakeLists.txt` 只建库(`libmlc_llm.so`/`mlc_llm_static`/`mlc_llm_module` + 可选 `mlc_llm_cpp_tests`),**无 CLI binary**。

### 命令风格

**子命令 `mlc_llm <cmd>`(argparse 手写)**,8 个:

| 子命令 | 作用 | 关键参数 |
|--------|------|---------|
| `compile` | 编译模型→model lib(.so/.dylib/.dll/.tar/.wasm) | `model`/`--quantization`/`--model-type`/`--device`/`--host`/`--opt`/`--output` |
| `convert_weight` | HF 权重→MLC 量化格式(params_shard_*.bin+tensor-cache.json) | `config`/`--quantization`/`--source`/`--source-format`/`--output`/`--lora-adapter` |
| `gen_config` | 生成 mlc-chat-config.json + tokenizer | `config`/`--quantization`/`--conv-template`/`--context-window-size`/`--output` |
| `chat` | 交互式 CLI 对话 | `model`/`--device`/`--model-lib`/`--overrides` |
| `serve` | OpenAI 兼容 REST server | `model`/`--device`/`--mode`(local/interactive/server)/`--speculative-mode`/`--prefix-cache-mode`/`--host`/`--port` |
| `package` | 打包 iOS/Android model lib+权重 | `--package-config`/`--output` |
| `calibrate` | 量化校准(ShareGPT) | `model`/`--model-lib`/`--dataset`/`--num-calibration-samples` |
| `router` | disaggregation 路由 | `model`/`--router-mode`(disagg/round-robin)/`--pd-balance-factor` |

辅助:`python -m mlc_llm.bench`(基准)/`cli/delivery`(HF 持续交付)/`cli/worker`(Disco 分布式)。

帮助文案集中 `interface/help.py`(HELP 字典,273 行),每参数 help 串接 `HELP[key]`。实际实现委派 `mlc_llm.interface.<cmd>`,CLI 层只解析+类型自动检测(`auto_config`/`auto_target`/`auto_device`)。

### 文档结构

**三层**:

| 层 | 路径 | 技术 | 部署 |
|----|------|------|------|
| README | `README.md` | Markdown | GitHub |
| Sphinx docs | `docs/` | Sphinx + RST + `sphinx_rtd_theme` | `llm.mlc.ai/docs` |
| Jekyll site | `site/` | Jekyll + `jekyll-theme-mlc` | `llm.mlc.ai`(首页) |

`docs/` toctree(`index.rst`):
- `get_started/`(quick_start/introduction)
- `install/`(mlc_llm/tvm/conda/gpu/emcc)
- `compilation/`(compile_models 1092 行/convert_weights/package_libraries/configure_quantization/define_new_models)
- `deploy/`(cli/rest/python_engine/webllm/ios/android/ide_integration/mlc_chat_config)
- `microserving/tutorial.rst`(跨引擎 disaggregation)
- `community/`(guideline/faq)

### 命令参考

- `--help`:每子命令独立 ArgumentParser,`mlc_llm <cmd> --help` 输出 usage;顶层 `mlc_llm --help` 列 8 子命令
- docs site RST 参数表(`docs/deploy/cli.rst` chat / `docs/deploy/rest.rst` serve / `docs/compilation/compile_models.rst` compile/convert_weight/gen_config 内嵌):**手写 RST 表格**,位置参数大写占位符(`MODEL`/`CONFIG`)+ `--flag`

### 主要能力

| 能力 | 实现 |
|------|------|
| compile(TVM 编译) | `cli/compile.py` + `interface/compile.py`;平台 cuda/metal/vulkan/rocm/opencl/webgpu/iphone/android/auto;cross-compile(`--host` LLVM triple);`--opt` O0-O3 + 细粒度 knob;`--system-lib-prefix` |
| convert_weight | `cli/convert_weight.py`;source-format huggingface-torch/safetensor/awq;量化 AWQ/GroupQuantize/FTQuantize/FP8/BlockScale/PerTensor/NoQuantize;`--lora-adapter`(PEFT 合并) |
| gen_config | `cli/gen_config.py`;conv-template/context-window/sliding-window/prefill-chunk/tensor-parallel/pipeline-parallel/disaggregation |
| serve(OpenAI 兼容) | `cli/serve.py`(264 行);mode local/interactive/server;speculative(small_draft/eagle/medusa);prefix cache(radix 树);hybrid/chunked prefill;tensor parallel;embedding model;API key;CORS |
| chat | `cli/chat.py`;REPL `/help`/`/exit`/`/stats`/`/metrics`/`/set temperature=0.5` |
| package | `cli/package.py`;iOS/Android model lib+权重 bundle |
| calibrate | `cli/calibrate.py`;ShareGPT 校准 |
| router | `cli/router.py`;prefill-decode disaggregation |
| bench | `python -m mlc_llm.bench`;固定并发/请求速率;TTFT;CUDA profile;CSV |
| JIT 编译 | `interface/jit.py`;`--model-lib` 未指定时自动 JIT(`MLC_JIT_POLICY` ON/OFF/REDO/READONLY) |
| 多平台 SDK | `ios/`(MLCChat+MLCSwift)/`android/`(MLCChat+mlc4j)/`web/`(WebLLM npm) |

### 关键文件

- `pyproject.toml`(无 [project.scripts],scikit-build-core)
- `python/mlc_llm/__main__.py`(CLI 总入口,argv[1] 分发)
- `python/mlc_llm/cli/`(8 子命令 + delivery/lib_delivery/model_metadata/worker)
- `python/mlc_llm/interface/`(实际实现 + help.py HELP 字典 + jit.py)
- `python/mlc_llm/support/argparse.py`(增强 ArgumentParser)
- `CMakeLists.txt`(只建库,无 CLI binary)
- `cpp/`(C++ 引擎:serve/metadata/json_ffi/tokenizers/multi_gpu)
- `docs/`(Sphinx RST)+ `site/`(Jekyll)
- `python/mlc_llm/model/`(45 模型架构)+ `quantization/`(7 量化算法)
- `ios/`/`android/`/`web/`(平台 SDK)

---

## 5. litert-lm(Google,Bazel)

### CLI 入口

**Bazel Python binary(click)**,非 C++:
- Bazel target:`//python/litert_lm_cli:litert-lm`(`pytype_strict_binary`,`main = "main.py"`)
- PyPI wheel `litert-lm`,`console_scripts` `litert-lm = litert_lm_cli.main:main`(`BUILD:301-305`)
- `python/litert_lm_cli/main.py` → `main()` → `cli()` click group

命令注册:**模块级 `register(cli)` + `cli.add_command()`**:
```python
# main.py
_serve_module.register(cli); _convert_module.register(cli); _list_module.register(cli)
_import_module.register(cli); _describe_module.register(cli); _delete_module.register(cli)
_rename_module.register(cli); _benchmark_module.register(cli); _run_module.register(cli)
_pack_module.register(cli); _unpack_module.register(cli)
```
共享 flag 用 `common.py` 装饰器工厂(`common_inference_options`/`config_option`/`huggingface_options`)。

### 命令风格

**单 binary + 子命令(click Group)**,11 个分 3 组(`help_formatter.py` COMMAND_SECTIONS):

| 分组 | 命令 | 文件 |
|------|------|------|
| Inference | `run`/`benchmark`/`serve` | `commands/run.py`/`benchmark.py`/`serve.py` |
| Model Management | `list`/`import`/`describe`/`rename`/`delete` | `commands/list.py`/`import.py`/`describe.py`/`rename.py`/`delete.py` |
| File Commands | `pack`/`unpack`/`convert`(stub) | `commands/pack.py`/`unpack.py`/`convert.py` |

各命令:
- `run`:推理(`--prompt`/`--backend` cpu|gpu|npu/`--speculative-decoding`/`--attachment` 多模态/`--thinking`/`--top-k/p/temperature`/`--cache`/`--android` ADB)
- `benchmark`:prefill/decode 计时(`--runs`,Android ADB)
- `serve`:HTTP(OpenAI/Gemini 兼容,`openai_handler.py`/`gemini_handler.py`)
- `list`/`import`/`describe`/`rename`/`delete`:模型管理(`~/.litert-lm`)
- `pack`/`unpack`:`.litertlm` 容器打包/拆包(`litert_lm_builder`)
- `convert`:**stub 未实现**(28 行,未 add_command)

### 文档结构

- `README.md`(产品总览,Release v0.7~v0.16)
- `docs/`(`README.md` 索引):
  - `getting-started/`(`build-and-run.md` Bazel 7.6.1 Win/Linux/Mac/Android;`cmake.md` 实验性)
  - `api/cpp/`(conversation/constrained-decoding/tool-use/tool-use-antlr)
  - `api/kotlin/`(getting_started.md)
  - `litert_lm_builder.md`(builder 库 CLI+Python API)
- 各 Python 包 README(指外链 `ai.google.dev/edge/litert-lm/*`)
- **完整 CLI 命令参考外链** `ai.google.dev/edge/litert-lm/cli`,仓库内只构建/API 指南

### 命令参考

- `--help`/`-h`(click `ColorHelpFormatter` 绿色标题/青色 usage,`show_default=True`,`max_content_width=120`)
- flag:`@click.option` 装饰器,`textwrap.dedent` 多行 help;`click.Choice`(`--backend` cpu|gpu|npu);`is_flag`/`flag_value`;`click.IntRange/FloatRange`;`multiple=True`(`--attachment`/`--cors-origin`)
- `config.json`(JSON Schema `config_schema.json`,v0.15 centralized CLI configuration):flags > config.json > 引擎默认;`--config` 指定路径(默认 `~/.litert-lm/config.json`)
- 无独立参数表 md,`--help` + docstring Args 段(`run.py:608-650`)为准

### 主要能力

| 能力 | 实现 |
|------|------|
| 导出(litert-torch) | **不在 litert-lm CLI**(无 litert-torch/torch.export 引用);定位**消费侧**,消费上游预导出 .litertlm/.tflite;`convert` 是 stub |
| 打包 .litertlm | `pack`/`unpack` + `litert_lm_builder`(LitertLmFileBuilder:tflite+tokenizer+metadata)+ `litert-lm-peek` |
| 量化 | 消费侧(预量化产物);`--activation-data-type` fp32|fp16|int16|int8(实验,hidden) |
| 推理 | `run` + `litert_lm.Engine`/`Session`/`Conversation`;prefill-decode(`session.run_prefill`→`run_decode_async` 流式) |
| 后端 | `--backend` cpu|gpu|npu;`--cpu-thread-count`/`--gpu-decode-steps-per-sync`/`--ringbuffers-local-attention`(KV 省内存)/`--cache` disk|memory|no |
| speculative | `--speculative-decoding` MTP(Multi-Token Prediction)drafters(Gemma4) |
| 多平台 | `prebuilt/`(android_arm64/x86_64/ios_arm64/sim/macos/linux_arm64/windows);Android ADB(`adb_engine`/`adb_benchmark`);v0.16 C API prebuilt 全平台 |
| 多模态/Tool use | `run --attachment`/`--preset`;`ToolEventHandler` 函数调用;constrained decoding(C++ ANTLR) |
| serve | OpenAI/Gemini 兼容(`/v1/chat/completions`/`/v1/embeddings`) |

### 关键文件

- `python/litert_lm_cli/main.py`(CLI 主入口,click group,命令注册)
- `python/litert_lm_cli/BUILD`(litert-lm binary/wheel target)
- `python/litert_lm_cli/{common,help_formatter,config,model}.py`(共享 flag/彩色 help/config 解析)
- `python/litert_lm_cli/commands/{run,benchmark,serve,list,import,describe,rename,delete,pack,unpack,convert}.py`
- `python/litert_lm/`(Engine/Session/Conversation/EmbeddingEngine/Benchmark)
- `python/litert_lm_builder/`(LitertLmFileBuilder/litertlm_peek)
- `docs/` + 各包 README
- `prebuilt/`(全平台 C API)
- `WORKSPACE`/`.bazelversion`/`.bazelrc`(Bazel)

---

## 6. coremltools(苹果)

### CLI 入口

**无 CLI**(纯 Python API):
- 无 `pyproject.toml`(仓库根)
- `setup.py`(115 行)只 `packages=find_packages()`+`install_requires`+`package_data`,**无 `entry_points`**
- 无 `__main__.py`(`coremltools/` 下 glob 确认)
- 无 argparse/click/typer(grep 零命中)
- `MANIFEST.in`(1 行 `include README.md`)
- `CMakeLists.txt`(285 行)建 C++ 扩展(`libmilstoragepython.so`/`libmodelpackage.so`/`libcoremlpython.so` 仅 macOS + `kmeans1d`),**无 CLI target**

### 命令风格

**纯 Python API**:`import coremltools as ct; ct.convert(...)`:
- `coremltools/__init__.py:129` → `from .converters import ... convert`
- `coremltools/converters/_converters_entry.py:69` `convert(model, source="auto", inputs=None, outputs=None, classifier_config=None, minimum_deployment_target=None, convert_to=None, compute_precision=None, skip_model_load=False, compute_units=_ComputeUnit.ALL, package_dir=None, debug=False, pass_pipeline=None, states=None)`(14 kwarg)
- 顶层 API:`ct.ComputeUnit`/`ct.target`(macOS13/iOS15)/`ct.precision`(FLOAT16/FLOAT32)/`ct.transform`(FP16ComputePrecision)/`ct.PassPipeline`/`ct.TensorType`/`ct.ImageType`/`ct.Shape`/`ct.RangeDim`/`ct.EnumeratedShapes`/`ct.ClassifierConfig`
- source: auto/tensorflow/pytorch/milinternal;convert_to: mlprogram/neuralnetwork/milinternal

### 文档结构

**三套 Sphinx 站点**:

| 站点 | 源码 | 主题 | 部署 |
|------|------|------|------|
| API Reference | `docs/` | `sphinx_rtd_theme` + autodoc/numpydoc/sphinx_gallery | `gh-pages/` 根 |
| Guide & Examples | `docs-guides/` | `sphinx_book_theme` + myst_parser(Markdown) + sphinx_copybutton | `gh-pages/docs-guides/` |
| Core ML Format Spec | `mlmodel/docs/` | protobuf 规范 | `gh-pages/mlmodel/` |

关键文档:
- `README.md`(安装/能力/资源)
- `BUILDING.md`(CMake/conda/zsh,`scripts/build.sh`)
- `docs-guides/source/`(约 90 篇:convert-pytorch/convert-tensorflow/opt-quantization/opt-palettization/opt-pruning/opt-joint-compression/flexible-inputs/stateful-models 等)
- `docs/source/`(autodoc stub:`coremltools.converters.convert.rst` 等)
- `docs/examples/optimize/torch/`(sphinx_gallery 可运行:pruning/palettization/quantization)
- `examples/README.md`(仅链接外部 readme.io)

### 命令参考

**无 `--help`**(无 CLI)。参数文档两种:
1. `convert()` numpy docstring(`_converters_entry.py:85-1190`,1100+ 行)— Sphinx autodoc + numpydoc 渲染 API 页(Parameters 段逐参数 `name : type (optional)` + 代码片段)
2. Guide Markdown 散文(`docs-guides/source/convert-*.md`/`conversion-options.rst`)讲工作流

### 主要能力

| 能力 | 实现 |
|------|------|
| 转换 PyTorch→CoreML | `converters/mil/frontend/torch/`(TorchScript + torch.export EDGE dialect;`source="pytorch"`) |
| 转换 TensorFlow→CoreML | `converters/mil/frontend/tensorflow/`(TF1.x frozen/`.pb`/`.h5`/SavedModel + TF2.x keras/`.h5`/concrete;`source="tensorflow"`) |
| 非神经网络 | `converters/{libsvm,sklearn,xgboost,lightgbm}/` |
| ML Program/NeuralNetwork | `convert_to`/`minimum_deployment_target`(iOS15+ mlprogram,旧 neuralnetwork;milinternal 调试) |
| 量化/调色板/剪枝(CoreML 侧) | `optimize/coreml/`(OpLinearQuantizerConfig/OpPalettizerConfig/OpMagnitudePrunerConfig + linear_quantize_weights/palettize_weights/prune_weights/decompress_weights/get_weights_metadata;MIL graph passes) |
| 量化/调色板/剪枝(PyTorch 训练侧) | `optimize/torch/`(base_model_optimizer/layerwise_compression/palettization/pruning/quantization/optimization_config,可选安装) |
| compute_precision | `convert(compute_precision=...)` + `ct.transform.FP16ComputePrecision(op_selector=...)` |
| 图 pass 管线 | `converters/mil/mil/passes/pass_pipeline.py` → `ct.PassPipeline`(EMPTY/CLEANUP/DEFAULT_PRUNING/DEFAULT_PALETTIZATION,remove_passes/set_options) |
| 预测/验证(仅 macOS) | `models/model.py` + `coremlpython/`(Obj-C++ 桥,调 Core ML framework);`skip_model_load=True` 非 macOS 跳加载只导出 |
| iOS 部署 | `modelrunner/`(Xcode 工程 `ModelRunner.xcworkspace`);产物 `.mlpackage`/`.mlmodel` Xcode 集成 |

### 关键文件

- `setup.py`(无 console_scripts)
- `CMakeLists.txt`/`Makefile`(C++ 扩展构建,`scripts/*.sh`)
- `coremltools/__init__.py`(顶层 API:convert/ComputeUnit/target/precision/transform/PassPipeline/类型描述符)
- `coremltools/converters/_converters_entry.py`(`convert()` 1190 行,docstring 命令参考)
- `coremltools/converters/mil/`(MIL + frontend torch/tf + backend milprogram/neuralnetwork + graph passes)
- `coremltools/optimize/{coreml,torch}/`(量化/调色板/剪枝)
- `coremltools/models/`(MLModel 类/ml_program/neural_network/utils/compute_plan)
- `coremltools/modelrunner/`(iOS 验证 app)
- `coremlpython/`/`milstoragepython/`/`mlmodel/`/`modelpackage/`(C++/Obj-C++ 原生模块)
- `docs/`(API Sphinx)+ `docs-guides/`(Guide Sphinx)+ `mlmodel/docs/`(Format Spec)

---

## 对比表(6 工具 + cann-lm)

| 维度 | mlx-lm | ollama | executorch | mlc-llm | litert-lm | coremltools | **cann-lm-converter** |
|------|--------|--------|-----------|---------|-----------|-------------|----------------------|
| **entry 注册** | console_scripts 18 个(setup.py) | Go binary(cobra) | console_scripts 仅 flatc | 无(alias python -m) | console_scripts 1 个(click) | 无(纯 API) | **console_scripts 1 个 ✅**(pyproject `[project.scripts]`) |
| **命令风格** | 双轨(点分+子命令,手写 dispatcher) | 统一 `ollama <cmd>`(cobra) | 分散 `python -m`(Hydra+argparse) | 子命令(argparse 手写) | 子命令(click Group) | `ct.convert()` 顶层函数 | **单命令 argparse**(无子命令) |
| **CLI 框架** | argparse(手写 dispatch) | cobra(Go) | argparse + Hydra | argparse(手写 dispatch) | click | 无 | argparse |
| **文档结构** | README + 特性 md(无 docs site) | README + Mintlify docs site | Sphinx docs/+examples/+.claude/+.wiki | README + Sphinx RST + Jekyll site | README + docs(外链 CLI) | 三 Sphinx 站(API/Guide/Spec) | **docs/ 分目录** + README 导航 + INDEX |
| **命令参考** | `--help` + 手写 md | `--help`(cobra)+ docs site | 各 README + 教程 + Makefile help | `--help` + RST 参数表 | `--help`(彩色)+ config.json | docstring(autodoc) | **cli/INDEX ✅** + cli_design 参数表 |
| **参数表风格** | 手写(argparse help=) | cobra 自动 + env 追加 | Hydra YAML + dataclass schema | 手写 RST(位置参数大写) | click 自动 + JSON Schema | numpy docstring(autodoc) | 手写(argparse help= + markdown 表) |
| **能力范围** | 转换+4 量化+推理+serve+LoRA+管理+GGUF | 运行+serve+pull+create+集成+多模态 | 导出+量化+推理+serve+微调+DevTools+12 backend | compile+convert+serve+chat+多平台 SDK | 消费侧 pack/unpack+推理+serve | 转换+量化+调色板+剪枝+iOS | **仅 convert**(导出 epair) |
| **环境变量** | — | `OLLAMA_*`(20+,serve --help 追加) | — | — | config.json(JSON Schema) | — | env 表(cli_design,无 `CANN_LM_ARG_*` 风格) |
| **定位** | 全链端侧运行时 | 全链端侧运行时 | 全链端侧部署引擎 | 全链端侧部署引擎 | 消费侧编排 | 转换工具 | **转换工具**(导出 epair 供 C++ delegate) |

---

## cann-lm-converter 对比 + 启示

### 现状对照

cann-lm-converter 是**转换工具**(导出 epair 供 C++ delegate 加载),定位接近 **coremltools**(转换/导出),非 ollama/mlc-llm 全链运行时。cann-lm-engine(独立 `scripts/cann_lm_engine/pyproject.toml`)承担 serve/推理。

| cann-lm 现状 | 6 工具共性 | 评估 |
|--------------|-----------|------|
| `cann-lm-converter` console_scripts ✅ | mlx-lm/executorch/litert-lm 用 console_scripts;ollama Go binary;mlc-llm alias;coremltools 无 | **规范**(比 mlc-llm alias/coremltools 无 CLI 更好,接近 mlx-lm/litert-lm) |
| 单命令 argparse(无子命令) | mlx-lm 双轨;ollama/mlc-llm/litert-lm 子命令;executorch 分散;coremltools API | **合理**(converter 单职责 convert;若扩展 serve/verify 再子命令化) |
| docs/ 分目录 + INDEX | llama.cpp docs/ + executorch Sphinx;mlx-lm 特性 md;ollama/mlc-llm docs site | **接近 llama.cpp/executorch**(可保留) |
| cli/INDEX 命令参考 ✅ | mlx-lm `--help`;ollama docs site;executorch 各 README;mlc-llm RST;litert-lm `--help`+config;coremltools docstring | **已有**(接近 mlx-lm/executorch) |
| 仅 convert | 6 工具多全链(转换+推理+serve+微调) | **合理**(cann-lm-engine 独立 serve/推理;converter 保持 convert) |

### 启示

1. **console_scripts `cann-lm-converter` ✅** 已注册(比 mlc-llm alias/coremltools 无 CLI 规范,接近 mlx-lm/executorch/litert-lm)
2. **单命令适合 converter 单职责**(convert 导出 epair);若扩展 serve/verify(像 mlx-lm/ollama/mlc-llm 全链),可子命令化:
   - mlx-lm 双轨(点分+子命令,手写 dispatcher)— 简单,但点分 entry 多
   - litert-lm click Group + `register(cli)` — 模块化,推荐
   - 参考 `cann-lm convert`/`cann-lm serve`/`cann-lm verify`
3. **docs/ 分目录接近 llama.cpp + executorch Sphinx**(可保留);可借鉴:
   - mlx-lm 特性 md(SERVER.md/LEARNED_QUANTS.md 风格,按特性分文件)
   - mlc-llm/ollama docs site(独立命令参考页,Mintlify/Sphinx)
4. **命令参考 cli/INDEX ✅** 已加;可借鉴:
   - llama.cpp `llama-gen-docs` 自动生成(`tools/cli/README.md` 从 `--help` 生成,避免手维护)
   - litert-lm `config.json`(JSON Schema,centralized CLI configuration)
   - ollama `OLLAMA_*` 环境变量(参数表标注 env,cann-lm 可 `CANN_LM_ARG_*` 风格)
5. **能力仅 convert**(端侧工具多全链):cann-lm-engine 独立(serve/推理,像 ollama/mlc-llm run/serve);converter 保持 convert 合理;若合并 CLI 参考 ollama/mlc-llm `convert`/`serve`/`run` 子命令

### 可选改进(低优先级,非阻塞)

| 改进 | 借鉴 | 优先级 |
|------|------|--------|
| 命令参考自动生成(`--help` → README) | llama.cpp `llama-gen-docs` | 中(避免手维护参数表) |
| 环境变量参数化(`CANN_LM_ARG_*`) | ollama `OLLAMA_*` | 低(env 表已够) |
| config.json CLI 配置 | litert-lm `config.json`(JSON Schema) | 低(converter 参数少) |
| 子命令化(若职责扩展) | litert-lm click `register(cli)` | 低(待 serve/verify 扩展) |
| 特性 md(SERVER.md 风格) | mlx-lm SERVER.md/LEARNED_QUANTS.md | 低(docs/ 分目录已覆盖) |

---

## 关键差异:导出工具 vs 端侧运行时

6 工具分两类:

| 类别 | 工具 | 定位 | cann-lm 对应 |
|------|------|------|-------------|
| **全链端侧运行时**(转换+推理+serve) | mlx-lm/ollama/executorch/mlc-llm/litert-lm | 转换+量化+推理+serve(+微调) | **cann-lm-engine**(独立,serve/推理) |
| **转换工具**(导出/转换,供下游运行时加载) | coremltools | PyTorch/TF→CoreML,无推理 | **cann-lm-converter**(HF→epair,供 C++ delegate 加载) |

cann-lm-converter 定位接近 **coremltools**(转换/导出),不是 ollama/mlc-llm 全链。所以:
- cann-lm-converter **单命令 + console_scripts** 合理(像 coremltools 的 `ct.convert()`,但 cann-lm 用 CLI 非 Python API)
- cann-lm-engine(独立 pyproject)承担 serve/推理(像 ollama/mlc-llm run/serve)
- 若 converter+engine 合并 CLI,参考 litert-lm click 子命令(`cann-lm convert`/`cann-lm serve`)

---

## 关联

- 遗留问题:[CLI-013](./cleanup_notes.md#cli-013-todo端侧-llm-工具-cli-与文档提供方式调研)(行动项全 ✅:子命令化评估保持单命令 + console_scripts 已注册 + 命令参考页 cli/INDEX 已加)
- cann-lm 打包:`scripts/cann_lm_converter/pyproject.toml`(console_scripts `cann-lm-converter`)+ `scripts/cann_lm_engine/pyproject.toml`(engine 独立)
- 设计文档:[cli_design.md](./cli/cli_design.md)「命令行参数 / 完整调用链」、[cli/INDEX.md](./cli/INDEX.md)「命令参考」
- 代码:`cann_lm/converter/cann_lm_converter.py`(CLI 入口)
